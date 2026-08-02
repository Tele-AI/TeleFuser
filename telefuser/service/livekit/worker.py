"""LiveKit worker lifecycle for TeleFuser stream pipelines."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator
from typing import Any, Protocol

from telefuser.service.api.stream_schema import StreamChunkMessage, StreamDoneMessage, serialisable_chunk
from telefuser.service.core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL, STREAM_MODE_SERVER_PUSH
from telefuser.utils.logging import logger

from .config import LiveKitServeConfig
from .data_protocol import TF_CONTROL_TOPIC, normalize_control_message
from .media_bridge import split_chunk_media
from .pipeline_adapter import LiveKitPipelineAdapter
from .room_client import LiveKitRoomClient, RoomClient
from .schemas import SessionStatus
from .session_registry import SessionRecord
from .token_service import LiveKitTokenService

_ROOM_DISCONNECT_TIMEOUT_SECONDS = 5.0
_CONTROLLER_JOIN_TIMEOUT_SECONDS = 60.0


class WorkerEventSink(Protocol):
    """Callbacks used by a worker to report lifecycle changes."""

    def on_worker_status(self, worker_id: str, status: str) -> None: ...
    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None: ...
    def on_session_status(self, session_id: str, status: SessionStatus, error: str | None = None) -> None: ...
    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None: ...
    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None: ...


class NullWorkerEventSink:
    """No-op worker event sink for tests and isolated worker use."""

    def on_worker_status(self, worker_id: str, status: str) -> None:
        return None

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        return None

    def on_session_status(self, session_id: str, status: SessionStatus, error: str | None = None) -> None:
        return None

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        return None

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        return None


class LiveKitWorker:
    """Owns one active LiveKit room and one active TeleFuser pipeline session."""

    def __init__(
        self,
        *,
        worker_id: str,
        config: LiveKitServeConfig,
        pipeline_file: str,
        token_service: LiveKitTokenService,
        event_sink: WorkerEventSink | None = None,
        pipeline_adapter: LiveKitPipelineAdapter | None = None,
        room_client: RoomClient | None = None,
        gpu_num: int = 1,
    ) -> None:
        self.worker_id = worker_id
        self.config = config
        self.pipeline_file = pipeline_file
        self.token_service = token_service
        self.event_sink = event_sink or NullWorkerEventSink()
        self.pipeline_adapter = pipeline_adapter or LiveKitPipelineAdapter()
        self.room_client = room_client or LiveKitRoomClient()
        self.gpu_num = gpu_num
        self._active_session_id: str | None = None
        self._pipeline_session_id: str | None = None
        self._stop_event = asyncio.Event()

    async def start(self, *, skip_validation: bool = False) -> None:
        """Load the stream pipeline owned by this worker."""
        self.pipeline_adapter.start(self.pipeline_file, skip_validation=skip_validation, gpu_num=self.gpu_num)
        self.event_sink.on_worker_status(self.worker_id, "idle")

    async def stop(self) -> None:
        """Stop the worker and close any active session."""
        self._stop_event.set()
        await self._close_active_session()
        await self.pipeline_adapter.aclose()
        self.event_sink.on_worker_status(self.worker_id, "stopped")

    async def run_session(self, record: SessionRecord) -> None:
        """Join a LiveKit room, create a pipeline session, and publish output chunks."""
        if self._active_session_id is not None:
            raise RuntimeError(f"Worker {self.worker_id} is already running a session")

        self._active_session_id = record.session_id
        self._stop_event.clear()
        error: str | None = None
        try:
            self.event_sink.on_worker_status(self.worker_id, "joining_room")
            self.event_sink.on_session_status(record.session_id, "joining_room")
            worker_token = self.token_service.create_token(
                identity=f"telefuser-{self.worker_id}",
                room_name=record.room_name,
                role="worker",
            )
            await self.room_client.connect(
                self.config.livekit_url,
                worker_token,
                lambda message, topic, identity: self._on_data_message(record, message, topic, identity),
            )
            await self.room_client.wait_for_participant(
                record.controller_identity,
                timeout_s=_CONTROLLER_JOIN_TIMEOUT_SECONDS,
            )

            self.event_sink.on_worker_status(self.worker_id, "starting_pipeline")
            self.event_sink.on_session_status(record.session_id, "starting_pipeline")
            if self.pipeline_adapter.stream_mode == STREAM_MODE_BIDIRECTIONAL:
                self._pipeline_session_id = self.pipeline_adapter.create_session(record.config)
                self.event_sink.on_pipeline_session(record.session_id, self._pipeline_session_id)
                chunks = self.pipeline_adapter.pull_chunks(self._pipeline_session_id)
            elif self.pipeline_adapter.stream_mode == STREAM_MODE_SERVER_PUSH:
                chunks = self.pipeline_adapter.stream_task(record.config)
            else:
                raise RuntimeError(f"Unsupported stream mode: {self.pipeline_adapter.stream_mode}")

            self.event_sink.on_worker_status(self.worker_id, "running")
            self.event_sink.on_session_status(record.session_id, "running")
            await self.room_client.publish_status(
                StreamChunkMessage(
                    session_id=record.session_id,
                    data={
                        "type": "status",
                        "stage": "worker_running",
                        "worker_id": self.worker_id,
                    },
                ).model_dump(mode="json")
            )
            await self._publish_pipeline_chunks(record.session_id, chunks)
        except asyncio.CancelledError:
            error = "cancelled"
            raise
        except Exception as exc:
            error = str(exc)
            logger.exception(f"LiveKit worker failed: worker={self.worker_id} session={record.session_id}")
            self.event_sink.on_session_status(record.session_id, "failed", error=error)
            with contextlib.suppress(Exception):
                await self.room_client.publish_status(
                    StreamChunkMessage(
                        session_id=record.session_id,
                        error=error,
                        data={
                            "type": "error",
                            "error": error,
                        },
                    ).model_dump(mode="json")
                )
        finally:
            await self._close_active_session()
            self.event_sink.on_session_finished(self.worker_id, record.session_id, error)

    async def stop_session(self, session_id: str) -> None:
        """Request the active session to stop."""
        if self._active_session_id != session_id:
            return
        self._stop_event.set()
        if self._pipeline_session_id is not None:
            with contextlib.suppress(Exception):
                self.pipeline_adapter.push_chunk(self._pipeline_session_id, {"type": "stop"})

    def _on_data_message(
        self,
        record: SessionRecord,
        message: bytes | str | dict[str, Any],
        topic: str,
        sender_identity: str,
    ) -> None:
        if self._pipeline_session_id is None:
            return
        try:
            chunk = normalize_control_message(
                message,
                topic=topic,
                session_id=record.session_id,
                sender_identity=sender_identity,
                controller_identity=record.controller_identity,
                max_bytes=self.config.max_data_message_bytes,
            )
        except Exception as exc:
            logger.warning(f"LiveKit control message rejected: session={record.session_id} error={exc}")
            return
        self.pipeline_adapter.push_chunk(self._pipeline_session_id, chunk)
        if chunk.get("type") == "stop":
            self._stop_event.set()

    async def _publish_pipeline_chunks(
        self,
        session_id: str,
        chunks: AsyncGenerator[dict, None],
    ) -> None:
        chunk_count = 0
        next_frame_at: float | None = None
        async for chunk in chunks:
            if self._stop_event.is_set():
                break

            frames, audio, metadata = split_chunk_media(chunk)
            decoded_ready_at = chunk.get("timestamp")
            publish_started_at = time.time()
            publish_started_monotonic = time.monotonic()
            chunk_data = chunk.get("data") if isinstance(chunk.get("data"), dict) else chunk
            fps_value = chunk_data.get("fps", chunk.get("fps", self.config.default_fps))
            try:
                fps = float(fps_value)
            except (TypeError, ValueError):
                fps = float(self.config.default_fps)
            if fps <= 0:
                fps = float(self.config.default_fps)
            frame_interval = 1.0 / fps

            for frame in frames:
                if self._stop_event.is_set():
                    break
                now = time.monotonic()
                if next_frame_at is None or now - next_frame_at > frame_interval:
                    next_frame_at = now
                delay = next_frame_at - now
                if delay > 0:
                    await asyncio.sleep(delay)
                await self.room_client.publish_video_frame(frame, fps=fps)
                next_frame_at += frame_interval

            if audio is not None:
                await self.room_client.publish_audio_frame(
                    audio.pcm,
                    sample_rate=audio.sample_rate,
                    channels=audio.channels,
                )

            if self._stop_event.is_set():
                break
            if frames:
                chunk_count += 1
                metadata["transport_measurement"] = {
                    "decoded_ready_at": decoded_ready_at if isinstance(decoded_ready_at, int | float) else None,
                    "publish_started_at": publish_started_at,
                    "publish_finished_at": time.time(),
                    "publish_seconds": time.monotonic() - publish_started_monotonic,
                    "frames": len(frames),
                    "pacing": "realtime",
                }
            if metadata:
                index = chunk.get("index")
                if index is None:
                    index = chunk_data.get("index")
                message = StreamChunkMessage(
                    session_id=session_id,
                    index=index if isinstance(index, int) else None,
                    data=serialisable_chunk(metadata),
                )
                await self.room_client.publish_status(message.model_dump(mode="json"))

        done = StreamDoneMessage(session_id=session_id, total_chunks=chunk_count).model_dump(mode="json")
        await self.room_client.publish_status(done)

    async def _close_active_session(self) -> None:
        pipeline_session_id = self._pipeline_session_id
        self._pipeline_session_id = None
        if pipeline_session_id is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.pipeline_adapter.close_session, pipeline_session_id)
        try:
            await asyncio.wait_for(
                self.room_client.disconnect(),
                timeout=_ROOM_DISCONNECT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                f"LiveKit room disconnect timed out after {_ROOM_DISCONNECT_TIMEOUT_SECONDS:g}s: "
                f"worker={self.worker_id}"
            )
        except Exception as exc:
            logger.warning(f"LiveKit room disconnect failed: worker={self.worker_id} error={exc}")
        self._active_session_id = None
