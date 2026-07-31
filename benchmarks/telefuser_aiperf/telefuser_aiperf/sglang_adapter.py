"""AIPerf adapter for SGLang realtime video WebSocket sessions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import msgspec
import websockets
from aiperf.common.redact import redact_string
from aiperf.streaming.adapters.common import StreamHttpClient, StreamTargetMetadataMixin
from aiperf.streaming.config import StreamProfileConfig
from aiperf.streaming.contracts import BenchmarkContract
from aiperf.streaming.events import StreamEventRecorder
from aiperf.streaming.models import ControlEventResult, SessionResult, StreamChunkMeasurement, StreamSessionPlan

from telefuser_aiperf.payload import build_sglang_realtime_init

_KEY_TO_ACTION = {
    "ArrowUp": "w",
    "ArrowDown": "s",
    "ArrowLeft": "a",
    "ArrowRight": "d",
}


def _websocket_url(server_url: str, endpoint: str) -> str:
    parsed = urlsplit(server_url)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parsed.scheme)
    if scheme is None:
        raise ValueError(f"Unsupported SGLang server URL scheme: {parsed.scheme}")
    path = f"{parsed.path.rstrip('/')}/{endpoint.lstrip('/')}"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


class _SGLangRealtimeSession:
    def __init__(self, *, adapter: SGLangRealtimeAdapter, plan: StreamSessionPlan) -> None:
        self.adapter = adapter
        self.plan = plan
        self.result = SessionResult(
            logical_session_index=plan.logical_session_index,
            phase=plan.phase,
            mode=plan.mode,
            planned_session_id=plan.planned_session_id,
            session_id=plan.planned_session_id,
        )
        self.events = StreamEventRecorder(
            artifacts_dir=adapter.artifacts_dir,
            phase=plan.phase,
            logical_session_index=plan.logical_session_index,
            planned_session_id=plan.planned_session_id,
            print_events=adapter.config.print_events,
        )
        self.started_at = 0.0
        self.active_started_at: float | None = None
        self.first_frame_at: float | None = None
        self.last_frame_at: float | None = None
        self.first_frame_batch_size = 0
        self.first_metadata_at: float | None = None
        self.websocket: Any = None
        self.control_task: asyncio.Task[None] | None = None
        self.control_sent_at: dict[int, float] = {}
        self.control_by_event_id: dict[int, int] = {}
        self.held_actions: set[str] = set()
        self.pending_frame_header: Mapping[str, Any] | None = None

    async def run(self) -> SessionResult:
        self.started_at = time.perf_counter()
        self.events.record(
            "session_start",
            transport="websocket",
            transport_provider="sglang",
            mode=self.plan.mode,
        )
        completed = False
        try:
            websocket_url = _websocket_url(self.plan.server_url, self.plan.endpoints.offer_path)
            connect_started_at = time.perf_counter()
            connect = self.adapter.websocket_connect_factory(
                websocket_url,
                open_timeout=float(self.adapter.options.connect_timeout_s),
                close_timeout=float(self.adapter.options.shutdown_timeout_s),
                max_size=None,
                proxy=None,
            )
            async with connect as websocket:
                self.websocket = websocket
                self.result.connected_latency_ms = (time.perf_counter() - self.started_at) * 1000.0
                self.result.offer_rtt_ms = (time.perf_counter() - connect_started_at) * 1000.0
                self.events.record("connected", websocket_url=websocket_url)
                await websocket.send(msgspec.msgpack.encode(build_sglang_realtime_init(self.plan)))
                self.events.record("init_sent")
                await self._receive()
                completed = True
        except Exception as exc:  # noqa: BLE001 - transport failures become results
            if self._is_clean_close(exc) and self.first_frame_at is not None:
                self.result.done_received = True
                completed = True
                self.events.record("generation_complete")
            else:
                self.result.error = redact_string(f"{type(exc).__name__}: {exc}")
                self.events.record("session_error", error=self.result.error)
        finally:
            if self.control_task is not None and not self.control_task.done():
                self.control_task.cancel()
                await asyncio.gather(self.control_task, return_exceptions=True)
            if self.websocket is not None:
                try:
                    await self.websocket.close()
                except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                    self.events.record_error("websocket_close_failed", exc)
            if completed:
                self._finalize_success()
            self.result.session_runtime_s = time.perf_counter() - self.started_at
            event_path = await self.events.export()
            self.result.artifacts_event_file = str(event_path)
        return self.result

    async def _receive(self) -> None:
        while True:
            timeout = float(self.adapter.options.frame_timeout_s)
            if self.active_started_at is not None:
                remaining = self.active_started_at + float(self.plan.session_duration_s) - time.perf_counter()
                if remaining <= 0:
                    self.events.record("session_duration_elapsed")
                    return
                timeout = min(timeout, remaining)
            try:
                raw_message = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                if self.first_frame_at is None:
                    raise TimeoutError("No SGLang video frame received before the frame timeout") from None
                self.events.record("session_duration_elapsed")
                return
            self._handle_message(raw_message)
            if self.control_task is not None and self.control_task.done():
                control_error = self.control_task.exception()
                if control_error is not None:
                    raise control_error

    def _handle_message(self, raw_message: bytes | str) -> None:
        now = time.perf_counter()
        if self.pending_frame_header is not None:
            if not isinstance(raw_message, bytes):
                raise ValueError("SGLang frame payload must be binary")
            header = self.pending_frame_header
            self.pending_frame_header = None
            self._handle_frame_batch(header, now, payload_bytes=len(raw_message))
            return
        if not isinstance(raw_message, bytes):
            raise ValueError("SGLang realtime messages must be MessagePack binary frames")
        message = msgspec.msgpack.decode(raw_message)
        if not isinstance(message, dict):
            raise ValueError("SGLang realtime message must decode to a mapping")
        self._mark_metadata(now)
        message_type = message.get("type")
        if message_type == "frame_batch_header":
            self.pending_frame_header = message
        elif message_type == "frame_batch":
            payload = message.get("payload")
            self._handle_frame_batch(
                message,
                now,
                payload_bytes=len(payload) if isinstance(payload, bytes) else None,
            )
        elif message_type == "chunk_stats":
            self._handle_chunk_stats(message, now)
        elif message_type == "error":
            raise RuntimeError(str(message.get("error") or message.get("message") or "SGLang realtime error"))
        else:
            self.events.record("sglang_message", message_type=message_type)

    def _mark_metadata(self, now: float) -> None:
        self.result.metadata_messages += 1
        if self.first_metadata_at is None:
            self.first_metadata_at = now
            self.result.first_metadata_latency_ms = (now - self.started_at) * 1000.0

    def _handle_frame_batch(self, message: Mapping[str, Any], now: float, *, payload_bytes: int | None) -> None:
        frames = int(message.get("num_frames", 0))
        if frames <= 0:
            raise ValueError("SGLang frame batch must report a positive num_frames")
        request_id = message.get("request_id")
        if isinstance(request_id, str) and request_id:
            self.result.session_id = request_id
            self.events.set_session_id(request_id)
        self.result.frames_received += frames
        if self.first_frame_at is None:
            self.first_frame_at = now
            self.first_frame_batch_size = frames
            self.active_started_at = now
            self.result.first_frame_latency_ms = (now - self.started_at) * 1000.0
            self.events.record("first_frame", frames=frames)
            if self.plan.control_trace:
                self.control_task = asyncio.create_task(self._send_control_trace())
        self.last_frame_at = now
        self._mark_control_frame(message.get("event_id"), now)
        self.events.record(
            "frame_batch",
            chunk_index=message.get("chunk_index"),
            frames=frames,
            payload_bytes=payload_bytes,
            content_type=message.get("content_type"),
        )

    def _handle_chunk_stats(self, message: Mapping[str, Any], now: float) -> None:
        self.result.status_messages += 1
        self.result.last_status_stage = "chunk_stats"
        self.result.chunk_measurements.append(
            StreamChunkMeasurement(
                index=int(message["chunk_index"]),
                frames=int(message.get("num_frames", 0)),
                request_prepare_seconds=self._milliseconds(message.get("request_prepare_ms")),
                compute_seconds=self._milliseconds(message.get("scheduler_forward_ms")) or 0.0,
                encode_seconds=self._milliseconds(message.get("raw_payload_build_ms")),
                output_pacing_seconds=self._milliseconds(message.get("pace_wait_ms")),
                output_header_write_seconds=self._milliseconds(message.get("header_write_ms")),
                output_payload_write_seconds=self._milliseconds(message.get("raw_write_ms")),
                output_write_seconds=self._milliseconds(message.get("ws_write_ms")),
                total_seconds=self._milliseconds(message.get("chunk_total_ms")),
                raw_output_bytes=self._optional_int(message.get("raw_bytes")),
                wire_output_bytes=self._optional_int(message.get("ws_payload_bytes")),
                output_batches=self._optional_int(message.get("num_batches")),
                output_content_type=str(message["content_type"]) if message.get("content_type") else None,
            )
        )
        self._mark_control_ack(message.get("event_id"), now)
        self.events.record(
            "chunk_stats",
            chunk_index=message.get("chunk_index"),
            event_id=message.get("event_id"),
        )

    async def _send_control_trace(self) -> None:
        active_started_at = self.active_started_at
        if active_started_at is None:
            return
        for event_index, entry in enumerate(self.plan.control_trace):
            deadline = active_started_at + float(entry["delay_s"])
            await asyncio.sleep(max(deadline - time.perf_counter(), 0.0))
            message = dict(entry["message"])
            key = str(message.get("key", ""))
            action = _KEY_TO_ACTION.get(key)
            if action is None:
                raise ValueError(f"Unsupported SGLang control key: {key}")
            operation = message.get("action")
            if operation == "press":
                self.held_actions.add(action)
            elif operation == "release":
                self.held_actions.discard(action)
            else:
                raise ValueError(f"Unsupported SGLang control action: {operation}")
            event_id = event_index + 1
            sent_at = time.perf_counter()
            self.result.control_events.append(
                ControlEventResult(
                    index=event_index,
                    scheduled_delay_s=float(entry["delay_s"]),
                    message=message,
                    sent_offset_s=sent_at - active_started_at,
                )
            )
            self.control_sent_at[event_index] = sent_at
            self.control_by_event_id[event_id] = event_index
            event = {
                "type": "event",
                "kind": "camera_actions",
                "event_id": event_id,
                "payload": {
                    "mode": "state",
                    "transitions": [
                        {
                            "actions": sorted(self.held_actions),
                            "client_ts_ms": (sent_at - active_started_at) * 1000.0,
                        }
                    ],
                },
            }
            await self.websocket.send(msgspec.msgpack.encode(event))
            self.events.record("control_sent", event_id=event_id, actions=sorted(self.held_actions))

    def _mark_control_ack(self, event_id: Any, now: float) -> None:
        index = self.control_by_event_id.get(event_id)
        if index is None:
            return
        control = self.result.control_events[index]
        if control.ack_latency_ms is None:
            control.ack_latency_ms = max((now - self.control_sent_at[index]) * 1000.0, 0.0)

    def _mark_control_frame(self, event_id: Any, now: float) -> None:
        index = self.control_by_event_id.get(event_id)
        if index is None:
            return
        control = self.result.control_events[index]
        if control.next_frame_latency_ms is None:
            control.next_frame_latency_ms = max((now - self.control_sent_at[index]) * 1000.0, 0.0)

    def _finalize_success(self) -> None:
        self.result.success = self.first_frame_at is not None and self.result.error is None
        if self.first_frame_at is None:
            self.result.error = self.result.error or "No SGLang video frame received"
            return
        if self.last_frame_at is not None and self.last_frame_at > self.first_frame_at:
            frames_after_first_batch = self.result.frames_received - self.first_frame_batch_size
            self.result.stream_fps = frames_after_first_batch / (self.last_frame_at - self.first_frame_at)

    @staticmethod
    def _milliseconds(value: Any) -> float | None:
        return float(value) / 1000.0 if value is not None else None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return int(value) if value is not None else None

    @staticmethod
    def _is_clean_close(exc: Exception) -> bool:
        return isinstance(exc, websockets.exceptions.ConnectionClosedOK)


class SGLangRealtimeAdapter(StreamTargetMetadataMixin):
    """AIPerf adapter for SGLang's MessagePack realtime video endpoint."""

    transport = "websocket"

    def __init__(
        self,
        *,
        contract: BenchmarkContract,
        config: StreamProfileConfig,
        artifacts_dir: str | Path,
        websocket_connect_factory: Callable[..., Any] = websockets.connect,
        http_client: StreamHttpClient | None = None,
    ) -> None:
        self.contract = contract
        self.config = config
        self.options = config.transport
        self.artifacts_dir = Path(artifacts_dir)
        self.websocket_connect_factory = websocket_connect_factory
        self.http = http_client or StreamHttpClient()

    async def check_health(self) -> None:
        """Check SGLang's contract-declared health endpoint."""

        health_path = str(self.contract.endpoint.get("health_path", "/health"))
        await self.http.check_health(
            f"{self.config.server_url}{health_path}",
            timeout_s=float(self.options.connect_timeout_s),
        )

    async def run_session(self, plan: StreamSessionPlan) -> SessionResult:
        """Execute one normalized plan through SGLang realtime video."""

        return await _SGLangRealtimeSession(adapter=self, plan=plan).run()

    async def aclose(self) -> None:
        """Close adapter-owned HTTP resources."""

        await self.http.aclose()
