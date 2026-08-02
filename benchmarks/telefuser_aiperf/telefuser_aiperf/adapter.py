"""AIPerf adapter for TeleFuser LiveKit streaming sessions."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import orjson
from aiperf.common.redact import redact_string
from aiperf.streaming.adapters.common import (
    StreamHttpClient,
    StreamTargetMetadataMixin,
)
from aiperf.streaming.adapters.target_measurements import record_target_measurement
from aiperf.streaming.config import StreamProfileConfig
from aiperf.streaming.contracts import BenchmarkContract
from aiperf.streaming.events import StreamEventRecorder
from aiperf.streaming.models import (
    ControlEventResult,
    SessionResult,
    StreamSessionPlan,
)

from telefuser_aiperf.livekit_room import (
    LiveKitRoomClient,
    LiveKitRoomClientProtocol,
)
from telefuser_aiperf.payload import build_telefuser_livekit_session_body

_CONTROL_TOPIC = "tf.control"
_STATUS_TOPIC = "tf.status"
_METRICS_TOPIC = "tf.metrics"


class _LiveKitSession:
    def __init__(
        self,
        *,
        adapter: TeleFuserLiveKitAdapter,
        plan: StreamSessionPlan,
        room: LiveKitRoomClientProtocol,
    ) -> None:
        self.adapter = adapter
        self.plan = plan
        self.room = room
        self.session_id = plan.planned_session_id
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
        self.first_metadata_at: float | None = None
        self.connected = False
        self.target_ready = False
        self.active_event = asyncio.Event()
        self.first_frame_event = asyncio.Event()
        self.done_event = asyncio.Event()
        self.control_task: asyncio.Task[None] | None = None
        self.pending_control_acks: deque[int] = deque()
        self.pending_control_frames: deque[int] = deque()
        self.control_sent_at: dict[int, float] = {}

    def _active_start(self) -> float:
        if self.active_started_at is None:
            raise RuntimeError("LiveKit active window has not started")
        return self.active_started_at

    def _try_start_active_window(self) -> None:
        if self.active_started_at is not None or not self.connected:
            return
        if not self.target_ready:
            return
        self.active_started_at = time.perf_counter()
        self.active_event.set()
        self.events.record("active_window_start")
        if self.control_task is None and self.plan.control_trace:
            self.control_task = asyncio.create_task(self._send_control_trace())

    def _handle_video_frame(self) -> None:
        now = time.perf_counter()
        self.target_ready = True
        self.result.frames_received += 1
        if self.first_frame_at is None:
            self.first_frame_at = now
            self.result.first_frame_latency_ms = (now - self.started_at) * 1000.0
            self.first_frame_event.set()
            self.events.record("first_frame")
        self.last_frame_at = now
        self._mark_control_frame(now)
        self._try_start_active_window()

    def _mark_control_frame(self, now: float) -> None:
        if not self.pending_control_frames:
            return
        index = self.pending_control_frames.popleft()
        control = self.result.control_events[index]
        if control.next_frame_latency_ms is None:
            control.next_frame_latency_ms = max(
                (now - self.control_sent_at[index]) * 1000.0,
                0.0,
            )

    def _handle_room_event(self, event: str, payload: Mapping[str, Any]) -> None:
        self.events.record(event, **dict(payload))
        if event == "disconnected":
            self.done_event.set()

    def _handle_data_message(
        self,
        raw_message: bytes | str,
        topic: str,
        sender_identity: str,
    ) -> None:
        if topic not in {_STATUS_TOPIC, _METRICS_TOPIC}:
            self.events.record("livekit_data_ignored", topic=topic)
            return
        now = time.perf_counter()
        self.result.metadata_messages += 1
        if self.first_metadata_at is None:
            self.first_metadata_at = now
            self.result.first_metadata_latency_ms = (now - self.started_at) * 1000.0
        try:
            payload = orjson.loads(raw_message)
        except orjson.JSONDecodeError:
            self.events.record("livekit_data_invalid", topic=topic)
            return
        if not isinstance(payload, dict):
            self.events.record("livekit_data_ignored", topic=topic, reason="not_mapping")
            return
        if payload.get("type") == "done":
            self.result.done_received = True
            self.done_event.set()
            self.events.record("done_message", topic=topic)
            return
        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        self._record_transport_profile(data)
        if data.get("type") == "error" or payload.get("error"):
            error = data.get("error") or payload.get("error")
            self.result.error = redact_string(str(error))
            self.done_event.set()
        stage = data.get("stage")
        if stage is not None:
            self._handle_status_stage(str(stage), data, now)
        self.events.record(
            "livekit_data",
            topic=topic,
            message_type=payload.get("type"),
            sender_identity=sender_identity,
            source_timestamp=payload.get("timestamp"),
        )

    def _handle_status_stage(
        self,
        stage: str,
        data: Mapping[str, Any],
        now: float,
    ) -> None:
        self.result.status_messages += 1
        self.result.last_status_stage = stage
        if stage in {"worker_running", "runtime_ready"}:
            self.target_ready = True
            self._try_start_active_window()
        if stage in {"runtime_ready", "chunk_sent"}:
            measurement = data.get("measurement")
            normalized_data = data
            if stage == "chunk_sent" and isinstance(measurement, Mapping):
                phases = measurement.get("phases")
                if isinstance(phases, Mapping):
                    self.events.record(
                        "target_phase_profile",
                        chunk_index=measurement.get("index"),
                        phases=dict(phases),
                    )
                    normalized_data = {
                        **data,
                        "measurement": {key: value for key, value in measurement.items() if key != "phases"},
                    }
            record_target_measurement(
                result=self.result,
                events=self.events,
                stage=stage,
                data=normalized_data,
            )
        if not self.pending_control_acks:
            return
        if stage not in {"control_state", "applying_direction_control"}:
            return
        index = self.pending_control_acks.popleft()
        control = self.result.control_events[index]
        if control.ack_latency_ms is None:
            control.ack_latency_ms = max(
                (now - self.control_sent_at[index]) * 1000.0,
                0.0,
            )

    def _record_transport_profile(self, data: Mapping[str, Any]) -> None:
        transport = data.get("transport_measurement")
        if not isinstance(transport, Mapping):
            return
        received_at = time.time()
        chunk_index = data.get("index")
        measurement = data.get("measurement")
        if chunk_index is None and isinstance(measurement, Mapping):
            chunk_index = measurement.get("index")
        publish_finished_at = transport.get("publish_finished_at")
        self.events.record(
            "target_transport_profile",
            chunk_index=chunk_index,
            transport=dict(transport),
            client_metadata_received_at=received_at,
            publish_to_client_metadata_seconds=(
                max(received_at - float(publish_finished_at), 0.0)
                if isinstance(publish_finished_at, int | float)
                else None
            ),
        )

    async def _send_control_trace(self) -> None:
        active_started_at = self._active_start()
        for event_index, entry in enumerate(self.plan.control_trace):
            deadline = active_started_at + float(entry["delay_s"])
            await asyncio.sleep(max(deadline - time.perf_counter(), 0.0))
            message = dict(entry["message"])
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
            self.pending_control_acks.append(event_index)
            self.pending_control_frames.append(event_index)
            await self.room.publish_data(
                message,
                topic=_CONTROL_TOPIC,
                reliable=True,
            )
            self.events.record("control_sent", event_id=event_index)

    async def _create_and_connect(self) -> None:
        create_started_at = time.perf_counter()
        response = await self.adapter.http.request_json(
            f"{self.plan.server_url}{self.plan.endpoints.offer_path}",
            method="POST",
            timeout_s=float(self.adapter.options.connect_timeout_s),
            payload=build_telefuser_livekit_session_body(self.plan),
        )
        self.result.offer_rtt_ms = (time.perf_counter() - create_started_at) * 1000.0
        required = ("session_id", "livekit_url", "token")
        missing = [name for name in required if not isinstance(response.get(name), str)]
        if missing:
            raise ValueError("LiveKit session response requires string fields: " + ", ".join(missing))
        self.session_id = response["session_id"]
        self.result.session_id = self.session_id
        self.events.set_session_id(self.session_id)
        self.events.record(
            "session_created",
            room=response.get("room"),
            status=response.get("status"),
            queue_position=response.get("queue_position"),
        )
        await self.room.connect(
            response["livekit_url"],
            response["token"],
            timeout_s=float(self.adapter.options.connect_timeout_s),
            on_data=self._handle_data_message,
            on_video_frame=self._handle_video_frame,
            on_event=self._handle_room_event,
        )
        self.connected = True
        self.result.connected_latency_ms = (time.perf_counter() - self.started_at) * 1000.0
        self.events.record("connected")
        self._try_start_active_window()

    async def _wait_for_media(self) -> None:
        await asyncio.wait_for(
            self.active_event.wait(),
            timeout=float(self.adapter.options.connect_timeout_s),
        )
        first_frame = asyncio.create_task(self.first_frame_event.wait())
        done = asyncio.create_task(self.done_event.wait())
        completed, pending = await asyncio.wait(
            {first_frame, done},
            timeout=float(self.adapter.options.frame_timeout_s),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if not completed or not self.first_frame_event.is_set():
            raise TimeoutError("No LiveKit video frame received before the frame timeout")
        remaining = max(
            self._active_start() + float(self.plan.session_duration_s) - time.perf_counter(),
            0.0,
        )
        try:
            await asyncio.wait_for(self.done_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            self.events.record("session_duration_elapsed")

    async def run(self) -> SessionResult:
        self.started_at = time.perf_counter()
        self.events.record(
            "session_start",
            transport="webrtc",
            transport_provider="livekit",
            mode=self.plan.mode,
        )
        completed = False
        try:
            await self._create_and_connect()
            await self._wait_for_media()
            completed = True
        except Exception as exc:  # noqa: BLE001 - transport failures become results
            self.result.error = redact_string(f"{type(exc).__name__}: {exc}")
            self.events.record("session_error", error=self.result.error)
        finally:
            await self._shutdown()
            if completed:
                self._finalize_success()
            self.result.session_runtime_s = time.perf_counter() - self.started_at
            event_path = await self.events.export()
            self.result.artifacts_event_file = str(event_path)
        return self.result

    def _finalize_success(self) -> None:
        self.result.success = self.first_frame_at is not None and self.result.error is None
        if self.first_frame_at is None:
            self.result.error = self.result.error or "No LiveKit video frame received"
            return
        if self.last_frame_at is not None and self.last_frame_at > self.first_frame_at:
            self.result.stream_fps = (self.result.frames_received - 1) / (self.last_frame_at - self.first_frame_at)

    async def _shutdown(self) -> None:
        if self.connected:
            try:
                await self.room.publish_data(
                    {"type": "stop"},
                    topic=_CONTROL_TOPIC,
                    reliable=True,
                )
                self.events.record("stop_sent")
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort
                self.events.record_error("stop_send_failed", exc)
        await self._delete_target_session()
        if self.control_task is not None and not self.control_task.done():
            self.control_task.cancel()
            await asyncio.gather(self.control_task, return_exceptions=True)
        try:
            await self.room.disconnect()
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            self.events.record_error("livekit_disconnect_failed", exc)

    async def _delete_target_session(self) -> None:
        template = self.plan.endpoints.delete_path_template
        if template is None or self.session_id == self.plan.planned_session_id:
            return
        delete_path = template.format(session_id=self.session_id)
        try:
            await self.adapter.http.request_json(
                f"{self.plan.server_url}{delete_path}",
                method="DELETE",
                timeout_s=float(self.adapter.options.shutdown_timeout_s),
                accepted_error_statuses=(404,),
            )
            self.events.record("session_delete")
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            self.events.record_error("session_delete_failed", exc)


class TeleFuserLiveKitAdapter(StreamTargetMetadataMixin):
    """AIPerf adapter for WebRTC media delivered through TeleFuser LiveKit rooms."""

    transport = "webrtc"

    def __init__(
        self,
        *,
        contract: BenchmarkContract,
        config: StreamProfileConfig,
        artifacts_dir: str | Path,
        room_client_factory: Callable[[], LiveKitRoomClientProtocol] = LiveKitRoomClient,
        http_client: StreamHttpClient | None = None,
    ) -> None:
        self.contract = contract
        self.config = config
        self.options = config.transport
        self.artifacts_dir = Path(artifacts_dir)
        self.room_client_factory = room_client_factory
        self.http = http_client or StreamHttpClient()

    async def check_health(self) -> None:
        """Check the contract-declared target health endpoint."""

        health_path = str(self.contract.endpoint.get("health_path", "/v1/service/health"))
        await self.http.check_health(
            f"{self.config.server_url}{health_path}",
            timeout_s=float(self.options.connect_timeout_s),
        )

    async def run_session(self, plan: StreamSessionPlan) -> SessionResult:
        """Execute one normalized plan through a LiveKit room."""

        return await _LiveKitSession(
            adapter=self,
            plan=plan,
            room=self.room_client_factory(),
        ).run()

    async def aclose(self) -> None:
        """Close adapter-owned HTTP resources."""

        await self.http.aclose()
