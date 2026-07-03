#!/usr/bin/env python3
"""Run an end-to-end WebRTC stream benchmark against ``telefuser stream-serve``.

This benchmark is intentionally separate from the vendored AIPerf core because
TeleFuser world-model streaming uses WebRTC session lifecycles instead of plain
HTTP request/response flows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
except ImportError as exc:  # pragma: no cover - runtime dependency check
    RTCConfiguration = None
    RTCIceServer = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    AIORTC_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised only when dependency exists
    AIORTC_IMPORT_ERROR = None

from telefuser.webrtc_ice import configure_ice_host_addresses


DEFAULT_CONTROL_TRACE = [
    {"delay_s": 1.0, "message": {"type": "control", "key": "ArrowUp", "action": "press"}},
    {"delay_s": 1.8, "message": {"type": "control", "key": "ArrowUp", "action": "release"}},
    {"delay_s": 2.8, "message": {"type": "control", "key": "ArrowLeft", "action": "press"}},
    {"delay_s": 3.6, "message": {"type": "control", "key": "ArrowLeft", "action": "release"}},
    {"delay_s": 4.6, "message": {"type": "control", "key": "ArrowRight", "action": "press"}},
    {"delay_s": 5.4, "message": {"type": "control", "key": "ArrowRight", "action": "release"}},
]


@dataclass
class BenchConfig:
    server_url: str = "http://127.0.0.1:8088"
    offer_path: str = "/v1/stream/webrtc/offer"
    delete_path_template: str = "/v1/stream/webrtc/{session_id}"
    mode: str = "bidirectional"
    task: str = "bidirectional"
    prompt: str = "walk forward through the scene"
    image_path: str | None = None
    fps: int = 16
    session_count: int = 1
    warmup_sessions: int = 0
    session_duration_s: float = 12.0
    stagger_s: float = 0.0
    connect_timeout_s: float = 30.0
    frame_timeout_s: float = 60.0
    ice_gather_timeout_s: float = 5.0
    shutdown_timeout_s: float = 5.0
    receive_audio: bool = False
    control_trace_path: str | None = None
    ice_host_ips: list[str] | None = None
    request_extra: dict[str, Any] = field(default_factory=dict)
    artifacts_dir: str = "artifacts/telefuser_aiperf/stream_bench"
    turn_url: str | None = None
    turn_username: str | None = None
    turn_credential: str | None = None
    force_turn_relay: bool = False
    print_events: bool = False


@dataclass
class ControlEventResult:
    index: int
    scheduled_delay_s: float
    message: dict[str, Any]
    sent_offset_s: float
    ack_latency_ms: float | None = None
    next_frame_latency_ms: float | None = None


@dataclass
class SessionResult:
    logical_session_index: int
    phase: str
    mode: str
    planned_session_id: str
    session_id: str
    success: bool = False
    error: str | None = None
    offer_rtt_ms: float | None = None
    connected_latency_ms: float | None = None
    first_frame_latency_ms: float | None = None
    first_metadata_latency_ms: float | None = None
    session_runtime_s: float | None = None
    frames_received: int = 0
    metadata_messages: int = 0
    status_messages: int = 0
    done_received: bool = False
    stream_fps: float | None = None
    last_status_stage: str | None = None
    control_events: list[ControlEventResult] = field(default_factory=list)
    artifacts_event_file: str | None = None


def _parse_json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional JSON config file")
    parser.add_argument("--server-url", default=argparse.SUPPRESS, help="TeleFuser stream server base URL")
    parser.add_argument("--mode", choices=("server_push", "bidirectional"), default=argparse.SUPPRESS)
    parser.add_argument("--task", default=argparse.SUPPRESS, help="Request task field sent in the WebRTC offer body")
    parser.add_argument("--prompt", default=argparse.SUPPRESS, help="Prompt sent in the initial offer body")
    parser.add_argument("--image-path", default=argparse.SUPPRESS, help="Image path visible to the server process")
    parser.add_argument("--fps", type=int, default=argparse.SUPPRESS, help="Requested stream FPS")
    parser.add_argument("--session-count", type=int, default=argparse.SUPPRESS, help="Number of profiled sessions")
    parser.add_argument("--warmup-sessions", type=int, default=argparse.SUPPRESS, help="Warmup sessions before profile")
    parser.add_argument("--session-duration-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--stagger-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--connect-timeout-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--frame-timeout-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--ice-gather-timeout-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--shutdown-timeout-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--control-trace-path", default=argparse.SUPPRESS, help="JSON file with timed control events")
    parser.add_argument(
        "--ice-host-ip",
        action="append",
        default=argparse.SUPPRESS,
        help="Allowlisted ICE host IP for candidate gathering. Repeat to allow multiple addresses.",
    )
    parser.add_argument("--request-extra-json", default=argparse.SUPPRESS, help="Inline JSON merged into request body")
    parser.add_argument("--artifacts-dir", default=argparse.SUPPRESS, help="Artifact root directory")
    parser.add_argument("--turn-url", default=argparse.SUPPRESS, help="TURN server URL for WebRTC ICE")
    parser.add_argument("--turn-username", default=argparse.SUPPRESS)
    parser.add_argument("--turn-credential", default=argparse.SUPPRESS)
    parser.add_argument("--receive-audio", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--force-turn-relay", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument("--print-events", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    return parser


def _load_config(args: argparse.Namespace) -> BenchConfig:
    merged: dict[str, Any] = asdict(BenchConfig())
    if args.config:
        file_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(file_data, dict):
            raise ValueError(f"Config file must contain a JSON object: {args.config}")
        merged.update(file_data)

    cli_values = vars(args).copy()
    cli_values.pop("config", None)
    request_extra_json = cli_values.pop("request_extra_json", None)
    if request_extra_json is not None:
        cli_values["request_extra"] = _parse_json_value(request_extra_json)
    if "ice_host_ip" in cli_values:
        cli_values["ice_host_ips"] = cli_values.pop("ice_host_ip")

    merged.update(cli_values)
    request_extra = merged.get("request_extra") or {}
    if not isinstance(request_extra, dict):
        raise ValueError("request_extra must be a JSON object")
    merged["request_extra"] = request_extra
    return BenchConfig(**merged)


def _detect_default_ice_host_ips() -> list[str] | None:
    """Return a small list of non-loopback host IPs for local ICE gathering."""

    try:
        result = subprocess.run(
            ["hostname", "-I"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    ips: list[str] = []
    for raw_ip in result.stdout.split():
        ip = raw_ip.strip()
        if not ip or ip.startswith("127.") or ip == "::1":
            continue
        if ip not in ips:
            ips.append(ip)

    return ips or None


def _build_rtc_configuration(config: BenchConfig):
    if RTCConfiguration is None or RTCIceServer is None:
        raise RuntimeError(
            "aiortc is required for stream benchmarking. Install TeleFuser with `pip install -e \".[webrtc]\"`."
        ) from AIORTC_IMPORT_ERROR

    ice_servers: list[Any] = []
    if config.turn_url:
        ice_server_kwargs = {"urls": config.turn_url}
        if config.turn_username:
            ice_server_kwargs["username"] = config.turn_username
        if config.turn_credential:
            ice_server_kwargs["credential"] = config.turn_credential
        ice_servers.append(RTCIceServer(**ice_server_kwargs))
    # aiortc 1.14.0 stalls on this benchmark's recvonly video + DataChannel
    # flow when MAX_BUNDLE is forced. Use the default bundle policy instead.
    return RTCConfiguration(iceServers=ice_servers or None)


async def _http_json_request(url: str, *, method: str, timeout_s: float, payload: dict | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)

    def _run() -> dict:
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                body = response.read()
                return json.loads(body.decode("utf-8")) if body else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

    return await asyncio.to_thread(_run)


async def _wait_for_ice_complete(peer_connection, timeout_s: float) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if peer_connection.iceGatheringState == "complete":
            return True
        await asyncio.sleep(0.05)
    return peer_connection.iceGatheringState == "complete"


def _load_control_trace(path: str | None, mode: str) -> list[dict[str, Any]]:
    if mode != "bidirectional":
        return []
    if path is None:
        return DEFAULT_CONTROL_TRACE

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    events = payload.get("events", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError(f"Control trace must be a list or {{\"events\": [...]}}: {path}")
    normalised: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"Control trace event #{idx} must be a JSON object")
        delay_s = float(event.get("delay_s", 0.0))
        message = event.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"Control trace event #{idx} is missing object field `message`")
        normalised.append({"delay_s": delay_s, "message": message})
    return sorted(normalised, key=lambda item: item["delay_s"])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summarise(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": float(len(values)),
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


class StreamBenchSession:
    """Single WebRTC session benchmark runner."""

    def __init__(
        self,
        *,
        config: BenchConfig,
        logical_session_index: int,
        phase: str,
        artifacts_dir: Path,
        rtc_configuration,
        control_trace: list[dict[str, Any]],
    ) -> None:
        self.config = config
        self.logical_session_index = logical_session_index
        self.phase = phase
        self.artifacts_dir = artifacts_dir
        self.rtc_configuration = rtc_configuration
        self.control_trace = control_trace
        self.planned_session_id = f"{phase}-{logical_session_index:03d}-{uuid.uuid4().hex[:10]}"
        self.session_id = self.planned_session_id
        self.result = SessionResult(
            logical_session_index=logical_session_index,
            phase=phase,
            mode=config.mode,
            planned_session_id=self.planned_session_id,
            session_id=self.session_id,
        )
        self._events: list[dict[str, Any]] = []
        self._pc = None
        self._data_channel = None
        self._session_started_at = 0.0
        self._offer_sent_at = 0.0
        self._first_frame_at: float | None = None
        self._last_frame_at: float | None = None
        self._first_metadata_at: float | None = None
        self._connected_at: float | None = None
        self._connected_event = asyncio.Event()
        self._done_event = asyncio.Event()
        self._state_monitor_task: asyncio.Task | None = None
        self._track_tasks: list[asyncio.Task] = []
        self._control_sender_task: asyncio.Task | None = None
        self._pending_control_ack_indices: deque[int] = deque()
        self._pending_control_frame_indices: deque[int] = deque()

    def _record_event(self, event_type: str, **payload: Any) -> None:
        event = {
            "event": event_type,
            "session_id": self.session_id,
            "logical_session_index": self.logical_session_index,
            "phase": self.phase,
            "timestamp": time.time(),
        }
        event.update(payload)
        self._events.append(event)
        if self.config.print_events:
            print(json.dumps(event, ensure_ascii=False))

    def _build_request_body(self) -> dict[str, Any]:
        request_options = dict(self.config.request_extra)
        request_fps = int(request_options.get("fps", self.config.fps))
        body = {
            "session_id": self.planned_session_id,
            "sdp": self._pc.localDescription.sdp,
            "type": self._pc.localDescription.type,
            "task": self.config.task,
            "prompt": self.config.prompt,
            "fps": request_fps,
            "config": dict(request_options),
        }
        body.update(request_options)
        if self.config.image_path:
            body["image_path"] = self.config.image_path
        return body

    def _current_state_snapshot(self) -> dict[str, str]:
        if self._pc is None:
            return {}
        return {
            "signaling_state": self._pc.signalingState,
            "ice_gathering_state": self._pc.iceGatheringState,
            "ice_connection_state": self._pc.iceConnectionState,
            "connection_state": self._pc.connectionState,
        }

    async def _monitor_peer_states(self) -> None:
        """Record state transitions even when aiortc callbacks stay quiet."""

        last_snapshot: dict[str, str] | None = None
        while not self._done_event.is_set():
            if self._pc is None:
                return
            snapshot = self._current_state_snapshot()
            if snapshot and snapshot != last_snapshot:
                self._record_event("peer_state_snapshot", **snapshot)
                last_snapshot = snapshot
            if snapshot.get("connection_state") == "connected":
                return
            await asyncio.sleep(0.5)

    async def _consume_video_track(self, track) -> None:
        try:
            while True:
                await track.recv()
                now = time.perf_counter()
                self.result.frames_received += 1
                if self._first_frame_at is None:
                    self._first_frame_at = now
                    self.result.first_frame_latency_ms = (now - self._session_started_at) * 1000.0
                    self._record_event("first_frame")
                self._last_frame_at = now
                if self._pending_control_frame_indices:
                    control_index = self._pending_control_frame_indices.popleft()
                    control_event = self.result.control_events[control_index]
                    if control_event.next_frame_latency_ms is None:
                        control_event.next_frame_latency_ms = (now - self._session_started_at) * 1000.0 - (
                            control_event.sent_offset_s * 1000.0
                        )
        except Exception as exc:
            self._record_event("video_track_ended", detail=str(exc))

    async def _consume_audio_track(self, track) -> None:
        try:
            while True:
                await track.recv()
        except Exception as exc:
            self._record_event("audio_track_ended", detail=str(exc))

    def _handle_datachannel_message(self, raw_message: str) -> None:
        now = time.perf_counter()
        self.result.metadata_messages += 1
        if self._first_metadata_at is None:
            self._first_metadata_at = now
            self.result.first_metadata_latency_ms = (now - self._session_started_at) * 1000.0

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            self._record_event("datachannel_message_invalid", raw_message=raw_message)
            return

        message_type = payload.get("type")
        if message_type == "done":
            self.result.done_received = True
            self._done_event.set()
            self._record_event("done_message", payload=payload)
            return

        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        stage = data.get("stage")
        if stage is not None:
            self.result.status_messages += 1
            self.result.last_status_stage = str(stage)
            if self._pending_control_ack_indices and stage in {"control_state", "applying_direction_control"}:
                control_index = self._pending_control_ack_indices.popleft()
                control_event = self.result.control_events[control_index]
                if control_event.ack_latency_ms is None:
                    control_event.ack_latency_ms = (now - self._session_started_at) * 1000.0 - (
                        control_event.sent_offset_s * 1000.0
                    )
        self._record_event("datachannel_message", payload=payload)

    async def _send_control_trace(self) -> None:
        if self._data_channel is None:
            return

        trace_started_at = time.perf_counter()
        for event_index, entry in enumerate(self.control_trace):
            deadline = trace_started_at + float(entry["delay_s"])
            delay = deadline - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if self._data_channel.readyState != "open":
                self._record_event("control_trace_aborted", reason=f"datachannel_state={self._data_channel.readyState}")
                return
            message = dict(entry["message"])
            payload = json.dumps(message)
            sent_at = time.perf_counter()
            control_event = ControlEventResult(
                index=event_index,
                scheduled_delay_s=float(entry["delay_s"]),
                message=message,
                sent_offset_s=sent_at - self._session_started_at,
            )
            self.result.control_events.append(control_event)
            self._pending_control_ack_indices.append(event_index)
            self._pending_control_frame_indices.append(event_index)
            self._data_channel.send(payload)
            self._record_event("control_sent", payload=message)

    def _register_peer_callbacks(self) -> None:
        @self._pc.on("signalingstatechange")
        async def _on_signalingstatechange() -> None:
            self._record_event("signaling_state", state=self._pc.signalingState)

        @self._pc.on("icegatheringstatechange")
        async def _on_icegatheringstatechange() -> None:
            self._record_event("ice_gathering_state", state=self._pc.iceGatheringState)

        @self._pc.on("iceconnectionstatechange")
        async def _on_iceconnectionstatechange() -> None:
            state = self._pc.iceConnectionState
            self._record_event("ice_connection_state", state=state)
            if state in {"failed", "closed", "disconnected"}:
                self._done_event.set()

        @self._pc.on("connectionstatechange")
        async def _on_connectionstatechange() -> None:
            state = self._pc.connectionState
            self._record_event("connection_state", state=state)
            if state == "connected" and self._connected_at is None:
                self._connected_at = time.perf_counter()
                self.result.connected_latency_ms = (self._connected_at - self._session_started_at) * 1000.0
                self._connected_event.set()
            if state in {"failed", "closed", "disconnected"}:
                self._done_event.set()

        @self._pc.on("track")
        def _on_track(track) -> None:
            self._record_event("remote_track", kind=track.kind)
            if track.kind == "video":
                self._track_tasks.append(asyncio.create_task(self._consume_video_track(track)))
            elif track.kind == "audio":
                self._track_tasks.append(asyncio.create_task(self._consume_audio_track(track)))

    def _create_bidirectional_channel(self) -> None:
        self._data_channel = self._pc.createDataChannel("telefuser")

        @self._data_channel.on("open")
        def _on_open() -> None:
            self._record_event("datachannel_open")
            if self._control_sender_task is None and self.control_trace:
                self._control_sender_task = asyncio.create_task(self._send_control_trace())

        @self._data_channel.on("message")
        def _on_message(message) -> None:
            if isinstance(message, bytes):
                message = message.decode("utf-8", errors="replace")
            self._handle_datachannel_message(str(message))

        @self._data_channel.on("close")
        def _on_close() -> None:
            self._record_event("datachannel_close")
            self._done_event.set()

    async def run(self) -> SessionResult:
        self._session_started_at = time.perf_counter()
        self._record_event("session_start", mode=self.config.mode)

        self._pc = RTCPeerConnection(configuration=self.rtc_configuration or RTCConfiguration())
        self._register_peer_callbacks()
        self._state_monitor_task = asyncio.create_task(self._monitor_peer_states())

        if self.config.mode == "bidirectional":
            self._create_bidirectional_channel()

        self._pc.addTransceiver("video", direction="recvonly")
        if self.config.receive_audio:
            self._pc.addTransceiver("audio", direction="recvonly")

        try:
            offer = await self._pc.createOffer()
            await self._pc.setLocalDescription(offer)
            ice_complete = await _wait_for_ice_complete(self._pc, self.config.ice_gather_timeout_s)
            self._record_event("ice_gathering_complete", complete=ice_complete)

            request_body = self._build_request_body()
            self._record_event("offer_request", body=request_body)
            self._offer_sent_at = time.perf_counter()
            answer = await _http_json_request(
                f"{self.config.server_url.rstrip('/')}{self.config.offer_path}",
                method="POST",
                timeout_s=self.config.connect_timeout_s,
                payload=request_body,
            )
            answered_at = time.perf_counter()
            self.result.offer_rtt_ms = (answered_at - self._offer_sent_at) * 1000.0
            self.session_id = str(answer.get("session_id", self.session_id))
            self.result.session_id = self.session_id
            self._record_event("offer_answer", answer=answer)
            await self._pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
            self._record_event("post_answer_state", **self._current_state_snapshot())

            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=self.config.connect_timeout_s)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"WebRTC connection did not reach connected state within {self.config.connect_timeout_s:.1f}s"
                )

            try:
                await asyncio.wait_for(self._done_event.wait(), timeout=self.config.session_duration_s)
            except asyncio.TimeoutError:
                self._record_event("session_duration_elapsed", duration_s=self.config.session_duration_s)

            if self._first_frame_at is None:
                wait_remaining = self.config.frame_timeout_s
                while self._first_frame_at is None and wait_remaining > 0:
                    started_wait = time.perf_counter()
                    try:
                        await asyncio.wait_for(self._done_event.wait(), timeout=min(wait_remaining, 0.25))
                    except asyncio.TimeoutError:
                        pass
                    wait_remaining -= time.perf_counter() - started_wait
                    if self._first_frame_at is not None:
                        break

            if self._first_frame_at is not None:
                self.result.success = True
                if self._last_frame_at is not None and self._last_frame_at > self._first_frame_at:
                    self.result.stream_fps = (self.result.frames_received - 1) / (self._last_frame_at - self._first_frame_at)
            elif self.result.error is None:
                self.result.error = "No video frame received before session shutdown"
        except Exception as exc:
            self.result.error = str(exc)
            self._record_event("session_error", error=str(exc))
        finally:
            await self._shutdown()
            finished_at = time.perf_counter()
            self.result.session_runtime_s = finished_at - self._session_started_at
            self._write_event_artifact()
        return self.result

    async def _shutdown(self) -> None:
        if self._data_channel is not None and self._data_channel.readyState == "open":
            try:
                self._data_channel.send(json.dumps({"type": "stop"}))
                self._record_event("stop_sent")
            except Exception as exc:
                self._record_event("stop_send_failed", error=str(exc))

        delete_url = f"{self.config.server_url.rstrip('/')}{self.config.delete_path_template.format(session_id=self.session_id)}"
        try:
            await _http_json_request(delete_url, method="DELETE", timeout_s=self.config.shutdown_timeout_s)
            self._record_event("session_delete")
        except Exception as exc:
            self._record_event("session_delete_failed", error=str(exc))

        if self._control_sender_task is not None and not self._control_sender_task.done():
            self._control_sender_task.cancel()
            try:
                await self._control_sender_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if self._state_monitor_task is not None and not self._state_monitor_task.done():
            self._state_monitor_task.cancel()
            try:
                await self._state_monitor_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        for task in self._track_tasks:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        if self._pc is not None:
            try:
                await self._pc.close()
            except Exception as exc:
                self._record_event("peer_close_failed", error=str(exc))

    def _write_event_artifact(self) -> None:
        events_dir = self.artifacts_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        event_file = events_dir / f"{self.phase}_{self.logical_session_index:03d}_{self.session_id}.jsonl"
        with event_file.open("w", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.result.artifacts_event_file = str(event_file)


async def _run_phase(
    *,
    config: BenchConfig,
    phase: str,
    session_count: int,
    artifacts_dir: Path,
    rtc_configuration,
    control_trace: list[dict[str, Any]],
) -> list[SessionResult]:
    sessions: list[StreamBenchSession] = []
    tasks: list[asyncio.Task] = []
    for index in range(session_count):
        session = StreamBenchSession(
            config=config,
            logical_session_index=index,
            phase=phase,
            artifacts_dir=artifacts_dir,
            rtc_configuration=rtc_configuration,
            control_trace=control_trace,
        )
        sessions.append(session)

        async def _start_session(current_session: StreamBenchSession, delay_s: float) -> SessionResult:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            return await current_session.run()

        tasks.append(asyncio.create_task(_start_session(session, config.stagger_s * index)))
    return list(await asyncio.gather(*tasks))


def _result_to_dict(result: SessionResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["control_events"] = [asdict(event) for event in result.control_events]
    return payload


def _build_summary(
    *,
    config: BenchConfig,
    warmup_results: list[SessionResult],
    profile_results: list[SessionResult],
    started_at_iso: str,
) -> dict[str, Any]:
    successful_profile_results = [result for result in profile_results if result.success]
    all_control_events = [
        control_event
        for result in profile_results
        for control_event in result.control_events
        if control_event.ack_latency_ms is not None or control_event.next_frame_latency_ms is not None
    ]
    summary = {
        "started_at_utc": started_at_iso,
        "config": asdict(config),
        "warmup": {
            "attempted_sessions": len(warmup_results),
            "successful_sessions": sum(1 for result in warmup_results if result.success),
            "failed_sessions": sum(1 for result in warmup_results if not result.success),
        },
        "profile": {
            "attempted_sessions": len(profile_results),
            "successful_sessions": len(successful_profile_results),
            "failed_sessions": len(profile_results) - len(successful_profile_results),
            "success_rate": (len(successful_profile_results) / len(profile_results)) if profile_results else 0.0,
            "metrics": {
                "offer_rtt_ms": _summarise([result.offer_rtt_ms for result in profile_results if result.offer_rtt_ms]),
                "connected_latency_ms": _summarise(
                    [result.connected_latency_ms for result in profile_results if result.connected_latency_ms]
                ),
                "first_frame_latency_ms": _summarise(
                    [result.first_frame_latency_ms for result in profile_results if result.first_frame_latency_ms]
                ),
                "first_metadata_latency_ms": _summarise(
                    [
                        result.first_metadata_latency_ms
                        for result in profile_results
                        if result.first_metadata_latency_ms is not None
                    ]
                ),
                "stream_fps": _summarise([result.stream_fps for result in profile_results if result.stream_fps]),
                "session_runtime_s": _summarise(
                    [result.session_runtime_s for result in profile_results if result.session_runtime_s]
                ),
                "frames_received": _summarise([float(result.frames_received) for result in profile_results]),
                "control_ack_latency_ms": _summarise(
                    [event.ack_latency_ms for event in all_control_events if event.ack_latency_ms is not None]
                ),
                "control_to_next_frame_latency_ms": _summarise(
                    [
                        event.next_frame_latency_ms
                        for event in all_control_events
                        if event.next_frame_latency_ms is not None
                    ]
                ),
            },
        },
    }
    return summary


def _print_console_summary(summary: dict[str, Any]) -> None:
    profile = summary["profile"]
    metrics = profile["metrics"]
    print(
        f"Profile sessions: {profile['successful_sessions']}/{profile['attempted_sessions']} succeeded "
        f"(success_rate={profile['success_rate']:.2%})"
    )
    for metric_name in (
        "offer_rtt_ms",
        "connected_latency_ms",
        "first_frame_latency_ms",
        "stream_fps",
        "control_ack_latency_ms",
        "control_to_next_frame_latency_ms",
    ):
        metric = metrics.get(metric_name)
        if metric is None:
            print(f"{metric_name}: n/a")
            continue
        print(
            f"{metric_name}: count={int(metric['count'])} "
            f"mean={metric['mean']:.2f} p50={metric['p50']:.2f} p90={metric['p90']:.2f} max={metric['max']:.2f}"
        )


async def _run_benchmark(config: BenchConfig) -> dict[str, Any]:
    if RTCPeerConnection is None:
        raise RuntimeError(
            "aiortc is required for stream benchmarking. Install TeleFuser with `pip install -e \".[webrtc]\"`."
        ) from AIORTC_IMPORT_ERROR

    resolved_ice_host_ips = config.ice_host_ips
    if not resolved_ice_host_ips:
        resolved_ice_host_ips = _detect_default_ice_host_ips()
    config.ice_host_ips = resolved_ice_host_ips
    configure_ice_host_addresses(resolved_ice_host_ips)
    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    control_trace = _load_control_trace(config.control_trace_path, config.mode)
    artifacts_root = Path(config.artifacts_dir) / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    artifacts_root.mkdir(parents=True, exist_ok=True)
    rtc_configuration = _build_rtc_configuration(config)

    warmup_results = await _run_phase(
        config=config,
        phase="warmup",
        session_count=config.warmup_sessions,
        artifacts_dir=artifacts_root,
        rtc_configuration=rtc_configuration,
        control_trace=control_trace,
    )
    profile_results = await _run_phase(
        config=config,
        phase="profile",
        session_count=config.session_count,
        artifacts_dir=artifacts_root,
        rtc_configuration=rtc_configuration,
        control_trace=control_trace,
    )

    summary = _build_summary(
        config=config,
        warmup_results=warmup_results,
        profile_results=profile_results,
        started_at_iso=started_at_iso,
    )
    summary_path = artifacts_root / "summary.json"
    sessions_path = artifacts_root / "sessions.jsonl"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with sessions_path.open("w", encoding="utf-8") as handle:
        for result in [*warmup_results, *profile_results]:
            handle.write(json.dumps(_result_to_dict(result), ensure_ascii=False) + "\n")

    summary["artifacts_dir"] = str(artifacts_root)
    summary["summary_path"] = str(summary_path)
    summary["sessions_path"] = str(sessions_path)
    return summary


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _load_config(args)
    summary = asyncio.run(_run_benchmark(config))
    _print_console_summary(summary)
    print(f"Artifacts: {summary['artifacts_dir']}")
    print(f"Summary JSON: {summary['summary_path']}")
    print(f"Session records: {summary['sessions_path']}")


if __name__ == "__main__":
    main()
