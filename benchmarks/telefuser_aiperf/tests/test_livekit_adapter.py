from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import orjson
import pytest
from aiperf.streaming.adapters import create_stream_adapter
from aiperf.streaming.config import StreamProfileConfig
from aiperf.streaming.contracts import BenchmarkContract
from aiperf.streaming.models import StreamEndpointPaths, StreamSessionPlan
from telefuser_aiperf import register_adapters
from telefuser_aiperf.adapter import TeleFuserLiveKitAdapter
from telefuser_aiperf.livekit_room import _disable_proxy_for_loopback
from telefuser_aiperf.payload import build_telefuser_livekit_session_body


class _FakeHttpClient:
    def __init__(self) -> None:
        self.health_urls: list[str] = []
        self.requests: list[tuple[str, str]] = []
        self.payloads: list[dict[str, Any] | None] = []

    async def check_health(self, url: str, *, timeout_s: float) -> None:
        self.health_urls.append(url)

    async def request_json(
        self,
        url: str,
        *,
        method: str,
        timeout_s: float,
        payload: dict[str, Any] | None = None,
        accepted_error_statuses: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        self.requests.append((method, url))
        self.payloads.append(payload)
        if method == "POST":
            return {
                "session_id": "livekit-session",
                "room": "tf-world-livekit-session",
                "livekit_url": "ws://127.0.0.1:7880",
                "token": "test-token",
                "status": "assigned",
            }
        return {}

    async def aclose(self) -> None:
        return None


class _FakeLiveKitRoom:
    instances: list[_FakeLiveKitRoom] = []

    def __init__(self) -> None:
        self.connected: tuple[str, str] | None = None
        self.published: list[tuple[dict[str, Any], str, bool]] = []
        self.disconnected = False
        self.__class__.instances.append(self)

    async def connect(
        self,
        url: str,
        token: str,
        *,
        timeout_s: float,
        on_data: Callable[[bytes | str, str, str], None],
        on_video_frame: Callable[[], None],
        on_event: Callable[[str, Mapping[str, Any]], None],
    ) -> None:
        self.connected = (url, token)
        on_event("remote_track", {"kind": "video"})
        on_data(
            orjson.dumps(
                {
                    "type": "chunk",
                    "data": {"type": "status", "stage": "worker_running"},
                }
            ),
            "tf.status",
            "telefuser-worker-0",
        )
        on_data(
            orjson.dumps(
                {
                    "type": "chunk",
                    "data": {
                        "stage": "runtime_ready",
                        "measurement": {"seconds": 2.0, "memory": []},
                        "runtime": {"cache": "enabled"},
                    },
                }
            ),
            "tf.metrics",
            "telefuser-worker-0",
        )
        on_data(
            orjson.dumps(
                {
                    "type": "chunk",
                    "data": {
                        "stage": "chunk_sent",
                        "measurement": {
                            "index": 0,
                            "frames": 3,
                            "compute_seconds": 0.5,
                            "encode_seconds": 0.1,
                            "memory": [],
                        },
                    },
                }
            ),
            "tf.metrics",
            "telefuser-worker-0",
        )
        on_video_frame()
        on_video_frame()
        on_data(
            orjson.dumps({"type": "done", "session_id": "livekit-session"}),
            "tf.status",
            "telefuser-worker-0",
        )

    async def publish_data(
        self,
        payload: dict[str, Any],
        *,
        topic: str,
        reliable: bool,
    ) -> None:
        self.published.append((payload, topic, reliable))

    async def disconnect(self) -> None:
        self.disconnected = True


class _ControlLiveKitRoom(_FakeLiveKitRoom):
    async def connect(
        self,
        url: str,
        token: str,
        *,
        timeout_s: float,
        on_data: Callable[[bytes | str, str, str], None],
        on_video_frame: Callable[[], None],
        on_event: Callable[[str, Mapping[str, Any]], None],
    ) -> None:
        self.connected = (url, token)
        self._on_data = on_data
        self._on_video_frame = on_video_frame
        on_data(
            orjson.dumps({"type": "chunk", "data": {"stage": "worker_running"}}),
            "tf.status",
            "telefuser-worker-0",
        )
        on_video_frame()

    async def publish_data(
        self,
        payload: dict[str, Any],
        *,
        topic: str,
        reliable: bool,
    ) -> None:
        await super().publish_data(payload, topic=topic, reliable=reliable)
        if payload.get("type") == "stop":
            return
        self._on_data(
            orjson.dumps({"type": "chunk", "data": {"stage": "control_state"}}),
            "tf.status",
            "telefuser-worker-0",
        )
        self._on_video_frame()
        self._on_data(
            orjson.dumps({"type": "done", "session_id": "livekit-session"}),
            "tf.status",
            "telefuser-worker-0",
        )


def _contract() -> BenchmarkContract:
    return BenchmarkContract(
        contract_version="v1",
        name="adapter-test",
        mode="stream_world",
        implementation="telefuser",
        model_family="world",
        model="world-model",
        supported_tasks=["bidirectional"],
        transport="webrtc",
        adapter="telefuser_livekit",
        transport_provider="livekit",
        endpoint={
            "health_path": "/health",
            "offer_path": "/stream",
            "delete_path_template": "/sessions/{session_id}",
        },
        request_encoding={"format": "json"},
        result_delivery={"media": "livekit_video_track"},
        workload={"size": "320x180"},
        metrics=["first_frame_latency_ms"],
        artifacts={"config": "stream.json"},
    )


def _config(tmp_path: Path) -> StreamProfileConfig:
    return StreamProfileConfig.model_validate(
        {
            "contract": "contract.yaml",
            "server_url": "http://127.0.0.1:30000",
            "prompt": "walk forward",
            "artifacts_dir": str(tmp_path),
            "transport": {
                "connect_timeout_s": 0.5,
                "message_timeout_s": 0.5,
                "frame_timeout_s": 0.5,
                "ice_gather_timeout_s": 0.5,
                "shutdown_timeout_s": 0.5,
            },
        }
    )


def _plan() -> StreamSessionPlan:
    return StreamSessionPlan(
        logical_session_index=0,
        phase="profiling",
        planned_session_id="planned",
        server_url="http://127.0.0.1:30000",
        endpoints=StreamEndpointPaths(
            health_path="/health",
            offer_path="/stream",
            delete_path_template="/sessions/{session_id}",
        ),
        mode="bidirectional",
        task="bidirectional",
        prompt="walk forward",
        fps=16,
        session_duration_s=0.03,
        request_extra={"benchmark_metrics": True},
    )


def test_registration_uses_unmodified_aiperf_registry(tmp_path: Path) -> None:
    register_adapters(replace=True)

    adapter = create_stream_adapter(
        contract=_contract(),
        config=_config(tmp_path),
        artifacts_dir=tmp_path,
    )

    assert isinstance(adapter, TeleFuserLiveKitAdapter)
    assert adapter.transport == "webrtc"


def test_payload_rejects_protocol_field_override() -> None:
    plan = _plan().model_copy(update={"request_extra": {"identity": "override"}})

    with pytest.raises(ValueError, match="identity"):
        build_telefuser_livekit_session_body(plan)


def test_proxy_bypass_is_limited_to_loopback_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")

    assert _disable_proxy_for_loopback("wss://livekit.example.test") is False
    assert "http_proxy" in os.environ
    assert _disable_proxy_for_loopback("ws://127.0.0.1:7880") is True
    assert "http_proxy" not in os.environ
    assert "HTTPS_PROXY" not in os.environ


@pytest.mark.asyncio
async def test_adapter_normalizes_room_events_and_deletes_target(
    tmp_path: Path,
) -> None:
    _FakeLiveKitRoom.instances.clear()
    http = _FakeHttpClient()
    adapter = TeleFuserLiveKitAdapter(
        contract=_contract(),
        config=_config(tmp_path),
        artifacts_dir=tmp_path,
        room_client_factory=_FakeLiveKitRoom,
        http_client=http,
    )

    await adapter.check_health()
    result = await adapter.run_session(_plan())

    room = _FakeLiveKitRoom.instances[0]
    assert result.success is True
    assert result.session_id == "livekit-session"
    assert result.frames_received == 2
    assert result.done_received is True
    assert result.phase_measurements[0].name == "runtime_creation"
    assert result.chunk_measurements[0].compute_seconds == 0.5
    assert result.runtime_metadata == {"cache": "enabled"}
    assert room.connected == ("ws://127.0.0.1:7880", "test-token")
    assert room.published[-1] == ({"type": "stop"}, "tf.control", True)
    assert room.disconnected is True
    assert http.payloads[0] == {
        "identity": "aiperf-planned",
        "role": "controller",
        "prompt": "walk forward",
        "config": {
            "task": "bidirectional",
            "fps": 16,
            "benchmark_metrics": True,
        },
    }
    assert (
        "DELETE",
        "http://127.0.0.1:30000/sessions/livekit-session",
    ) in http.requests
    assert Path(result.artifacts_event_file or "").is_file()


@pytest.mark.asyncio
async def test_adapter_maps_control_ack_and_next_frame(tmp_path: Path) -> None:
    _ControlLiveKitRoom.instances.clear()
    adapter = TeleFuserLiveKitAdapter(
        contract=_contract(),
        config=_config(tmp_path),
        artifacts_dir=tmp_path,
        room_client_factory=_ControlLiveKitRoom,
        http_client=_FakeHttpClient(),
    )
    plan = _plan().model_copy(
        update={
            "control_trace": [
                {
                    "delay_s": 0.0,
                    "message": {
                        "type": "control",
                        "key": "ArrowUp",
                        "action": "press",
                    },
                }
            ]
        }
    )

    result = await adapter.run_session(plan)

    assert result.success is True
    assert len(result.control_events) == 1
    assert result.control_events[0].ack_latency_ms is not None
    assert result.control_events[0].next_frame_latency_ms is not None
    room = _ControlLiveKitRoom.instances[0]
    assert room.published[0][1:] == ("tf.control", True)
