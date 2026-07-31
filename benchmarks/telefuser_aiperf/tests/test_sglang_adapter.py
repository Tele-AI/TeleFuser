from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import msgspec
import pytest
from aiperf.streaming.adapters import create_stream_adapter
from aiperf.streaming.config import StreamProfileConfig
from aiperf.streaming.contracts import BenchmarkContract
from aiperf.streaming.models import StreamEndpointPaths, StreamSessionPlan
from telefuser_aiperf import register_adapters
from telefuser_aiperf.payload import build_sglang_realtime_init
from telefuser_aiperf.sglang_adapter import SGLangRealtimeAdapter, _websocket_url


class _FakeHttpClient:
    def __init__(self) -> None:
        self.health_urls: list[str] = []

    async def check_health(self, url: str, *, timeout_s: float) -> None:
        self.health_urls.append(url)

    async def aclose(self) -> None:
        return None


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.messages = [
            msgspec.msgpack.encode(
                {
                    "type": "frame_batch",
                    "request_id": "sglang-request",
                    "chunk_index": 0,
                    "event_id": None,
                    "num_frames": 13,
                    "content_type": "image/webp",
                    "payload": b"first",
                }
            ),
            msgspec.msgpack.encode(
                {
                    "type": "chunk_stats",
                    "chunk_index": 0,
                    "event_id": 1,
                    "num_frames": 13,
                    "request_prepare_ms": 2.0,
                    "scheduler_forward_ms": 500.0,
                    "raw_payload_build_ms": 100.0,
                    "pace_wait_ms": 0.0,
                    "header_write_ms": 1.0,
                    "raw_write_ms": 3.0,
                    "ws_write_ms": 4.0,
                    "chunk_total_ms": 607.0,
                    "raw_bytes": 1000,
                    "ws_payload_bytes": 600,
                    "num_batches": 1,
                    "content_type": "image/webp",
                }
            ),
            msgspec.msgpack.encode(
                {
                    "type": "frame_batch_header",
                    "request_id": "sglang-request",
                    "chunk_index": 1,
                    "event_id": 1,
                    "num_frames": 16,
                    "content_type": "image/webp",
                }
            ),
            b"second",
        ]

    async def send(self, payload: bytes) -> None:
        self.sent.append(msgspec.msgpack.decode(payload))

    async def recv(self) -> bytes:
        await asyncio.sleep(0.001)
        if self.messages:
            return self.messages.pop(0)
        await asyncio.sleep(1.0)
        raise AssertionError("unreachable")

    async def close(self) -> None:
        self.closed = True


class _FakeConnect:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> _FakeWebSocket:
        return self.websocket

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


class _FakeConnectFactory:
    def __init__(self) -> None:
        self.websocket = _FakeWebSocket()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> _FakeConnect:
        self.calls.append((url, kwargs))
        return _FakeConnect(self.websocket)


def _contract() -> BenchmarkContract:
    return BenchmarkContract(
        contract_version="v1",
        name="sglang-adapter-test",
        mode="stream_world",
        implementation="sglang",
        model_family="lingbot_world_v2",
        model="world-model",
        supported_tasks=["bidirectional"],
        transport="websocket",
        adapter="sglang_realtime",
        transport_provider="sglang",
        endpoint={
            "health_path": "/health",
            "offer_path": "/v1/realtime_video/generate",
        },
        request_encoding={"format": "msgpack"},
        result_delivery={"media": "websocket_frame_batch"},
        workload={"size": "832x480"},
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
                "shutdown_timeout_s": 0.5,
            },
        }
    )


def _plan(image_path: Path) -> StreamSessionPlan:
    return StreamSessionPlan(
        logical_session_index=0,
        phase="profiling",
        planned_session_id="planned",
        server_url="http://127.0.0.1:30000",
        endpoints=StreamEndpointPaths(
            health_path="/health",
            offer_path="/v1/realtime_video/generate",
        ),
        mode="bidirectional",
        task="bidirectional",
        prompt="walk forward",
        fps=16,
        session_duration_s=0.02,
        image_path=str(image_path),
        request_extra={
            "size": "832x480",
            "max_chunks": 60,
            "num_inference_steps": 4,
        },
        control_trace=[
            {
                "delay_s": 0.0,
                "message": {
                    "type": "control",
                    "key": "ArrowUp",
                    "action": "press",
                },
            }
        ],
    )


def test_registration_uses_unmodified_aiperf_registry(tmp_path: Path) -> None:
    register_adapters(replace=True)

    adapter = create_stream_adapter(
        contract=_contract(),
        config=_config(tmp_path),
        artifacts_dir=tmp_path,
    )

    assert isinstance(adapter, SGLangRealtimeAdapter)
    assert adapter.transport == "websocket"


def test_init_payload_reads_first_frame_and_rejects_overrides(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpeg")
    plan = _plan(image_path)

    payload = build_sglang_realtime_init(plan)

    assert payload["first_frame"] == b"jpeg"
    assert payload["fps"] == 16
    assert payload["max_chunks"] == 60
    with pytest.raises(ValueError, match="prompt"):
        build_sglang_realtime_init(plan.model_copy(update={"request_extra": {"prompt": "override"}}))


def test_websocket_url_preserves_server_base_path() -> None:
    assert (
        _websocket_url("https://example.test/api/", "/v1/realtime_video/generate")
        == "wss://example.test/api/v1/realtime_video/generate"
    )


@pytest.mark.asyncio
async def test_adapter_maps_frames_controls_and_chunk_stats(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"jpeg")
    factory = _FakeConnectFactory()
    http = _FakeHttpClient()
    adapter = SGLangRealtimeAdapter(
        contract=_contract(),
        config=_config(tmp_path),
        artifacts_dir=tmp_path,
        websocket_connect_factory=factory,
        http_client=http,
    )

    await adapter.check_health()
    result = await adapter.run_session(_plan(image_path))

    assert result.success is True
    assert result.session_id == "sglang-request"
    assert result.frames_received == 29
    assert result.stream_fps is not None
    assert result.first_frame_latency_ms is not None
    assert result.chunk_measurements[0].compute_seconds == 0.5
    assert result.chunk_measurements[0].encode_seconds == 0.1
    assert result.chunk_measurements[0].wire_output_bytes == 600
    assert len(result.control_events) == 1
    assert result.control_events[0].ack_latency_ms is not None
    assert result.control_events[0].next_frame_latency_ms is not None
    assert factory.calls[0][0] == "ws://127.0.0.1:30000/v1/realtime_video/generate"
    assert factory.calls[0][1]["proxy"] is None
    assert factory.websocket.sent[0]["type"] == "init"
    assert factory.websocket.sent[1] == {
        "type": "event",
        "kind": "camera_actions",
        "event_id": 1,
        "payload": {
            "mode": "state",
            "transitions": [{"actions": ["w"], "client_ts_ms": pytest.approx(0.0, abs=20.0)}],
        },
    }
    assert factory.websocket.closed is True
    assert http.health_urls == ["http://127.0.0.1:30000/health"]
    assert Path(result.artifacts_event_file or "").is_file()
