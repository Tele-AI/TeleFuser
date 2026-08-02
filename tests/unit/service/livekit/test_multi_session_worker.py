from __future__ import annotations

import asyncio
import json

import pytest

from telefuser.service.core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL, STREAM_MODE_SERVER_PUSH
from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.multi_session_worker import MultiSessionLiveKitWorker
from telefuser.service.livekit.session_registry import SessionRecord


class _TokenService:
    def create_token(self, *, identity: str, room_name: str, role: str, **kwargs: object) -> str:
        return f"{role}:{identity}:{room_name}"


class _PipelineAdapter:
    stream_mode = STREAM_MODE_BIDIRECTIONAL

    def __init__(self) -> None:
        self.start_calls = 0
        self.closed_service = False
        self.created: list[str] = []
        self.pushed: list[tuple[str, dict]] = []
        self.closed: list[str] = []
        self.queues: dict[str, asyncio.Queue[dict | None]] = {}
        self.capacity_profile: dict[str, object] | None = None

    def start(self, pipeline_file: str, *, skip_validation: bool = False, gpu_num: int = 1) -> None:
        del pipeline_file, skip_validation, gpu_num
        self.start_calls += 1

    async def aclose(self) -> None:
        self.closed_service = True

    def create_session(self, config: dict) -> str:
        session_id = str(config["session_id"])
        self.created.append(session_id)
        self.queues[session_id] = asyncio.Queue()
        return session_id

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self.pushed.append((session_id, chunk))

    async def pull_chunks(self, session_id: str):
        while True:
            chunk = await self.queues[session_id].get()
            if chunk is None:
                return
            yield chunk

    def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    def configure_session_capacity(self, max_sessions: int | None) -> dict[str, object] | None:
        del max_sessions
        return self.capacity_profile


class _RoomClient:
    def __init__(self) -> None:
        self.connected = asyncio.Event()
        self.on_data = None
        self.statuses: list[dict] = []
        self.disconnected = False

    async def connect(self, url: str, token: str, on_data) -> None:
        del url, token
        self.on_data = on_data
        self.connected.set()

    async def wait_for_participant(self, identity: str, *, timeout_s: float) -> None:
        return None

    async def publish_video_track(self, name: str, width: int, height: int, *, fps: float = 16.0) -> None:
        return None

    async def publish_video_frame(self, frame_rgb, *, fps: float = 16.0) -> None:
        return None

    async def publish_audio_frame(self, pcm: bytes, *, sample_rate: int, channels: int) -> None:
        return None

    async def publish_status(self, payload: dict) -> None:
        self.statuses.append(payload)

    async def publish_metrics(self, payload: dict) -> None:
        return None

    async def disconnect(self) -> None:
        self.disconnected = True

    def emit_control(self, payload: dict, *, identity: str) -> None:
        assert self.on_data is not None
        self.on_data(json.dumps(payload), "tf.control", identity)


class _EventSink:
    def __init__(self) -> None:
        self.worker_statuses: list[str] = []

    def on_worker_status(self, worker_id: str, status: str) -> None:
        del worker_id
        self.worker_statuses.append(status)

    def on_session_status(self, session_id: str, status: str, error: str | None = None) -> None:
        return None

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        return None

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        return None


def _record(session_id: str) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        room_name=f"room-{session_id}",
        controller_identity=f"controller-{session_id}",
        status="assigned",
        worker_id="worker-0",
        config={"session_id": session_id},
        created_at=0,
        updated_at=0,
    )


async def _wait_for(predicate, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


def test_multi_session_worker_loads_one_pipeline_and_routes_two_rooms() -> None:
    async def _run() -> None:
        adapter = _PipelineAdapter()
        rooms: list[_RoomClient] = []

        def room_factory() -> _RoomClient:
            room = _RoomClient()
            rooms.append(room)
            return room

        worker = MultiSessionLiveKitWorker(
            worker_id="worker-0",
            config=LiveKitServeConfig(
                livekit_url="wss://livekit.example",
                livekit_api_key="key",
                livekit_api_secret="secret",
                max_sessions_per_worker=2,
            ),
            pipeline_file="pipeline.py",
            token_service=_TokenService(),
            pipeline_adapter=adapter,
            room_client_factory=room_factory,
        )
        await worker.start(skip_validation=True)
        tasks = [asyncio.create_task(worker.run_session(_record(session_id))) for session_id in ("a", "b")]
        await _wait_for(lambda: len(adapter.created) == 2 and len(rooms) == 2)

        rooms[0].emit_control({"type": "control_state", "controls": ["w"]}, identity="controller-a")
        rooms[1].emit_control({"type": "control_state", "controls": ["j"]}, identity="controller-b")
        await _wait_for(lambda: len(adapter.pushed) == 2)
        for session_id in ("a", "b"):
            await adapter.queues[session_id].put(None)
        await asyncio.gather(*tasks)
        await worker.stop()

        assert adapter.start_calls == 1
        assert adapter.created == ["a", "b"]
        assert adapter.pushed == [
            ("a", {"type": "control_state", "controls": ["w"]}),
            ("b", {"type": "control_state", "controls": ["j"]}),
        ]
        assert sorted(adapter.closed) == ["a", "b"]
        assert all(room.disconnected for room in rooms)
        assert adapter.closed_service is True

    asyncio.run(_run())


def test_multi_session_worker_uses_pipeline_calculated_capacity() -> None:
    async def _run() -> None:
        adapter = _PipelineAdapter()
        adapter.capacity_profile = {"effective_capacity": 3}
        worker = MultiSessionLiveKitWorker(
            worker_id="worker-0",
            config=LiveKitServeConfig(
                livekit_url="wss://livekit.example",
                livekit_api_key="key",
                livekit_api_secret="secret",
            ),
            pipeline_file="pipeline.py",
            token_service=_TokenService(),
            pipeline_adapter=adapter,
        )

        await worker.start(skip_validation=True)
        assert worker._session_capacity == 3
        await worker.stop()

    asyncio.run(_run())


def test_multi_session_worker_rejects_server_push_capacity_above_one() -> None:
    async def _run() -> None:
        adapter = _PipelineAdapter()
        adapter.stream_mode = STREAM_MODE_SERVER_PUSH
        worker = MultiSessionLiveKitWorker(
            worker_id="worker-0",
            config=LiveKitServeConfig(
                livekit_url="wss://livekit.example",
                livekit_api_key="key",
                livekit_api_secret="secret",
                max_sessions_per_worker=2,
            ),
            pipeline_file="pipeline.py",
            token_service=_TokenService(),
            pipeline_adapter=adapter,
        )

        with pytest.raises(
            RuntimeError,
            match="Multiple retained sessions require a BidirectionalService pipeline",
        ):
            await worker.start(skip_validation=True)

        assert adapter.closed_service is True

    asyncio.run(_run())


def test_multi_session_worker_aggregates_runner_statuses() -> None:
    event_sink = _EventSink()
    worker = MultiSessionLiveKitWorker(
        worker_id="worker-0",
        config=LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            max_sessions_per_worker=2,
        ),
        pipeline_file="pipeline.py",
        token_service=_TokenService(),
        pipeline_adapter=_PipelineAdapter(),
        event_sink=event_sink,
    )
    worker._session_worker_statuses.update({"a": "assigned", "b": "assigned"})

    worker._on_session_worker_status("a", "running")
    worker._on_session_worker_status("b", "joining_room")

    assert event_sink.worker_statuses[-1] == "running"
