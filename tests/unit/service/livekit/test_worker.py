from __future__ import annotations

import asyncio
import base64
import json

import cv2
import numpy as np
from PIL import Image

from telefuser.service.core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL, STREAM_MODE_SERVER_PUSH
from telefuser.service.livekit import worker as worker_module
from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.session_registry import SessionRecord
from telefuser.service.livekit.worker import LiveKitWorker


class FakeTokenService:
    def create_token(self, *, identity: str, room_name: str, role: str, **kwargs: object) -> str:
        return f"{role}:{identity}:{room_name}"


class FakePipelineAdapter:
    def __init__(self, stream_mode: str = STREAM_MODE_BIDIRECTIONAL) -> None:
        self.stream_mode = stream_mode
        self.started: list[dict[str, object]] = []
        self.created_config: dict | None = None
        self.served_config: dict | None = None
        self.pushed: list[tuple[str, dict]] = []
        self.closed: list[str] = []
        self.closed_service = False
        self.created = asyncio.Event()
        self.output_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def start(self, pipeline_file: str, *, skip_validation: bool = False, gpu_num: int = 1) -> None:
        self.started.append({"pipeline_file": pipeline_file, "skip_validation": skip_validation, "gpu_num": gpu_num})

    async def aclose(self) -> None:
        self.closed_service = True

    def create_session(self, config: dict) -> str:
        self.created_config = config
        self.created.set()
        return "pipeline-session-1"

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self.pushed.append((session_id, chunk))

    async def pull_chunks(self, session_id: str):
        while True:
            item = await self.output_queue.get()
            if item is None:
                break
            yield item

    async def stream_task(self, config: dict):
        self.served_config = config
        while True:
            item = await self.output_queue.get()
            if item is None:
                break
            yield item

    def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)


class FakeRoomClient:
    def __init__(self) -> None:
        self.connected = asyncio.Event()
        self.connect_args: tuple[str, str] | None = None
        self.on_data = None
        self.video_frames: list[np.ndarray] = []
        self.video_frame_fps: list[float] = []
        self.audio_frames: list[tuple[bytes, int, int]] = []
        self.statuses: list[dict] = []
        self.disconnected = False
        self.disconnect_gate: asyncio.Event | None = None
        self.participant_gate = asyncio.Event()
        self.participant_gate.set()
        self.waited_for_participants: list[tuple[str, float]] = []

    async def connect(self, url: str, token: str, on_data) -> None:
        self.connect_args = (url, token)
        self.on_data = on_data
        self.connected.set()

    async def wait_for_participant(self, identity: str, *, timeout_s: float) -> None:
        self.waited_for_participants.append((identity, timeout_s))
        await asyncio.wait_for(self.participant_gate.wait(), timeout=timeout_s)

    async def publish_video_track(self, name: str, width: int, height: int, *, fps: float = 16.0) -> None:
        return None

    async def publish_video_frame(self, frame_rgb: np.ndarray, *, fps: float = 16.0) -> None:
        self.video_frames.append(frame_rgb)
        self.video_frame_fps.append(fps)

    async def publish_audio_frame(self, pcm: bytes, *, sample_rate: int, channels: int) -> None:
        self.audio_frames.append((pcm, sample_rate, channels))

    async def publish_status(self, payload: dict) -> None:
        self.statuses.append(payload)

    async def publish_metrics(self, payload: dict) -> None:
        return None

    async def disconnect(self) -> None:
        if self.disconnect_gate is not None:
            await self.disconnect_gate.wait()
        self.disconnected = True

    def emit_control(self, payload: dict, *, identity: str = "controller") -> None:
        assert self.on_data is not None
        self.on_data(json.dumps(payload), "tf.control", identity)


class FakeSink:
    def __init__(self) -> None:
        self.worker_statuses: list[tuple[str, str]] = []
        self.session_statuses: list[tuple[str, str, str | None]] = []
        self.pipeline_sessions: list[tuple[str, str]] = []
        self.finished: list[tuple[str, str, str | None]] = []

    def on_worker_status(self, worker_id: str, status: str) -> None:
        self.worker_statuses.append((worker_id, status))

    def on_session_status(self, session_id: str, status: str, error: str | None = None) -> None:
        self.session_statuses.append((session_id, status, error))

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        self.pipeline_sessions.append((session_id, pipeline_session_id))

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        self.finished.append((worker_id, session_id, error))


def _jpeg_chunk() -> dict:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    return {
        "type": "chunk",
        "index": 0,
        "fps": 16,
        "frames_b64": [base64.b64encode(encoded.tobytes()).decode("ascii")],
    }


def _native_chunk() -> dict:
    return {
        "type": "chunk",
        "index": 1,
        "fps": 16,
        "timestamp": 123.0,
        "frames": [Image.new("RGB", (8, 8), color=(1, 2, 3))],
        "stream_progress": {"completed_chunks": 2},
    }


def _audio_chunk() -> dict:
    pcm = np.zeros(960, dtype=np.int16).tobytes()
    return {
        "type": "chunk",
        "index": 2,
        "audio_b64": base64.b64encode(pcm).decode("ascii"),
        "audio_sample_rate": 48_000,
        "audio_channels": 1,
    }


async def _wait_for(predicate, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.01)


def test_livekit_worker_runs_pipeline_and_forwards_control() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example", livekit_api_key="key", livekit_api_secret="secret"
        )
        adapter = FakePipelineAdapter()
        room = FakeRoomClient()
        sink = FakeSink()
        worker = LiveKitWorker(
            worker_id="worker-0",
            config=config,
            pipeline_file="pipeline.py",
            token_service=FakeTokenService(),
            event_sink=sink,
            pipeline_adapter=adapter,
            room_client=room,
        )
        record = SessionRecord(
            session_id="session-1",
            room_name="room-1",
            controller_identity="controller",
            status="assigned",
            worker_id="worker-0",
            config={"session_id": "session-1", "fps": 16},
            created_at=0,
            updated_at=0,
        )

        await worker.start(skip_validation=True)
        task = asyncio.create_task(worker.run_session(record))
        await room.connected.wait()
        await adapter.created.wait()
        room.emit_control({"type": "control", "event": "press", "key": "ArrowUp"})
        await _wait_for(lambda: len(adapter.pushed) == 1)
        await adapter.output_queue.put({"type": "status", "stage": "chunk_decoded", "frames": 13})
        await adapter.output_queue.put(_jpeg_chunk())
        await adapter.output_queue.put(_native_chunk())
        await adapter.output_queue.put(_audio_chunk())
        await adapter.output_queue.put(None)
        await task

        assert adapter.started == [{"pipeline_file": "pipeline.py", "skip_validation": True, "gpu_num": 1}]
        assert adapter.created_config == {"session_id": "session-1", "fps": 16}
        assert adapter.pushed == [("pipeline-session-1", {"type": "control", "event": "press", "key": "ArrowUp"})]
        assert adapter.closed == ["pipeline-session-1"]
        assert room.connect_args == ("wss://livekit.example", "worker:telefuser-worker-0:room-1")
        assert room.waited_for_participants == [("controller", worker_module._CONTROLLER_JOIN_TIMEOUT_SECONDS)]
        assert len(room.video_frames) == 2
        assert room.video_frame_fps == [16.0, 16.0]
        assert room.audio_frames == [(np.zeros(960, dtype=np.int16).tobytes(), 48_000, 1)]
        assert any(status.get("data", {}).get("frames") == 13 for status in room.statuses)
        assert any(status.get("data", {}).get("stream_progress") == {"completed_chunks": 2} for status in room.statuses)
        transport = next(
            status["data"]["transport_measurement"]
            for status in room.statuses
            if status.get("data", {}).get("index") == 1
        )
        assert transport["decoded_ready_at"] == 123.0
        assert transport["pacing"] == "realtime"
        assert transport["frames"] == 1
        assert transport["publish_started_at"] <= transport["publish_finished_at"]
        assert room.statuses[-1]["type"] == "done"
        assert room.disconnected is True
        assert sink.pipeline_sessions == [("session-1", "pipeline-session-1")]
        assert room.statuses[-1]["total_chunks"] == 2
        assert sink.finished == [("worker-0", "session-1", None)]

    asyncio.run(_run())


def test_livekit_worker_waits_for_controller_before_creating_pipeline() -> None:
    async def _run() -> None:
        adapter = FakePipelineAdapter()
        room = FakeRoomClient()
        room.participant_gate.clear()
        worker = LiveKitWorker(
            worker_id="worker-0",
            config=LiveKitServeConfig(
                livekit_url="wss://livekit.example",
                livekit_api_key="key",
                livekit_api_secret="secret",
            ),
            pipeline_file="pipeline.py",
            token_service=FakeTokenService(),
            pipeline_adapter=adapter,
            room_client=room,
        )
        record = SessionRecord(
            session_id="session-1",
            room_name="room-1",
            controller_identity="controller",
            status="assigned",
            worker_id="worker-0",
            config={"session_id": "session-1"},
            created_at=0,
            updated_at=0,
        )

        task = asyncio.create_task(worker.run_session(record))
        await room.connected.wait()
        await asyncio.sleep(0)
        assert adapter.created_config is None

        room.participant_gate.set()
        await adapter.created.wait()
        await adapter.output_queue.put(None)
        await task

    asyncio.run(_run())


def test_livekit_worker_runs_server_push_pipeline() -> None:
    async def _run() -> None:
        adapter = FakePipelineAdapter(stream_mode=STREAM_MODE_SERVER_PUSH)
        room = FakeRoomClient()
        sink = FakeSink()
        worker = LiveKitWorker(
            worker_id="worker-0",
            config=LiveKitServeConfig(
                livekit_url="wss://livekit.example",
                livekit_api_key="key",
                livekit_api_secret="secret",
            ),
            pipeline_file="pipeline.py",
            token_service=FakeTokenService(),
            event_sink=sink,
            pipeline_adapter=adapter,
            room_client=room,
        )
        record = SessionRecord(
            session_id="session-1",
            room_name="room-1",
            controller_identity="controller",
            status="assigned",
            worker_id="worker-0",
            config={"session_id": "session-1", "prompt": "sunset", "fps": 16},
            created_at=0,
            updated_at=0,
        )

        await worker.start(skip_validation=True)
        task = asyncio.create_task(worker.run_session(record))
        await room.connected.wait()
        await adapter.output_queue.put(_native_chunk())
        await adapter.output_queue.put(None)
        await task

        assert adapter.served_config == record.config
        assert adapter.created_config is None
        assert adapter.closed == []
        assert sink.pipeline_sessions == []
        assert len(room.video_frames) == 1
        assert room.statuses[-1]["type"] == "done"

    asyncio.run(_run())


def test_livekit_worker_bounds_room_disconnect(monkeypatch) -> None:
    async def _run() -> None:
        room = FakeRoomClient()
        room.disconnect_gate = asyncio.Event()
        worker = LiveKitWorker(
            worker_id="worker-0",
            config=LiveKitServeConfig(
                livekit_url="wss://livekit.example",
                livekit_api_key="key",
                livekit_api_secret="secret",
            ),
            pipeline_file="pipeline.py",
            token_service=FakeTokenService(),
            pipeline_adapter=FakePipelineAdapter(),
            room_client=room,
        )
        worker._active_session_id = "session-1"
        monkeypatch.setattr(worker_module, "_ROOM_DISCONNECT_TIMEOUT_SECONDS", 0.01)

        await asyncio.wait_for(worker._close_active_session(), timeout=1)

        assert worker._active_session_id is None
        assert room.disconnected is False

    asyncio.run(_run())
