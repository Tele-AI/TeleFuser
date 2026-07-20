"""Shared fixtures and helpers for stream integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------


class MockServerPushService:
    """Fake server-push service that yields chunks with optional audio."""

    def __init__(self, num_chunks: int = 3, include_audio: bool = False):
        self._num_chunks = num_chunks
        self._include_audio = include_audio

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def serve(self, request: dict) -> AsyncGenerator[dict, None]:
        import base64

        fake_jpeg = base64.b64encode(b"\xff\xd8\xff\xe0fake-jpeg-data").decode()
        num = request.get("num_chunks", self._num_chunks)
        for i in range(num):
            chunk: dict = {
                "type": "chunk",
                "index": i,
                "frames_b64": [fake_jpeg],
                "fps": 24,
                "prompt": request.get("prompt", ""),
            }
            if self._include_audio:
                import numpy as np

                silence = np.zeros(960, dtype=np.int16)
                chunk["audio_b64"] = base64.b64encode(silence.tobytes()).decode()
                chunk["audio_sample_rate"] = 48000
                chunk["audio_channels"] = 1
            yield chunk


class MockBidirectionalService:
    """Fake bidirectional service with in-memory session tracking."""

    def __init__(self):
        self._sessions: dict[str, list[dict]] = {}
        self._outputs: dict[str, asyncio.Queue] = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._sessions.clear()
        self._outputs.clear()

    def create_session(self, config: dict) -> str:
        import uuid

        sid = config.get("session_id", str(uuid.uuid4()))
        self._sessions[sid] = []
        self._outputs[sid] = asyncio.Queue()
        return sid

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        self._sessions[session_id].append(chunk)
        self._outputs[session_id].put_nowait(
            {"type": "chunk", "index": len(self._sessions[session_id]) - 1, "echo": chunk}
        )

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        if session_id not in self._outputs:
            return
        q = self._outputs[session_id]
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=2.0)
                if chunk.get("type") == "done":
                    break
                yield chunk
            except asyncio.TimeoutError:
                break

    def close_session(self, session_id: str) -> None:
        if session_id in self._outputs:
            self._outputs[session_id].put_nowait({"type": "done"})
        self._sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_stream_svc(service, stream_mode: str):
    from telefuser.service.core.stream_pipeline_service import StreamPipelineService

    svc = StreamPipelineService.__new__(StreamPipelineService)
    svc.is_running = True
    svc.service = service
    svc.stream_mode = stream_mode
    svc.ppl_file = "mock.py"
    svc._module = None
    svc._module_name = None
    svc._startup_measurement = None
    svc._runtime_environment = {}
    svc.security_level = None
    svc.security_validator = None
    return svc


def make_test_server(stream_svc):
    from telefuser.service.api.api_server import ApiServer
    from telefuser.service.core.task_manager import TaskManager

    task_manager = TaskManager(max_queue_size=10)
    server = ApiServer(max_queue_size=10, task_manager=task_manager, enable_openai_api=False)
    server.initialize_stream_service(stream_svc)
    return server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server_push_client():
    server = make_test_server(make_stream_svc(MockServerPushService(num_chunks=5), "server_push"))
    with TestClient(server.app) as client:
        yield client
    asyncio.run(server.cleanup())


@pytest.fixture
def bidirectional_client():
    server = make_test_server(make_stream_svc(MockBidirectionalService(), "bidirectional"))
    with TestClient(server.app) as client:
        yield client
    asyncio.run(server.cleanup())


@pytest.fixture
def audio_server_push_client():
    server = make_test_server(make_stream_svc(MockServerPushService(num_chunks=3, include_audio=True), "server_push"))
    with TestClient(server.app) as client:
        yield client
    asyncio.run(server.cleanup())
