from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from telefuser.service.core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL
from telefuser.service.livekit.pipeline_router import TurboServePipelineRouter


class _MigratableService:
    def __init__(self, *, fail_import: bool = False) -> None:
        self.fail_import = fail_import
        self.sessions: set[str] = set()
        self.queues: dict[str, asyncio.Queue[dict | None]] = {}
        self.prepared: set[str] = set()
        self.aborted: set[str] = set()
        self.on_import = None

    def create(self, session_id: str) -> str:
        self.sessions.add(session_id)
        self.queues[session_id] = asyncio.Queue()
        return session_id

    def prepare_migration(self, session_id: str) -> dict[str, str]:
        self.prepared.add(session_id)
        return {"session_id": session_id}

    def import_migration(self, bundle: dict[str, str], **_: object) -> str:
        if self.fail_import:
            raise RuntimeError("target import failed")
        session_id = self.create(bundle["session_id"])
        if self.on_import is not None:
            self.on_import()
        return session_id

    def commit_migration(self, session_id: str) -> None:
        self.close_session(session_id)

    def abort_migration(self, session_id: str) -> None:
        self.prepared.discard(session_id)
        self.aborted.add(session_id)

    def close_session(self, session_id: str) -> None:
        self.sessions.discard(session_id)
        queue = self.queues.get(session_id)
        if queue is not None:
            queue.put_nowait(None)


class _Backend:
    stream_mode = STREAM_MODE_BIDIRECTIONAL

    def __init__(self, *, fail_import: bool = False) -> None:
        self.service = _MigratableService(fail_import=fail_import)
        self.stream_service = SimpleNamespace(service=self.service)
        self.started: list[dict[str, object]] = []
        self.pushed: list[tuple[str, dict]] = []

    def start(self, pipeline_file: str, **kwargs: object) -> None:
        self.started.append({"pipeline_file": pipeline_file, **kwargs})

    async def aclose(self) -> None:
        return None

    def configure_session_capacity(self, max_sessions: int | None) -> dict[str, object]:
        return {"effective_capacity": max_sessions or 1}

    def create_session(self, config: dict) -> str:
        return self.service.create(str(config["session_id"]))

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self.pushed.append((session_id, chunk))

    async def pull_chunks(self, session_id: str):
        queue = self.service.queues[session_id]
        while True:
            chunk = await queue.get()
            if chunk is None:
                return
            yield chunk

    async def stream_task(self, config: dict):
        if False:
            yield config

    def close_session(self, session_id: str) -> None:
        self.service.close_session(session_id)


def test_router_switches_pull_and_push_without_recreating_transport() -> None:
    async def _run() -> None:
        source = _Backend()
        target = _Backend()
        router = TurboServePipelineRouter({"worker-0": source, "worker-1": target})
        view = router.worker_view("worker-0", gpu_ids=["0"])
        view.start("pipeline.py", skip_validation=True, gpu_num=1)
        session_id = view.create_session({"session_id": "session-a"})
        chunks = router.pull_chunks(session_id)

        source.service.queues[session_id].put_nowait({"index": 0})
        assert await anext(chunks) == {"index": 0}
        target.service.on_import = lambda: router.push_chunk(session_id, {"type": "during_migration"})
        ownership = router.migrate_session(session_id, "worker-1")
        target.service.queues[session_id].put_nowait({"index": 1})
        router.push_chunk(session_id, {"type": "control_state"})

        assert await anext(chunks) == {"index": 1}
        assert ownership.worker_id == "worker-1"
        assert ownership.epoch == 2
        assert target.pushed == [
            (session_id, {"type": "during_migration"}),
            (session_id, {"type": "control_state"}),
        ]
        assert router.snapshot()["routes"] == {session_id: "worker-1"}
        assert source.started[0]["gpu_ids"] == ["0"]

        router.close_session(session_id)
        with pytest.raises(StopAsyncIteration):
            await anext(chunks)

    asyncio.run(_run())


def test_router_aborts_source_when_target_import_fails() -> None:
    source = _Backend()
    target = _Backend(fail_import=True)
    router = TurboServePipelineRouter({"worker-0": source, "worker-1": target})
    session_id = router.create_session("worker-0", {"session_id": "session-a"})

    with pytest.raises(RuntimeError, match="target import failed"):
        router.migrate_session(session_id, "worker-1")

    assert router.snapshot()["routes"] == {session_id: "worker-0"}
    assert source.service.aborted == {session_id}
