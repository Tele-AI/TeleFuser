"""Worker pool implementations for LiveKit serving."""

from __future__ import annotations

import asyncio
from typing import Protocol

from telefuser.utils.logging import logger

from .session_registry import SessionRecord
from .worker import LiveKitWorker


class WorkerPool(Protocol):
    """Worker-pool operations used by the API runtime."""

    async def start(self, *, skip_validation: bool = False) -> None: ...

    def start_session(self, record: SessionRecord) -> None: ...
    async def stop_session(self, session_id: str) -> None: ...
    async def aclose(self) -> None: ...


class InProcessLiveKitWorkerPool:
    """Run LiveKit workers as asyncio tasks in the API server process."""

    def __init__(self, workers: dict[str, LiveKitWorker]) -> None:
        self._workers = workers
        self._started = False
        self._tasks: dict[str, asyncio.Task] = {}

    async def start(self, *, skip_validation: bool = False) -> None:
        """Load all worker-owned pipelines."""
        if self._started:
            return
        for worker in self._workers.values():
            await worker.start(skip_validation=skip_validation)

        self._started = True

    def start_session(self, record: SessionRecord) -> None:
        """Start a worker task for an assigned session."""
        if not self._started:
            raise RuntimeError("LiveKit worker pool is not started")
        if record.worker_id is None:
            raise RuntimeError(f"Session {record.session_id} has no assigned worker")
        if record.session_id in self._tasks:
            raise RuntimeError(f"Session {record.session_id} is already running")

        worker = self._workers[record.worker_id]
        task = asyncio.create_task(worker.run_session(record), name=f"livekit-worker-{record.worker_id}")
        self._tasks[record.session_id] = task
        task.add_done_callback(lambda done: self._on_task_done(record.session_id, done))

    async def stop_session(self, session_id: str) -> None:
        """Request an active session to stop and wait for cleanup."""
        task = self._tasks.get(session_id)
        if task is None:
            return
        for worker in self._workers.values():
            await worker.stop_session(session_id)
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def aclose(self) -> None:
        """Stop every active session and worker."""
        session_ids = list(self._tasks.keys())
        self._started = False
        for session_id in session_ids:
            await self.stop_session(session_id)
        for worker in self._workers.values():
            await worker.stop()

    def _on_task_done(self, session_id: str, task: asyncio.Task) -> None:
        self._tasks.pop(session_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(f"LiveKit worker task failed: session={session_id} error={exc}")
