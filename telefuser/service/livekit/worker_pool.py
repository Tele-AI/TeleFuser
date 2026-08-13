"""Worker pool implementations for LiveKit serving."""

from __future__ import annotations

import asyncio
from typing import Protocol

from telefuser.utils.logging import logger

from .pipeline_router import TurboServePipelineRouter
from .session_registry import SessionRecord
from .turboserve import TurboServeOwnership
from .worker import LiveKitWorker

_SESSION_STOP_GRACE_SECONDS = 8.0
_SESSION_CANCEL_GRACE_SECONDS = 1.0


class WorkerPool(Protocol):
    """Worker-pool operations used by the API runtime."""

    async def start(self, *, skip_validation: bool = False) -> None: ...

    def start_session(self, record: SessionRecord) -> None: ...
    async def stop_session(self, session_id: str) -> None: ...
    async def aclose(self) -> None: ...


class InProcessLiveKitWorkerPool:
    """Run independently device-bound LiveKit workers in the API process."""

    def __init__(
        self,
        workers: dict[str, LiveKitWorker],
        *,
        router: TurboServePipelineRouter | None = None,
        initial_workers: int | None = None,
    ) -> None:
        if initial_workers is not None and not 1 <= initial_workers <= len(workers):
            raise ValueError("initial_workers must be within the configured worker pool")
        self._workers = workers
        self.router = router
        self._initial_workers = initial_workers
        self._started = False
        self._skip_validation = False
        self._tasks: dict[str, asyncio.Task] = {}
        self._task_workers: dict[str, str] = {}
        self._active_workers: set[str] = set()
        self._scale_lock = asyncio.Lock()

    async def start(self, *, skip_validation: bool = False) -> None:
        """Load the configured initial replica set."""
        if self._started:
            return
        self._started = True
        self._skip_validation = skip_validation
        target = self._initial_workers or len(self._workers)
        try:
            await self.scale_to(target)
        except Exception:
            self._started = False
            raise
        for worker_id, worker in self._workers.items():
            if worker_id not in self._active_workers:
                worker.event_sink.on_worker_status(worker_id, "stopped")

    def start_session(self, record: SessionRecord) -> None:
        """Start a room runner on its assigned active worker."""
        if not self._started:
            raise RuntimeError("LiveKit worker pool is not started")
        if record.worker_id is None:
            raise RuntimeError(f"Session {record.session_id} has no assigned worker")
        if record.worker_id not in self._active_workers:
            raise RuntimeError(f"Worker {record.worker_id} is not active")
        if record.session_id in self._tasks:
            raise RuntimeError(f"Session {record.session_id} is already running")

        worker = self._workers[record.worker_id]
        task = asyncio.create_task(worker.run_session(record), name=f"livekit-worker-{record.worker_id}")
        self._tasks[record.session_id] = task
        self._task_workers[record.session_id] = record.worker_id
        task.add_done_callback(lambda done: self._on_task_done(record.session_id, done))

    async def stop_session(self, session_id: str) -> None:
        """Request an active session to stop and wait for cleanup."""
        task = self._tasks.get(session_id)
        if task is None:
            return
        worker_id = self._task_workers.get(session_id)
        if worker_id is not None:
            await self._workers[worker_id].stop_session(session_id)

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_SESSION_STOP_GRACE_SECONDS)
            return
        except asyncio.TimeoutError:
            logger.warning(
                f"LiveKit session did not stop within {_SESSION_STOP_GRACE_SECONDS:g}s; "
                f"cancelling runner: session={session_id}"
            )

        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(task, timeout=_SESSION_CANCEL_GRACE_SECONDS)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            logger.warning(
                f"LiveKit session runner did not acknowledge cancellation within "
                f"{_SESSION_CANCEL_GRACE_SECONDS:g}s: session={session_id}"
            )

    async def migrate_session(
        self, pipeline_session_id: str, target_worker_id: str
    ) -> TurboServeOwnership:
        """Move model state while the existing LiveKit runner keeps publishing."""
        if self.router is None:
            raise RuntimeError("Worker pool was created without TurboServe routing")
        if target_worker_id not in self._active_workers:
            raise RuntimeError(f"Migration target {target_worker_id} is not active")
        return await asyncio.to_thread(self.router.migrate_session, pipeline_session_id, target_worker_id)

    async def scale_to(self, target_workers: int) -> int:
        """Start replicas or retire idle replicas until the requested count is reached."""
        if not self._started:
            raise RuntimeError("LiveKit worker pool is not started")
        if not 1 <= target_workers <= len(self._workers):
            raise ValueError("target_workers must be within the configured worker pool")
        async with self._scale_lock:
            while len(self._active_workers) < target_workers:
                worker_id = next(worker_id for worker_id in self._workers if worker_id not in self._active_workers)
                worker = self._workers[worker_id]
                await worker.start(skip_validation=self._skip_validation)
                self._active_workers.add(worker_id)

            while len(self._active_workers) > target_workers:
                candidate = self._scale_in_candidate()
                if candidate is None:
                    break
                await self._workers[candidate].stop()
                self._active_workers.remove(candidate)
            return len(self._active_workers)

    def active_worker_count(self) -> int:
        return len(self._active_workers)

    def turboserve_snapshot(self) -> dict[str, object] | None:
        snapshot = self.router.snapshot() if self.router is not None else {}
        return {
            **snapshot,
            "active_workers": sorted(self._active_workers),
            "configured_workers": len(self._workers),
        }

    async def aclose(self) -> None:
        """Stop every active session and loaded worker."""
        session_ids = list(self._tasks)
        for session_id in session_ids:
            await self.stop_session(session_id)
        for worker_id in tuple(self._active_workers):
            await self._workers[worker_id].stop()
        self._active_workers.clear()
        self._started = False

    def _scale_in_candidate(self) -> str | None:
        busy_transport_workers = set(self._task_workers.values())
        retained_by_worker: dict[str, int] = {}
        if self.router is not None:
            retained_by_worker = self.router.snapshot()["retained_sessions_by_worker"]
        candidates = [
            worker_id
            for worker_id in reversed(tuple(self._workers))
            if worker_id in self._active_workers
            and worker_id not in busy_transport_workers
            and retained_by_worker.get(worker_id, 0) == 0
        ]
        return candidates[0] if candidates else None

    def _on_task_done(self, session_id: str, task: asyncio.Task) -> None:
        self._tasks.pop(session_id, None)
        self._task_workers.pop(session_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning(f"LiveKit worker task failed: session={session_id} error={exc}")
