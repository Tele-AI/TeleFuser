"""Worker admission control for LiveKit serving."""

from __future__ import annotations

import threading
from collections import deque
from typing import Literal

from pydantic import BaseModel, Field

from .schemas import utc_timestamp

WorkerStatus = Literal[
    "starting",
    "idle",
    "assigned",
    "joining_room",
    "starting_pipeline",
    "running",
    "draining",
    "failed",
    "stopped",
]
AdmissionStatus = Literal["assigned", "queued", "rejected"]


class WorkerState(BaseModel):
    """Scheduler-visible state for one model worker."""

    worker_id: str
    status: WorkerStatus
    gpu_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    room_name: str | None = None
    last_heartbeat_at: float
    error: str | None = None


class SchedulerAdmission(BaseModel):
    """Result of a scheduler admission attempt."""

    status: AdmissionStatus
    worker_id: str | None = None
    queue_position: int | None = None
    reason: str | None = None


class _QueuedSession(BaseModel):
    session_id: str
    room_name: str


class LiveKitScheduler:
    """Simple FIFO scheduler with one active session per worker."""

    def __init__(self, *, num_workers: int, gpu_groups: list[list[str]] | None = None, queue_size: int = 0) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if queue_size < 0:
            raise ValueError("queue_size must be >= 0")

        now = utc_timestamp()
        groups = gpu_groups or [[] for _ in range(num_workers)]
        if len(groups) != num_workers:
            raise ValueError(f"Expected {num_workers} GPU groups, got {len(groups)}")

        self._workers: dict[str, WorkerState] = {
            f"worker-{idx}": WorkerState(
                worker_id=f"worker-{idx}",
                status="idle",
                gpu_ids=list(groups[idx]),
                last_heartbeat_at=now,
            )
            for idx in range(num_workers)
        }
        self._queue_size = queue_size
        self._queue: deque[_QueuedSession] = deque()
        self._lock = threading.RLock()

    def assign(self, *, session_id: str, room_name: str) -> SchedulerAdmission:
        """Assign an idle worker or enqueue/reject the session."""
        with self._lock:
            worker = self._first_idle_worker()
            if worker is not None:
                worker.status = "assigned"
                worker.session_id = session_id
                worker.room_name = room_name
                worker.error = None
                worker.last_heartbeat_at = utc_timestamp()
                return SchedulerAdmission(status="assigned", worker_id=worker.worker_id)

            if self._queue_size == 0:
                return SchedulerAdmission(status="rejected", reason="no_idle_worker")
            if len(self._queue) >= self._queue_size:
                return SchedulerAdmission(status="rejected", reason="queue_full")

            self._queue.append(_QueuedSession(session_id=session_id, room_name=room_name))
            return SchedulerAdmission(status="queued", queue_position=len(self._queue))

    def release_session(self, session_id: str) -> SchedulerAdmission | None:
        """Release the worker owning ``session_id`` and assign the next queued session if present."""
        with self._lock:
            worker = self._worker_for_session(session_id)
            if worker is None:
                self._queue = deque(item for item in self._queue if item.session_id != session_id)
                return None

            worker.session_id = None
            worker.room_name = None
            worker.error = None
            worker.status = "idle"
            worker.last_heartbeat_at = utc_timestamp()

            if not self._queue:
                return None

            queued = self._queue.popleft()
            worker.status = "assigned"
            worker.session_id = queued.session_id
            worker.room_name = queued.room_name
            worker.last_heartbeat_at = utc_timestamp()
            return SchedulerAdmission(status="assigned", worker_id=worker.worker_id)

    def update_worker_status(self, worker_id: str, status: WorkerStatus) -> WorkerState:
        """Update a worker lifecycle status."""
        with self._lock:
            worker = self._workers[worker_id]
            worker.status = status
            worker.last_heartbeat_at = utc_timestamp()
            return worker.model_copy(deep=True)

    def heartbeat(self, worker_id: str) -> WorkerState:
        """Record a worker heartbeat."""
        with self._lock:
            worker = self._workers[worker_id]
            worker.last_heartbeat_at = utc_timestamp()
            return worker.model_copy(deep=True)

    def fail_worker(self, worker_id: str, error: str) -> WorkerState:
        """Mark a worker failed."""
        with self._lock:
            worker = self._workers[worker_id]
            worker.status = "failed"
            worker.error = error
            worker.last_heartbeat_at = utc_timestamp()
            return worker.model_copy(deep=True)

    def workers(self) -> list[WorkerState]:
        """Return copies of all workers."""
        with self._lock:
            return [worker.model_copy(deep=True) for worker in self._workers.values()]

    def health_snapshot(self) -> dict[str, int]:
        """Return scheduler capacity counts."""
        with self._lock:
            workers = list(self._workers.values())
            busy_statuses = {"assigned", "joining_room", "starting_pipeline", "running", "draining"}
            return {
                "workers_total": len(workers),
                "workers_idle": sum(1 for worker in workers if worker.status == "idle"),
                "workers_busy": sum(1 for worker in workers if worker.status in busy_statuses),
                "workers_failed": sum(1 for worker in workers if worker.status == "failed"),
                "queued_sessions": len(self._queue),
            }

    def _first_idle_worker(self) -> WorkerState | None:
        for worker in self._workers.values():
            if worker.status == "idle":
                return worker
        return None

    def _worker_for_session(self, session_id: str) -> WorkerState | None:
        for worker in self._workers.values():
            if worker.session_id == session_id:
                return worker
        return None
