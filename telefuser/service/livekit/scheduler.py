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
    session_ids: list[str] = Field(default_factory=list)
    session_capacity: int = 1
    session_id: str | None = None
    room_name: str | None = None
    last_heartbeat_at: float
    error: str | None = None


class SchedulerAdmission(BaseModel):
    """Result of a scheduler admission attempt."""

    status: AdmissionStatus
    worker_id: str | None = None
    session_id: str | None = None
    queue_position: int | None = None
    reason: str | None = None


class _QueuedSession(BaseModel):
    session_id: str
    room_name: str


class LiveKitScheduler:
    """FIFO scheduler with bounded retained-session capacity per worker."""

    def __init__(
        self,
        *,
        num_workers: int,
        gpu_groups: list[list[str]] | None = None,
        queue_size: int = 0,
        max_sessions_per_worker: int = 1,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if queue_size < 0:
            raise ValueError("queue_size must be >= 0")
        if max_sessions_per_worker < 1:
            raise ValueError("max_sessions_per_worker must be >= 1")

        now = utc_timestamp()
        groups = gpu_groups or [[] for _ in range(num_workers)]
        if len(groups) != num_workers:
            raise ValueError(f"Expected {num_workers} GPU groups, got {len(groups)}")

        self._workers: dict[str, WorkerState] = {
            f"worker-{idx}": WorkerState(
                worker_id=f"worker-{idx}",
                status="idle",
                gpu_ids=list(groups[idx]),
                session_capacity=max_sessions_per_worker,
                last_heartbeat_at=now,
            )
            for idx in range(num_workers)
        }
        self._queue_size = queue_size
        self._queue: deque[_QueuedSession] = deque()
        self._room_names: dict[str, str] = {}
        self._lock = threading.RLock()

    def assign(self, *, session_id: str, room_name: str) -> SchedulerAdmission:
        """Assign a worker with capacity or enqueue/reject the session."""
        with self._lock:
            worker = self._least_loaded_worker()
            if worker is not None:
                worker.status = "assigned"
                worker.session_ids.append(session_id)
                self._room_names[session_id] = room_name
                worker.session_id = session_id
                worker.room_name = room_name
                worker.error = None
                worker.last_heartbeat_at = utc_timestamp()
                return SchedulerAdmission(status="assigned", worker_id=worker.worker_id, session_id=session_id)

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

            worker.session_ids.remove(session_id)
            self._room_names.pop(session_id, None)
            worker.session_id = worker.session_ids[-1] if worker.session_ids else None
            worker.room_name = self._room_names.get(worker.session_id) if worker.session_id is not None else None
            worker_failed = worker.status in {"failed", "stopped"}
            if not worker_failed:
                worker.error = None
                worker.status = "assigned" if worker.session_ids else "idle"
            worker.last_heartbeat_at = utc_timestamp()

            if not self._queue or worker_failed:
                return None

            queued = self._queue.popleft()
            worker.status = "assigned"
            worker.session_ids.append(queued.session_id)
            self._room_names[queued.session_id] = queued.room_name
            worker.session_id = queued.session_id
            worker.room_name = queued.room_name
            worker.last_heartbeat_at = utc_timestamp()
            return SchedulerAdmission(status="assigned", worker_id=worker.worker_id, session_id=queued.session_id)

    def drain_queue(self) -> list[SchedulerAdmission]:
        """Assign as many queued sessions as newly active capacity permits."""
        admissions: list[SchedulerAdmission] = []
        with self._lock:
            while self._queue and (worker := self._least_loaded_worker()) is not None:
                queued = self._queue.popleft()
                worker.status = "assigned"
                worker.session_ids.append(queued.session_id)
                self._room_names[queued.session_id] = queued.room_name
                worker.session_id = queued.session_id
                worker.room_name = queued.room_name
                worker.last_heartbeat_at = utc_timestamp()
                admissions.append(
                    SchedulerAdmission(status="assigned", worker_id=worker.worker_id, session_id=queued.session_id)
                )
        return admissions

    def reassign_session(self, session_id: str, target_worker_id: str) -> SchedulerAdmission:
        """Move admission ownership after the pipeline router commits migration."""
        with self._lock:
            source = self._worker_for_session(session_id)
            if source is None:
                raise KeyError(f"Session {session_id} is not assigned")
            target = self._workers[target_worker_id]
            if source.worker_id == target.worker_id:
                return SchedulerAdmission(status="assigned", worker_id=target.worker_id, session_id=session_id)
            if target.status in {"failed", "stopped", "draining"}:
                raise RuntimeError(f"Migration target {target_worker_id} is not available")
            if len(target.session_ids) >= target.session_capacity:
                raise RuntimeError(f"Migration target {target_worker_id} has no retained-session capacity")
            source.session_ids.remove(session_id)
            source.session_id = source.session_ids[-1] if source.session_ids else None
            source.room_name = self._room_names.get(source.session_id) if source.session_id is not None else None
            if source.status not in {"failed", "stopped"}:
                source.status = "assigned" if source.session_ids else "idle"
                source.error = None
            source.last_heartbeat_at = utc_timestamp()
            target.session_ids.append(session_id)
            target.session_id = session_id
            target.room_name = self._room_names.get(session_id)
            target.status = "assigned"
            target.error = None
            target.last_heartbeat_at = utc_timestamp()
            return SchedulerAdmission(status="assigned", worker_id=target.worker_id, session_id=session_id)

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

    def update_worker_capacity(self, worker_id: str, capacity: int) -> WorkerState:
        """Apply hardware-calculated retained-session capacity before admission starts."""
        if capacity < 1:
            raise ValueError("worker session capacity must be positive")
        with self._lock:
            worker = self._workers[worker_id]
            if len(worker.session_ids) > capacity:
                raise RuntimeError(
                    f"Worker {worker_id} already owns {len(worker.session_ids)} sessions, "
                    f"cannot reduce capacity to {capacity}"
                )
            worker.session_capacity = capacity
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
            return {
                "workers_total": len(workers),
                "workers_idle": sum(1 for worker in workers if worker.status == "idle"),
                "workers_busy": sum(1 for worker in workers if bool(worker.session_ids)),
                "workers_failed": sum(1 for worker in workers if worker.status == "failed"),
                "queued_sessions": len(self._queue),
            }

    def _least_loaded_worker(self) -> WorkerState | None:
        candidates = [
            worker
            for worker in self._workers.values()
            if worker.status not in {"failed", "stopped", "draining"}
            and len(worker.session_ids) < worker.session_capacity
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda worker: (
                len(worker.session_ids) / worker.session_capacity,
                len(worker.session_ids),
                worker.worker_id,
            ),
        )

    def _worker_for_session(self, session_id: str) -> WorkerState | None:
        for worker in self._workers.values():
            if session_id in worker.session_ids:
                return worker
        return None
