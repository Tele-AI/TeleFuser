"""Task manager for tracking and managing generation tasks."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from telefuser.metrics import get_service_metrics
from telefuser.service.core.pipeline_contract import infer_media_type_for_task
from telefuser.utils.logging import logger


class TaskStatus(Enum):
    """Task status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ACTIVE_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.PROCESSING, TaskStatus.STREAMING})


@dataclass
class TaskInfo:
    """Task information data class."""

    task_id: str
    status: TaskStatus
    message: Any
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    error: str | None = None
    output_path: str | None = None
    peak_memory_mb: float | None = None
    inference_time_s: float | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class TaskManager:
    """Manages task lifecycle from creation to completion.

    Thread-safe task queue with automatic cleanup of old completed tasks.
    Supports multi-slot concurrent processing for data-parallel pipeline pools.
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        cleanup_keep_count: int = 1000,
        cancel_timeout: float = 5.0,
        processing_lock_timeout: float = 1.0,
        max_concurrent_processing: int = 1,
    ) -> None:
        self.max_queue_size = max_queue_size
        self.cleanup_keep_count = cleanup_keep_count
        self.cancel_timeout = cancel_timeout
        self.processing_lock_timeout = processing_lock_timeout
        self._max_concurrent_processing = max(1, max_concurrent_processing)

        self._tasks: OrderedDict[str, TaskInfo] = OrderedDict()
        self._lock = threading.RLock()

        self._processing_lock = threading.Lock()
        self._current_processing_tasks: OrderedDict[str, bool] = OrderedDict()
        self._zero_capacity: bool = False

        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

    def create_task(self, message: Any) -> str:
        """Create a new task with validation and metrics."""
        with self._lock:
            if self._zero_capacity:
                raise RuntimeError("No inference capacity: all pipeline replicas are dead")

            if hasattr(message, "task_id") and message.task_id in self._tasks:
                raise RuntimeError(f"Task ID {message.task_id} already exists")

            active_tasks = sum(1 for t in self._tasks.values() if t.status in _ACTIVE_STATUSES)
            if active_tasks >= self.max_queue_size:
                raise RuntimeError(f"Task queue is full (max {self.max_queue_size} tasks)")

            task_id = getattr(message, "task_id", str(uuid.uuid4()))
            task_info = TaskInfo(
                task_id=task_id,
                status=TaskStatus.PENDING,
                message=message,
                output_path=getattr(message, "output_path", None),
            )

            self._tasks[task_id] = task_info
            self.total_tasks += 1

            get_service_metrics().record_task_created()

            self._cleanup_old_tasks()

            return task_id

    def start_task(self, task_id: str) -> TaskInfo:
        """Mark task as started."""
        return self._transition_to(task_id, TaskStatus.PROCESSING, from_statuses=(TaskStatus.PENDING,))

    def start_streaming(self, task_id: str) -> TaskInfo:
        """Mark task as streaming (continuous output in progress)."""
        return self._transition_to(
            task_id, TaskStatus.STREAMING, from_statuses=(TaskStatus.PENDING, TaskStatus.PROCESSING)
        )

    def _transition_to(self, task_id: str, target: TaskStatus, from_statuses: tuple[TaskStatus, ...]) -> TaskInfo:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"Task {task_id} not found")

            task = self._tasks[task_id]
            if task.status not in from_statuses:
                return task

            task.status = target
            task.start_time = datetime.now()
            self._tasks.move_to_end(task_id)
            return task

    def complete_task(
        self,
        task_id: str,
        output_path: str | None = None,
        *,
        peak_memory_mb: float | None = None,
        inference_time_s: float | None = None,
    ) -> None:
        """Mark task as completed with metrics."""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found for completion")
                return

            task = self._tasks[task_id]
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                logger.info(f"Ignoring completion for task {task_id} in terminal state {task.status.value}")
                return

            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            if output_path:
                task.output_path = output_path

            self.completed_tasks += 1

            if task.start_time:
                duration = (task.end_time - task.start_time).total_seconds()
                task.inference_time_s = inference_time_s if inference_time_s is not None else duration
                get_service_metrics().record_task_completed(duration)
            task.peak_memory_mb = peak_memory_mb

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed with metrics."""
        with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Task {task_id} not found for failure")
                return

            task = self._tasks[task_id]
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                logger.info(f"Ignoring failure for task {task_id} in terminal state {task.status.value}")
                return

            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            task.error = error

            self.failed_tasks += 1
            get_service_metrics().record_task_failed()

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or processing task."""
        thread_to_join: threading.Thread | None = None

        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                return False

            task.stop_event.set()
            task.status = TaskStatus.CANCELLED
            task.end_time = datetime.now()
            task.error = "Task cancelled by user"

            get_service_metrics().record_task_cancelled()

            if task.thread and task.thread.is_alive():
                thread_to_join = task.thread

        if thread_to_join is not None:
            thread_to_join.join(timeout=self.cancel_timeout)

        return True

    def cancel_all_tasks(self) -> None:
        """Cancel all pending or processing tasks."""
        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if task.status in _ACTIVE_STATUSES:
                    self.cancel_task(task_id)

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task info by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get task status dictionary."""
        task = self.get_task(task_id)
        if not task:
            return None

        status = {
            "task_id": task.task_id,
            "status": task.status.value,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "error": task.error,
            "output_path": task.output_path,
            "peak_memory_mb": task.peak_memory_mb,
            "inference_time_s": task.inference_time_s,
        }
        status.update(self._serialize_task_message(task.message))
        return status

    def get_all_tasks(self) -> dict[str, dict[str, Any] | None]:
        """Get all tasks as status dictionaries."""
        with self._lock:
            return {task_id: self.get_task_status(task_id) for task_id in self._tasks}

    def get_active_task_count(self) -> int:
        """Get count of pending and processing tasks."""
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status in _ACTIVE_STATUSES)

    def get_pending_task_count(self) -> int:
        """Get count of pending tasks."""
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

    def is_processing(self) -> bool:
        """Check if any task is currently processing."""
        with self._lock:
            return len(self._current_processing_tasks) > 0

    def acquire_processing_lock(self, task_id: str, timeout: float | None = None) -> bool:
        """Acquire exclusive processing lock for a task."""
        timeout = timeout if timeout is not None else self.processing_lock_timeout
        acquired = self._processing_lock.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                logger.info(f"Task {task_id} acquired processing lock")
        return acquired

    def release_processing_lock(self, task_id: str) -> None:
        """Release processing lock for a task."""
        try:
            self._processing_lock.release()
            logger.info(f"Task {task_id} released processing lock")
        except RuntimeError:
            pass  # Lock was not held by this thread

    def get_next_pending_task(self) -> str | None:
        """Get next pending task ID (FIFO order).

        Deprecated: kept for backward compatibility. This is a non-atomic read
        (it does not mark the task PROCESSING), so concurrent callers may pick
        the same task. Use claim_next_pending_task() instead.
        """
        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status == TaskStatus.PENDING:
                    return task_id
        return None

    def claim_next_pending_task(self) -> str | None:
        """Atomically find the next PENDING task and mark it PROCESSING.

        Multi-slot semantics: returns None if all processing slots are full
        or if zero capacity (all replicas dead). When zero capacity, also
        drains any remaining pending tasks with a failure error.
        """
        with self._lock:
            if self._zero_capacity:
                self._fail_all_pending("No inference capacity: all pipeline replicas are dead")
                return None
            if len(self._current_processing_tasks) >= self._max_concurrent_processing:
                return None
            for task_id, task in self._tasks.items():
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.PROCESSING
                    task.start_time = datetime.now()
                    self._tasks.move_to_end(task_id)
                    self._current_processing_tasks[task_id] = True
                    return task_id
        return None

    def release_processing_slot(self, task_id: str) -> None:
        """Release the processing slot after task completion.

        Companion to claim_next_pending_task(). Clears the slot so the next
        PENDING task can be claimed.
        """
        with self._lock:
            self._current_processing_tasks.pop(task_id, None)

    def get_service_status(self) -> dict[str, Any]:
        """Get overall service status."""
        with self._lock:
            active_tasks = []
            pending_count = 0
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.PROCESSING, TaskStatus.STREAMING):
                    active_tasks.append(task_id)
                elif task.status == TaskStatus.PENDING:
                    pending_count += 1

            return {
                "service_status": "busy" if self._current_processing_tasks else "idle",
                "current_task": next(iter(self._current_processing_tasks), None),
                "current_tasks": list(self._current_processing_tasks.keys()),
                "processing_count": len(self._current_processing_tasks),
                "max_concurrent_processing": self._max_concurrent_processing,
                "active_tasks": active_tasks,
                "pending_tasks": pending_count,
                "queue_size": self.max_queue_size,
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "failed_tasks": self.failed_tasks,
            }

    def set_max_concurrent_processing(self, n: int) -> None:
        """Dynamically adjust max concurrent processing slots.

        Called by PipelinePool when a replica dies. Thread-safe.
        When n=0, marks pool as having zero capacity and fails all pending tasks.
        """
        with self._lock:
            old = self._max_concurrent_processing
            self._max_concurrent_processing = max(0, n)
            if self._max_concurrent_processing == 0:
                self._zero_capacity = True
                self._fail_all_pending("No inference capacity: all pipeline replicas are dead")
            logger.info(f"Max concurrent processing adjusted: {old} → {self._max_concurrent_processing}")

    @property
    def max_concurrent_processing(self) -> int:
        """Current max concurrent processing slots (may differ from initial value after replica death)."""
        with self._lock:
            return self._max_concurrent_processing

    def _fail_all_pending(self, error: str) -> None:
        """Fail all pending tasks. Caller must hold self._lock."""
        for task_id, task in list(self._tasks.items()):
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.FAILED
                task.end_time = datetime.now()
                task.error = error
                self.failed_tasks += 1
                get_service_metrics().record_task_failed()
                logger.warning(f"Task {task_id} failed: {error}")

    def _cleanup_old_tasks(self, keep_count: int | None = None) -> None:
        """Remove old completed tasks to prevent memory growth."""
        if keep_count is None:
            keep_count = self.cleanup_keep_count
        if len(self._tasks) <= keep_count:
            return

        completed_tasks = [
            (task_id, task)
            for task_id, task in self._tasks.items()
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]

        completed_tasks.sort(key=lambda x: x[1].end_time or x[1].start_time)

        remove_count = len(self._tasks) - keep_count
        for task_id, _ in completed_tasks[:remove_count]:
            del self._tasks[task_id]
            logger.debug(f"Cleaned up old task: {task_id}")

    def _serialize_task_message(self, message: Any) -> dict[str, Any]:
        """Extract stable task metadata needed by server APIs from the original request object."""
        if message is None:
            return {}

        data: dict[str, Any] = {}
        for field_name in (
            "task",
            "prompt",
            "negative_prompt",
            "resolution",
            "target_video_length",
            "aspect_ratio",
            "output_format",
            "model",
            "first_image_path",
            "last_image_path",
            "ref_video_path",
        ):
            value = getattr(message, field_name, None)
            if value not in (None, ""):
                data[field_name] = value

        task_name = data.get("task") or getattr(message, "task", None)
        if task_name:
            data["media_type"] = infer_media_type_for_task(task_name)

        return data
