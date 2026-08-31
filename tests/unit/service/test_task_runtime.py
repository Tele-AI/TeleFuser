from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

from telefuser.service.api.schema import StructuredTaskRequest, TaskRequest, TaskResponse
from telefuser.service.core.task_manager import TaskManager, TaskStatus
from telefuser.service.core.task_processor import AsyncTaskProcessor


def test_complete_task_does_not_override_cancelled_status() -> None:
    """Cancelled tasks must remain cancelled even if completion is reported later."""
    task_manager = TaskManager(max_queue_size=10)
    message = TaskRequest(task="t2i")

    task_id = task_manager.create_task(message)

    assert task_manager.cancel_task(task_id) is True

    task_manager.complete_task(task_id, output_path="ignored.png")

    status = task_manager.get_task_status(task_id)
    assert status is not None
    assert status["status"] == TaskStatus.CANCELLED.value
    assert status["error"] == "Task cancelled by user"
    assert status["output_path"] == message.output_path


def test_complete_task_exports_runtime_metrics() -> None:
    task_manager = TaskManager(max_queue_size=10)
    task_id = task_manager.create_task(TaskRequest(task="t2v"))
    task = task_manager.start_task(task_id)
    task.start_time -= timedelta(seconds=2)

    task_manager.complete_task(
        task_id,
        output_path="result.mp4",
        peak_memory_mb=2048.0,
        inference_time_s=1.5,
    )

    status = task_manager.get_task_status(task_id)
    assert status is not None
    assert status["peak_memory_mb"] == 2048.0
    assert status["inference_time_s"] == 1.5


def test_artifact_cleanup_snapshot_splits_active_and_terminal_tasks() -> None:
    task_manager = TaskManager()
    active_id = task_manager.create_task(SimpleNamespace(output_path="active.mp4"))
    terminal_id = task_manager.create_task(SimpleNamespace(output_path="done.mp4"))
    task_manager.start_task(active_id)
    task_manager.complete_task(terminal_id)

    snapshot = task_manager.get_artifact_cleanup_snapshot()

    assert snapshot["active_task_ids"] == {active_id}
    assert terminal_id in snapshot["terminal_task_end_times"]
    assert snapshot["terminal_task_statuses"] == {terminal_id: TaskStatus.COMPLETED.value}
    assert active_id not in snapshot["terminal_task_end_times"]


class _ControlledMediaService:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def generate_media_with_stop_event(self, message: TaskRequest, stop_event) -> TaskResponse:
        self.started.set()
        await self.finish.wait()
        return TaskResponse(
            task_id=message.task_id,
            task_status="completed",
            output_path=message.output_path,
        )


def test_async_task_processor_preserves_cancelled_status() -> None:
    """Cancellation should win even if the media service returns after cancellation."""

    async def scenario() -> None:
        task_manager = TaskManager(max_queue_size=10)
        media_service = _ControlledMediaService()
        processor = AsyncTaskProcessor(task_manager=task_manager, media_service=media_service, max_concurrent=1)
        message = TaskRequest(task="t2i")

        task_id = task_manager.create_task(message)

        await processor.start()
        try:
            await asyncio.wait_for(media_service.started.wait(), timeout=1.0)

            assert task_manager.cancel_task(task_id) is True

            media_service.finish.set()

            async def wait_for_cancelled_status() -> None:
                while True:
                    status = task_manager.get_task_status(task_id)
                    if status and status["status"] == TaskStatus.CANCELLED.value:
                        return
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_cancelled_status(), timeout=1.0)

            status = task_manager.get_task_status(task_id)
            assert status is not None
            assert status["status"] == TaskStatus.CANCELLED.value
            assert status["error"] == "Task cancelled by user"
        finally:
            await processor.stop()

    asyncio.run(scenario())


def test_async_task_processor_wakes_when_task_becomes_available() -> None:
    """A newly submitted task should not wait for the idle polling timeout."""

    async def scenario() -> None:
        task_manager = TaskManager(max_queue_size=10)
        media_service = _ControlledMediaService()
        media_service.finish.set()
        processor = AsyncTaskProcessor(task_manager=task_manager, media_service=media_service, max_concurrent=1)

        await processor.start()
        try:
            await asyncio.sleep(0)
            task_manager.create_task(TaskRequest(task="t2i"))
            processor.notify_task_available()

            await asyncio.wait_for(media_service.started.wait(), timeout=0.5)
        finally:
            await processor.stop()

    asyncio.run(scenario())


def test_claim_next_pending_task_atomic_single_winner() -> None:
    """Two PENDING tasks, single slot: only one is claimed, the second claim returns None."""
    task_manager = TaskManager(max_queue_size=10)

    first_id = task_manager.create_task(TaskRequest(task="t2i"))
    second_id = task_manager.create_task(TaskRequest(task="t2i"))

    claimed = task_manager.claim_next_pending_task()
    # FIFO: the first-created task is claimed.
    assert claimed == first_id
    assert task_manager.get_task(first_id).status == TaskStatus.PROCESSING

    # Single slot occupied -> next claim returns None even though a task is PENDING.
    assert task_manager.claim_next_pending_task() is None
    assert task_manager.get_task(second_id).status == TaskStatus.PENDING


def test_claim_release_cycle_advances_to_next_pending() -> None:
    """After releasing the slot, the next PENDING task becomes claimable."""
    task_manager = TaskManager(max_queue_size=10)

    first_id = task_manager.create_task(TaskRequest(task="t2i"))
    second_id = task_manager.create_task(TaskRequest(task="t2i"))

    assert task_manager.claim_next_pending_task() == first_id
    assert task_manager.claim_next_pending_task() is None

    task_manager.release_processing_slot(first_id)

    assert task_manager.claim_next_pending_task() == second_id
    assert task_manager.get_task(second_id).status == TaskStatus.PROCESSING


def test_concurrent_claims_pick_distinct_tasks() -> None:
    """Concurrent claims under a single slot never hand the same task to two callers."""
    import threading

    task_manager = TaskManager(max_queue_size=50)
    for _ in range(20):
        task_manager.create_task(TaskRequest(task="t2i"))

    claimed: list[str] = []
    claimed_lock = threading.Lock()

    def worker() -> None:
        # Claim, then immediately release to let others proceed (single-slot).
        for _ in range(20):
            task_id = task_manager.claim_next_pending_task()
            if task_id is not None:
                with claimed_lock:
                    claimed.append(task_id)
                task_manager.release_processing_slot(task_id)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Every task claimed at most once; all 20 eventually claimed.
    assert len(claimed) == len(set(claimed))
    assert set(claimed) == set(task_manager._tasks.keys())


def test_terminal_tasks_release_original_structured_request_payloads() -> None:
    task_manager = TaskManager(max_queue_size=10)

    def request() -> StructuredTaskRequest:
        return StructuredTaskRequest(
            task="vla_action",
            instruction="pick up the block",
            state=[0.0] * 14,
            camera_high="encoded-camera-payload",
            camera_left_wrist="encoded-camera-payload",
            camera_right_wrist="encoded-camera-payload",
        )

    completed_id = task_manager.create_task(request())
    task_manager.complete_task(completed_id, result={"horizon": 1})
    completed = task_manager.get_task(completed_id)
    assert completed is not None
    assert completed.message is None
    completed_status = task_manager.get_task_status(completed_id)
    assert completed_status is not None
    assert completed_status["task"] == "vla_action"
    assert completed_status["media_type"] == "structured"
    assert "camera_high" not in completed_status

    failed_id = task_manager.create_task(request())
    task_manager.fail_task(failed_id, "inference failed")
    failed = task_manager.get_task(failed_id)
    assert failed is not None
    assert failed.message is None

    pending_cancel_id = task_manager.create_task(request())
    assert task_manager.cancel_task(pending_cancel_id) is True
    pending_cancelled = task_manager.get_task(pending_cancel_id)
    assert pending_cancelled is not None
    assert pending_cancelled.message is None

    processing_cancel_id = task_manager.create_task(request())
    assert task_manager.claim_next_pending_task() == processing_cancel_id
    processing_cancelled = task_manager.get_task(processing_cancel_id)
    assert processing_cancelled is not None
    assert task_manager.cancel_task(processing_cancel_id) is True
    assert processing_cancelled.message is not None
    task_manager.release_processing_slot(processing_cancel_id)
    assert processing_cancelled.message is None


def test_timeout_stop_signal_fails_task_instead_of_leaving_it_processing() -> None:
    class TimedOutMediaService:
        async def generate_media_with_stop_event(self, message: TaskRequest, stop_event):
            stop_event.set()
            return None

    async def scenario() -> None:
        task_manager = TaskManager(max_queue_size=10)
        processor = AsyncTaskProcessor(
            task_manager=task_manager,
            media_service=TimedOutMediaService(),
            max_concurrent=1,
        )
        task_id = task_manager.create_task(TaskRequest(task="t2i"))
        assert task_manager.claim_next_pending_task() == task_id

        await processor._process_task(task_id)

        status = task_manager.get_task_status(task_id)
        assert status is not None
        assert status["status"] == TaskStatus.FAILED.value
        assert status["error"] == "Generation failed"

    asyncio.run(scenario())
