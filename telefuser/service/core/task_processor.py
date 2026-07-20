"""
Async Task Processor for TeleFuser

Provides pure async task processing without threading.
Replaces the threading + asyncio.run() anti-pattern.
"""

from __future__ import annotations

import asyncio

from telefuser.utils.logging import logger

from .task_manager import TaskManager, TaskStatus
from .task_service import MediaGenerationService


class AsyncTaskProcessor:
    """Pure async task processor using asyncio.Queue.

    This replaces the previous threading-based approach that used
    asyncio.run() inside threads, which is an anti-pattern.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        media_service: MediaGenerationService,
        max_concurrent: int = 1,
    ) -> None:
        """Initialize the async task processor."""
        self.task_manager = task_manager
        self.media_service = media_service
        self.max_concurrent = max_concurrent

        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._stop_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_running(self) -> bool:
        """Whether the processor workers are running."""
        return self._running

    async def start(self) -> None:
        """Start the task processor workers."""
        if self._running:
            logger.warning("Task processor already running")
            return

        self._running = True
        self._stop_event.clear()
        self._loop = asyncio.get_running_loop()

        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"), name=f"task-processor-{i}")
            self._workers.append(worker)

        logger.info(f"Started {self.max_concurrent} task processor workers")

    async def stop(self) -> None:
        """Stop the task processor gracefully."""
        if not self._running:
            return

        logger.info("Stopping task processor...")
        self._running = False
        self._stop_event.set()

        for worker in self._workers:
            worker.cancel()

        if self._workers:
            current_loop = asyncio.get_running_loop()
            worker_loops = {worker.get_loop() for worker in self._workers}

            if any(loop.is_closed() for loop in worker_loops):
                logger.warning("Skipping worker await during shutdown because the worker event loop is already closed")
            elif worker_loops == {current_loop}:
                await asyncio.gather(*self._workers, return_exceptions=True)
            else:
                logger.warning(
                    "Skipping worker await during shutdown because stop() was called from a different event loop"
                )

        self._workers.clear()
        self._loop = None
        logger.info("Task processor stopped")

    async def _worker_loop(self, worker_name: str) -> None:
        """Worker loop that processes tasks."""
        logger.info(f"{worker_name} started")

        while self._running and not self._stop_event.is_set():
            try:
                task_id = self.task_manager.claim_next_pending_task()

                if task_id is None:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=1.0)
                    continue

                await self._process_task(task_id)

            except asyncio.CancelledError:
                logger.info(f"{worker_name} cancelled")
                break
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.exception(f"{worker_name} error: {e}")

        logger.info(f"{worker_name} stopped")

    async def _process_task(self, task_id: str) -> None:
        """Process a single task.

        The task has already been atomically claimed (marked PROCESSING) by
        claim_next_pending_task(). This method must always release the
        processing slot when it returns.
        """
        try:
            task_info = self.task_manager.get_task(task_id)
            if not task_info:
                logger.warning(f"Task {task_id} disappeared before processing")
                return

            logger.info(f"Processing task {task_id}")

            if task_info.status == TaskStatus.CANCELLED:
                logger.info(f"Task {task_id} was cancelled before processing started")
                return

            if task_info.stop_event.is_set():
                logger.info(f"Task {task_id} cancelled before processing")
                return

            try:
                result = await self.media_service.generate_media_with_stop_event(
                    task_info.message, task_info.stop_event
                )

                if result:
                    self.task_manager.complete_task(
                        task_id,
                        result.output_path,
                        peak_memory_mb=result.peak_memory_mb,
                        inference_time_s=result.inference_time_s,
                    )
                    logger.info(f"Task {task_id} completed successfully")
                else:
                    if task_info.stop_event.is_set():
                        logger.info(f"Task {task_id} cancelled during processing")
                    else:
                        self.task_manager.fail_task(task_id, "Generation failed")
                        logger.error(f"Task {task_id} generation failed")

            except Exception as e:
                if task_info.stop_event.is_set():
                    logger.info(f"Task {task_id} exited after cancellation")
                    return
                logger.exception(f"Task {task_id} processing failed")
                self.task_manager.fail_task(task_id, str(e))

        finally:
            self.task_manager.release_processing_slot(task_id)
