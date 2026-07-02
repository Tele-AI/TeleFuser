"""Pipeline stage wrapper for request processing.

Wraps individual pipeline stages with queue-based execution,
result routing, and metrics collection.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    import ray
except ImportError:
    ray = None

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger


@dataclass
class StageConfig:
    """Configuration for a pipeline stage.

    Attributes:
        stage_id: Unique stage identifier
        stage_name: Human-readable stage name
        pipeline_attr: Attribute name on pipeline object
        param_builder: Optional function to build stage parameters
        result_processor: Optional function to process results
        shared_lock_group: Optional lock group for exclusive access
        parallel_group: Optional group ID for parallel execution
        metadata: Additional stage metadata
    """

    stage_id: int
    stage_name: str
    pipeline_attr: str

    param_builder: Optional[Callable[[Dict, Any], tuple[list, dict]]] = None
    result_processor: Optional[Callable[[Any], Any]] = None
    shared_lock_group: Optional[str] = None
    parallel_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.stage_id < 0:
            raise ValueError(f"stage_id must be non-negative, got {self.stage_id}")

        if not self.stage_name:
            raise ValueError("stage_name cannot be empty")

        if not self.pipeline_attr:
            raise ValueError("pipeline_attr cannot be empty")


@dataclass
class StageTask:
    """Task submitted to a pipeline stage."""

    request_id: str
    stage_id: int
    inputs: Any
    context: Dict[str, Any] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StageResult:
    """Result from a pipeline stage execution."""

    request_id: str
    stage_id: int
    outputs: Any
    finished: bool = False
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class EnhancedPipelineStageWrapper:
    """Wrapper for pipeline stage with async queue processing."""

    def __init__(
        self,
        stage_id: int,
        stage_name: str,
        stage_callable: Any,
        stage_config: StageConfig,
        result_queue: asyncio.Queue,
        shared_call_lock: Optional[threading.Lock] = None,
    ):
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.stage_callable = stage_callable
        self.stage_config = stage_config
        self._result_queue = result_queue
        self._shared_call_lock = shared_call_lock

        self.in_queue: asyncio.Queue = asyncio.Queue()

        self._exec_lock = asyncio.Lock()

        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"[Stage-{stage_id}] {stage_name} initialized "
            f"(single-request mode, pipeline_attr={stage_config.pipeline_attr})"
        )

    async def start(self):
        """Start the stage worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"[Stage-{self.stage_id}] {self.stage_name} worker started")

    async def stop(self):
        """Stop the stage worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info(f"[Stage-{self.stage_id}] {self.stage_name} worker stopped")

    async def submit(self, task: StageTask):
        """Submit a task to this stage."""
        await self.in_queue.put(task)
        logger.debug(
            f"[Stage-{self.stage_id}] Task {task.request_id} submitted to queue (queue_size={self.in_queue.qsize()})"
        )

    async def _worker_loop(self):
        """Main worker loop processing tasks from queue."""
        logger.info(f"[Stage-{self.stage_id}] {self.stage_name} worker loop started")

        while self._running:
            try:
                task = await self.in_queue.get()

                logger.info(
                    f"[Stage-{self.stage_id}] Processing {task.request_id} (queue_remaining={self.in_queue.qsize()})"
                )

                async with self._exec_lock:
                    result = await self._execute_stage(task)

                await self._result_queue.put(result)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"[Stage-{self.stage_id}] Worker loop error: {e}")
                await asyncio.sleep(0.1)

    async def _execute_stage(self, task: StageTask) -> StageResult:
        """Execute a single stage task."""
        try:
            logger.info(f"[Stage-{self.stage_id}] Executing {task.request_id} at stage {self.stage_name}")

            exec_start = time.time()

            devices = self._get_stage_devices()
            for device in devices:
                try:
                    current_platform.reset_peak_memory_stats(device)
                except Exception:
                    pass

            loop = asyncio.get_event_loop()

            def _run_sync():
                if self.stage_config.param_builder:
                    args, kwargs = self.stage_config.param_builder(task.context, task.inputs)
                else:
                    args = task.inputs if isinstance(task.inputs, (list, tuple)) else [task.inputs]
                    kwargs = task.params

                if hasattr(self.stage_callable, "process"):
                    call_fn = self.stage_callable.process
                else:
                    call_fn = self.stage_callable

                use_ray = bool(self.stage_config.metadata.get("use_ray", False))

                def _call_and_wait():
                    out = None
                    if use_ray:
                        # Ray actor/remote function: call .remote and ray.get the ObjectRef.
                        if ray is None:
                            raise RuntimeError("use_ray=True but ray is not available")
                        if not hasattr(call_fn, "remote"):
                            raise RuntimeError("use_ray=True but call target has no .remote")
                        out = call_fn.remote(*args, **kwargs)
                        out = ray.get(out)
                        return out

                    out = call_fn(*args, **kwargs)
                    # ParallelWorker returns a waiter callable; run it here.
                    if callable(out):
                        out = out()
                    return out

                lock = self._shared_call_lock
                if lock is None:
                    return _call_and_wait()

                lock.acquire()
                try:
                    return _call_and_wait()
                finally:
                    lock.release()

            outputs = await loop.run_in_executor(None, _run_sync)

            if self.stage_config.result_processor:
                outputs = self.stage_config.result_processor(outputs)

            exec_time = time.time() - exec_start

            peak_mem_str = ""
            for device in devices:
                try:
                    peak_bytes = current_platform.max_memory_allocated(device)
                    peak_gb = peak_bytes / (1024**3)
                    peak_mem_str += f"device={device}: {peak_gb:.3f}GB "
                except Exception:
                    pass

            result = StageResult(
                request_id=task.request_id,
                stage_id=self.stage_id,
                outputs=outputs,
                finished=True,
                metrics={
                    "stage_time_ms": exec_time * 1000,
                    "stage_name": self.stage_name,
                    "peak_mem": peak_mem_str,
                },
            )

            logger.info(f"[Stage-{self.stage_id}] {task.request_id} completed in {exec_time:.3f}s ({peak_mem_str})")

            return result

        except Exception as e:
            logger.exception(f"[Stage-{self.stage_id}] Task {task.request_id} failed: {e}")
            return StageResult(
                request_id=task.request_id,
                stage_id=self.stage_id,
                outputs=None,
                finished=True,
                error=str(e),
            )

    def _get_stage_devices(self) -> List[Optional[int]]:
        """Get list of devices used by this stage."""
        if hasattr(self.stage_callable, "device_ids"):
            return list(self.stage_callable.device_ids)

        device = getattr(self.stage_callable, "device", None)
        if isinstance(device, torch.device) and device.type == "cuda":
            return [device.index]

        return [None]
