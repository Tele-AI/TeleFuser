"""Pipeline runner abstraction for TeleFuser service.

This module provides a thin compatibility layer between the service task schema
and various pipeline implementations:

- Legacy synchronous `run_with_file(pipeline, **task_data)` entrypoints
- Async `run_with_file(...)` entrypoints (common for orchestrator-based pipelines)
- Entry points that use different parameter names (e.g. `req_id`, `image`)

The service layer should not need to know whether a pipeline is orchestrator-based
or not; it only calls `PipelineRunner.run(...)`.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from telefuser.service_types import PipelineRunStatus
from telefuser.utils.logging import logger


@dataclass(frozen=True)
class PipelineRunResult:
    """Normalized result returned by PipelineRunner."""

    status: PipelineRunStatus
    output_path: str | None = None
    message: str = ""
    raw: Any | None = None


def _is_awaitable(obj: Any) -> bool:
    return inspect.isawaitable(obj) or isinstance(obj, asyncio.Future)


def _coerce_output_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    return str(value)


def _select_kwargs(
    fn: Any,
    *,
    task_data: dict[str, Any],
    module: ModuleType | None,
    stop_event: Any | None = None,
) -> dict[str, Any]:
    """Build kwargs for calling run_with_file based on signature inspection.

    Prefer passing full task_data when **kwargs is supported; otherwise pass a compatible subset and
    apply a small set of common alias conversions for orchestrator-based scripts.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(task_data)

    params = list(sig.parameters.values())
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    call_data = dict(task_data)
    if stop_event is not None and "stop_event" in sig.parameters:
        call_data["stop_event"] = stop_event

    if accepts_var_kw:
        return call_data

    kwargs: dict[str, Any] = {}
    for p in params:
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.VAR_POSITIONAL):
            continue

        name = p.name
        if name in call_data:
            kwargs[name] = call_data[name]
            continue

        # Common aliases used in examples/orchestrator runners.
        if name in ("req_id", "request_id"):
            if "task_id" in task_data:
                kwargs[name] = task_data["task_id"]
            continue

        if name in ("image", "input_image"):
            first_path = task_data.get("first_image_path") or task_data.get("image_path")
            if first_path:
                from PIL import Image

                kwargs[name] = Image.open(first_path).convert("RGB")
            continue

        if name == "ppl_config" and module is not None and hasattr(module, "PPL_CONFIG"):
            kwargs[name] = getattr(module, "PPL_CONFIG")
            continue

    return kwargs


class PipelineRunner:
    """Runner that normalizes pipeline execution for service calls."""

    def __init__(
        self,
        *,
        pipeline: Any,
        run_with_file: Any,
        module: ModuleType | None = None,
        output_root_env: str | None = "TELEAI_EXAMPLE_OUTPUT_DIR",
    ) -> None:
        self._pipeline = pipeline
        self._run_with_file = run_with_file
        self._module = module
        self._output_root_env = output_root_env

        self._started = False
        self._start_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()

    async def ensure_started(self) -> None:
        """Best-effort pipeline startup (once)."""
        if self._started:
            return
        async with self._start_lock:
            if self._started:
                return

            if hasattr(self._pipeline, "astart"):
                logger.info("Starting pipeline via astart()")
                await self._pipeline.astart()
            elif hasattr(self._pipeline, "start"):
                logger.info("Starting pipeline via start()")
                await asyncio.to_thread(self._pipeline.start)

            self._started = True

    async def shutdown(self) -> None:
        """Best-effort pipeline shutdown."""
        if not self._started:
            return

        async with self._run_lock:
            if hasattr(self._pipeline, "astop"):
                await self._pipeline.astop()
            elif hasattr(self._pipeline, "stop"):
                await asyncio.to_thread(self._pipeline.stop)
            elif hasattr(self._pipeline, "close"):
                await asyncio.to_thread(self._pipeline.close)

            self._started = False

    def _release_run_lock(self, task: asyncio.Task[Any]) -> None:
        """Release a timed-out invocation only after its underlying work exits."""
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        if self._run_lock.locked():
            self._run_lock.release()

    async def run(
        self,
        *,
        task_data: dict[str, Any],
        stop_event: Any | None = None,
        timeout_s: float | None = None,
        output_root: str | None = None,
    ) -> PipelineRunResult:
        """Run a single task.

        Args:
            task_data: Service-normalized task dict.
            stop_event: Optional threading.Event used for cooperative cancellation.
            timeout_s: Optional timeout for a single task execution.
            output_root: Optional output root to export as env var (for example scripts).
        """
        if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
            return PipelineRunResult(status=PipelineRunStatus.CANCELLED, message="Task cancelled before start")

        await self.ensure_started()

        if output_root and self._output_root_env:
            os.environ[self._output_root_env] = str(output_root)

        inference_stop_event = stop_event if stop_event is not None else threading.Event()
        kwargs = _select_kwargs(
            self._run_with_file,
            task_data=task_data,
            module=self._module,
            stop_event=inference_stop_event,
        )

        async def _invoke() -> Any:
            if inspect.iscoroutinefunction(self._run_with_file):
                return await self._run_with_file(self._pipeline, **kwargs)

            result = await asyncio.to_thread(self._run_with_file, self._pipeline, **kwargs)
            if _is_awaitable(result):
                return await result
            return result

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s if timeout_s else None

        def remaining_timeout() -> float | None:
            if deadline is None:
                return None
            return max(0.0, deadline - loop.time())

        try:
            if deadline is None:
                await self._run_lock.acquire()
            else:
                await asyncio.wait_for(self._run_lock.acquire(), timeout=remaining_timeout())
        except asyncio.TimeoutError:
            inference_stop_event.set()
            return PipelineRunResult(
                status=PipelineRunStatus.ERROR, output_path=None, message="Task processing timeout"
            )

        if inference_stop_event.is_set():
            self._run_lock.release()
            return PipelineRunResult(status=PipelineRunStatus.CANCELLED, message="Task cancelled before inference")

        invocation = asyncio.create_task(_invoke())
        release_on_completion = False
        try:
            if deadline is None:
                raw = await asyncio.shield(invocation)
            else:
                raw = await asyncio.wait_for(asyncio.shield(invocation), timeout=remaining_timeout())

            output_path = None
            if isinstance(raw, dict):
                output_path = raw.get("output_path") or raw.get("uri")
            elif isinstance(raw, (str, Path)):
                output_path = str(raw)

            output_path = _coerce_output_path(output_path) or _coerce_output_path(task_data.get("output_path"))
            return PipelineRunResult(
                status=PipelineRunStatus.SUCCESS, output_path=output_path, message="Inference completed", raw=raw
            )

        except asyncio.TimeoutError:
            inference_stop_event.set()
            release_on_completion = True
            invocation.add_done_callback(self._release_run_lock)
            return PipelineRunResult(
                status=PipelineRunStatus.ERROR, output_path=None, message="Task processing timeout"
            )
        except asyncio.CancelledError:
            inference_stop_event.set()
            release_on_completion = True
            invocation.add_done_callback(self._release_run_lock)
            raise
        except Exception as e:
            logger.exception(f"Pipeline run failed: {e}")
            return PipelineRunResult(status=PipelineRunStatus.ERROR, output_path=None, message=str(e))
        finally:
            if not release_on_completion:
                self._run_lock.release()
