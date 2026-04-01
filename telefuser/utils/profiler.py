import asyncio
import os
import time
from functools import wraps
from pathlib import Path

import torch

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger

_DEVICE_ACTIVITY_MAP = {
    "cuda": torch.profiler.ProfilerActivity.CUDA,
    "xpu": torch.profiler.ProfilerActivity.XPU,
    "npu": torch.profiler.ProfilerActivity.PrivateUse1,
}


def _get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _should_enable_profiler(name: str) -> bool:
    enabled_names = os.getenv("ENABLE_PROFILER_NAMES", "").split(",")
    return name in {n.strip() for n in enabled_names if n.strip()}


def _create_profiler() -> torch.profiler.profile:
    activities = [torch.profiler.ProfilerActivity.CPU]
    device_activity = _DEVICE_ACTIVITY_MAP.get(current_platform.device_type)
    if device_activity is not None:
        activities.append(device_activity)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )


def _log_profiler_summary(profiler: torch.profiler.profile, name: str, rank_info: str) -> None:
    if not hasattr(profiler, "key_averages"):
        return
    try:
        summary = profiler.key_averages()
        logger.info(f"{rank_info}Profiler summary for '{name}': Total operations: {len(summary)}")

        top_ops = sorted(
            summary,
            key=lambda x: getattr(x, "cpu_time_total", 0) + getattr(x, "cuda_time_total", 0),
            reverse=True,
        )[:10]
        for i, op in enumerate(top_ops):
            cpu_time_ms = getattr(op, "cpu_time_total", getattr(op, "cpu_time", 0)) / 1000
            cuda_time_ms = getattr(op, "cuda_time_total", getattr(op, "cuda_time", 0)) / 1000
            cuda_str = f"{cuda_time_ms:.2f} ms" if cuda_time_ms > 0 else "N/A"
            logger.info(f"{rank_info}  {i + 1}. {op.key}: CPU={cpu_time_ms:.2f} ms, CUDA={cuda_str}")
    except Exception as e:
        logger.warning(f"{rank_info}Failed to generate profiler summary: {e}")


# Global counter per profiler name for unique output filenames
_profiler_run_counts: dict[str, int] = {}


class _ProfilingContext:
    """Profiling context manager and decorator.

    When used as a decorator, each function invocation creates fresh profiling state,
    avoiding shared-state bugs in concurrent or repeated calls.
    """

    def __init__(self, name: str, *, reset_peak_memory: bool = True):
        self.name = name
        self.reset_peak_memory = reset_peak_memory
        self._rank = _get_rank()
        self._rank_info = f"Rank {self._rank} - "
        self._enable_profiler = _should_enable_profiler(name)
        self._profiler_output_dir = Path(os.getenv("PROFILER_OUTPUT_DIR", "./profiler_output"))
        # Per-invocation state
        self._profiler: torch.profiler.profile | None = None
        self._start_time: float = 0.0

    def _get_profiler_output_path(self) -> Path:
        count = _profiler_run_counts.get(self.name, 0) + 1
        _profiler_run_counts[self.name] = count
        return self._profiler_output_dir / f"{self.name}_rank{self._rank}_run{count}.json"

    def _start(self) -> None:
        current_platform.synchronize()
        if self.reset_peak_memory:
            current_platform.reset_peak_memory_stats()
        self._start_time = time.perf_counter()

        if self._enable_profiler:
            logger.info(f"{self._rank_info}Starting PyTorch profiler for '{self.name}'")
            self._profiler = _create_profiler()
            self._profiler.start()

    def _stop(self) -> None:
        current_platform.synchronize()

        if self._enable_profiler and self._profiler:
            self._profiler.stop()
            self._profiler_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self._get_profiler_output_path()
            self._profiler.export_chrome_trace(str(output_path))
            logger.info(f"{self._rank_info}PyTorch profiler trace saved to: {output_path}")
            _log_profiler_summary(self._profiler, self.name, self._rank_info)
            self._profiler = None

        peak_memory = current_platform.max_memory_allocated() / (1024**3)
        elapsed = time.perf_counter() - self._start_time
        logger.info(f"{self._rank_info}Function '{self.name}' Peak Memory: {peak_memory:.2f} GB")
        logger.info(f"[Profile] {self.name} cost {elapsed:.6f} seconds")

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop()
        return False

    async def __aenter__(self):
        self._start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._stop()
        return False

    def __call__(self, func):
        name = self.name
        reset_peak_memory = self.reset_peak_memory

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with _ProfilingContext(name, reset_peak_memory=reset_peak_memory):
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with _ProfilingContext(name, reset_peak_memory=reset_peak_memory):
                    return func(*args, **kwargs)

            return sync_wrapper


class _NullContext:
    """No-op context manager / decorator for disabled profiling."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __call__(self, func):
        return func


# Public API — backward compatible
ProfilingContext = _ProfilingContext
_DEBUG = os.getenv("TELEFUSER_PROFILE_DEBUG", "false").lower() == "true"
ProfilingContext4Debug = _ProfilingContext if _DEBUG else _NullContext


def enable_profiler_for_names(names: str) -> None:
    """Set the list of names to enable profiler for."""
    os.environ["ENABLE_PROFILER_NAMES"] = names


def set_profiler_output_dir(path: str) -> None:
    """Set profiler output directory."""
    os.environ["PROFILER_OUTPUT_DIR"] = path


def get_enabled_profiler_names() -> set[str]:
    """Get the set of currently enabled profiler names."""
    enabled_names = os.getenv("ENABLE_PROFILER_NAMES", "").split(",")
    return {name.strip() for name in enabled_names if name.strip()}
