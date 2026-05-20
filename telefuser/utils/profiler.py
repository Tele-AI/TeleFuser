"""
TeleFuser Profiler Module

Three-layer profiling (progressive enablement):

Layer 1 (Gated by TELEFUSER_PROFILE_DEBUG=true):
    - Stage timing (ms)
    - Peak GPU memory (GB)
    - Memory snapshots at checkpoints
    - Auto export to JSON
    - Enable: TELEFUSER_PROFILE_DEBUG=true (and optionally TELEFUSER_TIMING_REPORT=path.json)

Layer 2 (ON by env var):
    - torch.profiler trace
    - Kernel categorization
    - Chrome trace visualization
    - Enable: ENABLE_PROFILER_NAMES=stage_name1,stage_name2
    - Optional torch.profiler flags:
      TELEFUSER_TORCH_PROFILER_RECORD_SHAPES=true|false
      TELEFUSER_TORCH_PROFILER_PROFILE_MEMORY=true|false
      TELEFUSER_TORCH_PROFILER_WITH_STACK=true|false

Layer 3 (External tool):
    - ncu deep kernel analysis
    - Enable: ncu --set full python inference.py

Usage:
    # Layer 1: Auto export timing report
    TELEFUSER_TIMING_REPORT=timing.json python inference.py

    # Or programmatically
    reset_timing_registry("req_001")
    pipeline.run(...)
    dump_timing_report("timing.json")

    # Layer 2: Enable profiler for specific stages
    ENABLE_PROFILER_NAMES=denoising,vae_decode python inference.py

    # Layer 3: Use ncu externally
    ncu --set full python inference.py
"""

from __future__ import annotations

import asyncio
import atexit
import inspect
import json
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import torch

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.memory_snapshot import (
    MemorySnapshot,
    capture_memory_snapshot,
    get_memory_analysis,
    get_memory_snapshots,
    print_memory_analysis,
    record_memory_snapshot,
    reset_memory_registry,
    reset_peak_memory_stats,
    set_memory_baseline,
)

F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# Configuration
# =============================================================================

_DEVICE_ACTIVITY_MAP = {
    "cuda": torch.profiler.ProfilerActivity.CUDA,
    "xpu": torch.profiler.ProfilerActivity.XPU,
    "npu": torch.profiler.ProfilerActivity.PrivateUse1,
}


# =============================================================================
# Stage I/O Signature (Layer 1 Capture)
# =============================================================================


@dataclass
class TensorSignature:
    """Signature for a tensor input/output."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool = False

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "requires_grad": self.requires_grad,
        }

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorSignature":
        """Create signature from a tensor."""
        return cls(
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
            device=str(tensor.device),
            requires_grad=tensor.requires_grad,
        )


@dataclass
class StageIOSignature:
    """Signature for a stage's inputs and outputs."""

    stage_name: str
    input_signatures: dict[str, TensorSignature | list[TensorSignature] | str | float | int | None]
    output_signature: TensorSignature | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        inputs = {}
        for name, sig in self.input_signatures.items():
            if isinstance(sig, TensorSignature):
                inputs[name] = sig.to_dict()
            elif isinstance(sig, list) and sig and isinstance(sig[0], TensorSignature):
                inputs[name] = [s.to_dict() for s in sig]
            else:
                inputs[name] = sig
        return {
            "stage_name": self.stage_name,
            "input_signatures": inputs,
            "output_signature": self.output_signature.to_dict() if self.output_signature else None,
            "metadata": self.metadata,
        }


def _extract_signature_from_value(value: Any) -> TensorSignature | list[TensorSignature] | str | float | int | None:
    """Extract signature from a value (tensor, list of tensors, or primitive)."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return TensorSignature.from_tensor(value)
    if isinstance(value, (list, tuple)):
        return (
            [TensorSignature.from_tensor(t) for t in value]
            if value and isinstance(value[0], torch.Tensor)
            else f"list[{type(value[0]).__name__}]"
        )
    if isinstance(value, (str, float, int, bool)):
        return value
    return str(type(value).__name__)


def _get_rank() -> int:
    """Get current process rank for distributed inference."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _should_enable_profiler(name: str) -> bool:
    """Check if torch.profiler should be enabled for this name (Layer 2)."""
    enabled_names = os.getenv("ENABLE_PROFILER_NAMES", "")
    if not enabled_names:
        return False
    enabled_set = {n.strip() for n in enabled_names.split(",") if n.strip()}
    # Support wildcard to enable all stages
    if "*" in enabled_set:
        return True
    return name in enabled_set


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var while preserving default behavior on unset/invalid values."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    logger.warning(f"[Profiler] Invalid boolean value for {name}={value!r}; using default {default}.")
    return default


def _get_torch_profiler_options() -> dict[str, bool]:
    """Get configurable torch.profiler options.

    Defaults intentionally match the historical TeleFuser profiler behavior.
    """
    return {
        "record_shapes": _env_bool("TELEFUSER_TORCH_PROFILER_RECORD_SHAPES", True),
        "profile_memory": _env_bool("TELEFUSER_TORCH_PROFILER_PROFILE_MEMORY", True),
        "with_stack": _env_bool("TELEFUSER_TORCH_PROFILER_WITH_STACK", True),
    }


def _get_timing_report_path() -> str | None:
    """Get Layer 1 timing report output path from env."""
    return os.getenv("TELEFUSER_TIMING_REPORT")


def _get_pipeline_name() -> str:
    """Get pipeline name from env, default to 'default'."""
    return os.getenv("TELEFUSER_PIPELINE_NAME", "default")


def _get_profiler_output_dir() -> Path:
    """Get profiler output directory.

    Default: work_dirs/profiler_output/{pipeline_name}/{YYYYMMDD_HHMM}
    """
    explicit_dir = os.getenv("TELEFUSER_PROFILER_OUTPUT_DIR")
    if explicit_dir:
        return Path(explicit_dir)
    return Path("work_dirs") / "profiler_output" / _get_pipeline_name() / datetime.now().strftime("%Y%m%d_%H%M")


def set_pipeline_name(name: str) -> None:
    """Set pipeline name for profiler output directory.

    Args:
        name: Pipeline name (e.g., 'wan21_t2v', 'wan22_i2v')
    """
    os.environ["TELEFUSER_PIPELINE_NAME"] = name


# =============================================================================
# Worker Timing File-based IPC
# =============================================================================


def _get_worker_timing_dir() -> Path:
    """Get worker timing directory, shared between main and worker processes via env var."""
    dir_path = os.environ.get("_TELEFUSER_WORKER_TIMING_DIR")
    if dir_path:
        return Path(dir_path)
    # Default: use temp directory scoped to main process PID
    dir_path = str(Path(tempfile.gettempdir()) / f"telefuser_worker_timing_{os.getpid()}")
    os.environ["_TELEFUSER_WORKER_TIMING_DIR"] = dir_path
    return Path(dir_path)


def _clean_worker_timing_dir() -> None:
    """Remove all worker timing files from previous runs."""
    worker_dir = _get_worker_timing_dir()
    if worker_dir.exists():
        for f in worker_dir.glob("*.json"):
            f.unlink(missing_ok=True)


def _write_worker_timing_file(name: str, rank: int, duration_ms: float, memory_snapshot: MemorySnapshot | None) -> None:
    """Write a timing record file from a worker process."""
    worker_dir = _get_worker_timing_dir()
    worker_dir.mkdir(parents=True, exist_ok=True)
    timing_file = worker_dir / f"rank{rank}_{name}.json"
    data = {
        "rank": rank,
        "name": name,
        "duration_ms": duration_ms,
        "peak_memory_gb": memory_snapshot.peak_reserved_gb if memory_snapshot else 0.0,
    }
    try:
        with open(timing_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        logger.warning(f"[Profiler] Failed to write worker timing file: {timing_file}")


# =============================================================================
# Global Timing Registry (Layer 1)
# =============================================================================


class GlobalTimingRegistry:
    """
    Global registry for collecting stage timing across the pipeline.

    Layer 1: Always collects timing + memory + I/O signatures.
    Supports auto-export via TELEFUSER_TIMING_REPORT env var.

    Memory tracking is delegated to the memory_snapshot module.
    """

    _instance: "GlobalTimingRegistry | None" = None

    def __init__(self):
        self._records: list[dict[str, Any]] = []
        self._request_id: str = ""
        self._metadata: dict[str, Any] = {}
        self._parallel_info: dict[str, Any] = {}
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._io_signatures: dict[str, StageIOSignature] = {}  # stage_name -> signature

    @classmethod
    def get_instance(cls) -> "GlobalTimingRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls, request_id: str = "", metadata: dict[str, Any] | None = None) -> "GlobalTimingRegistry":
        """Reset registry for new request."""
        instance = cls.get_instance()
        instance._records = []
        instance._request_id = request_id or f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        instance._metadata = metadata or {}
        instance._parallel_info = {}
        instance._io_signatures = {}
        instance._start_time = time.perf_counter()
        instance._end_time = 0.0
        # Reset memory registry
        reset_memory_registry()
        _clean_worker_timing_dir()
        return instance

    def record(self, name: str, duration_ms: float, memory_snapshot: MemorySnapshot | None = None) -> None:
        """Record a stage timing with memory snapshot."""
        self._records.append(
            {
                "name": name,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat(),
                "memory_snapshot": memory_snapshot,
            }
        )

    def record_parallel_stage(self, name: str, rank_records: dict[int, dict]) -> None:
        """Record a parallel stage with max duration across ranks.

        Args:
            name: Stage name
            rank_records: {rank: {"duration_ms": ..., "peak_memory_gb": ...}}
        """
        max_duration = max(r["duration_ms"] for r in rank_records.values())
        self._records.append(
            {
                "name": name,
                "duration_ms": max_duration,
                "timestamp": datetime.now().isoformat(),
                "memory_snapshot": None,
            }
        )
        self._parallel_info[name] = {
            "num_ranks": len(rank_records),
            "ranks": [{"rank": rank, **data} for rank, data in sorted(rank_records.items())],
        }

    def record_signature(self, signature: StageIOSignature) -> None:
        """Record a stage I/O signature for Layer 2 harness."""
        self._io_signatures[signature.stage_name] = signature

    def get_io_signatures(self) -> dict[str, StageIOSignature]:
        """Get all recorded I/O signatures."""
        return self._io_signatures.copy()

    def dump_io_signatures(self, path: str) -> None:
        """Dump I/O signatures to JSON file for Layer 2 harness."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "request_id": self._request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stages": {name: sig.to_dict() for name, sig in self._io_signatures.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"[Profiler] Stage I/O signatures saved to: {path}")

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata for the report."""
        self._metadata[key] = value

    def finalize(self) -> None:
        """Mark the end of timing collection (idempotent)."""
        if self._end_time == 0.0:
            self._end_time = time.perf_counter()

    def _compute_totals(self) -> tuple[float, float]:
        """Compute total duration and wall clock time."""
        total_ms = sum(r["duration_ms"] for r in self._records)
        wall_clock_ms = (self._end_time - self._start_time) * 1000 if self._end_time > 0 else total_ms
        return total_ms, wall_clock_ms

    def _aggregate_worker_timing(self) -> None:
        """Aggregate timing records from worker processes via file-based IPC.

        Scans the worker timing directory for per-rank JSON files and merges
        them into the registry using record_parallel_stage().
        """
        worker_dir = _get_worker_timing_dir()
        if not worker_dir.exists():
            return

        timing_files = list(worker_dir.glob("*.json"))
        if not timing_files:
            return

        # Group by stage name
        stage_rank_data: dict[str, dict[int, dict]] = defaultdict(dict)
        for timing_file in timing_files:
            try:
                with open(timing_file, encoding="utf-8") as f:
                    data = json.load(f)
                stage_rank_data[data["name"]][data["rank"]] = {
                    "duration_ms": data["duration_ms"],
                    "peak_memory_gb": data["peak_memory_gb"],
                }
            except (OSError, json.JSONDecodeError, KeyError):
                continue

        for stage_name, rank_records in stage_rank_data.items():
            self.record_parallel_stage(stage_name, rank_records)

        # Clean up after aggregation
        for timing_file in timing_files:
            timing_file.unlink(missing_ok=True)

    def get_report(self) -> dict:
        """Get timing report as dict."""
        self._aggregate_worker_timing()
        self.finalize()

        total_ms, wall_clock_ms = self._compute_totals()

        # Stage breakdown
        stages_dict = {}
        for r in self._records:
            pct = round(r["duration_ms"] / total_ms * 100, 1) if total_ms > 0 else 0
            stage_data: dict[str, Any] = {
                "duration_ms": round(r["duration_ms"], 2),
                "percentage": pct,
            }
            if r["memory_snapshot"] is not None:
                stage_data["memory"] = r["memory_snapshot"].to_dict()
            if r["name"] in self._parallel_info:
                stage_data["parallel_info"] = self._parallel_info[r["name"]]
            stages_dict[r["name"]] = stage_data

        report = {
            "request_id": self._request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wall_clock_ms": round(wall_clock_ms, 2),
            "total_stages_ms": round(total_ms, 2),
            "num_stages": len(self._records),
            "stages": stages_dict,
            "metadata": self._metadata,
        }

        # Integrate full memory analysis from memory_snapshot module
        memory_analysis = get_memory_analysis()
        if memory_analysis:
            report["memory"] = memory_analysis

        return report

    def dump(self, path: str) -> None:
        """Dump timing report to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        report = self.get_report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[Profiler] Layer 1 timing report saved to: {path}")

    def summary(self) -> str:
        """Get human-readable summary string."""
        total_ms, wall_clock_ms = self._compute_totals()
        memory_analysis = get_memory_analysis()

        lines = [
            "",
            "=" * 60,
            f"Pipeline Timing Summary ({self._request_id})",
            "=" * 60,
            f"Wall Clock: {wall_clock_ms:.1f} ms ({wall_clock_ms / 1000:.2f} s)",
            f"Stage Total: {total_ms:.1f} ms",
        ]

        # Memory section from memory_analysis
        if memory_analysis:
            peak_res_gb = memory_analysis["peak_reserved_gb"]
            peak_res_mb = memory_analysis["peak_reserved_mb"]
            peak_alloc_gb = memory_analysis["peak_allocated_gb"]
            peak_alloc_mb = memory_analysis["peak_allocated_mb"]
            pool_mb = memory_analysis["pool_overhead_mb"]
            pool_pct = memory_analysis["pool_overhead_pct"]
            remaining_gb = memory_analysis["remaining_memory_gb"]
            lines.extend(
                [
                    "",
                    "Memory Statistics:",
                    f"  Peak Reserved: {peak_res_gb:.2f} GB ({peak_res_mb:.0f} MB)",
                    f"  Peak Allocated: {peak_alloc_gb:.2f} GB ({peak_alloc_mb:.0f} MB)",
                    f"  Pool Overhead: {pool_mb:.0f} MB ({pool_pct:.1f}%)",
                    f"  Remaining: {remaining_gb:.2f} GB",
                ]
            )
            if "memory_increase_gb" in memory_analysis:
                lines.append(f"  Increase from Baseline: {memory_analysis['memory_increase_gb']:.2f} GB")

        lines.append("-" * 60)
        lines.append("Stage Breakdown:")

        for r in self._records:
            pct = r["duration_ms"] / total_ms * 100 if total_ms > 0 else 0
            mem_info = ""
            if r["memory_snapshot"]:
                mem_info = (
                    f"  mem: {r['memory_snapshot'].allocated_mb:.0f}/{r['memory_snapshot'].reserved_mb:.0f} MB "
                    f"(peak: {r['memory_snapshot'].peak_allocated_mb:.0f}/{r['memory_snapshot'].peak_reserved_mb:.0f})"
                )
            lines.append(f"  {r['name']:30s} {r['duration_ms']:8.1f} ms ({pct:5.1f}%){mem_info}")
            if r["name"] in self._parallel_info:
                for rank_data in self._parallel_info[r["name"]]["ranks"]:
                    lines.append(
                        f"    rank {rank_data['rank']:2d}: {rank_data['duration_ms']:8.1f} ms  "
                        f"peak: {rank_data['peak_memory_gb']:.2f} GB"
                    )

        # Memory checkpoints from memory_snapshot module
        memory_snapshots = get_memory_snapshots()
        if memory_snapshots:
            lines.append("-" * 60)
            lines.append("Memory Checkpoints:")
            for name, snapshot in memory_snapshots.items():
                lines.append(
                    f"  {name}: {snapshot.allocated_mb:.0f}/{snapshot.reserved_mb:.0f} MB "
                    f"(peak: {snapshot.peak_allocated_mb:.0f}/{snapshot.peak_reserved_mb:.0f} MB)"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# Flag to suppress atexit export in worker processes
_IS_WORKER_PROCESS = False


def mark_as_worker_process() -> None:
    """Mark the current process as a worker. Suppresses atexit timing export."""
    global _IS_WORKER_PROCESS
    _IS_WORKER_PROCESS = True


def _auto_export_timing_report() -> None:
    """Auto export timing report at exit if TELEFUSER_PROFILE_DEBUG is true."""
    if _IS_WORKER_PROCESS:
        return

    registry = GlobalTimingRegistry.get_instance()
    if not registry._records:
        return

    # Get explicit path or use default
    path = _get_timing_report_path()
    if not path:
        # Auto-generate path: work_dirs/profiler_output/{pipeline_name}/{date}/timing.json
        output_dir = _get_profiler_output_dir()
        path = str(output_dir / "timing.json")

    registry.dump(path)
    # Also export I/O signatures for Layer 2 harness
    if registry._io_signatures:
        sig_path = str(path).replace(".json", "_io_signature.json")
        registry.dump_io_signatures(sig_path)


# Register auto-export at program exit
atexit.register(_auto_export_timing_report)


# =============================================================================
# Public API for Layer 1
# =============================================================================


def reset_timing_registry(request_id: str = "", metadata: dict[str, Any] | None = None) -> GlobalTimingRegistry:
    """
    Reset timing registry for a new request.

    Args:
        request_id: Optional request identifier
        metadata: Optional metadata dict to include in report

    Returns:
        The GlobalTimingRegistry instance
    """
    return GlobalTimingRegistry.reset(request_id, metadata)


def get_timing_report() -> dict:
    """Get current timing report as dict."""
    return GlobalTimingRegistry.get_instance().get_report()


def dump_timing_report(path: str) -> None:
    """
    Dump timing report to JSON file.

    Args:
        path: Output file path
    """
    GlobalTimingRegistry.get_instance().dump(path)


def print_timing_summary() -> None:
    """Print timing summary to logger."""
    logger.info(GlobalTimingRegistry.get_instance().summary())


def set_timing_metadata(key: str, value: Any) -> None:
    """
    Set metadata for timing report.

    Args:
        key: Metadata key
        value: Metadata value
    """
    GlobalTimingRegistry.get_instance().set_metadata(key, value)


def get_io_signatures() -> dict[str, StageIOSignature]:
    """
    Get all recorded stage I/O signatures.

    Returns:
        Dict mapping stage names to their I/O signatures
    """
    return GlobalTimingRegistry.get_instance().get_io_signatures()


def dump_io_signatures(path: str) -> None:
    """
    Dump stage I/O signatures to JSON file.

    Args:
        path: Output file path
    """
    GlobalTimingRegistry.get_instance().dump_io_signatures(path)


# =============================================================================
# Profiling Context (Layer 1 + Layer 2)
# =============================================================================


class _ProfilingContext:
    """
    Profiling context manager and decorator.

    Layer 1 (Always): Record timing + peak memory directly
    Layer 2 (Optional): torch.profiler trace + kernel breakdown

    Usage:
        @ProfilingContext4Debug("denoising")
        def forward(self, batch):
            ...
    """

    def __init__(
        self,
        name: str,
        *,
        reset_peak_memory: bool = True,
        capture_memory: bool = False,
    ):
        self.name = name
        self.reset_peak_memory = reset_peak_memory
        self.capture_memory = capture_memory
        self._rank = _get_rank()
        self._rank_info = f"[Rank {self._rank}]"
        self._enable_profiler = _should_enable_profiler(name)
        self._profiler: torch.profiler.profile | None = None
        self._output_dir = _get_profiler_output_dir()
        self._start_time: float = 0.0
        self._duration_ms: float = 0.0
        self._memory_snapshot: MemorySnapshot | None = None

    def _get_output_path(self, suffix: str) -> Path:
        """Get output file path with suffix."""
        return self._output_dir / f"{self.name}_rank{self._rank}.{suffix}"

    def _format_memory_info(self) -> str:
        """Format memory info string for logging."""
        if self._memory_snapshot:
            return (
                f", memory: {self._memory_snapshot.allocated_mb:.0f}/"
                f"{self._memory_snapshot.reserved_mb:.0f} MB "
                f"(peak: {self._memory_snapshot.peak_allocated_mb:.0f}/"
                f"{self._memory_snapshot.peak_reserved_mb:.0f} MB)"
            )
        return ""

    def __enter__(self):
        current_platform.synchronize()

        if self.capture_memory:
            record_memory_snapshot(f"before_{self.name}")

        if self.reset_peak_memory:
            reset_peak_memory_stats()

        self._start_time = time.perf_counter()

        if self._enable_profiler:
            activities = [torch.profiler.ProfilerActivity.CPU]
            device_activity = _DEVICE_ACTIVITY_MAP.get(current_platform.device_type)
            if device_activity:
                activities.append(device_activity)

            self._profiler = torch.profiler.profile(
                activities=activities,
                **_get_torch_profiler_options(),
            )
            self._profiler.start()
            logger.info(f"{self._rank_info} [Profiler] Started torch.profiler for '{self.name}'")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_platform.synchronize()

        # Compute timing and memory
        self._duration_ms = (time.perf_counter() - self._start_time) * 1000
        self._memory_snapshot = capture_memory_snapshot()

        if self.capture_memory:
            record_memory_snapshot(f"after_{self.name}")

        # Register with global registry
        GlobalTimingRegistry.get_instance().record(self.name, self._duration_ms, self._memory_snapshot)

        # In worker processes, write timing to file for main process aggregation
        if _IS_WORKER_PROCESS:
            _write_worker_timing_file(self.name, self._rank, self._duration_ms, self._memory_snapshot)

        # Log with memory details
        mem_info = self._format_memory_info()
        logger.info(f"{self._rank_info} [Timing] {self.name}: {self._duration_ms:.2f} ms{mem_info}")

        if self._enable_profiler and self._profiler:
            self._profiler.stop()
            self._output_dir.mkdir(parents=True, exist_ok=True)

            # Export Chrome trace
            trace_path = self._get_output_path("trace.json.gz")
            self._profiler.export_chrome_trace(str(trace_path))
            logger.info(f"{self._rank_info} [Profiler] Chrome trace: {trace_path}")

            # Export kernel breakdown
            self._export_kernel_breakdown()
            self._profiler = None

        return False

    def _export_kernel_breakdown(self) -> None:
        """Export top kernels directly without categorization."""
        if not self._profiler or not hasattr(self._profiler, "key_averages"):
            return

        try:
            events = self._profiler.key_averages()
        except Exception:
            return

        kernels = []
        for event in events:
            cuda_ms = getattr(event, "cuda_time_total", 0) / 1000
            cpu_ms = getattr(event, "cpu_time_total", 0) / 1000
            total_ms = max(cuda_ms, cpu_ms)
            if total_ms < 0.01:
                continue
            kernels.append(
                {"name": event.key, "ms": round(total_ms, 2), "cuda_ms": round(cuda_ms, 2), "cpu_ms": round(cpu_ms, 2)}
            )

        kernels.sort(key=lambda x: -x["ms"])
        report = {
            "name": self.name,
            "rank": self._rank,
            "total_kernel_time_ms": round(sum(k["ms"] for k in kernels), 2),
            "num_kernels": len(kernels),
            "top_kernels": kernels[:50],
        }

        breakdown_path = self._get_output_path("breakdown.json")
        with open(breakdown_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"{self._rank_info} [Profiler] Kernel breakdown: {breakdown_path}")

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func: F) -> F:
        """Decorator support for wrapping functions."""
        is_async = asyncio.iscoroutinefunction(func)
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            input_signatures = _capture_input_signatures(args, kwargs, param_names)
            async with _ProfilingContext(self.name, **self._ctx_kwargs):
                result = await func(*args, **kwargs)
            _record_stage_signature(self.name, input_signatures, result, kwargs)
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            input_signatures = _capture_input_signatures(args, kwargs, param_names)
            with _ProfilingContext(self.name, **self._ctx_kwargs):
                result = func(*args, **kwargs)
            _record_stage_signature(self.name, input_signatures, result, kwargs)
            return result

        return cast(F, async_wrapper if is_async else sync_wrapper)

    @property
    def _ctx_kwargs(self) -> dict:
        """Context kwargs for recreating context in wrapper."""
        return {"reset_peak_memory": self.reset_peak_memory, "capture_memory": self.capture_memory}


def _capture_input_signatures(args: tuple, kwargs: dict, param_names: list[str]) -> dict:
    """Capture input signatures from args and kwargs."""
    # Skip 'self' parameter (first arg in methods)
    start_idx = 1 if param_names and param_names[0] == "self" else 0
    input_signatures = {
        param_names[i]: _extract_signature_from_value(arg)
        for i, arg in enumerate(args[start_idx:], start=start_idx)
        if i < len(param_names)
    }
    input_signatures.update({k: _extract_signature_from_value(v) for k, v in kwargs.items()})
    return input_signatures


def _extract_metadata_from_kwargs(kwargs: dict) -> dict:
    """Extract metadata (non-tensor numeric/string parameters) for harness."""
    return {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool))}


def _record_stage_signature(
    name: str,
    input_signatures: dict,
    result: Any,
    kwargs: dict,
) -> None:
    """Record stage I/O signature to registry (main process only)."""
    if _IS_WORKER_PROCESS or not input_signatures:
        return
    output_sig = _extract_signature_from_value(result) if isinstance(result, torch.Tensor) else None
    GlobalTimingRegistry.get_instance().record_signature(
        StageIOSignature(
            stage_name=name,
            input_signatures=input_signatures,
            output_signature=output_sig if isinstance(output_sig, TensorSignature) else None,
            metadata=_extract_metadata_from_kwargs(kwargs),
        )
    )


class _NullContext:
    """
    No-op context manager. No synchronization, no timing, no memory tracking.

    Used when TELEFUSER_PROFILE_DEBUG is false (default).
    """

    def __init__(self, name: str, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def __call__(self, func: F) -> F:
        """Decorator support - returns the function unchanged."""
        return func


# =============================================================================
# Public API - ProfilingContext
# =============================================================================

# Backward compatible: ProfilingContext4Debug
# - Always records Layer 1 timing + memory
# - Enables Layer 2 torch.profiler when ENABLE_PROFILER_NAMES matches
_DEBUG = os.getenv("TELEFUSER_PROFILE_DEBUG", "false").lower() == "true"
ProfilingContext4Debug = _ProfilingContext if _DEBUG else _NullContext

# Full profiler context (always uses torch.profiler when enabled)
ProfilingContext = _ProfilingContext


# =============================================================================
# Convenience Functions
# =============================================================================


def enable_profiler_for_names(names: str) -> None:
    """
    Set the list of names to enable torch.profiler for (Layer 2).

    Args:
        names: Comma-separated list of stage names, or "*" for all
    """
    os.environ["ENABLE_PROFILER_NAMES"] = names


def set_profiler_output_dir(path: str) -> None:
    """
    Set profiler output directory for Layer 2 traces.

    Args:
        path: Output directory path
    """
    os.environ["TELEFUSER_PROFILER_OUTPUT_DIR"] = path


def get_profiler_output_dir() -> Path:
    """
    Get the current profiler output directory.

    Returns:
        Path to the profiler output directory
    """
    return _get_profiler_output_dir()


# =============================================================================
# Layer 3: NCU Helpers
# =============================================================================

NCU_KEY_METRICS = [
    "gpu__time_duration.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]


def analyze_ncu_report(report_path: str) -> dict[str, Any]:
    """
    Analyze NCU report and extract key metrics (Layer 3).

    Args:
        report_path: Path to .ncu-rep file

    Returns:
        Dict with metrics and bottleneck analysis
    """
    import csv
    import io
    import subprocess

    cmd = ["ncu", "--import", report_path, "--page", "details", "--csv"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    metrics = {}
    reader = csv.DictReader(io.StringIO(result.stdout))

    for row in reader:
        metric_name = row.get("Metric Name", "")
        for key_metric in NCU_KEY_METRICS:
            if key_metric in metric_name:
                metrics[key_metric] = row.get("Metric Value", "")
                break

    def parse_pct(val: str) -> float:
        """Parse percentage value."""
        if not val:
            return 0.0
        return float(val.replace("%", "").replace(",", ""))

    dram = parse_pct(metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", "0"))
    sm = parse_pct(metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "0"))

    # Determine bottleneck type
    if dram > 80:
        bottleneck = "memory_bound"
        recommendation = (
            "Kernel saturates memory bandwidth. Consider fusing with adjacent ops to reduce memory traffic."
        )
    elif sm > 60:
        bottleneck = "compute_bound"
        recommendation = "Kernel is compute-heavy. Consider reducing arithmetic or using faster instructions."
    else:
        bottleneck = "latency_bound"
        recommendation = "Kernel underutilizes hardware. Check occupancy and memory coalescing."

    return {
        "metrics": metrics,
        "dram_throughput_pct": dram,
        "sm_throughput_pct": sm,
        "bottleneck": bottleneck,
        "recommendation": recommendation,
    }


# =============================================================================
# Performance Comparison
# =============================================================================


def compare_timing_reports(baseline_path: str, new_path: str) -> str:
    """
    Compare two timing reports and generate markdown diff.

    Args:
        baseline_path: Path to baseline timing JSON
        new_path: Path to new timing JSON

    Returns:
        Markdown formatted comparison string
    """
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(new_path, encoding="utf-8") as f:
        new = json.load(f)

    lines = ["### Performance Comparison\n"]

    # Total time comparison
    base_total = baseline.get("total_stages_ms", baseline.get("total_ms", 0))
    new_total = new.get("total_stages_ms", new.get("total_ms", 0))
    diff_pct = ((new_total - base_total) / base_total * 100) if base_total else 0
    emoji = "🔴" if diff_pct > 5 else "🟢" if diff_pct < -5 else "⚪"

    lines.append("| Metric | Baseline | New | Diff |")
    lines.append("|--------|----------|-----|------|")
    lines.append(f"| Total (ms) | {base_total:.1f} | {new_total:.1f} | {diff_pct:+.1f}% {emoji} |")

    # Peak memory comparison
    base_peak = baseline.get("peak_memory_gb", 0)
    new_peak = new.get("peak_memory_gb", 0)
    lines.append(f"| Peak Memory (GB) | {base_peak:.2f} | {new_peak:.2f} | - |")

    # Stage breakdown
    lines.append("\n### Stage Breakdown")
    lines.append("| Stage | Baseline (ms) | New (ms) | Diff (%) |")
    lines.append("|-------|---------------|----------|----------|")

    base_stages = baseline.get("stages", {})
    new_stages = new.get("stages", {})

    all_names = sorted(set(base_stages.keys()) | set(new_stages.keys()))
    for name in all_names:
        b = base_stages.get(name, {}).get("duration_ms", 0)
        n = new_stages.get(name, {}).get("duration_ms", 0)
        d = ((n - b) / b * 100) if b else 0
        e = "🔴" if d > 5 else "🟢" if d < -5 else "⚪"
        lines.append(f"| {name} | {b:.1f} | {n:.1f} | {d:+.1f}% {e} |")

    return "\n".join(lines)
