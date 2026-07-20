"""Lightweight target-side measurements for external benchmark collectors."""

from __future__ import annotations

import platform
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class RuntimeMeasurement:
    """State needed to finish one synchronized runtime measurement."""

    started_at_s: float
    devices: tuple[str, ...]
    capture_peak_memory: bool
    clock: Callable[[], float] = field(repr=False, compare=False)


def canonical_cuda_devices(
    values: Sequence[str | torch.device],
) -> tuple[str, ...]:
    """Return unique canonical CUDA device names, ignoring non-CUDA values."""

    devices: list[str] = []
    for value in values:
        device = torch.device(value)
        if device.type != "cuda":
            continue
        index = device.index if device.index is not None else 0
        canonical = f"cuda:{index}"
        if canonical not in devices:
            devices.append(canonical)
    return tuple(devices)


def visible_cuda_devices() -> tuple[str, ...]:
    """Return every CUDA device visible to the current process."""

    if not torch.cuda.is_available():
        return ()
    return tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))


def _synchronize(devices: Sequence[str]) -> None:
    if not torch.cuda.is_available():
        return
    for device in devices:
        torch.cuda.synchronize(device)


def start_runtime_measurement(
    devices: Sequence[str | torch.device],
    *,
    capture_peak_memory: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> RuntimeMeasurement:
    """Synchronize devices, optionally reset peaks, and start a phase timer."""

    canonical = canonical_cuda_devices(devices)
    _synchronize(canonical)
    if capture_peak_memory and torch.cuda.is_available():
        for device in canonical:
            torch.cuda.reset_peak_memory_stats(device)
    return RuntimeMeasurement(
        started_at_s=clock(),
        devices=canonical,
        capture_peak_memory=capture_peak_memory,
        clock=clock,
    )


def finish_runtime_measurement(measurement: RuntimeMeasurement) -> dict[str, Any]:
    """Finish a measurement and return AIPerf-compatible phase facts."""

    _synchronize(measurement.devices)
    seconds = max(measurement.clock() - measurement.started_at_s, 0.0)
    memory: list[dict[str, int | str]] = []
    if measurement.capture_peak_memory and torch.cuda.is_available():
        memory = [
            {
                "device": device,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            for device in measurement.devices
        ]
    return {"seconds": seconds, "memory": memory}


def _git_commit(repo_root: Path | None) -> str | None:
    if repo_root is None:
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def collect_runtime_environment(
    devices: Sequence[str | torch.device],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Collect stable software and accelerator identity for benchmark artifacts."""

    canonical = canonical_cuda_devices(devices)
    gpu_info: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for device in canonical:
            properties = torch.cuda.get_device_properties(device)
            gpu_info.append(
                {
                    "device": device,
                    "name": properties.name,
                    "compute_capability": (f"{properties.major}.{properties.minor}"),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "telefuser_git_commit": _git_commit(repo_root),
        "gpus": gpu_info,
    }
