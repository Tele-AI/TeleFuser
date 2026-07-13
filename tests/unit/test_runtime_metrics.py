from __future__ import annotations

from pathlib import Path

import torch

from telefuser.metrics.runtime import (
    canonical_cuda_devices,
    collect_runtime_environment,
    finish_runtime_measurement,
    start_runtime_measurement,
)


def test_runtime_measurement_uses_injected_monotonic_clock() -> None:
    ticks = iter([10.0, 12.5])
    measurement = start_runtime_measurement([], clock=lambda: next(ticks))

    result = finish_runtime_measurement(measurement)

    assert result == {"seconds": 2.5, "memory": []}


def test_canonical_cuda_devices_deduplicates_and_ignores_cpu() -> None:
    assert canonical_cuda_devices([torch.device("cpu"), "cuda", "cuda:1", "cuda:1"]) == ("cuda:0", "cuda:1")


def test_runtime_environment_has_stable_software_identity() -> None:
    environment = collect_runtime_environment([], repo_root=Path("/missing"))

    assert environment["python_version"]
    assert environment["platform"]
    assert environment["torch_version"]
    assert environment["gpus"] == []
