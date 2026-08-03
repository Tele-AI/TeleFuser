from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from telefuser.core.config import RayConfig, RayGPUConfig
from telefuser.worker.ray_worker import RayWorker


def _ray_worker(*, num_gpus: int, memory_limit: float = 0.0) -> RayWorker:
    worker = RayWorker.__new__(RayWorker)
    worker.worker_id = "test-stage"
    worker.ray_config = RayConfig(
        gpu_config=RayGPUConfig(num_gpus=num_gpus, memory_limit=memory_limit),
        memory_gb=0,
    )
    return worker


def test_setup_resources_preserves_ray_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    worker = _ray_worker(num_gpus=2, memory_limit=0.5)

    with (
        patch("telefuser.worker.ray_worker.current_platform") as current_platform,
        patch("telefuser.worker.ray_worker.torch.cuda.set_per_process_memory_fraction") as set_fraction,
    ):
        current_platform.device_type = "cuda"
        current_platform.device_count.return_value = 2
        worker._setup_resources()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"
    current_platform.set_device.assert_called_once_with(0)
    set_fraction.assert_called_once_with(0.5, device=0)


def test_setup_resources_rejects_insufficient_ray_gpu_assignment() -> None:
    worker = _ray_worker(num_gpus=2)

    with (
        patch("telefuser.worker.ray_worker.current_platform.device_count", return_value=1),
        patch("telefuser.worker.ray_worker.current_platform.set_device") as set_device,
        pytest.raises(RuntimeError, match="assigned 1 visible GPUs"),
    ):
        worker._setup_resources()

    set_device.assert_not_called()
