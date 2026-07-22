"""CPU platform implementation."""

from __future__ import annotations

import torch

from .interface import BasePlatform


class CpuPlatform(BasePlatform):
    """CPU platform for fallback execution without accelerator."""

    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    device_control_env_var = "CPU_VISIBLE_MEMORY_NODES"
    dist_backend: str = "gloo"
    full_dist_backend: str = "cpu:gloo"

    @staticmethod
    def empty_cache() -> None:
        """CPU has no device allocator cache to release."""

    @staticmethod
    def default_device():
        return torch.device("cpu")

    @staticmethod
    def get_device_name():
        return "CPU"

    @staticmethod
    def is_accelerator_available():
        return False
