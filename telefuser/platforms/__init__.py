"""Platform abstraction layer for device-specific operations.

Provides unified interface for CUDA, ROCm, NPU, and CPU backends.
"""

from __future__ import annotations

import logging
import re

import torch

from .cpu import CpuPlatform
from .cuda import CudaPlatform
from .interface import BasePlatform
from .npu import NPUPlatform  # noqa: F401
from .rocm import RocmPlatform

logger = logging.getLogger(__name__)


def _is_cuda_available() -> bool:
    """Check if NVIDIA CUDA is available (not ROCm)."""
    # ROCm uses torch.cuda but is not NVIDIA CUDA
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        return False
    return torch.cuda.is_available()


def _is_rocm_available() -> bool:
    """Check if ROCm is available."""
    if not (hasattr(torch.version, "hip") and torch.version.hip is not None):
        return False
    # A HIP build without a visible GPU falls back to CPU like the CUDA check below.
    return torch.cuda.is_available()


def _is_npu_available() -> bool:
    """Check if NPU is available."""
    try:
        import torch_npu  # type: ignore  # noqa

        return torch.npu.is_available()
    except (ImportError, OSError, RuntimeError):
        return False


def _resolve_current_platform() -> BasePlatform:
    """Determine and instantiate the appropriate platform based on available hardware.

    Detection order: ROCm -> CUDA -> NPU -> CPU
    """
    # Check ROCm first (it also sets torch.cuda.is_available() to True)
    if _is_rocm_available():
        logger.debug("ROCm platform detected")
        return RocmPlatform()

    # Check CUDA (NVIDIA)
    if _is_cuda_available():
        logger.debug("CUDA platform detected")
        # _init_cuda_optimizations()
        return CudaPlatform()

    # Check NPU
    if _is_npu_available():
        logger.debug("NPU platform detected")
        return NPUPlatform()

    # Fall back to CPU
    logger.debug("CPU platform detected")
    return CpuPlatform()


current_platform: BasePlatform = _resolve_current_platform()


def normalize_device_name(device: str | torch.device) -> str:
    """Parse a TeleFuser device without importing its vendor runtime."""
    match = re.fullmatch(r"(cpu|cuda|rocm|npu)(?::([0-9]+))?", str(device))
    if match is None:
        raise ValueError(f"Expected cpu, cuda, rocm or npu with an optional index, got {device!r}")
    kind, index = match.groups()
    if kind == "cpu":
        return "cpu"
    return f"{'cuda' if kind == 'rocm' else kind}:{int(index or 0)}"


def resolve_device(device: str | torch.device) -> torch.device:
    """Validate a compute device against this process before starting workers."""
    name = normalize_device_name(device)
    if name == "cpu":
        return torch.device(name)
    kind, index = name.split(":")
    if kind == "npu":
        try:
            import torch_npu  # noqa: F401
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(f"Cannot initialize torch_npu for {name}: {exc}") from exc
    if kind != current_platform.device_type or (
        str(device).split(":")[0] == "rocm" and current_platform.device_name != "rocm"
    ):
        raise RuntimeError(f"Requested {device} but detected {current_platform.device_name}; check device visibility")
    count = current_platform.device_count()
    if int(index) >= count:
        raise ValueError(f"{name} is outside the {count} visible {current_platform.device_name} devices")
    return torch.device(name)


def requires_host_tensor_queue(device_type: str) -> bool:
    """Ascend tensors cross multiprocessing queues through host memory."""
    return device_type == "npu"


__all__ = [
    "BasePlatform",
    "current_platform",
    "normalize_device_name",
    "resolve_device",
    "requires_host_tensor_queue",
    "CudaPlatform",
    "RocmPlatform",
    "NPUPlatform",
    "CpuPlatform",
]
