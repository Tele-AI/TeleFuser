"""Platform abstraction layer for device-specific operations.

Provides unified interface for CUDA, ROCm, NPU, and CPU backends.
"""

from __future__ import annotations

import logging

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
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def _is_npu_available() -> bool:
    """Check if NPU is available."""
    try:
        import torch_npu  # type: ignore  # noqa

        return torch.npu.is_available()
    except ImportError:
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


__all__ = [
    "BasePlatform",
    "current_platform",
    "CudaPlatform",
    "RocmPlatform",
    "NPUPlatform",
    "CpuPlatform",
]
