from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from telefuser.platforms import (
    CpuPlatform,
    CudaPlatform,
    RocmPlatform,
    _resolve_current_platform,
)
from telefuser.platforms.rocm import RocmPlatform as RocmPlatformClass


def test_resolve_prefers_rocm_over_cuda() -> None:
    with (
        patch.object(torch.version, "hip", "test-hip", create=True),
        patch("telefuser.platforms.torch.cuda.is_available", return_value=True),
    ):
        platform = _resolve_current_platform()
    assert isinstance(platform, RocmPlatformClass)


def test_resolve_rocm_without_visible_gpu_falls_back_to_cpu() -> None:
    with (
        patch.object(torch.version, "hip", "test-hip", create=True),
        patch("telefuser.platforms.torch.cuda.is_available", return_value=False),
    ):
        platform = _resolve_current_platform()
    assert isinstance(platform, CpuPlatform)


def test_resolve_cuda_when_no_hip() -> None:
    with (
        patch.object(torch.version, "hip", None, create=True),
        patch("telefuser.platforms.torch.cuda.is_available", return_value=True),
    ):
        platform = _resolve_current_platform()
    assert isinstance(platform, CudaPlatform)


@pytest.mark.parametrize("platform_cls", [CudaPlatform, RocmPlatform])
def test_accelerator_methods_delegate_to_torch_cuda(platform_cls: type) -> None:
    with (
        patch("telefuser.platforms.cuda.torch.cuda.device_count", return_value=3),
        patch("telefuser.platforms.cuda.torch.cuda.current_device", return_value=1),
    ):
        assert platform_cls.device_count() == 3
        assert platform_cls.current_device() == 1


@pytest.mark.parametrize("platform_cls", [CudaPlatform, RocmPlatform])
def test_is_accelerator_available_requires_visible_device(platform_cls: type) -> None:
    for available, count, expected in ((True, 2, True), (True, 0, False), (False, 0, False)):
        with (
            patch("telefuser.platforms.cuda.torch.cuda.is_available", return_value=available),
            patch("telefuser.platforms.cuda.torch.cuda.device_count", return_value=count),
        ):
            assert platform_cls.is_accelerator_available() is expected


def test_rocm_platform_implements_full_interface() -> None:
    assert {"device_count", "is_accelerator_available", "current_device"} <= set(RocmPlatformClass.__dict__)


def test_is_accelerator_available_matches_mocked_torch_cuda() -> None:
    torch_cuda = MagicMock()
    torch_cuda.is_available.return_value = True
    torch_cuda.device_count.return_value = 1
    with patch("telefuser.platforms.rocm.torch.cuda", torch_cuda):
        assert RocmPlatform.is_accelerator_available() is True
        assert RocmPlatform.device_count() == 1
        assert RocmPlatform.current_device() is torch_cuda.current_device()
