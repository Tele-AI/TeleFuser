"""Tests for the tf-kernel FP8 GEMM wrapper."""

import pytest
import torch
import torch.nn as nn

from telefuser.ops import fp8_gemm


def test_fp8_linear_requires_tf_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fp8_gemm, "tf_kernel", None)

    with pytest.raises(ImportError, match="tf-kernel is required"):
        fp8_gemm.FP8Linear(nn.Linear(8, 4), options=fp8_gemm.FP8GemmOptions())


def test_fp8_linear_keeps_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fp8_gemm, "tf_kernel", object())
    linear = nn.Linear(8, 4)
    inputs = torch.randn(2, 8)
    expected = linear(inputs)

    wrapped = fp8_gemm.FP8Linear(
        linear,
        options=fp8_gemm.FP8GemmOptions(
            fp16_weight_storage="keep",
            materialize_fp8_on_wrap=False,
        ),
    )

    torch.testing.assert_close(wrapped(inputs), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(fp8_gemm.tf_kernel is None, reason="tf-kernel is required")
def test_fp8_linear_tf_kernel_forward() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(64, 128, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(2, 64, 3, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    assert not inputs.is_contiguous()
    expected = linear(inputs)
    wrapped = fp8_gemm.FP8Linear(
        linear,
        options=fp8_gemm.FP8GemmOptions(fp16_weight_storage="keep"),
    )

    actual = wrapped(inputs)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.1, rtol=0.1)
