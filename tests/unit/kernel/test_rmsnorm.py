"""Unit tests for RMSNorm Triton kernels.

Test markers:
    @pytest.mark.gpu - Tests requiring GPU
"""

import pytest
import torch
import torch.nn as nn

try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    from telefuser.kernel.triton import (
        fused_add_rms_norm,
        norm_infer,
        triton_one_pass_rms_norm,
    )
except (ImportError, RuntimeError) as exc:
    pytest.skip(
        f"Skipping Triton RMSNorm tests: {exc}",
        allow_module_level=True,
    )

# =============================================================================
# Reference Implementations
# =============================================================================


def torch_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm implementation using PyTorch."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def torch_fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference fused add + RMSNorm using PyTorch."""
    residual_out = x + residual
    variance = residual_out.pow(2).mean(dim=-1, keepdim=True)
    x_normed = residual_out * torch.rsqrt(variance + eps)
    return x_normed * weight, residual_out


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cuda_device():
    """Fixture for CUDA device, skips test if CUDA unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# =============================================================================
# RMSNorm Tests
# =============================================================================


@pytest.mark.gpu
class TestRMSNorm:
    """Test RMSNorm kernel implementations."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 128])
    @pytest.mark.parametrize("hidden_size", [64, 256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rms_norm_correctness(self, cuda_device, batch_size, seq_len, hidden_size, dtype):
        """Test RMSNorm kernel produces correct results."""
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        weight = torch.randn(hidden_size, dtype=dtype, device=cuda_device)

        # Triton kernel output
        output_triton = triton_one_pass_rms_norm(x, weight)

        # PyTorch reference
        output_torch = torch_rms_norm(x.float(), weight.float()).to(dtype)

        assert output_triton.shape == x.shape
        assert output_triton.dtype == x.dtype
        assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)

    def test_rms_norm_all_ones(self, cuda_device):
        """Test RMSNorm with all-ones input."""
        dtype = torch.bfloat16
        hidden_size = 64

        x_ones = torch.ones(2, 10, hidden_size, dtype=dtype, device=cuda_device)
        weight = torch.ones(hidden_size, dtype=dtype, device=cuda_device)
        output = triton_one_pass_rms_norm(x_ones, weight)

        # RMS of all ones is 1, so output should be ones
        assert torch.allclose(output, x_ones, atol=1e-3)

    def test_rms_norm_all_zeros(self, cuda_device):
        """Test RMSNorm with all-zeros input should not produce NaN."""
        dtype = torch.bfloat16
        hidden_size = 64

        x_zeros = torch.zeros(2, 10, hidden_size, dtype=dtype, device=cuda_device)
        weight = torch.ones(hidden_size, dtype=dtype, device=cuda_device)
        output = triton_one_pass_rms_norm(x_zeros, weight)

        assert not torch.isnan(output).any()


@pytest.mark.gpu
class TestOnePassRMSNorm:
    """Test single-pass RMSNorm kernel."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [64, 256])
    @pytest.mark.parametrize("hidden_size", [512, 2048])
    def test_one_pass_rms_norm_correctness(self, cuda_device, batch_size, seq_len, hidden_size):
        """Test single-pass RMSNorm kernel produces correct results."""
        dtype = torch.bfloat16
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        weight = torch.randn(hidden_size, dtype=dtype, device=cuda_device)

        # Triton kernel output
        output_triton = triton_one_pass_rms_norm(x, weight)

        # PyTorch reference
        output_torch = torch_rms_norm(x.float(), weight.float()).to(dtype)

        assert output_triton.shape == x.shape
        assert output_triton.dtype == x.dtype
        assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)


# =============================================================================
# Fused Add RMSNorm Tests
# =============================================================================


@pytest.mark.gpu
class TestFusedAddRMSNorm:
    """Test fused add + RMSNorm kernel."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 128])
    @pytest.mark.parametrize("hidden_size", [64, 256, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_add_rms_norm_correctness(self, cuda_device, batch_size, seq_len, hidden_size, dtype):
        """Test fused add + RMSNorm kernel produces correct results."""
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        residual = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        weight = torch.randn(hidden_size, dtype=dtype, device=cuda_device)

        # Clone residual since it will be modified in-place
        residual_copy = residual.clone()

        # Triton kernel output
        output_triton, residual_out = fused_add_rms_norm(x, residual_copy, weight)

        # PyTorch reference
        output_torch, residual_ref = torch_fused_add_rms_norm(x.float(), residual.float(), weight.float())
        output_torch = output_torch.to(dtype)
        residual_ref = residual_ref.to(dtype)

        assert output_triton.shape == x.shape
        assert residual_out.shape == residual.shape
        assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)
        assert torch.allclose(residual_out, residual_ref, atol=1e-2, rtol=1e-2)

    def test_fused_add_rms_norm_inplace(self, cuda_device):
        """Test that fused_add_rms_norm modifies residual in-place."""
        dtype = torch.bfloat16

        x = torch.randn(2, 10, 64, dtype=dtype, device=cuda_device)
        residual = torch.randn(2, 10, 64, dtype=dtype, device=cuda_device)
        weight = torch.randn(64, dtype=dtype, device=cuda_device)

        residual_ptr = residual.data_ptr()
        output, residual_out = fused_add_rms_norm(x, residual, weight)

        # Check residual is modified in-place
        assert residual_out.data_ptr() == residual_ptr
        assert torch.allclose(residual_out, residual)


# =============================================================================
# Comparison with torch.nn.RMSNorm
# =============================================================================


@pytest.mark.gpu
class TestRMSNormAgainstTorch:
    """Compare RMSNorm kernel against torch.nn.RMSNorm if available."""

    def test_against_nn_rmsnorm(self, cuda_device):
        """Test against torch.nn.RMSNorm (PyTorch 2.4+)."""
        if not hasattr(nn, "RMSNorm"):
            pytest.skip("torch.nn.RMSNorm not available (requires PyTorch 2.4+)")

        dtype = torch.bfloat16
        hidden_size = 64

        x = torch.randn(2, 10, hidden_size, dtype=dtype, device=cuda_device)

        # torch.nn.RMSNorm
        nn_norm = nn.RMSNorm(hidden_size, eps=1e-6).to(device=cuda_device, dtype=dtype)
        output_nn = nn_norm(x)

        # Our kernel
        output_kernel = triton_one_pass_rms_norm(x, nn_norm.weight)

        assert torch.allclose(output_kernel, output_nn, atol=1e-2, rtol=1e-2)


# =============================================================================
# Norm Infer Tests
# =============================================================================


def torch_norm_infer(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
    is_rms_norm: bool = False,
) -> torch.Tensor:
    """Reference norm implementation for inference (no mean/rstd storage)."""
    if is_rms_norm:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
    else:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_normed = (x - mean) / torch.sqrt(var + eps)

    if weight is not None:
        x_normed = x_normed * weight
    if bias is not None:
        x_normed = x_normed + bias
    return x_normed


@pytest.mark.gpu
class TestNormInfer:
    """Test inference-optimized norm kernel."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 128])
    @pytest.mark.parametrize("hidden_size", [64, 256, 1024])
    @pytest.mark.parametrize("is_rms_norm", [True, False])
    def test_norm_infer_correctness(self, cuda_device, batch_size, seq_len, hidden_size, is_rms_norm):
        """Test norm_infer kernel produces correct results."""
        dtype = torch.bfloat16
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        weight = torch.randn(hidden_size, dtype=dtype, device=cuda_device)
        bias = torch.randn(hidden_size, dtype=dtype, device=cuda_device)

        # Triton kernel output
        output_triton = norm_infer(x, weight, bias, eps=1e-6, is_rms_norm=is_rms_norm)

        # PyTorch reference
        output_torch = torch_norm_infer(x.float(), weight.float(), bias.float(), eps=1e-6, is_rms_norm=is_rms_norm).to(
            dtype
        )

        assert output_triton.shape == x.shape
        assert output_triton.dtype == x.dtype
        assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)

    def test_norm_infer_no_weight(self, cuda_device):
        """Test norm_infer without weight (elementwise_affine=False)."""
        dtype = torch.bfloat16
        torch.manual_seed(42)

        x = torch.randn(2, 16, 64, dtype=dtype, device=cuda_device)

        output = norm_infer(x, None, None, eps=1e-6, is_rms_norm=True)

        # RMS should be approximately 1
        rms = torch.sqrt(output.pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    def test_norm_infer_with_out_tensor(self, cuda_device):
        """Test norm_infer with pre-allocated output tensor."""
        dtype = torch.bfloat16
        torch.manual_seed(42)

        x = torch.randn(2, 16, 64, dtype=dtype, device=cuda_device)
        weight = torch.randn(64, dtype=dtype, device=cuda_device)
        out = torch.empty_like(x)

        result = norm_infer(x, weight, None, eps=1e-6, is_rms_norm=True, out=out)

        assert result.data_ptr() == out.data_ptr()
        assert result.shape == x.shape

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_norm_infer_uses_tensor_device_not_current_device(self):
        """Test norm_infer launches on the input tensor device when current CUDA device differs."""
        original_device = torch.cuda.current_device()
        try:
            tensor_device = torch.device("cuda:1")
            with torch.cuda.device(0):
                dtype = torch.bfloat16
                x = torch.randn(2, 16, 64, dtype=dtype, device=tensor_device)
                weight = torch.randn(64, dtype=dtype, device=tensor_device)

                output_triton = norm_infer(x, weight, None, eps=1e-6, is_rms_norm=True)
                output_torch = torch_norm_infer(
                    x.float(),
                    weight.float(),
                    None,
                    eps=1e-6,
                    is_rms_norm=True,
                ).to(dtype)

            assert output_triton.device == tensor_device
            assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)
        finally:
            torch.cuda.set_device(original_device)
