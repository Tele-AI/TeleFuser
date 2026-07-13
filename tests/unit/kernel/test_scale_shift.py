"""Unit tests for fused scale and shift Triton kernels.

Test markers:
    @pytest.mark.gpu - Tests requiring GPU
"""

import pytest
import torch

try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    from telefuser.kernel.triton import (
        fused_layernorm_scale_shift_gate_select01,
        fused_residual_layernorm_scale_shift_gate_select01,
        fused_scale_shift,
        fused_scale_shift_gate_select,
    )
except (ImportError, RuntimeError) as exc:
    pytest.skip(
        f"Skipping Triton scale/shift tests: {exc}",
        allow_module_level=True,
    )

# =============================================================================
# Reference Implementations
# =============================================================================


def torch_fused_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float = 1.0,
) -> torch.Tensor:
    """Reference scale and shift using PyTorch."""
    return x * (scale_constant + scale) + shift


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
# Fused Scale/Shift Tests
# =============================================================================


@pytest.mark.gpu
class TestFusedScaleShift:
    """Test fused scale and shift kernel."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("hidden_size", [64, 256, 512])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_scale_shift_2d_scale(self, cuda_device, batch_size, seq_len, hidden_size, dtype):
        """Test with 2D scale/shift [B, C]."""
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        scale = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)

        # Triton kernel output
        output_triton = fused_scale_shift(x, scale, shift)

        # PyTorch reference (broadcast scale/shift)
        scale_exp = scale[:, None, :].expand(batch_size, seq_len, hidden_size)
        shift_exp = shift[:, None, :].expand(batch_size, seq_len, hidden_size)
        output_torch = torch_fused_scale_shift(x.float(), scale_exp.float(), shift_exp.float()).to(dtype)

        assert output_triton.shape == x.shape
        assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [64, 128])
    @pytest.mark.parametrize("hidden_size", [256, 512])
    def test_fused_scale_shift_3d_scale(self, cuda_device, batch_size, seq_len, hidden_size):
        """Test with 3D scale/shift [B, L, C]."""
        dtype = torch.bfloat16
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        scale = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        shift = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)

        output_triton = fused_scale_shift(x, scale, shift)
        output_torch = torch_fused_scale_shift(x.float(), scale.float(), shift.float()).to(dtype)

        assert output_triton.shape == x.shape
        assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)

    def test_fused_scale_shift_scalar(self, cuda_device):
        """Test with scalar scale/shift."""
        dtype = torch.bfloat16

        x = torch.randn(2, 16, 64, dtype=dtype, device=cuda_device)
        scale = torch.tensor(0.5, dtype=dtype, device=cuda_device)
        shift = torch.tensor(0.1, dtype=dtype, device=cuda_device)

        output = fused_scale_shift(x, scale, shift)

        output_torch = x * (1.0 + scale) + shift
        assert torch.allclose(output, output_torch, atol=1e-2, rtol=1e-2)

    def test_fused_scale_shift_with_scale_constant(self, cuda_device):
        """Test with custom scale_constant."""
        dtype = torch.bfloat16

        x = torch.randn(2, 16, 64, dtype=dtype, device=cuda_device)
        scale = torch.randn(2, 64, dtype=dtype, device=cuda_device)
        shift = torch.randn(2, 64, dtype=dtype, device=cuda_device)
        scale_constant = 0.0  # No identity term

        output = fused_scale_shift(x, scale, shift, scale_constant=scale_constant)

        scale_exp = scale[:, None, :].expand(2, 16, 64)
        shift_exp = shift[:, None, :].expand(2, 16, 64)
        output_torch = x * (scale_constant + scale_exp) + shift_exp

        assert torch.allclose(output, output_torch, atol=1e-2, rtol=1e-2)


# =============================================================================
# Fused Scale/Shift Gate Select Tests
# =============================================================================


@pytest.mark.gpu
class TestFusedScaleShiftGateSelect:
    """Test fused scale, shift, and gate selection kernel."""

    def test_fused_scale_shift_gate_select_basic(self, cuda_device):
        """Test basic gate selection."""
        dtype = torch.bfloat16
        batch_size, seq_len, hidden_size = 2, 16, 64

        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)

        # Two sets of scale/shift/gate
        scale0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)

        # Index: all zeros (select set 0)
        index_zeros = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cuda_device)

        output, gate_out = fused_scale_shift_gate_select(x, scale0, shift0, gate0, scale1, shift1, gate1, index_zeros)

        assert output.shape == x.shape
        assert gate_out.shape == x.shape

    def test_fused_scale_shift_gate_select_select1(self, cuda_device):
        """Test gate selection with index=1."""
        dtype = torch.bfloat16
        batch_size, seq_len, hidden_size = 2, 16, 64

        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)

        scale0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)

        # Index: all ones (select set 1)
        index_ones = torch.ones(batch_size, seq_len, dtype=torch.bool, device=cuda_device)

        output, gate_out = fused_scale_shift_gate_select(x, scale0, shift0, gate0, scale1, shift1, gate1, index_ones)

        # Verify set 1 is selected
        scale1_exp = scale1[:, None, :].expand(batch_size, seq_len, hidden_size)
        shift1_exp = shift1[:, None, :].expand(batch_size, seq_len, hidden_size)
        expected = x * (1 + scale1_exp) + shift1_exp

        assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)
        gate1_exp = gate1[:, None, :].expand(batch_size, seq_len, hidden_size)
        assert torch.allclose(gate_out, gate1_exp, atol=1e-2, rtol=1e-2)

    def test_fused_scale_shift_gate_select_mixed(self, cuda_device):
        """Test gate selection with mixed indices."""
        dtype = torch.bfloat16
        batch_size, seq_len, hidden_size = 2, 16, 64

        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)

        scale0 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)  # Identity
        shift0 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.ones(batch_size, hidden_size, dtype=dtype, device=cuda_device)  # Scale by 2
        shift1 = torch.ones(batch_size, hidden_size, dtype=dtype, device=cuda_device)  # Shift by 1
        gate1 = torch.ones(batch_size, hidden_size, dtype=dtype, device=cuda_device)

        # Mixed index: first half 0, second half 1
        index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cuda_device)
        index[:, seq_len // 2 :] = True

        output, gate_out = fused_scale_shift_gate_select(x, scale0, shift0, gate0, scale1, shift1, gate1, index)

        # First half should be identity (scale0=0, shift0=0 => x * 1 + 0 = x)
        assert torch.allclose(output[:, : seq_len // 2], x[:, : seq_len // 2], atol=1e-5)

        # Second half should be scaled by 2 and shifted by 1
        expected_second = x[:, seq_len // 2 :] * 2 + 1
        assert torch.allclose(output[:, seq_len // 2 :], expected_second, atol=1e-2, rtol=1e-2)


# =============================================================================
# Edge Cases Tests
# =============================================================================


@pytest.mark.gpu
class TestFusedScaleShiftEdgeCases:
    """Test edge cases for fused scale and shift."""

    def test_small_tensors(self, cuda_device):
        """Test with small tensor sizes."""
        dtype = torch.bfloat16

        x = torch.randn(1, 4, 16, dtype=dtype, device=cuda_device)
        scale = torch.randn(1, 16, dtype=dtype, device=cuda_device)
        shift = torch.randn(1, 16, dtype=dtype, device=cuda_device)

        output = fused_scale_shift(x, scale, shift)
        assert output.shape == x.shape

    def test_large_hidden_size(self, cuda_device):
        """Test with large hidden size."""
        dtype = torch.bfloat16

        x = torch.randn(1, 16, 4096, dtype=dtype, device=cuda_device)
        scale = torch.randn(1, 4096, dtype=dtype, device=cuda_device)
        shift = torch.randn(1, 4096, dtype=dtype, device=cuda_device)

        output = fused_scale_shift(x, scale, shift)
        assert output.shape == x.shape

    def test_contiguity(self, cuda_device):
        """Test that output is contiguous."""
        dtype = torch.bfloat16

        x = torch.randn(2, 16, 64, dtype=dtype, device=cuda_device)
        scale = torch.randn(2, 64, dtype=dtype, device=cuda_device)
        shift = torch.randn(2, 64, dtype=dtype, device=cuda_device)

        output = fused_scale_shift(x, scale, shift)
        assert output.is_contiguous()


# =============================================================================
# Fused LayerNorm + Scale/Shift + Gate Select Tests
# =============================================================================


def torch_layernorm_scale_shift_gate_select(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference LayerNorm + scale/shift + gate selection using PyTorch."""
    B, L, C = x.shape

    # LayerNorm
    x_flat = x.reshape(B * L, C)
    mean = x_flat.mean(dim=-1, keepdim=True)
    var = x_flat.var(dim=-1, unbiased=False, keepdim=True)
    x_hat = (x_flat - mean) / torch.sqrt(var + eps)

    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias

    x_hat = x_hat.reshape(B, L, C)

    # Scale/shift with gate selection
    scale0_exp = scale0[:, None, :].expand(B, L, C)
    shift0_exp = shift0[:, None, :].expand(B, L, C)
    scale1_exp = scale1[:, None, :].expand(B, L, C)
    shift1_exp = shift1[:, None, :].expand(B, L, C)
    gate0_exp = gate0[:, None, :].expand(B, L, C)
    gate1_exp = gate1[:, None, :].expand(B, L, C)

    index_exp = index[:, :, None].expand(B, L, C)
    scale = torch.where(index_exp, scale1_exp, scale0_exp)
    shift = torch.where(index_exp, shift1_exp, shift0_exp)
    gate = torch.where(index_exp, gate1_exp, gate0_exp)

    output = x_hat * (1.0 + scale) + shift
    return output, gate


@pytest.mark.gpu
class TestFusedLayerNormScaleShiftGateSelect01:
    """Test fused LayerNorm + scale/shift + gate selection kernel."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [16, 32])
    @pytest.mark.parametrize("hidden_size", [64, 128])
    def test_basic_correctness(self, cuda_device, batch_size, seq_len, hidden_size):
        """Test basic correctness without weight/bias."""
        dtype = torch.bfloat16
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        scale0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)

        # Index: first half 0, second half 1
        index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cuda_device)
        index[:, seq_len // 2 :] = True

        output, gate_out = fused_layernorm_scale_shift_gate_select01(
            x, None, None, scale0, shift0, gate0, scale1, shift1, gate1, index, eps=1e-6
        )

        # PyTorch reference
        output_ref, gate_ref = torch_layernorm_scale_shift_gate_select(
            x.float(),
            None,
            None,
            scale0.float(),
            shift0.float(),
            gate0.float(),
            scale1.float(),
            shift1.float(),
            gate1.float(),
            index,
            eps=1e-6,
        )

        assert output.shape == x.shape
        assert gate_out.shape == x.shape
        assert torch.allclose(output, output_ref.to(dtype), atol=1e-2, rtol=1e-2)

    def test_with_weight_and_bias(self, cuda_device):
        """Test with LayerNorm weight and bias."""
        dtype = torch.bfloat16
        batch_size, seq_len, hidden_size = 2, 16, 64
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        weight = torch.randn(hidden_size, dtype=dtype, device=cuda_device)
        bias = torch.randn(hidden_size, dtype=dtype, device=cuda_device)
        scale0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cuda_device)

        output, gate_out = fused_layernorm_scale_shift_gate_select01(
            x, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index, eps=1e-6
        )

        output_ref, gate_ref = torch_layernorm_scale_shift_gate_select(
            x.float(),
            weight.float(),
            bias.float(),
            scale0.float(),
            shift0.float(),
            gate0.float(),
            scale1.float(),
            shift1.float(),
            gate1.float(),
            index,
            eps=1e-6,
        )

        assert torch.allclose(output, output_ref.to(dtype), atol=1e-2, rtol=1e-2)

    def test_select_set0(self, cuda_device):
        """Test that index=0 selects scale0/shift0/gate0."""
        dtype = torch.bfloat16
        batch_size, seq_len, hidden_size = 2, 16, 64
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        scale0 = torch.ones(batch_size, hidden_size, dtype=dtype, device=cuda_device)  # Scale by 2
        shift0 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.ones(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift1 = torch.ones(batch_size, hidden_size, dtype=dtype, device=cuda_device)  # Shift by 1
        gate1 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cuda_device)  # All 0

        output, gate_out = fused_layernorm_scale_shift_gate_select01(
            x, None, None, scale0, shift0, gate0, scale1, shift1, gate1, index, eps=1e-6
        )

        # Gate should be all ones (gate0)
        assert torch.allclose(gate_out, gate0[:, None, :].expand(batch_size, seq_len, hidden_size), atol=1e-3)


# =============================================================================
# Fused Residual + LayerNorm + Scale/Shift + Gate Select Tests
# =============================================================================


def torch_residual_layernorm_scale_shift_gate_select(
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_gate: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    gate0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
    gate1: torch.Tensor,
    index: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference residual + LayerNorm + scale/shift + gate selection using PyTorch."""
    B, L, C = x.shape

    # Gated residual add
    residual_out = residual + residual_gate * x

    # LayerNorm
    residual_flat = residual_out.reshape(B * L, C)
    mean = residual_flat.mean(dim=-1, keepdim=True)
    var = residual_flat.var(dim=-1, unbiased=False, keepdim=True)
    x_hat = (residual_flat - mean) / torch.sqrt(var + eps)

    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias

    x_hat = x_hat.reshape(B, L, C)

    # Scale/shift with gate selection
    scale0_exp = scale0[:, None, :].expand(B, L, C)
    shift0_exp = shift0[:, None, :].expand(B, L, C)
    scale1_exp = scale1[:, None, :].expand(B, L, C)
    shift1_exp = shift1[:, None, :].expand(B, L, C)
    gate0_exp = gate0[:, None, :].expand(B, L, C)
    gate1_exp = gate1[:, None, :].expand(B, L, C)

    index_exp = index[:, :, None].expand(B, L, C)
    scale = torch.where(index_exp, scale1_exp, scale0_exp)
    shift = torch.where(index_exp, shift1_exp, shift0_exp)
    gate = torch.where(index_exp, gate1_exp, gate0_exp)

    output = x_hat * (1.0 + scale) + shift
    return output, residual_out, gate


@pytest.mark.gpu
class TestFusedResidualLayerNormScaleShiftGateSelect01:
    """Test fused residual + LayerNorm + scale/shift + gate selection kernel."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [16, 32])
    @pytest.mark.parametrize("hidden_size", [64, 128])
    def test_basic_correctness(self, cuda_device, batch_size, seq_len, hidden_size):
        """Test basic correctness."""
        dtype = torch.bfloat16
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        residual = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        residual_gate = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        scale0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cuda_device)
        index[:, seq_len // 2 :] = True

        output, residual_out, gate_out = fused_residual_layernorm_scale_shift_gate_select01(
            x, residual, residual_gate, None, None, scale0, shift0, gate0, scale1, shift1, gate1, index, eps=1e-6
        )

        output_ref, residual_ref, gate_ref = torch_residual_layernorm_scale_shift_gate_select(
            x.float(),
            residual.float(),
            residual_gate.float(),
            None,
            None,
            scale0.float(),
            shift0.float(),
            gate0.float(),
            scale1.float(),
            shift1.float(),
            gate1.float(),
            index,
            eps=1e-6,
        )

        assert output.shape == x.shape
        assert residual_out.shape == x.shape
        assert gate_out.shape == x.shape
        assert torch.allclose(output, output_ref.to(dtype), atol=1e-2, rtol=1e-2)
        assert torch.allclose(residual_out, residual_ref.to(dtype), atol=1e-2, rtol=1e-2)

    def test_with_weight_and_bias(self, cuda_device):
        """Test with LayerNorm weight and bias."""
        dtype = torch.bfloat16
        batch_size, seq_len, hidden_size = 2, 16, 64
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        residual = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        residual_gate = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        weight = torch.randn(hidden_size, dtype=dtype, device=cuda_device)
        bias = torch.randn(hidden_size, dtype=dtype, device=cuda_device)
        scale0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate1 = torch.randn(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cuda_device)

        output, residual_out, gate_out = fused_residual_layernorm_scale_shift_gate_select01(
            x, residual, residual_gate, weight, bias, scale0, shift0, gate0, scale1, shift1, gate1, index, eps=1e-6
        )

        output_ref, residual_ref, gate_ref = torch_residual_layernorm_scale_shift_gate_select(
            x.float(),
            residual.float(),
            residual_gate.float(),
            weight.float(),
            bias.float(),
            scale0.float(),
            shift0.float(),
            gate0.float(),
            scale1.float(),
            shift1.float(),
            gate1.float(),
            index,
            eps=1e-6,
        )

        assert torch.allclose(output, output_ref.to(dtype), atol=1e-2, rtol=1e-2)
        assert torch.allclose(residual_out, residual_ref.to(dtype), atol=1e-2, rtol=1e-2)

    def test_gated_residual(self, cuda_device):
        """Test that gated residual is computed correctly."""
        dtype = torch.bfloat16
        batch_size, seq_len, hidden_size = 2, 16, 64

        x = torch.ones(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        residual = torch.zeros(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        residual_gate = torch.ones(batch_size, seq_len, hidden_size, dtype=dtype, device=cuda_device)
        scale0 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift0 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate0 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        scale1 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        shift1 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        gate1 = torch.zeros(batch_size, hidden_size, dtype=dtype, device=cuda_device)
        index = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=cuda_device)

        output, residual_out, gate_out = fused_residual_layernorm_scale_shift_gate_select01(
            x, residual, residual_gate, None, None, scale0, shift0, gate0, scale1, shift1, gate1, index, eps=1e-6
        )

        # residual_out = residual + residual_gate * x = 0 + 1 * 1 = 1
        assert torch.allclose(residual_out, torch.ones_like(residual_out), atol=1e-3)
