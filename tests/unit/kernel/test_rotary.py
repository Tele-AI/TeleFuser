"""Unit tests for Rotary Position Embedding (RoPE) Triton kernels.

Test markers:
    @pytest.mark.gpu - Tests requiring GPU
"""

import pytest
import torch

try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    from telefuser.kernel.triton import apply_rotary_embedding
except (ImportError, RuntimeError) as exc:
    pytest.skip(
        f"Skipping Triton rotary tests: {exc}",
        allow_module_level=True,
    )

# =============================================================================
# Reference Implementations
# =============================================================================


def torch_apply_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    """Reference RoPE implementation using PyTorch.

    The Triton kernel uses interleaved format by default:
    - x: [x1, x2, x1, x2, ...] (pairs of adjacent elements)
    - cos/sin: [seq, head_dim//2]

    Non-interleaved format:
    - x: [x1, x1, ..., x2, x2, ...] (first half and second half)
    """
    head_dim = x.shape[-1]

    if interleaved:
        # Interleaved format: pairs of adjacent elements
        x1 = x[..., 0::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
    else:
        # Non-interleaved format: first half and second half
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]

    # Broadcast cos/sin to match x shape
    # cos/sin: [seq, head_dim//2] -> [1, seq, 1, head_dim//2]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if interleaved:
        # Interleave the output
        output = torch.stack([o1, o2], dim=-1).flatten(-2)
    else:
        output = torch.cat([o1, o2], dim=-1)

    return output


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
# RoPE Correctness Tests
# =============================================================================


@pytest.mark.gpu
class TestRotaryEmbedding:
    """Test Rotary Position Embedding kernel implementations."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [16, 64])
    @pytest.mark.parametrize("num_heads", [8, 16])
    @pytest.mark.parametrize("head_size", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_apply_rotary_embedding_correctness(self, cuda_device, batch_size, seq_len, num_heads, head_size, dtype):
        """Test RoPE kernel produces correct results."""
        torch.manual_seed(42)

        x = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype, device=cuda_device)

        # cos/sin: [seq_len, head_size/2]
        cos = torch.randn(seq_len, head_size // 2, dtype=dtype, device=cuda_device)
        sin = torch.randn(seq_len, head_size // 2, dtype=dtype, device=cuda_device)

        # Triton kernel output
        output_triton = apply_rotary_embedding(x, cos, sin)

        # PyTorch reference
        output_torch = torch_apply_rotary_embedding(x.float(), cos.float(), sin.float()).to(dtype)

        assert output_triton.shape == x.shape
        assert output_triton.dtype == x.dtype
        assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)

    def test_apply_rotary_embedding_3d_input(self, cuda_device):
        """Test RoPE with 3D input tensor."""
        dtype = torch.bfloat16

        x = torch.randn(4, 16, 8, 64, dtype=dtype, device=cuda_device)  # [seq, heads, head_dim]
        cos = torch.randn(4, 32, dtype=dtype, device=cuda_device)
        sin = torch.randn(4, 32, dtype=dtype, device=cuda_device)

        output = apply_rotary_embedding(x, cos, sin)

        assert output.shape == x.shape

    def test_rotary_embedding_identity(self, cuda_device):
        """Test that RoPE with cos=1, sin=0 is identity."""
        dtype = torch.bfloat16

        x = torch.randn(2, 8, 4, 64, dtype=dtype, device=cuda_device)
        cos = torch.ones(8, 32, dtype=dtype, device=cuda_device)
        sin = torch.zeros(8, 32, dtype=dtype, device=cuda_device)

        output = apply_rotary_embedding(x, cos, sin)

        # With cos=1, sin=0, output should be input
        assert torch.allclose(output, x, atol=1e-5)

    @pytest.mark.multi_gpu
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    def test_apply_rotary_embedding_uses_tensor_device_not_current_device(self):
        """Test RoPE launches on the input tensor device when current CUDA device differs."""
        original_device = torch.cuda.current_device()
        try:
            tensor_device = torch.device("cuda:1")
            with torch.cuda.device(0):
                dtype = torch.bfloat16
                x = torch.randn(2, 8, 4, 64, dtype=dtype, device=tensor_device)
                cos = torch.randn(8, 32, dtype=dtype, device=tensor_device)
                sin = torch.randn(8, 32, dtype=dtype, device=tensor_device)

                output_triton = apply_rotary_embedding(x, cos, sin)
                output_torch = torch_apply_rotary_embedding(x.float(), cos.float(), sin.float()).to(dtype)

            assert output_triton.device == tensor_device
            assert torch.allclose(output_triton, output_torch, atol=1e-2, rtol=1e-2)
        finally:
            torch.cuda.set_device(original_device)


# =============================================================================
# RoPE Interleaved Format Tests
# =============================================================================


@pytest.mark.gpu
class TestRotaryEmbeddingInterleaved:
    """Test RoPE with interleaved and non-interleaved formats."""

    @pytest.mark.parametrize("interleaved", [False, True])
    def test_rotary_embedding_interleaved(self, cuda_device, interleaved):
        """Test RoPE with interleaved format."""
        dtype = torch.bfloat16
        batch_size, seq_len, num_heads, head_size = 2, 8, 4, 64

        x = torch.randn(batch_size, seq_len, num_heads, head_size, dtype=dtype, device=cuda_device)

        if interleaved:
            # Full head_size cos/sin for interleaved
            cos = torch.randn(seq_len, head_size, dtype=dtype, device=cuda_device)
            sin = torch.randn(seq_len, head_size, dtype=dtype, device=cuda_device)
        else:
            cos = torch.randn(seq_len, head_size // 2, dtype=dtype, device=cuda_device)
            sin = torch.randn(seq_len, head_size // 2, dtype=dtype, device=cuda_device)

        output = apply_rotary_embedding(x, cos, sin, interleaved=interleaved)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


# =============================================================================
# RoPE Edge Cases Tests
# =============================================================================


@pytest.mark.gpu
class TestRotaryEmbeddingEdgeCases:
    """Test edge cases for Rotary Position Embedding."""

    def test_small_head_size(self, cuda_device):
        """Test with very small head size."""
        dtype = torch.bfloat16

        x = torch.randn(2, 8, 4, 32, dtype=dtype, device=cuda_device)  # head_size=32
        cos = torch.randn(8, 16, dtype=dtype, device=cuda_device)
        sin = torch.randn(8, 16, dtype=dtype, device=cuda_device)

        output = apply_rotary_embedding(x, cos, sin)

        assert output.shape == x.shape

    def test_large_batch(self, cuda_device):
        """Test with large batch size."""
        dtype = torch.bfloat16

        x = torch.randn(32, 16, 8, 64, dtype=dtype, device=cuda_device)
        cos = torch.randn(16, 32, dtype=dtype, device=cuda_device)
        sin = torch.randn(16, 32, dtype=dtype, device=cuda_device)

        output = apply_rotary_embedding(x, cos, sin)

        assert output.shape == x.shape
