"""Triton kernels for Rotary Position Embedding (RoPE).

Adapted from sglang diffusion kernels for TeleFuser.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HS_HALF": 32}, num_warps=2),
        triton.Config({"BLOCK_HS_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 128}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 256}, num_warps=8),
    ],
    key=["head_size", "interleaved", "is_inplace"],
)
@triton.jit
def _rotary_embedding_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,  # Only used when is_inplace=False
    num_heads,
    head_size,
    num_tokens,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    interleaved: tl.constexpr,
    is_inplace: tl.constexpr,
    BLOCK_HS_HALF: tl.constexpr,
):
    """Unified Rotary Position Embedding kernel (in-place and out-of-place).

    Args:
        x_ptr: Input tensor pointer (also output for in-place)
        cos_ptr: Cosine values pointer
        sin_ptr: Sine values pointer
        output_ptr: Output tensor pointer (only used when is_inplace=False)
        num_heads: Number of attention heads
        head_size: Size of each head
        num_tokens: Number of tokens
        stride_x_row: Row stride for input
        stride_cos_row: Row stride for cos
        stride_sin_row: Row stride for sin
        interleaved: Whether cos/sin are interleaved
        is_inplace: Whether to operate in-place
        BLOCK_HS_HALF: Block size for half of head dimension
    """
    row_idx = tl.program_id(0)
    token_idx = (row_idx // num_heads) % num_tokens

    x_row_ptr = x_ptr + row_idx * stride_x_row
    cos_row_ptr = cos_ptr + token_idx * stride_cos_row
    sin_row_ptr = sin_ptr + token_idx * stride_sin_row

    if is_inplace:
        store_ptr = x_row_ptr
    else:
        store_ptr = output_ptr + row_idx * stride_x_row

    head_size_half = head_size // 2

    for block_start in range(0, head_size_half, BLOCK_HS_HALF):
        offsets_half = block_start + tl.arange(0, BLOCK_HS_HALF)
        mask = offsets_half < head_size_half

        cos_vals = tl.load(cos_row_ptr + offsets_half, mask=mask, other=0.0)
        sin_vals = tl.load(sin_row_ptr + offsets_half, mask=mask, other=0.0)

        offsets_x1 = 2 * offsets_half
        offsets_x2 = 2 * offsets_half + 1

        x1_vals = tl.load(x_row_ptr + offsets_x1, mask=mask, other=0.0)
        x2_vals = tl.load(x_row_ptr + offsets_x2, mask=mask, other=0.0)

        x1_fp32 = x1_vals.to(tl.float32)
        x2_fp32 = x2_vals.to(tl.float32)
        cos_fp32 = cos_vals.to(tl.float32)
        sin_fp32 = sin_vals.to(tl.float32)
        o1_vals = tl.fma(-x2_fp32, sin_fp32, x1_fp32 * cos_fp32)
        o2_vals = tl.fma(x1_fp32, sin_fp32, x2_fp32 * cos_fp32)

        tl.store(store_ptr + offsets_x1, o1_vals.to(x1_vals.dtype), mask=mask)
        tl.store(store_ptr + offsets_x2, o2_vals.to(x2_vals.dtype), mask=mask)


def _apply_rotary_embedding_kernel(
    x_reshaped: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    head_size: int,
    num_tokens: int,
    bsz: int,
    interleaved: bool,
    is_inplace: bool,
) -> torch.Tensor | None:
    """Apply rotary embedding using unified Triton kernel."""
    grid = (bsz * num_tokens * num_heads,)

    if interleaved and cos.shape[-1] == head_size:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos.contiguous()
        sin = sin.contiguous()

    output = torch.empty_like(x_reshaped) if not is_inplace else x_reshaped

    _rotary_embedding_kernel[grid](
        x_reshaped,
        cos,
        sin,
        output,
        num_heads,
        head_size,
        num_tokens,
        x_reshaped.stride(0),
        cos.stride(0) if cos.dim() > 1 else 0,
        sin.stride(0) if sin.dim() > 1 else 0,
        interleaved,
        is_inplace,
    )

    return output if not is_inplace else None


def apply_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to input tensor.

    Args:
        x: Input tensor of shape [batch, seq_len, num_heads, head_size] or
           [seq_len, num_heads, head_size]
        cos: Cosine values of shape [seq_len, head_size] or [batch, seq_len, head_size]
        sin: Sine values of shape [seq_len, head_size] or [batch, seq_len, head_size]
        interleaved: If True, use interleaved format where cos/sin have full head_size

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x_reshaped = x.view(-1, head_size)

    output_reshaped = _apply_rotary_embedding_kernel(
        x_reshaped, cos, sin, num_heads, head_size, num_tokens, bsz, interleaved, is_inplace=False
    )

    return output_reshaped.view(x.shape)


def apply_rotary_embedding_inplace(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Apply Rotary Position Embedding (RoPE) in-place.

    Args:
        x: Input tensor to be modified in-place
        cos: Cosine values
        sin: Sine values
    """
    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x_reshaped = x.view(-1, head_size)

    _apply_rotary_embedding_kernel(
        x_reshaped, cos, sin, num_heads, head_size, num_tokens, bsz, interleaved=False, is_inplace=True
    )
