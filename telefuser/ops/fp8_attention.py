"""Block-scaled FP8 activation helpers for attention boundaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F

FP8_ATTENTION_BLOCK_SIZE = 64


def quantize_fp8_per_block(
    x: torch.Tensor,
    block_size: int = FP8_ATTENTION_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [B, T, H, D] tensor with one E4M3 scale per block/head."""
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("FP8 attention quantization expects a floating-point [B, T, H, D] tensor")
    batch, tokens, heads, head_dim = x.shape
    blocks = (tokens + block_size - 1) // block_size
    padded_tokens = blocks * block_size
    padded = F.pad(x, (0, 0, 0, 0, 0, padded_tokens - tokens))
    blocked = padded.reshape(batch, blocks, block_size, heads, head_dim)
    scale = blocked.detach().abs().amax(dim=(2, 4)).float().clamp_min(1e-6) / 448.0
    quantized = (blocked / scale.to(x.dtype)[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.reshape(batch, padded_tokens, heads, head_dim)[:, :tokens].contiguous(), scale


def dequantize_fp8_per_block(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = FP8_ATTENTION_BLOCK_SIZE,
) -> torch.Tensor:
    """Restore block-scaled FP8 activations to ``dtype``."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn activations")
    token_scale = scale.repeat_interleave(block_size, dim=1)[:, : x.shape[1]]
    return x.to(dtype) * token_scale.to(dtype).unsqueeze(-1)


__all__ = ["FP8_ATTENTION_BLOCK_SIZE", "dequantize_fp8_per_block", "quantize_fp8_per_block"]
