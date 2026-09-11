"""Block-scaled FP8 activation helpers for attention boundaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F

FP8_ATTENTION_BLOCK_SIZE = 64


def quantize_fp8_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SM90 block-scaled Q/K and layout-aware V-channel E4M3 quantization."""

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k, and v must share shape [B, T, H, D]")
    if not (q.is_cuda and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("fused FP8 QKV quantization requires contiguous CUDA tensors")
    if q.shape[-1] != 128:
        raise ValueError("fused FP8 QKV quantization requires head dimension 128")
    from telefuser.kernel.triton.fp8_attention import quantize_fp8_qkv_triton

    return quantize_fp8_qkv_triton(q, k, v, FP8_ATTENTION_BLOCK_SIZE)


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


def dequantize_fp8_per_token(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore token-scaled FP8 [B, T, H, D] activations."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn activations")
    if scale.shape[0] != x.shape[0] or scale.shape[1] < x.shape[1] or scale.shape[2] != x.shape[2]:
        raise ValueError("scale must have shape [B, padded_T, H] with padded_T >= T")
    return x.to(dtype) * scale[:, : x.shape[1]].to(dtype).unsqueeze(-1)


def quantize_fp8_per_channel(
    x: torch.Tensor,
    *,
    token_contiguous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BTHD with one E4M3 scale per head/channel.

    ``token_contiguous`` stores the same BTHD view over B,H,D,T-contiguous
    backing memory, matching the SM90 K-major PV WGMMA operand.
    """
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("FP8 attention quantization expects a floating-point [B, T, H, D] tensor")
    scale = x.detach().abs().amax(dim=1).float().clamp_min(1e-6) / 448.0
    quantized = (x / scale.to(x.dtype).unsqueeze(1)).to(torch.float8_e4m3fn)
    if token_contiguous:
        quantized = quantized.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    else:
        quantized = quantized.contiguous()
    return quantized, scale.contiguous()


def dequantize_fp8_per_channel(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore channel-scaled FP8 [B, T, H, D] activations."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn activations")
    if scale.shape != (x.shape[0], x.shape[2], x.shape[3]):
        raise ValueError("scale must have shape [B, H, D]")
    return x.to(dtype) * scale.to(dtype).unsqueeze(1)


__all__ = [
    "FP8_ATTENTION_BLOCK_SIZE",
    "dequantize_fp8_per_block",
    "dequantize_fp8_per_channel",
    "dequantize_fp8_per_token",
    "quantize_fp8_qkv",
    "quantize_fp8_per_block",
    "quantize_fp8_per_channel",
]
