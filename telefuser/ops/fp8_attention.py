"""Block-scaled FP8 activation helpers for attention boundaries."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

FP8_ATTENTION_BLOCK_SIZE = 64


@triton.jit
def _quantize_qkv_fp8_stage1(
    q,
    k,
    v,
    q_out,
    k_out,
    q_scale,
    k_scale,
    v_scale,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    token_offsets = block_idx * block + tl.arange(0, block)
    dim_offsets = tl.arange(0, head_dim)
    valid = token_offsets < tokens
    offsets = ((batch * tokens + token_offsets[:, None]) * heads + head) * head_dim + dim_offsets[None, :]
    q_values = tl.load(q + offsets, mask=valid[:, None], other=0.0).to(tl.float32)
    k_values = tl.load(k + offsets, mask=valid[:, None], other=0.0).to(tl.float32)
    v_values = tl.load(v + offsets, mask=valid[:, None], other=0.0).to(tl.float32)

    q_s = tl.maximum(tl.max(tl.max(tl.abs(q_values), axis=1), axis=0), 1.0e-6) / 448.0
    k_s = tl.maximum(tl.max(tl.max(tl.abs(k_values), axis=1), axis=0), 1.0e-6) / 448.0
    scale_offset = (batch * tl.cdiv(tokens, block) + block_idx) * heads + head
    tl.store(q_scale + scale_offset, q_s)
    tl.store(k_scale + scale_offset, k_s)
    tl.store(q_out + offsets, q_values / q_s, mask=valid[:, None])
    tl.store(k_out + offsets, k_values / k_s, mask=valid[:, None])

    v_s = tl.max(tl.abs(v_values), axis=0) / 448.0
    v_scale_offsets = (batch * heads + head) * head_dim + dim_offsets
    tl.atomic_max(v_scale + v_scale_offsets, v_s)


@triton.jit
def _quantize_qkv_fp8_stage2_v(
    v,
    v_out,
    v_scale,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    token_offsets = block_idx * block + tl.arange(0, block)
    dim_offsets = tl.arange(0, head_dim)
    valid = token_offsets < tokens
    input_offsets = ((batch * tokens + token_offsets[:, None]) * heads + head) * head_dim + dim_offsets[None, :]
    output_offsets = ((batch * heads + head) * head_dim + dim_offsets[None, :]) * tokens + token_offsets[:, None]
    scale_offsets = (batch * heads + head) * head_dim + dim_offsets
    scale = tl.maximum(tl.load(v_scale + scale_offsets), 1.0e-6 / 448.0)
    values = tl.load(v + input_offsets, mask=valid[:, None], other=0.0).to(tl.float32)
    tl.store(v_out + output_offsets, values / scale[None, :], mask=valid[:, None])


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
    batch, tokens, heads, head_dim = q.shape
    if head_dim != 128:
        raise ValueError("fused FP8 QKV quantization requires head dimension 128")
    blocks = triton.cdiv(tokens, FP8_ATTENTION_BLOCK_SIZE)
    q_out = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
    k_out = torch.empty_like(q_out)
    v_storage = torch.empty((batch, heads, head_dim, tokens), device=q.device, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((batch, blocks, heads), device=q.device, dtype=torch.float32)
    k_scale = torch.ones_like(q_scale)
    v_scale = torch.zeros((batch, heads, head_dim), device=q.device, dtype=torch.float32)
    grid = (blocks, batch * heads)
    _quantize_qkv_fp8_stage1[grid](
        q,
        k,
        v,
        q_out,
        k_out,
        q_scale,
        k_scale,
        v_scale,
        tokens,
        heads,
        head_dim,
        FP8_ATTENTION_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    _quantize_qkv_fp8_stage2_v[grid](
        v,
        v_storage,
        v_scale,
        tokens,
        heads,
        head_dim,
        FP8_ATTENTION_BLOCK_SIZE,
        num_warps=8,
        num_stages=1,
    )
    v_out = v_storage.permute(0, 3, 1, 2)
    return q_out, k_out, v_out, q_scale, k_scale, v_scale


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
