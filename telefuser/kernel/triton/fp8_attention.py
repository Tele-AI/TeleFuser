"""Fused block-scaled FP8 Q/K/V quantization Triton kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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


def quantize_fp8_qkv_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the fused Q/K/V quantization kernels on validated CUDA inputs."""
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, block_size)
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
        block_size,
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
        block_size,
        num_warps=8,
        num_stages=1,
    )
    v_out = v_storage.permute(0, 3, 1, 2)
    return q_out, k_out, v_out, q_scale, k_scale, v_scale
