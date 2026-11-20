"""Triton kernels for FP8 attention quantization and output correction."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _sequence_mean_kernel(
    x,
    output,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_batch,
    stride_token,
    stride_head,
    stride_dim,
    block_tokens: tl.constexpr,
    block_dim: tl.constexpr,
):
    batch_head = tl.program_id(0)
    dim_block = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    dims = dim_block * block_dim + tl.arange(0, block_dim)
    valid_dims = dims < head_dim
    total = tl.zeros((block_dim,), dtype=tl.float32)
    for start in range(0, tokens, block_tokens):
        token_offsets = start + tl.arange(0, block_tokens)
        valid_tokens = token_offsets < tokens
        offsets = (
            batch * stride_batch
            + token_offsets[:, None] * stride_token
            + head * stride_head
            + dims[None, :] * stride_dim
        )
        values = tl.load(
            x + offsets,
            mask=valid_tokens[:, None] & valid_dims[None, :],
            other=0.0,
        ).to(tl.float32)
        total += tl.sum(values, axis=0)
    tl.store(output + batch_head * head_dim + dims, total / tokens, mask=valid_dims)


@triton.jit
def _sequence_mean_kv_kernel(
    k,
    v,
    k_output,
    v_output,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    k_stride_batch,
    k_stride_token,
    k_stride_head,
    k_stride_dim,
    v_stride_batch,
    v_stride_token,
    v_stride_head,
    v_stride_dim,
    block_tokens: tl.constexpr,
    block_dim: tl.constexpr,
):
    batch_head = tl.program_id(0)
    dim_block = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    dims = dim_block * block_dim + tl.arange(0, block_dim)
    valid_dims = dims < head_dim
    k_total = tl.zeros((block_dim,), dtype=tl.float32)
    v_total = tl.zeros((block_dim,), dtype=tl.float32)
    for start in range(0, tokens, block_tokens):
        token_offsets = start + tl.arange(0, block_tokens)
        valid_tokens = token_offsets < tokens
        k_offsets = (
            batch * k_stride_batch
            + token_offsets[:, None] * k_stride_token
            + head * k_stride_head
            + dims[None, :] * k_stride_dim
        )
        v_offsets = (
            batch * v_stride_batch
            + token_offsets[:, None] * v_stride_token
            + head * v_stride_head
            + dims[None, :] * v_stride_dim
        )
        mask = valid_tokens[:, None] & valid_dims[None, :]
        k_values = tl.load(k + k_offsets, mask=mask, other=0.0).to(tl.float32)
        v_values = tl.load(v + v_offsets, mask=mask, other=0.0).to(tl.float32)
        k_total += tl.sum(k_values, axis=0)
        v_total += tl.sum(v_values, axis=0)
    output_offsets = batch_head * head_dim + dims
    tl.store(k_output + output_offsets, k_total / tokens, mask=valid_dims)
    tl.store(v_output + output_offsets, v_total / tokens, mask=valid_dims)


@triton.jit
def _add_output_correction_kernel(
    output,
    correction,
    elements,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < elements
    batch = offsets // (tokens * heads * head_dim)
    head = (offsets // head_dim) % heads
    dim = offsets % head_dim
    correction_offsets = (batch * heads + head) * head_dim + dim
    values = tl.load(output + offsets, mask=valid).to(tl.float32)
    shift = tl.load(correction + correction_offsets, mask=valid)
    tl.store(output + offsets, values + shift, mask=valid)


@triton.jit
def _merge_prefix_output_correction_kernel(
    prefix,
    output,
    correction,
    merged,
    elements,
    prefix_tokens: tl.constexpr,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    valid = offsets < elements
    batch = offsets // (tokens * heads * head_dim)
    token = (offsets // (heads * head_dim)) % tokens
    head = (offsets // head_dim) % heads
    dim = offsets % head_dim
    prefix_offsets = ((batch * prefix_tokens + token) * heads + head) * head_dim + dim
    correction_offsets = (batch * heads + head) * head_dim + dim
    prefix_values = tl.load(prefix + prefix_offsets, mask=valid & (token < prefix_tokens), other=0.0)
    output_values = tl.load(output + offsets, mask=valid & (token >= prefix_tokens), other=0.0).to(tl.float32)
    shift = tl.load(correction + correction_offsets, mask=valid & (token >= prefix_tokens), other=0.0)
    values = tl.where(token < prefix_tokens, prefix_values, output_values + shift)
    tl.store(merged + offsets, values, mask=valid)


@triton.jit
def _quantize_qkv_fp8_stage1(
    q,
    k,
    v,
    k_mean,
    v_mean,
    q_out,
    k_out,
    q_scale,
    k_scale,
    v_scale,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
    smooth_k: tl.constexpr,
    smooth_v: tl.constexpr,
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
    mean_offsets = batch_head * head_dim + dim_offsets
    if smooth_k:
        k_values = tl.where(
            valid[:, None],
            k_values - tl.load(k_mean + mean_offsets)[None, :],
            0.0,
        )
    if smooth_v:
        v_values = tl.where(
            valid[:, None],
            v_values - tl.load(v_mean + mean_offsets)[None, :],
            0.0,
        )

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
    v_mean,
    v_out,
    v_scale,
    tokens: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block: tl.constexpr,
    smooth_v: tl.constexpr,
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
    if smooth_v:
        values = values - tl.load(v_mean + scale_offsets)[None, :]
    tl.store(v_out + output_offsets, values / scale[None, :], mask=valid[:, None])
