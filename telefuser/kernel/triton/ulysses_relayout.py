# SPDX-License-Identifier: Apache-2.0
"""Destination-major Ulysses input and output relayout kernels."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_qkv_destination_major_kernel(
    output_ptr,
    query_ptr,
    key_ptr,
    value_ptr,
    total_elements,
    rows,
    local_heads,
    total_local_heads,
    local_head_start,
    head_dim,
    stride_query_row,
    stride_query_head,
    stride_key_row,
    stride_key_head,
    stride_value_row,
    stride_value_head,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    channel = offsets % head_dim
    head_slot = offsets // head_dim
    local_head = head_slot % local_heads
    row_slot = head_slot // local_heads
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * total_local_heads + local_head_start + local_head

    query = tl.load(
        query_ptr + row * stride_query_row + global_head * stride_query_head + channel,
        mask=mask,
    )
    key = tl.load(
        key_ptr + row * stride_key_row + global_head * stride_key_head + channel,
        mask=mask,
    )
    value = tl.load(
        value_ptr + row * stride_value_row + global_head * stride_value_head + channel,
        mask=mask,
    )
    output_base = head_slot * (3 * head_dim) + channel
    tl.store(output_ptr + output_base, query, mask=mask)
    tl.store(output_ptr + output_base + head_dim, key, mask=mask)
    tl.store(output_ptr + output_base + 2 * head_dim, value, mask=mask)


@triton.jit
def _round_bf16_to_fp32(value):
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _pack_qkv_qknorm_rope_destination_major_kernel(
    output_ptr,
    qkv_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cache_ptr,
    total_vectors,
    rows,
    packed_local_heads,
    total_local_heads,
    local_head_start,
    head_dim,
    rope_dim,
    eps,
    stride_qkv_row,
    stride_qkv_section,
    stride_qkv_head,
    stride_cache_row,
    BLOCK_N: tl.constexpr,
):
    vector_id = tl.program_id(0)
    valid_vector = vector_id < total_vectors
    local_head = vector_id % packed_local_heads
    row_slot = vector_id // packed_local_heads
    row = row_slot % rows
    destination = row_slot // rows
    global_head = destination * total_local_heads + local_head_start + local_head
    columns = tl.arange(0, BLOCK_N)
    mask = valid_vector & (columns < head_dim)
    q_base = qkv_ptr + row * stride_qkv_row + global_head * stride_qkv_head
    k_base = q_base + stride_qkv_section
    v_base = k_base + stride_qkv_section

    query = tl.load(q_base + columns, mask=mask, other=0.0).to(tl.float32)
    key = tl.load(k_base + columns, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(v_base + columns, mask=mask, other=0.0)
    q_weight = tl.load(q_weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + columns, mask=mask, other=0.0).to(tl.float32)
    q_inv_rms = tl.rsqrt(tl.sum(query * query, axis=0) / head_dim + eps)
    k_inv_rms = tl.rsqrt(tl.sum(key * key, axis=0) / head_dim + eps)
    query = _round_bf16_to_fp32(query * q_inv_rms * q_weight)
    key = _round_bf16_to_fp32(key * k_inv_rms * k_weight)

    rope_half = rope_dim // 2
    rotary_mask = mask & (columns < rope_dim)
    partner_columns = tl.where(columns < rope_half, columns + rope_half, columns - rope_half)
    q_partner = tl.load(q_base + partner_columns, mask=rotary_mask, other=0.0).to(tl.float32)
    k_partner = tl.load(k_base + partner_columns, mask=rotary_mask, other=0.0).to(tl.float32)
    q_partner_weight = tl.load(q_weight_ptr + partner_columns, mask=rotary_mask, other=0.0).to(tl.float32)
    k_partner_weight = tl.load(k_weight_ptr + partner_columns, mask=rotary_mask, other=0.0).to(tl.float32)
    q_partner = _round_bf16_to_fp32(q_partner * q_inv_rms * q_partner_weight)
    k_partner = _round_bf16_to_fp32(k_partner * k_inv_rms * k_partner_weight)
    frequency_column = columns % rope_half
    cosine = tl.load(cache_ptr + row * stride_cache_row + frequency_column, mask=rotary_mask, other=1.0).to(tl.float32)
    sine = tl.load(
        cache_ptr + row * stride_cache_row + rope_half + frequency_column,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    query = tl.where(columns < rope_half, query * cosine - q_partner * sine, query * cosine + q_partner * sine)
    key = tl.where(columns < rope_half, key * cosine - k_partner * sine, key * cosine + k_partner * sine)

    output_base = vector_id * (3 * head_dim)
    tl.store(output_ptr + output_base + columns, query, mask=mask)
    tl.store(output_ptr + output_base + head_dim + columns, key, mask=mask)
    tl.store(output_ptr + output_base + 2 * head_dim + columns, value, mask=mask)


@triton.jit
def _merge_ulysses_heads_kernel(
    output_ptr,
    input_ptr,
    total_vectors,
    batch,
    sequence,
    world_size: tl.constexpr,
    local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    vector_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    channels = tl.arange(0, VECTOR_SIZE)
    mask = vector_ids < total_vectors
    channel_vector = vector_ids % (head_dim // VECTOR_SIZE)
    rest = vector_ids // (head_dim // VECTOR_SIZE)
    local_head = rest % local_heads
    rest //= local_heads
    destination = rest % world_size
    rest //= world_size
    row = rest % sequence
    batch_index = rest // sequence
    source = (
        (((destination * sequence + row) * batch + batch_index) * local_heads + local_head) * head_dim
    ) + channel_vector * VECTOR_SIZE
    values = tl.load(input_ptr + source[:, None] + channels[None, :], mask=mask[:, None])
    tl.store(
        output_ptr + vector_ids[:, None] * VECTOR_SIZE + channels[None, :],
        values,
        mask=mask[:, None],
    )


@triton.jit
def _merge_ulysses_head_chunk_kernel(
    output_ptr,
    input_ptr,
    total_vectors,
    batch,
    sequence,
    output_sequence,
    total_local_heads,
    local_head_start,
    world_size: tl.constexpr,
    chunk_local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    ZERO_TAIL: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
):
    vector_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_HEAD_DIM)
    vector_mask = vector_ids < total_vectors
    channel_mask = channels < head_dim
    mask = vector_mask[:, None] & channel_mask[None, :]
    chunk_head = vector_ids % chunk_local_heads
    rest = vector_ids // chunk_local_heads
    batch_index = rest % batch
    rest //= batch
    if ZERO_TAIL:
        row = rest % output_sequence
        destination = rest // output_sequence
        input_vector = ((destination * sequence + row) * batch + batch_index) * chunk_local_heads + chunk_head
        load_mask = mask & (row[:, None] < sequence)
    else:
        row = rest % sequence
        destination = rest // sequence
        input_vector = vector_ids
        load_mask = mask
    output_head = destination * total_local_heads + local_head_start + chunk_head
    values = tl.load(
        input_ptr + input_vector[:, None] * head_dim + channels[None, :],
        mask=load_mask,
        other=0.0,
    )
    output_base = (batch_index * output_sequence + row) * (world_size * total_local_heads) + output_head
    tl.store(output_ptr + output_base[:, None] * head_dim + channels[None, :], values, mask=mask)


def pack_qkv_destination_major(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    world_size: int,
    *,
    local_head_start: int = 0,
    local_head_count: int | None = None,
) -> torch.Tensor:
    """Pack 3D Q/K/V directly as ``[destination, row, local_head, 3 * dim]``."""
    rows, global_heads, head_dim = query.shape
    total_local_heads = global_heads // world_size
    local_heads = total_local_heads - local_head_start if local_head_count is None else local_head_count
    if local_head_start < 0 or local_heads <= 0 or local_head_start + local_heads > total_local_heads:
        raise ValueError("Ulysses QKV head chunk falls outside the destination-local heads")
    output = torch.empty(
        world_size,
        rows,
        local_heads,
        3 * head_dim,
        dtype=query.dtype,
        device=query.device,
    )
    total_elements = rows * world_size * local_heads * head_dim
    if total_elements == 0:
        return output
    block_size = 1024
    _pack_qkv_destination_major_kernel[(triton.cdiv(total_elements, block_size),)](
        output,
        query,
        key,
        value,
        total_elements,
        rows,
        local_heads,
        total_local_heads,
        local_head_start,
        head_dim,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output


def pack_qkv_qknorm_rope_destination_major(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    world_size: int,
    eps: float,
    *,
    local_head_start: int = 0,
    local_head_count: int | None = None,
) -> torch.Tensor:
    """Normalize/rotate Q/K and pack Q/K/V directly for Ulysses all-to-all."""
    if qkv.ndim != 4 or qkv.shape[1] != 3:
        raise ValueError("fused Ulysses QKV pack requires shape [rows, 3, heads, dim]")
    rows, _, global_heads, head_dim = qkv.shape
    if world_size <= 0 or global_heads % world_size:
        raise ValueError("QKV heads must divide the Ulysses world size")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError("Q/K RMSNorm weights must match the QKV head dimension")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[0] != rows:
        raise ValueError("RoPE cache must have shape [rows, rotary_dim]")
    rope_dim = cos_sin_cache.shape[1]
    if rope_dim <= 0 or rope_dim % 2 or rope_dim > head_dim:
        raise ValueError("RoPE dimension must be positive, even, and no larger than head_dim")
    if qkv.dtype != torch.bfloat16 or q_weight.dtype != torch.bfloat16 or k_weight.dtype != torch.bfloat16:
        raise ValueError("fused Ulysses QKV pack requires BF16 QKV and RMSNorm weights")
    if not qkv.is_cuda or not q_weight.is_cuda or not k_weight.is_cuda or not cos_sin_cache.is_cuda:
        raise ValueError("fused Ulysses QKV pack requires CUDA tensors")
    total_local_heads = global_heads // world_size
    packed_local_heads = total_local_heads - local_head_start if local_head_count is None else local_head_count
    if local_head_start < 0 or packed_local_heads <= 0 or local_head_start + packed_local_heads > total_local_heads:
        raise ValueError("fused Ulysses QKV head chunk falls outside the destination-local heads")
    output = torch.empty(world_size, rows, packed_local_heads, 3 * head_dim, dtype=qkv.dtype, device=qkv.device)
    total_vectors = rows * world_size * packed_local_heads
    block_n = triton.next_power_of_2(head_dim)
    _pack_qkv_qknorm_rope_destination_major_kernel[(total_vectors,)](
        output,
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        total_vectors,
        rows,
        packed_local_heads,
        total_local_heads,
        local_head_start,
        head_dim,
        rope_dim,
        eps,
        qkv.stride(0),
        qkv.stride(1),
        qkv.stride(2),
        cos_sin_cache.stride(0),
        BLOCK_N=block_n,
        num_warps=1,
    )
    return output


def merge_ulysses_head_chunk(
    tensor: torch.Tensor,
    output: torch.Tensor,
    *,
    local_head_start: int,
    zero_tail: bool = False,
) -> torch.Tensor:
    """Merge one received head chunk into a shared full output tensor."""
    world_size, sequence, batch, chunk_local_heads, head_dim = tensor.shape
    if output.ndim != 5 or not output.is_contiguous():
        raise ValueError("Ulysses chunk destination must be a contiguous 5D tensor")
    total_local_heads = output.shape[3]
    if (
        output.shape[0] != batch
        or output.shape[1] < sequence
        or output.shape[2] != world_size
        or output.shape[-1] != head_dim
    ):
        raise ValueError("invalid Ulysses chunk destination shape")
    if local_head_start < 0 or local_head_start + chunk_local_heads > total_local_heads:
        raise ValueError("Ulysses gather head chunk falls outside the destination tensor")
    written_sequence = output.shape[1] if zero_tail else sequence
    total_vectors = world_size * written_sequence * batch * chunk_local_heads
    block_head_dim = triton.next_power_of_2(head_dim)
    block_rows = max(1, min(8, 1024 // block_head_dim))
    _merge_ulysses_head_chunk_kernel[(triton.cdiv(total_vectors, block_rows),)](
        output,
        tensor,
        total_vectors,
        batch,
        sequence,
        output.shape[1],
        total_local_heads,
        local_head_start,
        world_size=world_size,
        chunk_local_heads=chunk_local_heads,
        head_dim=head_dim,
        ZERO_TAIL=zero_tail,
        BLOCK_ROWS=block_rows,
        BLOCK_HEAD_DIM=block_head_dim,
        num_warps=8,
    )
    return output


def merge_ulysses_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Relayout ``[world, sequence, batch, local_head, dim]`` to destination-last order."""
    world_size, sequence, batch, local_heads, head_dim = tensor.shape
    output = torch.empty(
        batch,
        sequence,
        world_size,
        local_heads,
        head_dim,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if tensor.numel() == 0:
        return output
    vector_size = 8 if head_dim % 8 == 0 else 1
    total_vectors = tensor.numel() // vector_size
    block_size = 256
    _merge_ulysses_heads_kernel[(triton.cdiv(total_vectors, block_size),)](
        output,
        tensor,
        total_vectors,
        batch,
        sequence,
        world_size=world_size,
        local_heads=local_heads,
        head_dim=head_dim,
        VECTOR_SIZE=vector_size,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return output
