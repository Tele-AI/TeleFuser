# SPDX-License-Identifier: Apache-2.0
"""Fused BF16 RMSNorm and indexed AdaLN scale/shift for MiniMax-H3."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _round_bf16_to_fp32(value):
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _indexed_rmsnorm_scale_shift_bf16_kernel(
    output_ptr,
    x_ptr,
    weight_ptr,
    shift_ptr,
    scale_ptr,
    indices_ptr,
    hidden_size,
    parameter_rows,
    eps,
    stride_x_row,
    stride_output_row,
    stride_shift_row,
    stride_scale_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / hidden_size
    inv_rms = 1.0 / tl.sqrt(variance + eps)
    weight = tl.load(weight_ptr + columns, mask=mask, other=1.0).to(tl.float32)

    # Match the standalone RMSNorm BF16 output store/load boundary exactly.
    normalized = _round_bf16_to_fp32(x * inv_rms * weight)
    index = tl.load(indices_ptr + row * stride_indices)
    parameter_mask = mask & (index >= 0) & (index < parameter_rows)
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns,
        mask=parameter_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(
        scale_ptr + index * stride_scale_row + columns,
        mask=parameter_mask,
        other=0.0,
    ).to(tl.float32)
    one_plus_scale = _round_bf16_to_fp32(1.0 + scale)
    scaled = _round_bf16_to_fp32(normalized * one_plus_scale)
    tl.store(output_ptr + row * stride_output_row + columns, scaled + shift, mask=mask)


def indexed_rmsnorm_scale_shift_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Return the bit-compatible fusion of RMSNorm and indexed AdaLN modulation."""
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("fused indexed RMSNorm requires a contiguous 2D input")
    if x.dtype != torch.bfloat16 or weight.dtype != x.dtype or shift.dtype != x.dtype or scale.dtype != x.dtype:
        raise TypeError("fused indexed RMSNorm requires matching BF16 tensors")
    if weight.shape != (x.shape[1],):
        raise ValueError("RMSNorm weight must match the hidden size")
    if shift.ndim != 2 or scale.shape != shift.shape or shift.shape[1] != x.shape[1]:
        raise ValueError("indexed shift/scale must match the input hidden size")
    if indices.ndim != 1 or indices.numel() != x.shape[0]:
        raise ValueError("indices must contain one entry per input row")
    if any(tensor.device != x.device for tensor in (weight, shift, scale, indices)):
        raise ValueError("all fused indexed RMSNorm tensors must share a device")
    output = torch.empty_like(x)
    hidden_size = x.shape[1]
    block_n = triton.next_power_of_2(hidden_size)
    _indexed_rmsnorm_scale_shift_bf16_kernel[(x.shape[0],)](
        output,
        x,
        weight,
        shift,
        scale,
        indices,
        hidden_size,
        shift.shape[0],
        eps,
        x.stride(0),
        output.stride(0),
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        BLOCK_N=block_n,
        num_warps=min(max(block_n // 256, 1), 8),
    )
    return output
