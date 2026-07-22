"""Mixture-of-Experts routing and grouped expert primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def route_topk(
    logits: torch.Tensor,
    *,
    num_expert_groups: int = 1,
    experts_per_group: int | None = None,
    top_k_groups: int | None = None,
    top_k: int = 1,
    correction_bias: torch.Tensor | None = None,
    routing_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts using LingBot's group-limited sigmoid router semantics.

    Correction bias participates only in discrete selection. Returned weights
    are gathered from the unbiased sigmoid scores and normalized over top-k.
    """
    if logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    tokens, experts = logits.shape
    if top_k < 1 or top_k > experts:
        raise ValueError("top_k must be within the number of experts")
    if experts_per_group is None:
        if experts % num_expert_groups:
            raise ValueError("experts must divide evenly across groups")
        experts_per_group = experts // num_expert_groups
    if num_expert_groups * experts_per_group != experts:
        raise ValueError("group dimensions do not match expert count")
    if top_k_groups is None:
        top_k_groups = num_expert_groups
    if not 1 <= top_k_groups <= num_expert_groups:
        raise ValueError("top_k_groups must be within the number of groups")
    scores = logits.sigmoid()
    selection_scores = scores if correction_bias is None else scores + correction_bias.reshape(1, -1)
    grouped = selection_scores.reshape(tokens, num_expert_groups, experts_per_group)
    group_values = grouped.topk(min(2, experts_per_group), dim=-1, sorted=False).values.sum(dim=-1)
    selected_groups = group_values.topk(top_k_groups, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_values, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups, True)
    score_mask = group_mask.unsqueeze(-1).expand(tokens, num_expert_groups, experts_per_group).reshape(tokens, experts)
    masked = selection_scores.masked_fill(~score_mask, float("-inf"))
    indices = masked.topk(top_k, dim=-1, sorted=False).indices
    weights = scores.gather(-1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    return indices, weights * routing_scale


def grouped_expert_forward(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    *,
    alignment: int = 8,
) -> torch.Tensor:
    """Run LingBot routed experts with PyTorch's native grouped GEMM.

    The token packing, alignment, BF16 GEMMs, and FP32 route reduction mirror
    the optimized upstream LingBot-Video implementation. Callers should retain
    an eager fallback because torch._grouped_mm is only available in recent
    CUDA-enabled PyTorch builds.
    """
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("grouped expert execution requires torch._grouped_mm")
    if tokens.device.type != "cuda":
        raise RuntimeError("grouped expert execution requires CUDA tensors")
    if indices.shape != weights.shape or indices.ndim != 2:
        raise ValueError("expert indices and weights must have matching [tokens, top_k] shapes")
    if tokens.ndim != 2:
        raise ValueError("expert tokens must have shape [tokens, hidden_size]")
    if alignment <= 0:
        raise ValueError("grouped expert alignment must be positive")

    num_tokens, hidden_size = tokens.shape
    top_k = indices.shape[1]
    num_experts = w1.shape[0]
    flat_weights = weights.reshape(-1)
    flat_indices = indices.reshape(-1)
    active_positions = torch.where(flat_weights != 0)[0]
    if active_positions.numel() == 0:
        return tokens.new_zeros(tokens.shape)

    active_experts = flat_indices[active_positions]
    counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.int64)
    counts.scatter_add_(0, active_experts, torch.ones_like(active_experts, dtype=torch.int64))
    sort_order = torch.argsort(active_experts, stable=True)
    sorted_positions = active_positions[sort_order]
    sorted_tokens = tokens[sorted_positions // top_k]

    padded_tokens, padded_indices, aligned_counts = _pad_grouped_tokens(
        sorted_tokens,
        counts,
        alignment=alignment,
    )
    offsets = torch.cumsum(aligned_counts, dim=0, dtype=torch.int32)
    grouped_mm = torch._grouped_mm
    gate = grouped_mm(
        padded_tokens.bfloat16(),
        w1.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    up = grouped_mm(
        padded_tokens.bfloat16(),
        w3.bfloat16().transpose(-2, -1),
        offs=offsets,
    )
    expert_output = grouped_mm(
        F.silu(gate) * up,
        w2.bfloat16().transpose(-2, -1),
        offs=offsets,
    ).to(tokens.dtype)
    expert_output = _unpad_grouped_tokens(expert_output, padded_indices, sorted_tokens.shape[0])

    routed = torch.zeros(
        num_tokens * top_k,
        hidden_size,
        dtype=expert_output.dtype,
        device=expert_output.device,
    )
    routed[sorted_positions] = expert_output
    return (
        (routed.reshape(num_tokens, top_k, hidden_size).float() * weights.float().unsqueeze(-1))
        .sum(dim=1)
        .to(tokens.dtype)
    )


def quantize_expert_weight_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [experts, out, in] weights with per-output-channel FP8 scales."""
    if weight.ndim != 3:
        raise ValueError("expert weight must have shape [experts, out_features, in_features]")
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    source = weight.float()
    scales = source.abs().amax(dim=2).clamp_min(torch.finfo(torch.float32).tiny) / fp8_info.max
    quantized = (source / scales.unsqueeze(2)).clamp(min=fp8_info.min, max=fp8_info.max).to(fp8_dtype)
    return quantized, scales


def fp8_expert_forward(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w3_scale: torch.Tensor,
) -> torch.Tensor:
    """Run sorted LingBot experts with native dynamic W8A8 scaled GEMMs."""
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("FP8 expert execution requires torch._scaled_mm")
    if tokens.device.type != "cuda":
        raise RuntimeError("FP8 expert execution requires CUDA tensors")
    if w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn or w3.dtype != torch.float8_e4m3fn:
        raise RuntimeError("FP8 expert weights must be quantized before execution")
    if indices.shape != weights.shape or indices.ndim != 2:
        raise ValueError("expert indices and weights must have matching [tokens, top_k] shapes")

    top_k = indices.shape[1]
    flat_weights = weights.reshape(-1)
    flat_indices = indices.reshape(-1)
    active_positions = torch.where(flat_weights != 0)[0]
    if active_positions.numel() == 0:
        return tokens.new_zeros(tokens.shape)

    active_experts = flat_indices[active_positions]
    sort_order = torch.argsort(active_experts, stable=True)
    sorted_positions = active_positions[sort_order]
    sorted_experts = active_experts[sort_order]
    sorted_tokens = tokens[sorted_positions // top_k]
    counts = torch.bincount(sorted_experts, minlength=w1.shape[0])

    outputs: list[torch.Tensor] = []
    offset = 0
    for expert_index, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        selected = sorted_tokens[offset : offset + count]
        offset += count
        selected_fp8, selected_scale = _quantize_rows_fp8(selected)
        gate = _scaled_mm_fp8(selected_fp8, w1[expert_index], selected_scale, w1_scale[expert_index])
        up = _scaled_mm_fp8(selected_fp8, w3[expert_index], selected_scale, w3_scale[expert_index])
        activation_fp8, activation_scale = _quantize_rows_fp8(F.silu(gate) * up)
        outputs.append(_scaled_mm_fp8(activation_fp8, w2[expert_index], activation_scale, w2_scale[expert_index]))

    expert_output = torch.cat(outputs, dim=0).to(tokens.dtype)
    routed = torch.zeros(
        tokens.shape[0] * top_k,
        tokens.shape[-1],
        dtype=tokens.dtype,
        device=tokens.device,
    )
    routed[sorted_positions] = expert_output
    return (
        (routed.reshape(tokens.shape[0], top_k, -1).float() * weights.float().unsqueeze(-1)).sum(dim=1).to(tokens.dtype)
    )


def _quantize_rows_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamically quantize matrix rows for torch._scaled_mm row-wise scaling."""
    fp8_dtype = torch.float8_e4m3fn
    fp8_info = torch.finfo(fp8_dtype)
    scale = tensor.float().abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    scale = (scale / fp8_info.max).contiguous()
    quantized = (tensor.float() / scale).clamp(min=fp8_info.min, max=fp8_info.max).to(fp8_dtype)
    return quantized, scale


def _scaled_mm_fp8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Apply one row-wise scaled FP8 matrix multiplication."""
    return torch._scaled_mm(
        activation,
        weight.transpose(0, 1),
        scale_a=activation_scale,
        scale_b=weight_scale.unsqueeze(0).contiguous(),
        out_dtype=torch.bfloat16,
    )


def _pad_grouped_tokens(
    tokens: torch.Tensor,
    counts: torch.Tensor,
    *,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad each expert group without synchronizing counts back to the CPU."""
    num_tokens = tokens.shape[0]
    num_experts = counts.shape[0]
    max_length = ((num_tokens + num_experts * alignment + alignment - 1) // alignment) * alignment
    counts_i64 = counts.to(torch.int64)
    aligned_counts_i64 = (torch.clamp_min(counts_i64, alignment) + alignment - 1) // alignment * alignment
    write_offsets = torch.cumsum(aligned_counts_i64, dim=0) - aligned_counts_i64
    end_offsets = torch.cumsum(aligned_counts_i64, dim=0)
    start_indices = torch.cumsum(counts_i64, dim=0) - counts_i64

    slots = torch.arange(max_length, dtype=torch.int64, device=tokens.device)
    expert_indices = torch.bucketize(slots, end_offsets, right=True)
    valid_expert = expert_indices < num_experts
    safe_expert_indices = expert_indices.clamp(max=num_experts - 1)
    local_indices = slots - write_offsets[safe_expert_indices]
    source_indices = start_indices[safe_expert_indices] + local_indices
    valid = valid_expert & (local_indices < counts_i64[safe_expert_indices])
    padded_indices = torch.where(valid, source_indices, torch.full_like(source_indices, num_tokens))

    tokens_with_sentinel = torch.vstack((tokens, tokens.new_zeros((1, tokens.shape[-1]))))
    return tokens_with_sentinel[padded_indices], padded_indices, aligned_counts_i64.to(torch.int32)


def _unpad_grouped_tokens(
    output: torch.Tensor,
    padded_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Restore expert-sorted rows from the aligned grouped-MM output."""
    unpadded = output.new_empty((num_tokens + 1, output.shape[-1]))
    unpadded[padded_indices] = output
    return unpadded[:num_tokens]
