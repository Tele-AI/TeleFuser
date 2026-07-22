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

    sorted_weights = flat_weights[sorted_positions].to(expert_output.dtype)
    weighted_output = expert_output * sorted_weights.unsqueeze(-1)
    routed = torch.zeros(
        num_tokens * top_k,
        hidden_size,
        dtype=expert_output.dtype,
        device=expert_output.device,
    )
    routed[sorted_positions] = weighted_output
    return routed.reshape(num_tokens, top_k, hidden_size).sum(dim=1).to(tokens.dtype)


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
