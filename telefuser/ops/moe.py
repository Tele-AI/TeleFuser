"""Correctness-first Mixture-of-Experts routing primitives."""

from __future__ import annotations

import torch


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
    group_values = grouped.topk(min(2, experts_per_group), dim=-1).values.sum(dim=-1)
    selected_groups = group_values.topk(top_k_groups, dim=-1).indices
    group_mask = torch.zeros_like(grouped, dtype=torch.bool)
    group_mask.scatter_(1, selected_groups.unsqueeze(-1).expand(-1, -1, experts_per_group), True)
    masked = selection_scores.masked_fill(~group_mask.reshape(tokens, experts), float("-inf"))
    indices = masked.topk(top_k, dim=-1).indices
    weights = scores.gather(-1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
    return indices, weights * routing_scale
