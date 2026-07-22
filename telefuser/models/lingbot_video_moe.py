"""Checkpoint-compatible LingBot-Video MoE transformer modules.

Numerical behavior is adapted from the Apache-2.0 licensed upstream
LingBot-Video transformer implementation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from telefuser.ops.moe import route_topk

from .lingbot_video_dit import LingBotVideoBlock, LingBotVideoMLP, LingBotVideoTransformer3DModel


class LingBotVideoRouter(nn.Module):
    """Checkpoint-compatible group-limited sigmoid MoE router."""

    def __init__(
        self, hidden_size: int, num_experts: int, top_k: int, n_group: int, topk_group: int, route_scale: float
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(num_experts), persistent=True)
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.route_scale = route_scale

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(tokens.float(), self.weight.float())
        indices, weights = route_topk(
            logits,
            num_expert_groups=self.n_group,
            top_k_groups=self.topk_group,
            top_k=self.top_k,
            correction_bias=self.e_score_correction_bias,
            routing_scale=self.route_scale,
        )
        return indices, weights.to(tokens.dtype)


class LingBotVideoGroupedExperts(nn.Module):
    """Official grouped expert layout with sorted eager execution."""

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        execution_backend: str = "sorted",
    ) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.set_execution_backend(execution_backend)

    def set_execution_backend(self, backend: str) -> None:
        """Select the source-style sorted path or the diagnostic where fallback."""
        if backend not in {"sorted", "where"}:
            raise ValueError("execution backend must be 'sorted' or 'where'")
        self.execution_backend = backend

    def forward(self, tokens: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.execution_backend == "where":
            return self._forward_where(tokens, indices, weights)
        return self._forward_sorted(tokens, indices, weights)

    def _forward_sorted(self, tokens: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        top_k = indices.shape[1]
        flat_weights = weights.reshape(-1)
        flat_indices = indices.reshape(-1)
        active_positions = torch.where(flat_weights != 0)[0]
        if active_positions.numel() == 0:
            return tokens.new_zeros(tokens.shape)
        active_experts = flat_indices[active_positions]
        counts = torch.zeros(self.w1.shape[0], device=tokens.device, dtype=torch.int64)
        counts.scatter_add_(0, active_experts, torch.ones_like(active_experts, dtype=torch.int64))
        sort_order = torch.argsort(active_experts, stable=True)
        sorted_positions = active_positions[sort_order]
        sorted_tokens = tokens[sorted_positions // top_k]

        outputs: list[torch.Tensor] = []
        for expert_index, selected in enumerate(torch.split(sorted_tokens, counts.tolist(), dim=0)):
            if selected.numel() == 0:
                continue
            activation = F.silu(F.linear(selected, self.w1[expert_index])) * F.linear(selected, self.w3[expert_index])
            outputs.append(F.linear(activation, self.w2[expert_index]))
        expert_output = torch.cat(outputs, dim=0)
        routed = torch.zeros(tokens.shape[0] * top_k, tokens.shape[-1], dtype=tokens.dtype, device=tokens.device)
        routed[sorted_positions] = expert_output
        return (
            (routed.reshape(tokens.shape[0], top_k, -1).float() * weights.float().unsqueeze(-1))
            .sum(dim=1)
            .to(tokens.dtype)
        )

    def _forward_where(self, tokens: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        top_k = indices.shape[1]
        output = torch.zeros(
            tokens.shape[0] * top_k,
            tokens.shape[-1],
            dtype=tokens.dtype,
            device=tokens.device,
        )
        for expert_index in range(self.w1.shape[0]):
            token_indices, topk_indices = torch.where(indices == expert_index)
            if token_indices.numel() == 0:
                continue
            selected = tokens[token_indices]
            activation = F.silu(F.linear(selected, self.w1[expert_index])) * F.linear(selected, self.w3[expert_index])
            output[token_indices * top_k + topk_indices] = F.linear(activation, self.w2[expert_index])
        return (
            (output.reshape(tokens.shape[0], top_k, -1).float() * weights.float().unsqueeze(-1))
            .sum(dim=1)
            .to(tokens.dtype)
        )


class LingBotVideoSparseMoeBlock(nn.Module):
    """Correctness-first MoE block with routed and optional shared experts."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        n_group: int,
        topk_group: int,
        route_scale: float,
        n_shared_experts: int = 1,
    ) -> None:
        super().__init__()
        self.router = LingBotVideoRouter(hidden_size, num_experts, top_k, n_group, topk_group, route_scale)
        self.experts = LingBotVideoGroupedExperts(num_experts, hidden_size, intermediate_size)
        self.shared_experts = (
            LingBotVideoMLP(hidden_size, intermediate_size * n_shared_experts) if n_shared_experts > 0 else None
        )

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, sequence, hidden_size = hidden_states.shape
        tokens = hidden_states.reshape(-1, hidden_size)
        indices, weights = self.router(tokens)
        if padding_mask is not None:
            weights = weights * padding_mask.reshape(-1, 1).to(weights.dtype)
        output = self.experts(tokens, indices, weights).reshape(batch, sequence, hidden_size)
        if self.shared_experts is not None:
            output = output + self.shared_experts(hidden_states)
        return output


class LingBotVideoMoeBlock(LingBotVideoBlock):
    """LingBot block selecting official sparse or dense FFN by layer index."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        *,
        norm_eps: float,
        qkv_bias: bool,
        out_bias: bool,
        layer_index: int,
        num_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        decoder_sparse_step: int,
        mlp_only_layers: tuple[int, ...],
        n_group: int,
        topk_group: int,
        route_scale: float,
        n_shared_experts: int,
    ) -> None:
        super().__init__(hidden_size, num_attention_heads, intermediate_size, norm_eps, qkv_bias, out_bias)
        if layer_index not in mlp_only_layers and num_experts > 0 and (layer_index + 1) % decoder_sparse_step == 0:
            self.ffn = LingBotVideoSparseMoeBlock(
                hidden_size,
                num_experts,
                top_k,
                moe_intermediate_size,
                n_group,
                topk_group,
                route_scale,
                n_shared_experts,
            )


class LingBotVideoMoeTransformer3DModel(LingBotVideoTransformer3DModel):
    """Official MoE/refiner transformer using eager expert execution for correctness."""

    def __init__(
        self,
        *,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        decoder_sparse_step: int = 1,
        mlp_only_layers: tuple[int, ...] = (),
        n_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        n_shared_experts: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        hidden_size = self.patch_embedder.out_features
        num_attention_heads = self.blocks[0].attn.num_heads
        intermediate_size = self.blocks[0].ffn.gate_proj.out_features
        norm_eps = self.blocks[0].norm1.variance_epsilon
        qkv_bias = self.blocks[0].attn.to_q.bias is not None
        out_bias = self.blocks[0].attn.to_out.bias is not None
        self.blocks = nn.ModuleList(
            [
                LingBotVideoMoeBlock(
                    hidden_size,
                    num_attention_heads,
                    intermediate_size,
                    norm_eps=norm_eps,
                    qkv_bias=qkv_bias,
                    out_bias=out_bias,
                    layer_index=index,
                    num_experts=num_experts,
                    top_k=num_experts_per_tok,
                    moe_intermediate_size=moe_intermediate_size,
                    decoder_sparse_step=decoder_sparse_step,
                    mlp_only_layers=tuple(mlp_only_layers),
                    n_group=n_group,
                    topk_group=topk_group,
                    route_scale=routed_scaling_factor,
                    n_shared_experts=n_shared_experts,
                )
                for index in range(len(self.blocks))
            ]
        )

    def set_expert_execution_backend(self, backend: str) -> None:
        """Set one eager expert backend consistently across all sparse blocks."""
        for module in self.modules():
            if isinstance(module, LingBotVideoGroupedExperts):
                module.set_execution_backend(backend)
