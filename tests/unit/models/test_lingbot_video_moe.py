from __future__ import annotations

import torch
import torch.nn.functional as F

from telefuser.models.lingbot_video_moe import LingBotVideoGroupedExperts, LingBotVideoMoeTransformer3DModel


def test_moe_transformer_uses_official_router_and_expert_key_layout() -> None:
    model = LingBotVideoMoeTransformer3DModel(
        hidden_size=16,
        num_attention_heads=2,
        depth=1,
        intermediate_size=32,
        text_dim=12,
        freq_dim=8,
        axes_dims=(2, 2, 4),
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
        n_shared_experts=1,
    )

    output = model(torch.randn(1, 16, 1, 4, 4), torch.tensor([1]), torch.randn(1, 3, 12))

    assert output.shape == (1, 16, 1, 4, 4)
    assert "blocks.0.ffn.router.weight" in model.state_dict()
    assert "blocks.0.ffn.experts.w1" in model.state_dict()
    assert "blocks.0.ffn.shared_experts.up_proj.weight" in model.state_dict()


def test_grouped_experts_restores_route_outputs_with_fp32_weighted_sum() -> None:
    torch.manual_seed(7)
    experts = LingBotVideoGroupedExperts(num_experts=2, hidden_size=4, intermediate_size=6).bfloat16()
    with torch.no_grad():
        experts.w1.uniform_(-0.1, 0.1)
        experts.w2.uniform_(-0.1, 0.1)
        experts.w3.uniform_(-0.1, 0.1)
    tokens = torch.randn(3, 4, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1], [1, 0], [0, 1]])
    weights = torch.tensor([[0.625, 0.375], [0.25, 0.75], [0.5, 0.5]], dtype=torch.bfloat16)

    actual = experts(tokens, indices, weights)
    routed = torch.zeros(tokens.shape[0] * indices.shape[1], tokens.shape[1], dtype=tokens.dtype)
    for expert_index in range(experts.w1.shape[0]):
        token_indices, topk_indices = torch.where(indices == expert_index)
        selected = tokens[token_indices]
        activation = F.silu(F.linear(selected, experts.w1[expert_index])) * F.linear(selected, experts.w3[expert_index])
        routed[token_indices * indices.shape[1] + topk_indices] = F.linear(activation, experts.w2[expert_index])
    expected = (routed.reshape(tokens.shape[0], indices.shape[1], -1).float() * weights.float().unsqueeze(-1)).sum(1)

    assert torch.equal(actual, expected.to(torch.bfloat16))


def test_grouped_experts_skips_zero_weight_routes_without_changing_output() -> None:
    torch.manual_seed(11)
    experts = LingBotVideoGroupedExperts(num_experts=3, hidden_size=4, intermediate_size=6).bfloat16()
    with torch.no_grad():
        experts.w1.uniform_(-0.1, 0.1)
        experts.w2.uniform_(-0.1, 0.1)
        experts.w3.uniform_(-0.1, 0.1)
    tokens = torch.randn(3, 4, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1], [2, 0], [1, 2]])
    weights = torch.tensor([[0.625, 0.375], [0.0, 0.0], [0.5, 0.5]], dtype=torch.bfloat16)

    actual = experts(tokens, indices, weights)
    routed = torch.zeros(tokens.shape[0] * indices.shape[1], tokens.shape[1], dtype=tokens.dtype)
    for expert_index in range(experts.w1.shape[0]):
        token_indices, topk_indices = torch.where(indices == expert_index)
        selected = tokens[token_indices]
        activation = F.silu(F.linear(selected, experts.w1[expert_index])) * F.linear(selected, experts.w3[expert_index])
        routed[token_indices * indices.shape[1] + topk_indices] = F.linear(activation, experts.w2[expert_index])
    expected = (routed.reshape(tokens.shape[0], indices.shape[1], -1).float() * weights.float().unsqueeze(-1)).sum(1)

    assert torch.equal(actual, expected.to(torch.bfloat16))


def test_grouped_expert_backends_are_exactly_equivalent() -> None:
    torch.manual_seed(13)
    experts = LingBotVideoGroupedExperts(num_experts=4, hidden_size=4, intermediate_size=6).bfloat16()
    with torch.no_grad():
        experts.w1.uniform_(-0.1, 0.1)
        experts.w2.uniform_(-0.1, 0.1)
        experts.w3.uniform_(-0.1, 0.1)
    tokens = torch.randn(5, 4, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 1], [2, 3], [3, 0], [1, 2], [0, 3]])
    weights = torch.tensor([[0.625, 0.375], [0.5, 0.5], [0.25, 0.75], [0.75, 0.25], [0.5, 0.5]], dtype=torch.bfloat16)

    sorted_output = experts(tokens, indices, weights)
    experts.set_execution_backend("where")
    where_output = experts(tokens, indices, weights)

    assert torch.equal(sorted_output, where_output)
