from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.models.lingbot_video_dit import (
    LingBotVideoAttention,
    LingBotVideoPatchEmbed,
    LingBotVideoTransformer3DModel,
    apply_lingbot_video_complex_rope,
)
from telefuser.pipelines.lingbot_video.data import LingBotVideoModelConfig


def test_patchify_matches_official_feature_order_and_round_trips() -> None:
    config = LingBotVideoModelConfig(in_channels=2, hidden_size=8)
    patch_embed = LingBotVideoPatchEmbed(config)
    latent = torch.arange(8).reshape(1, 2, 1, 2, 2)

    tokens = patch_embed.patchify(latent)

    assert tokens.tolist() == [[[0, 4, 1, 5, 2, 6, 3, 7]]]
    assert torch.equal(patch_embed.unpatchify(tokens, frames=1, height=2, width=2), latent)


def test_complex_rope_identity_table_preserves_values() -> None:
    states = torch.randn(1, 4, 2, 8)
    identity = torch.ones(4, 4, dtype=torch.complex64)

    assert torch.equal(apply_lingbot_video_complex_rope(states, identity), states)


def test_attention_dispatcher_sdpa_preserves_native_attention_result() -> None:
    torch.manual_seed(7)
    module = LingBotVideoAttention(hidden_size=8, num_heads=2, norm_eps=1e-6, qkv_bias=True, out_bias=True)
    hidden_states = torch.randn(1, 3, 8)
    rotary = torch.ones(3, 2, dtype=torch.complex64)
    module.set_attention_config(AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA))

    actual = module(hidden_states, rotary)
    query = apply_lingbot_video_complex_rope(module.norm_q(module.to_q(hidden_states).view(1, 3, 2, 4)), rotary)
    key = apply_lingbot_video_complex_rope(module.norm_k(module.to_k(hidden_states).view(1, 3, 2, 4)), rotary)
    value = module.to_v(hidden_states).view(1, 3, 2, 4)
    expected = module.to_out(
        F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
        .transpose(1, 2)
        .reshape(1, 3, 8)
    )

    assert torch.equal(actual, expected)


def test_ulysses_submits_all_qkv_collectives_before_waiting() -> None:
    module = LingBotVideoAttention(hidden_size=8, num_heads=2, norm_eps=1e-6, qkv_bias=True, out_bias=True)
    module.set_attention_config(AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA))
    module.set_ulysses_group(object())
    hidden_states = torch.randn(1, 3, 8)
    rotary = torch.ones(3, 2, dtype=torch.complex64)
    events: list[str] = []
    submit_index = 0

    def submit(tensor: torch.Tensor, group: object):
        nonlocal submit_index
        del group
        name = ("q", "k", "v")[submit_index]
        submit_index += 1
        events.append(f"submit-{name}")

        def wait() -> torch.Tensor:
            events.append(f"wait-{name}")
            return tensor

        return wait

    def gather(tensor: torch.Tensor, group: object, *, num_heads: int):
        del group, num_heads
        events.append("submit-output")

        def wait() -> torch.Tensor:
            events.append("wait-output")
            return tensor

        return wait

    with (
        patch("telefuser.models.lingbot_video_dit.dist.is_initialized", return_value=True),
        patch("telefuser.models.lingbot_video_dit.dist.get_world_size", return_value=2),
        patch("telefuser.models.lingbot_video_dit.ulysses_scatter_heads", side_effect=submit),
        patch("telefuser.models.lingbot_video_dit.ulysses_gather_heads", side_effect=gather),
    ):
        module(hidden_states, rotary)

    assert events == ["submit-q", "submit-k", "submit-v", "wait-q", "wait-k", "wait-v", "submit-output", "wait-output"]


def test_transformer_rejects_non_patch_aligned_latent_geometry() -> None:
    model = LingBotVideoTransformer3DModel(
        in_channels=16,
        hidden_size=16,
        num_attention_heads=2,
        depth=1,
        intermediate_size=32,
        text_dim=12,
        freq_dim=8,
        axes_dims=(2, 2, 4),
    )

    with pytest.raises(ValueError, match="divisible by the checkpoint patch size"):
        model(torch.randn(1, 16, 1, 3, 4), torch.tensor([1]), torch.randn(1, 3, 12))


def test_transformer_packed_batch_matches_independent_samples_with_different_text_lengths() -> None:
    torch.manual_seed(11)
    model = LingBotVideoTransformer3DModel(
        in_channels=2,
        out_channels=2,
        hidden_size=16,
        num_attention_heads=2,
        depth=1,
        intermediate_size=32,
        text_dim=12,
        freq_dim=8,
        axes_dims=(2, 2, 4),
    ).eval()
    latents = torch.randn(2, 2, 1, 2, 2)
    timesteps = torch.tensor([700.0, 300.0])
    text = torch.randn(2, 3, 12)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)

    actual = model(latents, timesteps, text, mask)
    expected = torch.cat(
        [
            model(latents[index : index + 1], timesteps[index : index + 1], text[index : index + 1, :length])
            for index, length in enumerate((3, 1))
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_transformer_rejects_packed_batch_with_non_sdpa_attention() -> None:
    model = LingBotVideoTransformer3DModel(
        in_channels=2,
        out_channels=2,
        hidden_size=16,
        num_attention_heads=2,
        depth=1,
        intermediate_size=32,
        text_dim=12,
        freq_dim=8,
        axes_dims=(2, 2, 4),
    ).eval()
    model.set_attention_config(AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2))

    with pytest.raises(ValueError, match="packed sequence attention currently requires TORCH_SDPA"):
        model(
            torch.randn(2, 2, 1, 2, 2),
            torch.tensor([700.0, 300.0]),
            torch.randn(2, 3, 12),
            torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool),
        )
