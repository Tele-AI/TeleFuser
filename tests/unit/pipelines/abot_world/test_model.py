from __future__ import annotations

import pytest
import torch

from telefuser.models.abot_world_dit import (
    ABotWorldDiT,
    CausalWanSelfAttention,
    _rope_apply,
)
from telefuser.models.wan_video_dit import precompute_freqs_cis_3d


def _tiny_dit() -> ABotWorldDiT:
    return ABotWorldDiT(
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        downscale_factor_control_adapter=2,
    )


def test_official_state_dict_converter_is_native() -> None:
    state_dict = {"blocks.0.self_attn.q.weight": torch.zeros(4, 4)}

    converted, metadata = ABotWorldDiT.state_dict_converter().from_official(state_dict)

    assert converted is state_dict
    assert metadata == {}


def test_causal_window_updates_every_transformer_block() -> None:
    model = _tiny_dit()

    model.set_causal_attention_window(local_attn_size=18, sink_size=6)

    assert model.local_attn_size == 18
    assert model.sink_size == 6
    assert all(block.self_attn.local_attn_size == 18 for block in model.blocks)
    assert all(block.self_attn.sink_size == 6 for block in model.blocks)


def test_causal_window_rejects_invalid_sink_configuration() -> None:
    model = _tiny_dit()

    with pytest.raises(ValueError, match="sink_size"):
        model.set_causal_attention_window(local_attn_size=6, sink_size=6)


def test_sink_cache_retains_prefix_and_rolls_tail() -> None:
    attention = CausalWanSelfAttention(dim=4, num_heads=1, local_attn_size=3, sink_size=1)
    cache = {
        "k": torch.zeros(1, 3, 1, 4),
        "v": torch.zeros(1, 3, 1, 4),
        "global_end_index": torch.zeros(1, dtype=torch.long),
        "local_end_index": torch.zeros(1, dtype=torch.long),
    }

    for frame in range(4):
        value = torch.full((1, 1, 1, 4), float(frame))
        attention._update_cache(cache, value, value, current_start=frame, frame_tokens=1)

    # The first frame is the sink; the rolling tail contains the newest frames.
    assert torch.equal(cache["k"][0, :, 0, 0], torch.tensor([0.0, 2.0, 3.0]))
    assert int(cache["global_end_index"].item()) == 4
    assert int(cache["local_end_index"].item()) == 3


def test_rope_applies_frame_indices_at_the_supported_boundary() -> None:
    freqs = torch.cat(precompute_freqs_cis_3d(8), dim=1)
    values = torch.randn(1, 4, 2, 8)

    output = _rope_apply(values, (2, 1, 2), freqs, torch.tensor([0, 1023]))

    assert output.shape == values.shape
    assert torch.isfinite(output).all()


def test_rope_rejects_positions_outside_the_precomputed_table() -> None:
    freqs = torch.cat(precompute_freqs_cis_3d(8), dim=1)
    values = torch.randn(1, 2, 1, 8)

    with pytest.raises(ValueError, match="frame indices"):
        _rope_apply(values, (2, 1, 1), freqs, torch.tensor([0, 1024]))


def test_sink_attention_uses_bounded_positions_for_long_sessions() -> None:
    attention = CausalWanSelfAttention(dim=8, num_heads=1, local_attn_size=3, sink_size=1)
    freqs = torch.cat(precompute_freqs_cis_3d(8), dim=1)
    cache = {
        "k": torch.zeros(1, 3, 1, 8),
        "v": torch.zeros(1, 3, 1, 8),
        "global_end_index": torch.tensor([2048]),
        "local_end_index": torch.tensor([3]),
    }

    output = attention(
        torch.randn(1, 1, 8),
        (1, 1, 1),
        freqs,
        cache,
        current_start=2048,
    )

    assert output.shape == (1, 1, 8)
    assert torch.isfinite(output).all()


def test_small_dit_forward_preserves_latent_and_cache_contract() -> None:
    """Exercise the complete DiT path with the same cache shape as the stage."""
    model = _tiny_dit().eval()
    model.set_causal_attention_window(local_attn_size=3, sink_size=1)
    batch, frames, latent_height, latent_width = 1, 1, 8, 8
    frame_tokens = (latent_height // 2) * (latent_width // 2)
    self_cache = []
    cross_cache = []
    for _ in range(model.num_layers):
        self_cache.append(
            {
                "k": torch.zeros(batch, 3 * frame_tokens, model.num_heads, model.dim // model.num_heads),
                "v": torch.zeros(batch, 3 * frame_tokens, model.num_heads, model.dim // model.num_heads),
                "global_end_index": torch.zeros(1, dtype=torch.long),
                "local_end_index": torch.zeros(1, dtype=torch.long),
            }
        )
        cross_cache.append(
            {
                "k": torch.zeros(batch, model.text_len, model.num_heads, model.dim // model.num_heads),
                "v": torch.zeros(batch, model.text_len, model.num_heads, model.dim // model.num_heads),
                "is_init": False,
                "sequence_length": 0,
            }
        )

    output = model(
        x=torch.randn(batch, model.in_dim, frames, latent_height, latent_width),
        timestep=torch.tensor([[0.5]]),
        context=torch.randn(batch, model.text_len, 16),
        # The action map is pixel-space, while x is VAE-latent space.
        act_context=torch.randn(batch, 32, frames, latent_height * 2, latent_width * 2),
        kv_cache=self_cache,
        crossattn_cache=cross_cache,
        current_start=0,
    )

    assert output.shape == (batch, model.out_dim, frames, latent_height, latent_width)
    assert torch.isfinite(output).all()
    assert all(int(cache["global_end_index"].item()) == frame_tokens for cache in self_cache)
    assert all(bool(cache["is_init"]) for cache in cross_cache)


def test_full_window_steady_state_matches_regular_relative_rope_continuation() -> None:
    """The graph-safe path must preserve the normal rolling-cache result."""
    torch.manual_seed(7)
    model = _tiny_dit().eval()
    model.set_causal_attention_window(local_attn_size=3, sink_size=1)
    batch, frames, latent_height, latent_width = 1, 1, 8, 8
    frame_tokens = (latent_height // 2) * (latent_width // 2)
    head_dim = model.dim // model.num_heads
    self_cache = [
        {
            "k": torch.zeros(batch, 3 * frame_tokens, model.num_heads, head_dim),
            "v": torch.zeros(batch, 3 * frame_tokens, model.num_heads, head_dim),
            "global_end_index": torch.zeros(1, dtype=torch.long),
            "local_end_index": torch.zeros(1, dtype=torch.long),
        }
        for _ in range(model.num_layers)
    ]
    cross_cache = [
        {
            "k": torch.zeros(batch, model.text_len, model.num_heads, head_dim),
            "v": torch.zeros(batch, model.text_len, model.num_heads, head_dim),
            "is_init": False,
            "sequence_length": 0,
        }
        for _ in range(model.num_layers)
    ]
    context = torch.randn(batch, model.text_len, 16)

    def model_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.randn(batch, model.in_dim, frames, latent_height, latent_width),
            torch.randn(batch, 32, frames, latent_height * 2, latent_width * 2),
            torch.tensor([[0.5]]),
        )

    # Fill the local [sink, rolling-tail] window through the original path.
    for frame_index in range(model.local_attn_size):
        latent, action_context, timestep = model_inputs()
        model(
            x=latent,
            timestep=timestep,
            context=context,
            act_context=action_context,
            kv_cache=self_cache,
            crossattn_cache=cross_cache,
            current_start=frame_index * frame_tokens,
        )

    def clone_cache(cache_list: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in layer.items()}
            for layer in cache_list
        ]

    def assert_cache_equal(actual: list[dict[str, object]], expected: list[dict[str, object]]) -> None:
        for actual_layer, expected_layer in zip(actual, expected, strict=True):
            assert actual_layer.keys() == expected_layer.keys()
            for key, actual_value in actual_layer.items():
                expected_value = expected_layer[key]
                if isinstance(actual_value, torch.Tensor):
                    assert isinstance(expected_value, torch.Tensor)
                    torch.testing.assert_close(actual_value, expected_value)
                else:
                    assert actual_value == expected_value

    reference_self = clone_cache(self_cache)
    reference_cross = clone_cache(cross_cache)
    steady_self = clone_cache(self_cache)
    steady_cross = clone_cache(cross_cache)
    roll_tokens = (model.local_attn_size - model.sink_size - frames) * frame_tokens
    roll_scratch_k = torch.empty(batch, roll_tokens, model.num_heads, head_dim)
    roll_scratch_v = torch.empty_like(roll_scratch_k)
    current_end = torch.tensor([(model.local_attn_size + frames) * frame_tokens], dtype=torch.long)
    latent, action_context, timestep = model_inputs()

    expected = model(
        x=latent,
        timestep=timestep,
        context=context,
        act_context=action_context,
        kv_cache=reference_self,
        crossattn_cache=reference_cross,
        current_start=model.local_attn_size * frame_tokens,
    )
    actual = model.forward_steady_state(
        x=latent,
        timestep=timestep,
        context=context,
        act_context=action_context,
        kv_cache=steady_self,
        crossattn_cache=steady_cross,
        current_end=current_end,
        roll_scratch_k=roll_scratch_k,
        roll_scratch_v=roll_scratch_v,
        update_cache=True,
    )

    torch.testing.assert_close(actual, expected)
    assert_cache_equal(steady_self, reference_self)
    assert_cache_equal(steady_cross, reference_cross)

    # Remaining sampler calls use the same logical chunk and overwrite only
    # its fixed tail slot; they must not roll the window again.
    latent, action_context, timestep = model_inputs()
    expected = model(
        x=latent,
        timestep=timestep,
        context=context,
        act_context=action_context,
        kv_cache=reference_self,
        crossattn_cache=reference_cross,
        current_start=model.local_attn_size * frame_tokens,
    )
    actual = model.forward_steady_state(
        x=latent,
        timestep=timestep,
        context=context,
        act_context=action_context,
        kv_cache=steady_self,
        crossattn_cache=steady_cross,
        current_end=current_end,
        roll_scratch_k=roll_scratch_k,
        roll_scratch_v=roll_scratch_v,
        update_cache=False,
    )

    torch.testing.assert_close(actual, expected)
    assert_cache_equal(steady_self, reference_self)
    assert_cache_equal(steady_cross, reference_cross)
