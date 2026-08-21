"""Sequence-parallel contracts for the isolated LTX-2.5 transformer."""

from __future__ import annotations

import torch

from telefuser.core.config import AttnImplType
from telefuser.models.ltx25.transformer import Attention, LTXModel, LTXRopeType, TransformerArgs


def _tiny_model() -> LTXModel:
    return LTXModel(
        num_attention_heads=4,
        attention_head_dim=8,
        in_channels=8,
        out_channels=8,
        num_layers=1,
        cross_attention_dim=16,
        audio_num_attention_heads=4,
        audio_attention_head_dim=8,
        audio_in_channels=8,
        audio_out_channels=8,
        audio_cross_attention_dim=16,
    )


def test_enable_usp_wires_self_and_av_attention_only(monkeypatch) -> None:
    model = _tiny_model()
    mesh = object()
    group = object()
    monkeypatch.setattr("telefuser.models.ltx25.transformer.get_attention_strategy", lambda _: "ulysses")
    monkeypatch.setattr("telefuser.models.ltx25.transformer.get_ulysses_world_size", lambda _: 4)
    monkeypatch.setattr("telefuser.models.ltx25.transformer.get_ulysses_group", lambda _: group)

    model.enable_usp(mesh)  # type: ignore[arg-type]

    block = model.transformer_blocks[0]
    assert model.usp_flag
    assert block.attn1.ulysses_group is group
    assert block.audio_attn1.ulysses_group is group
    assert block.audio_to_video_attn.ulysses_group is group
    assert block.video_to_audio_attn.ulysses_group is group
    assert block.attn2.ulysses_group is None
    assert block.audio_attn2.ulysses_group is None


def test_shard_transformer_args_masks_sequence_padding(monkeypatch) -> None:
    model = _tiny_model()
    model.device_mesh = object()  # type: ignore[assignment]
    monkeypatch.setattr("telefuser.models.ltx25.transformer.get_ulysses_world_size", lambda _: 4)
    shard_calls = []
    monkeypatch.setattr(
        "telefuser.models.ltx25.transformer.sequence_parallel_shard",
        lambda mesh, tensors, dimensions: shard_calls.append((mesh, tensors, dimensions)),
    )
    args = TransformerArgs(
        x=torch.zeros(1, 5, 32),
        context=torch.zeros(1, 2, 16),
        context_mask=torch.zeros(1, 1, 1, 2),
        timesteps=torch.zeros(1, 5, 6, 32),
        embedded_timestep=torch.zeros(1, 5, 32),
        positional_embeddings=(torch.zeros(1, 5, 32), torch.zeros(1, 5, 32)),
        cross_positional_embeddings=(torch.zeros(1, 5, 16), torch.zeros(1, 5, 16)),
        cross_scale_shift_timestep=torch.zeros(1, 5, 4, 32),
        cross_gate_timestep=torch.zeros(1, 1, 32),
        enabled=True,
    )

    sharded, sequence_length = model._shard_transformer_args(args)

    assert sequence_length == 5
    assert shard_calls[0][2] == [1] * 8
    assert sharded.key_padding_mask.shape == (1, 1, 1, 8)
    assert torch.all(sharded.key_padding_mask[..., :5] == 0)
    assert torch.all(sharded.key_padding_mask[..., 5:] == torch.finfo(torch.float32).min)
    assert sharded.self_attention_mask is sharded.key_padding_mask


def test_split_rope_shards_the_token_dimension_when_heads_match_sequence_length(monkeypatch) -> None:
    model = _tiny_model()
    model.rope_type = LTXRopeType.SPLIT
    model.device_mesh = object()  # type: ignore[assignment]
    monkeypatch.setattr("telefuser.models.ltx25.transformer.get_ulysses_world_size", lambda _: 2)
    shard_calls = []
    monkeypatch.setattr(
        "telefuser.models.ltx25.transformer.sequence_parallel_shard",
        lambda mesh, tensors, dimensions: shard_calls.append((mesh, tensors, dimensions)),
    )
    args = TransformerArgs(
        x=torch.zeros(1, 4, 32),
        context=torch.zeros(1, 2, 16),
        context_mask=torch.zeros(1, 1, 1, 2),
        timesteps=torch.zeros(1, 4, 6, 32),
        embedded_timestep=torch.zeros(1, 4, 32),
        positional_embeddings=(torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 4, 4)),
        cross_positional_embeddings=(torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 4, 4)),
        cross_scale_shift_timestep=torch.zeros(1, 4, 4, 32),
        cross_gate_timestep=torch.zeros(1, 1, 32),
        enabled=True,
    )

    model._shard_transformer_args(args)

    assert shard_calls[0][2] == [1, 1, 1, 1, 2, 2, 2, 2]


def test_padding_mask_forces_mask_aware_attention(monkeypatch) -> None:
    selected_backends = []

    def fake_attention(query, key, value, *, attention_config, **kwargs):
        del key, value, kwargs
        selected_backends.append(attention_config.attn_impl)
        return query

    monkeypatch.setattr("telefuser.models.ltx25.transformer.attn_func", fake_attention)
    attention = Attention(query_dim=8, heads=2, dim_head=4)

    output = attention(torch.zeros(1, 4, 8), mask=torch.zeros(1, 1, 1, 4), enforce_mask=True)

    assert output.shape == (1, 4, 8)
    assert selected_backends == [AttnImplType.TORCH_SDPA]
