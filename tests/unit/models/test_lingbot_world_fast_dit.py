from unittest.mock import patch

import torch

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.models.lingbot_world_fast_dit import (
    CachedCrossAttention,
    CausalSelfAttention,
    LingBotWorldFastDiT,
)
from telefuser.models.wan_video_dit import precompute_freqs_cis_3d


def test_causal_self_attention_uses_unified_attention() -> None:
    attention = CausalSelfAttention(dim=32, num_heads=4)
    attention_config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90)
    attention.attention_config = attention_config

    freqs = precompute_freqs_cis_3d(8)
    freqs_cos = torch.cat([freq.real for freq in freqs], dim=-1)
    freqs_sin = torch.cat([freq.imag for freq in freqs], dim=-1)
    cache = {
        "k": torch.zeros(1, 12, 4, 8),
        "v": torch.zeros(1, 12, 4, 8),
        "global_end_index": 0,
        "local_end_index": 0,
    }
    captured: dict[str, object] = {}

    def fake_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
        captured.update(q_shape=q.shape, k_shape=k.shape, v_shape=v.shape, **kwargs)
        return q

    with patch("telefuser.models.lingbot_world_fast_dit.attn_func", side_effect=fake_attention):
        output = attention(
            torch.randn(1, 6, 32),
            freqs_cos,
            freqs_sin,
            (1, 2, 3),
            cache,
            current_start=0,
            max_attention_size=12,
        )

    assert output.shape == (1, 6, 32)
    assert captured["q_shape"] == torch.Size([1, 6, 4, 8])
    assert captured["k_shape"] == torch.Size([1, 6, 4, 8])
    assert captured["v_shape"] == torch.Size([1, 6, 4, 8])
    assert captured["attention_config"] is attention_config
    assert captured["input_layout"] == "BSND"
    assert captured["output_layout"] == "BSND"


def test_causal_self_attention_packs_qkv_into_one_ulysses_collective() -> None:
    attention = CausalSelfAttention(dim=32, num_heads=4)
    freqs = precompute_freqs_cis_3d(8)
    freqs_cos = torch.cat([freq.real for freq in freqs], dim=-1)
    freqs_sin = torch.cat([freq.imag for freq in freqs], dim=-1)
    cache = {
        "k": torch.zeros(1, 12, 4, 8),
        "v": torch.zeros(1, 12, 4, 8),
        "global_end_index": 0,
        "local_end_index": 0,
    }

    def fake_scatter(tensor: torch.Tensor, _group: object):
        return lambda: tensor

    def fake_gather(tensor: torch.Tensor, _group: object, *, num_heads: int):
        assert num_heads == 4
        return lambda: tensor

    with (
        patch("telefuser.models.lingbot_world_fast_dit.get_ulysses_group", return_value=object()),
        patch("telefuser.models.lingbot_world_fast_dit.get_ulysses_world_size", return_value=4),
        patch(
            "telefuser.models.lingbot_world_fast_dit.ulysses_scatter_heads",
            side_effect=fake_scatter,
        ) as scatter,
        patch("telefuser.models.lingbot_world_fast_dit.ulysses_gather_heads", side_effect=fake_gather),
        patch("telefuser.models.lingbot_world_fast_dit.attn_func", side_effect=lambda q, _k, _v, **_kwargs: q),
    ):
        attention(
            torch.randn(1, 4, 32),
            freqs_cos,
            freqs_sin,
            (1, 2, 2),
            cache,
            current_start=0,
            max_attention_size=4,
        )

    scatter.assert_called_once()
    packed_qkv = scatter.call_args.args[0]
    assert packed_qkv.shape == (1, 4, 4, 24)


def test_cached_cross_attention_uses_unified_attention_and_bsnd_cache() -> None:
    attention = CachedCrossAttention(dim=32, num_heads=4)
    attention_config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90)
    attention.attention_config = attention_config
    cache: dict[str, torch.Tensor | bool | int] = {"is_init": False}
    calls: list[tuple[torch.Size, torch.Size, AttentionConfig]] = []

    def fake_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> torch.Tensor:
        assert k.shape == v.shape
        calls.append((q.shape, k.shape, kwargs["attention_config"]))
        return q

    with patch("telefuser.models.lingbot_world_fast_dit.attn_func", side_effect=fake_attention):
        output = attention(torch.randn(1, 6, 32), torch.randn(1, 5, 32), cache)
        cached_output = attention(torch.randn(1, 6, 32), torch.randn(1, 5, 32), cache)

    assert output.shape == cached_output.shape == (1, 6, 32)
    assert calls == [
        (torch.Size([1, 6, 4, 8]), torch.Size([1, 5, 4, 8]), attention_config),
        (torch.Size([1, 6, 4, 8]), torch.Size([1, 5, 4, 8]), attention_config),
    ]
    assert cache["k"].shape == torch.Size([1, 5, 4, 8])
    assert cache["v"].shape == torch.Size([1, 5, 4, 8])
    assert cache["is_init"] is True


def test_cached_cross_attention_writes_into_preallocated_storage() -> None:
    attention = CachedCrossAttention(dim=32, num_heads=4)
    cache: dict[str, torch.Tensor | bool | int] = {
        "k": torch.empty(1, 8, 4, 8),
        "v": torch.empty(1, 8, 4, 8),
        "is_init": False,
        "sequence_length": 0,
    }
    k_pointer = cache["k"].data_ptr()
    v_pointer = cache["v"].data_ptr()

    with patch("telefuser.models.lingbot_world_fast_dit.attn_func", side_effect=lambda q, _k, _v, **_kwargs: q):
        attention(torch.randn(1, 6, 32), torch.randn(1, 5, 32), cache)

    assert cache["k"].data_ptr() == k_pointer
    assert cache["v"].data_ptr() == v_pointer
    assert cache["sequence_length"] == 5
    assert cache["is_init"] is True


def test_set_attention_config_updates_all_blocks() -> None:
    model = LingBotWorldFastDiT(
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
    )
    attention_config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90)

    model.set_attention_config(attention_config)

    for block in model.blocks:
        assert block.self_attn.attention_config is attention_config
        assert block.cross_attn.attention_config is attention_config


def test_scalar_timestep_modulation_stays_broadcastable() -> None:
    model = LingBotWorldFastDiT(
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=1,
    )

    scalar_head, scalar_modulation = model._build_timestep_embeddings(torch.tensor([5.0]), seq_len=3)
    token_head, token_modulation = model._build_timestep_embeddings(torch.full((1, 3), 5.0), seq_len=3)

    assert scalar_head.shape == (1, 1, 32)
    assert scalar_modulation.shape == (1, 6, 32)
    assert token_head.shape == (1, 3, 32)
    assert token_modulation.shape == (1, 3, 6, 32)
    torch.testing.assert_close(token_head, scalar_head.expand_as(token_head))
    torch.testing.assert_close(token_modulation, scalar_modulation.unsqueeze(1).expand_as(token_modulation))


def test_camera_conditioner_runs_after_sequence_sharding() -> None:
    model = LingBotWorldFastDiT(
        patch_size=(1, 2, 2),
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
    ).eval()
    model.device_mesh = object()
    control = torch.randn(1, 6 * 64, 1, 4, 4)

    def fake_shard(_mesh: object, tensors: list[torch.Tensor], dims: list[int]) -> None:
        assert dims == [1]
        local = tensors[0][:, :1].clone()
        tensors[0].resize_(local.shape)
        tensors[0].copy_(local)

    with (
        torch.inference_mode(),
        patch(
            "telefuser.models.lingbot_world_fast_dit.sequence_parallel_shard",
            side_effect=fake_shard,
        ) as shard,
    ):
        prepared = model._prepare_control(control, shard_for_usp=True)

    assert prepared is not None
    control_tokens, camera_modulations = prepared
    assert control_tokens.shape == (1, 1, 32)
    assert len(camera_modulations) == 2
    assert all(scale.shape == shift.shape == (1, 1, 32) for scale, shift in camera_modulations)
    shard.assert_called_once()


def _kv_cache() -> tuple[list[dict[str, torch.Tensor | int]], list[dict[str, torch.Tensor | bool | int]]]:
    self_cache = [
        {
            "k": torch.zeros(1, 1, 4, 8),
            "v": torch.zeros(1, 1, 4, 8),
            "global_end_index": 0,
            "local_end_index": 0,
        }
    ]
    cross_cache = [
        {
            "k": torch.empty(1, 2, 4, 8),
            "v": torch.empty(1, 2, 4, 8),
            "is_init": False,
            "sequence_length": 0,
        }
    ]
    return self_cache, cross_cache


def test_prepared_control_and_cache_only_forward_preserve_transformer_execution() -> None:
    model = LingBotWorldFastDiT(
        patch_size=(1, 2, 2),
        in_dim=8,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=1,
    ).eval()
    x = torch.randn(1, 4, 1, 2, 2)
    condition = torch.randn_like(x)
    context = torch.randn(1, 2, 16)
    control = torch.randn(1, 6 * 64, 1, 2, 2)
    timestep = torch.zeros(1)

    self_cache, cross_cache = _kv_cache()
    baseline = model(
        x,
        timestep,
        context,
        y=condition,
        control_tensor=control,
        kv_cache=self_cache,
        crossattn_cache=cross_cache,
        max_attention_size=1,
    )
    prepared_control = model._prepare_control(control)
    projected_context = model._project_text_context(context)
    self_cache, cross_cache = _kv_cache()
    cached = model(
        x,
        timestep,
        context,
        y=condition,
        control_tensor=control,
        kv_cache=self_cache,
        crossattn_cache=cross_cache,
        max_attention_size=1,
        _prepared_control=prepared_control,
        _projected_context=projected_context,
    )

    torch.testing.assert_close(cached, baseline, rtol=0, atol=0)

    session_input_cache: dict[str, object] = {}
    with (
        patch.object(model, "_project_text_context", wraps=model._project_text_context) as project_context,
        patch.object(model, "_prepare_control", wraps=model._prepare_control) as prepare_control,
        patch.object(
            model.blocks[0].self_attn,
            "_prepare_causal_rope",
            wraps=model.blocks[0].self_attn._prepare_causal_rope,
        ) as prepare_causal_rope,
    ):
        self_cache, cross_cache = _kv_cache()
        first_cached = model(
            x,
            timestep,
            context,
            y=condition,
            control_tensor=control,
            kv_cache=self_cache,
            crossattn_cache=cross_cache,
            max_attention_size=1,
            _session_input_cache=session_input_cache,
        )
        self_cache, cross_cache = _kv_cache()
        second_cached = model(
            x,
            timestep,
            context,
            y=condition,
            control_tensor=control,
            kv_cache=self_cache,
            crossattn_cache=cross_cache,
            max_attention_size=1,
            _session_input_cache=session_input_cache,
        )

    torch.testing.assert_close(first_cached, baseline, rtol=0, atol=0)
    torch.testing.assert_close(second_cached, baseline, rtol=0, atol=0)
    project_context.assert_called_once()
    prepare_control.assert_called_once()
    prepare_causal_rope.assert_called_once()

    self_cache, cross_cache = _kv_cache()
    with (
        patch.object(model.blocks[0].self_attn.o, "forward", wraps=model.blocks[0].self_attn.o.forward) as self_out,
        patch.object(model.blocks[0].cross_attn, "forward", wraps=model.blocks[0].cross_attn.forward) as cross_attn,
        patch.object(model.blocks[0].ffn, "forward", wraps=model.blocks[0].ffn.forward) as ffn,
        patch.object(model.head, "forward", wraps=model.head.forward) as head_forward,
    ):
        output = model(
            x,
            timestep,
            context,
            y=condition,
            control_tensor=control,
            kv_cache=self_cache,
            crossattn_cache=cross_cache,
            max_attention_size=1,
            _prepared_control=prepared_control,
            _projected_context=projected_context,
            update_cache_only=True,
        )

    assert output is None
    self_out.assert_not_called()
    cross_attn.assert_not_called()
    ffn.assert_not_called()
    head_forward.assert_not_called()
    assert self_cache[0]["global_end_index"] == 1
    assert cross_cache[0]["is_init"] is False
