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


def test_cached_cross_attention_uses_unified_attention_and_bsnd_cache() -> None:
    attention = CachedCrossAttention(dim=32, num_heads=4)
    attention_config = AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90)
    attention.attention_config = attention_config
    cache: dict[str, torch.Tensor | bool] = {"is_init": False}
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
