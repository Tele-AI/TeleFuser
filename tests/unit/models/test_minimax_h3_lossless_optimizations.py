from unittest.mock import MagicMock, patch

import pytest
import torch

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.models.minimax_h3_dit import (
    MiniMaxH3Attention,
    MiniMaxH3DiTConfig,
    _modulate,
    _norm_modulate,
    _rms_norm,
)


def _small_config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        hidden_size=32,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=1,
    )


def test_attention_config_validates_lossless_ulysses_options() -> None:
    default = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_4)
    assert default.attention_chunks == 1
    assert default.ulysses_sequence_mode == "padded"

    optimized = AttentionConfig.dense_attention(
        AttnImplType.FLASH_ATTN_4,
        attention_chunks=2,
        ulysses_sequence_mode="valid_only",
    )
    assert optimized.attention_chunks == 2
    assert optimized.ulysses_sequence_mode == "valid_only"

    with pytest.raises(ValueError, match="attention_chunks"):
        AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_4, attention_chunks=3)
    with pytest.raises(ValueError, match="ulysses_sequence_mode"):
        AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_4, ulysses_sequence_mode="unknown")


@pytest.mark.parametrize(
    ("chunks", "sequence_mode", "expected_ranges", "pad_output", "zero_tail"),
    [
        (1, "padded", [(0, 2)], True, False),
        (2, "valid_only", [(0, 1), (1, 1)], False, True),
    ],
)
def test_fused_ulysses_path_obeys_public_configuration(
    chunks: int,
    sequence_mode: str,
    expected_ranges: list[tuple[int, int]],
    pad_output: bool,
    zero_tail: bool,
) -> None:
    module = MiniMaxH3Attention(_small_config()).eval()
    module.ulysses_group = MagicMock()
    hidden = torch.randn(8, 32, dtype=torch.bfloat16)
    rope = torch.randn(8, 2, dtype=torch.bfloat16)
    scatter_ranges: list[tuple[int, int]] = []
    attention_options: list[tuple[bool, bool]] = []
    gather_options: list[bool] = []

    def scatter(*_args: object, local_head_start: int, local_head_count: int, **_kwargs: object):
        scatter_ranges.append((local_head_start, local_head_count))
        tensor = torch.zeros(1, 8, local_head_count, 8, dtype=torch.bfloat16)
        return lambda: (tensor, tensor, tensor)

    def run_attention(query: torch.Tensor, *_args: object, **kwargs: object) -> torch.Tensor:
        fixed_valid = bool(kwargs["fixed_valid"])
        pad_fixed_valid_output = bool(kwargs["pad_fixed_valid_output"])
        attention_options.append((fixed_valid, pad_fixed_valid_output))
        return query if pad_fixed_valid_output else query[:, :5]

    def gather(*_args: object, destination: torch.Tensor, zero_tail: bool, **_kwargs: object):
        gather_options.append(zero_tail)

        def wait() -> torch.Tensor:
            destination.zero_()
            return destination

        return wait

    config = AttentionConfig.dense_attention(
        AttnImplType.FLASH_ATTN_4,
        attention_chunks=chunks,
        ulysses_sequence_mode=sequence_mode,
    )
    with (
        patch("telefuser.models.minimax_h3_dit.dist.get_world_size", return_value=2),
        patch("telefuser.models.minimax_h3_dit._can_run_fused_ulysses_attention", return_value=True),
        patch(
            "telefuser.models.minimax_h3_dit.ulysses_scatter_qkv_qknorm_rope_chunk_async",
            side_effect=scatter,
        ),
        patch("telefuser.models.minimax_h3_dit.ulysses_gather_heads_chunk_async", side_effect=gather),
        patch("telefuser.models.minimax_h3_dit.attention", side_effect=run_attention),
    ):
        output = module(
            hidden,
            sequence_lengths=[5, 3],
            rope_cos_sin_cache=rope,
            attention_config=config,
            cu_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
        )

    assert output.shape == hidden.shape
    assert scatter_ranges == expected_ranges
    assert attention_options == [(True, pad_output)] * chunks
    assert gather_options == [zero_tail] * chunks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernels")
def test_fused_norm_modulation_is_bit_exact() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    norm = _rms_norm(32, 1e-5).to(device=device, dtype=torch.bfloat16)
    hidden = torch.randn(17, 32, device=device, dtype=torch.bfloat16)
    shift = torch.randn(3, 32, device=device, dtype=torch.bfloat16)
    scale = torch.randn(3, 32, device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, 3, (17,), device=device)

    expected = _modulate(norm(hidden.clone()), shift, scale, indices)
    actual = _norm_modulate(norm, hidden.clone(), shift, scale, indices)

    assert torch.equal(actual, expected)
