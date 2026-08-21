"""Tests for isolated LTX-2.5 DiffVAE checkpoint mapping."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from telefuser.models.ltx25.diff_vae import diffusion_tiling
from telefuser.models.ltx25.diff_vae.diffusion_video_decoder import (
    DiffusionVideoDecoder,
    _configure_chunked_compile_mode,
    _configure_chunked_eager_mode,
    ltx25_diffusion_vae_checkpoint_key_coverage,
)
from telefuser.models.ltx25.diff_vae.transformer import compiling as diffvae_compiling
from telefuser.models.ltx25.diff_vae.transformer.chunked.block import ChunkedDiffusionNABlock
from telefuser.models.ltx25.diff_vae.transformer.config import DiffVAEMode
from telefuser.ops import neighborhood_attention as attention_ops


def test_diffvae_mapping_splits_qkv_and_ignores_verified_unused_type_embedding(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "diffusion_video_vae.safetensors"
    save_file(
        {
            "decoder.block.qkv.weight": torch.ones(12, 4),
            "decoder.block.qkv.bias": torch.ones(12),
            "decoder.type_emb": torch.ones(4),
            "per_channel_statistics.mean-of-means": torch.zeros(4),
            "per_channel_statistics.std-of-means": torch.ones(4),
        },
        checkpoint_path,
    )
    model_keys = {
        "block.qkv.to_q.weight",
        "block.qkv.to_k.weight",
        "block.qkv.to_v.weight",
        "block.qkv.to_q.bias",
        "block.qkv.to_k.bias",
        "block.qkv.to_v.bias",
        "per_channel_statistics.mean-of-means",
        "per_channel_statistics.std-of-means",
    }
    assert ltx25_diffusion_vae_checkpoint_key_coverage(checkpoint_path, model_keys) == (set(), set())


def test_chunked_eager_configuration_uses_deferred_stage4_and_width_chunks() -> None:
    model = DiffusionVideoDecoder(
        in_channels=8,
        out_channels=3,
        patch_size=1,
        head_dim=8,
        rope_dim_split=(2, 2, 4),
        stage_channels=(8, 8, 8, 8, 8),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((1, 1, 1),) * 5,
        upsamples=(((1, 1, 1), 1),) * 4,
        stage5_kernel=(1, 1, 1),
        t_emb_dim=8,
    )

    _configure_chunked_eager_mode(model)

    assert model.deferred_stage4_upsample
    for block in model.diff_blocks:
        assert isinstance(block, ChunkedDiffusionNABlock)
        assert block.stage4_upsample is model.upsamples[3]
        assert block.attn.w_chunks == 4
        assert block.attn.rope_num_tiles == 1


def test_chunked_compile_configuration_compiles_only_diffusion_residuals(monkeypatch) -> None:
    compiled: list[object] = []
    kv_parallelism: list[bool] = []
    model = DiffusionVideoDecoder(
        in_channels=8,
        out_channels=3,
        patch_size=1,
        head_dim=8,
        rope_dim_split=(2, 2, 4),
        stage_channels=(8, 8, 8, 8, 8),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((1, 1, 1),) * 5,
        upsamples=(((1, 1, 1), 1),) * 4,
        stage5_kernel=(1, 1, 1),
        t_emb_dim=8,
    )

    def fake_compile(function, **kwargs):
        compiled.append((function, kwargs))
        return function

    monkeypatch.setattr(diffvae_compiling.torch, "compile", fake_compile)
    monkeypatch.setattr(
        diffvae_compiling,
        "configure_neighborhood_attention_kv_parallelism",
        lambda enabled: kv_parallelism.append(enabled),
    )

    _configure_chunked_compile_mode(model)

    assert model.deferred_stage4_upsample
    assert model.mark_dynamic_shapes
    assert kv_parallelism == [False]
    assert len(compiled) == len(model.diff_blocks)
    for block in model.diff_blocks:
        assert isinstance(block, ChunkedDiffusionNABlock)
        assert block.stage4_upsample is model.upsamples[3]
        assert block.attn.w_chunks == 4
        assert block.attn.natten_backend == "cutlass-fna"


def test_chunked_compile_tiling_uses_the_conservative_memory_budget() -> None:
    assert diffusion_tiling.stage5_mem_coef(DiffVAEMode.CHUNKED_COMPILE) == 7
    assert diffusion_tiling.budget_safety_bytes(DiffVAEMode.CHUNKED_COMPILE) == 2 << 30


def test_chunked_compile_tiling_caps_natten_workspace_tile_geometry() -> None:
    tiling = diffusion_tiling.recommended_decode_tiling_config(
        tile_halos=((0, 0, 0), (0, 0, 0)),
        pixel_scale=diffusion_tiling.VIDEO_SCALE_FACTORS,
        min_tile_size_s4=(1, 1, 1),
        patch_size=1,
        height=1536,
        width=1536,
        num_frames=121,
        mode=DiffVAEMode.CHUNKED_COMPILE,
        free_bytes=1 << 40,
        stage5_channels=8,
        stage4_channels=8,
        upsample_strides=((1, 1, 1),) * 4,
    )

    assert tiling.frames.tile_size <= 80
    assert tiling.height.tile_size <= 320
    assert tiling.width.tile_size <= 320


def test_natten_availability_requires_the_cuda_extension(monkeypatch) -> None:
    monkeypatch.setattr(attention_ops, "_NATTEN_AVAILABLE", True)
    monkeypatch.setattr(attention_ops, "natten", SimpleNamespace(HAS_LIBNATTEN=False))
    assert not attention_ops.natten_available()

    monkeypatch.setattr(attention_ops, "natten", SimpleNamespace(HAS_LIBNATTEN=True))
    assert attention_ops.natten_available()


def test_neighborhood_attention_dispatch_normalizes_qkv_dtype(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_na3d(query, key, value, **kwargs):
        captured.update(query=query, key=key, value=value, **kwargs)
        return value

    monkeypatch.setattr(attention_ops, "_NATTEN_AVAILABLE", True)
    monkeypatch.setattr(attention_ops, "natten", SimpleNamespace(HAS_LIBNATTEN=True, na3d=fake_na3d))

    value = torch.zeros((1, 1, 1, 1, 1, 2), dtype=torch.bfloat16)
    output = attention_ops.neighborhood_attention_3d(
        torch.ones_like(value, dtype=torch.float32),
        torch.ones_like(value, dtype=torch.float32),
        value,
        kernel_size=(1, 3, 3),
        backend="cutlass-fna",
    )

    assert output is value
    assert captured["query"].dtype == captured["key"].dtype == captured["value"].dtype == torch.bfloat16
    assert captured["kernel_size"] == (1, 3, 3)


def test_diffvae_recommends_auto_tiling_from_decoder_configuration() -> None:
    model = DiffusionVideoDecoder(
        in_channels=8,
        out_channels=3,
        patch_size=1,
        head_dim=8,
        rope_dim_split=(2, 2, 4),
        stage_channels=(8, 8, 8, 8, 8),
        stage_depths=(1, 1, 1, 1, 1),
        stage_kernels=((1, 1, 1),) * 5,
        upsamples=(((1, 1, 1), 1),) * 4,
        stage5_kernel=(1, 1, 1),
        t_emb_dim=8,
    )

    tiling = model.recommended_tiling_config(height=64, width=64, num_frames=9, free_bytes=1 << 40, model_bytes=0)

    assert tiling.frames.tile_size >= 9
    assert tiling.height.tile_size >= 64
    assert tiling.width.tile_size >= 64
