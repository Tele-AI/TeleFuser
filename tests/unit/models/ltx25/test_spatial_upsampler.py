"""Tests for the isolated LTX-2.5 spatial upsampler."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from telefuser.models.ltx25.spatial_upsampler import (
    LTX25PixelShuffleND,
    LTX25SpatialUpsampler,
    LTX25SpatialUpsamplerConfig,
    load_video_latent_statistics,
    upsample_video_latent,
)


def test_spatial_pixel_shuffle_matches_torch_2d_pixel_shuffle() -> None:
    value = torch.arange(2 * 12 * 3 * 5, dtype=torch.float32).reshape(2, 12, 3, 5)
    actual = LTX25PixelShuffleND(2)(value)
    expected = torch.nn.functional.pixel_shuffle(value, upscale_factor=2)
    torch.testing.assert_close(actual, expected)


def test_spatial_upsampler_loads_exact_checkpoint_and_statistics(tmp_path: Path) -> None:
    config = LTX25SpatialUpsamplerConfig(
        in_channels=4,
        mid_channels=32,
        num_blocks_per_stage=1,
        dims=3,
        spatial_upsample=True,
        temporal_upsample=False,
    )
    source = LTX25SpatialUpsampler(config)
    upsampler_path = tmp_path / "upsampler.safetensors"
    save_file(source.state_dict(), upsampler_path, metadata={"config": json.dumps(asdict(config))})
    loaded = LTX25SpatialUpsampler.from_checkpoint(upsampler_path, torch_dtype=torch.float32)
    latent = torch.randn(1, 4, 2, 3, 5)
    torch.testing.assert_close(loaded(latent), source(latent))

    vae_path = tmp_path / "video_vae.safetensors"
    save_file(
        {
            "per_channel_statistics.std-of-means": torch.full((4,), 2.0),
            "per_channel_statistics.mean-of-means": torch.full((4,), 0.25),
        },
        vae_path,
    )
    statistics = load_video_latent_statistics(vae_path)
    expected = statistics.normalize(source(statistics.un_normalize(latent)))
    torch.testing.assert_close(upsample_video_latent(latent, source, statistics), expected)
