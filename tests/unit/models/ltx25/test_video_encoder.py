"""Tests for isolated LTX-2.5 video encoder checkpoint mapping."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from telefuser.models.ltx25.video_encoder import _video_encoder_kwargs, ltx25_video_encoder_checkpoint_key_coverage


def test_video_encoder_mapping_accepts_encoder_and_shared_statistics(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "video_vae.safetensors"
    save_file(
        {
            "encoder.conv_in.weight": torch.ones(1),
            "per_channel_statistics.mean-of-means": torch.zeros(1),
            "per_channel_statistics.std-of-means": torch.ones(1),
        },
        checkpoint_path,
    )
    model_keys = {
        "conv_in.weight",
        "per_channel_statistics.mean-of-means",
        "per_channel_statistics.std-of-means",
    }
    assert ltx25_video_encoder_checkpoint_key_coverage(checkpoint_path, model_keys) == (set(), set())


def test_video_encoder_uses_nested_vae_latent_channels() -> None:
    kwargs = _video_encoder_kwargs(
        {
            "vae": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "encoder_blocks": [["res_x", {"num_layers": 4}]],
            }
        }
    )

    assert kwargs["in_channels"] == 3
    assert kwargs["out_channels"] == 128
