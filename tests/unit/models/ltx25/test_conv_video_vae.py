"""Tests for the isolated LTX-2.5 Conv VAE checkpoint mapping."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from telefuser.models.ltx25.conv_video_vae import ltx25_conv_video_vae_checkpoint_key_coverage


def test_conv_vae_mapping_duplicates_shared_statistics_for_encoder_and_decoder(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "conv_video_vae.safetensors"
    save_file(
        {
            "encoder.conv.weight": torch.ones(1),
            "decoder.conv.weight": torch.ones(1),
            "per_channel_statistics.mean-of-means": torch.zeros(1),
            "per_channel_statistics.std-of-means": torch.ones(1),
        },
        checkpoint_path,
    )
    model_keys = {
        "encoder.conv.weight",
        "decoder.conv.weight",
        "encoder.per_channel_statistics.mean-of-means",
        "encoder.per_channel_statistics.std-of-means",
        "decoder.per_channel_statistics.mean-of-means",
        "decoder.per_channel_statistics.std-of-means",
    }
    assert ltx25_conv_video_vae_checkpoint_key_coverage(checkpoint_path, model_keys) == (set(), set())
