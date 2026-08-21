"""Tests for the isolated LTX-2.5 audio decoder and vocoder loader."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from telefuser.models.ltx25.audio import (
    AudioDecoderConfigurator,
    VocoderConfigurator,
    load_ltx25_audio_decoder_and_vocoder,
    ltx25_audio_checkpoint_key_coverage,
)


def _audio_config() -> dict:
    return {
        "audio_vae": {
            "model": {
                "params": {
                    "ddconfig": {
                        "attn_resolutions": [],
                        "causality_axis": "height",
                        "ch": 32,
                        "ch_mult": [1, 2],
                        "mel_bins": 64,
                        "mid_block_add_attention": False,
                        "norm_type": "pixel",
                        "num_res_blocks": 1,
                        "out_ch": 2,
                        "resolution": 32,
                        "z_channels": 8,
                    },
                    "sampling_rate": 16000,
                }
            },
            "preprocessing": {"stft": {"causal": True, "hop_length": 160}},
        },
        "vocoder": {
            "vocoder": {
                "activation": "snakebeta",
                "resblock": "AMP1",
                "resblock_dilation_sizes": [[1, 3, 5]],
                "resblock_kernel_sizes": [3],
                "upsample_initial_channel": 64,
                "upsample_kernel_sizes": [4],
                "upsample_rates": [2],
                "use_bias_at_final": False,
                "use_tanh_at_final": False,
            },
            "bwe": {
                "activation": "snakebeta",
                "hop_length": 2,
                "input_sampling_rate": 16000,
                "n_fft": 8,
                "num_mels": 4,
                "output_sampling_rate": 48000,
                "resblock": "AMP1",
                "resblock_dilation_sizes": [[1, 3, 5]],
                "resblock_kernel_sizes": [3],
                "upsample_initial_channel": 32,
                "upsample_kernel_sizes": [4],
                "upsample_rates": [2],
                "use_bias_at_final": False,
                "use_tanh_at_final": False,
            },
        },
    }


def test_audio_loader_strictly_maps_decoder_and_bwe_vocoder_weights(tmp_path: Path) -> None:
    config = _audio_config()
    decoder = AudioDecoderConfigurator.from_config(config)
    vocoder = VocoderConfigurator.from_config(config)
    state_dict = {
        **{
            ("audio_vae.per_channel_statistics." if key.startswith("per_channel_statistics.") else "audio_vae.decoder.")
            + key.removeprefix("per_channel_statistics."): value
            for key, value in decoder.state_dict().items()
        },
        **{f"vocoder.{key}": value for key, value in vocoder.state_dict().items()},
    }
    checkpoint_path = tmp_path / "audio.safetensors"
    save_file(state_dict, checkpoint_path, metadata={"config": json.dumps(config)})

    loaded_decoder, loaded_vocoder = load_ltx25_audio_decoder_and_vocoder(checkpoint_path, torch_dtype=torch.float32)
    coverage = ltx25_audio_checkpoint_key_coverage(
        checkpoint_path, set(loaded_decoder.state_dict()), set(loaded_vocoder.state_dict())
    )
    assert coverage == (set(), set(), set(), set())
    assert loaded_vocoder.output_sampling_rate == 48000
    for expected, actual in zip(decoder.state_dict().values(), loaded_decoder.state_dict().values(), strict=True):
        torch.testing.assert_close(actual, expected, equal_nan=True)
