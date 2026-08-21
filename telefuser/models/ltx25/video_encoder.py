"""Isolated LTX-2.5 video encoder split-checkpoint loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .checkpoint import inspect_checkpoint
from .conv_video_vae import LogVarianceType, NormLayerType, PaddingModeType, VideoEncoder


class LTX25VideoEncoder(VideoEncoder):
    """LTX-2.5 video encoder built from the split VAE metadata."""

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LTX25VideoEncoder":
        """Construct and strictly load the video encoder from either official VAE variant."""
        checkpoint = inspect_checkpoint(checkpoint_path)
        kwargs = _video_encoder_kwargs(checkpoint.config)
        with torch.device("meta"):
            model = cls(**kwargs)
        unexpected, missing = ltx25_video_encoder_checkpoint_key_coverage(checkpoint.path, set(model.state_dict()))
        if unexpected or missing:
            raise ValueError(
                "LTX-2.5 video encoder checkpoint coverage mismatch: "
                f"unexpected={sorted(unexpected)[:5]}, missing={sorted(missing)[:5]}"
            )
        state_dict = _load_video_encoder_state_dict(checkpoint.path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True, assign=True)
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"LTX-2.5 video encoder load mismatch: missing={missing_keys[:5]}, unexpected={unexpected_keys[:5]}"
            )
        return model.to(device=device, dtype=torch_dtype).eval()


def _video_encoder_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    vae_config = config.get("vae")
    if not isinstance(vae_config, dict):
        raise ValueError("LTX-2.5 VAE checkpoint is missing object vae config")
    encoder_config = vae_config.get("encoder", vae_config)
    if not isinstance(encoder_config, dict):
        raise ValueError("LTX-2.5 VAE encoder config must be an object")
    blocks = encoder_config.get("blocks", encoder_config.get("encoder_blocks"))
    if not isinstance(blocks, list):
        raise ValueError("LTX-2.5 VAE encoder config is missing blocks")
    return {
        "convolution_dimensions": encoder_config.get("dims", vae_config.get("dims", 3)),
        "in_channels": encoder_config.get("in_channels", 3),
        "out_channels": encoder_config.get("latent_channels", vae_config.get("latent_channels", 128)),
        "encoder_blocks": blocks,
        "patch_size": encoder_config.get("patch_size", 4),
        "norm_layer": NormLayerType(encoder_config.get("norm_layer", "pixel_norm")),
        "latent_log_var": LogVarianceType(encoder_config.get("latent_log_var", "uniform")),
        "encoder_spatial_padding_mode": PaddingModeType(encoder_config.get("spatial_padding_mode", "zeros")),
    }


def _video_encoder_target_key(key: str) -> str | None:
    if key.startswith("encoder."):
        return key
    if key.startswith("per_channel_statistics."):
        return "per_channel_statistics." + key.removeprefix("per_channel_statistics.")
    return None


def _load_video_encoder_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            target = _video_encoder_target_key(key)
            if target is not None:
                state_dict[target.removeprefix("encoder.")] = checkpoint.get_tensor(key)
    return state_dict


def ltx25_video_encoder_checkpoint_key_coverage(
    checkpoint_path: str | Path,
    model_keys: set[str],
) -> tuple[set[str], set[str]]:
    """Return unexplained source keys and missing isolated video encoder keys."""
    mapped: set[str] = set()
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            target = _video_encoder_target_key(key)
            if target is not None:
                mapped.add(target.removeprefix("encoder."))
    return mapped - model_keys, model_keys - mapped
