"""ModuleManager loading helpers for the LTX-2.5 distilled model pack."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx25 import (
    DiffusionVideoDecoder,
    LTX25AVTransformer,
    LTX25ConvVideoVAE,
    LTX25DurationHead,
    LTX25EmbeddingsProcessor,
    LTX25Gemma4TextEncoder,
    LTX25ModelPaths,
    LTX25SpatialUpsampler,
    LTX25VideoEncoder,
    load_ltx25_audio_decoder_and_vocoder,
)
from telefuser.models.ltx25.spatial_upsampler import load_video_latent_statistics


def load_ltx25_distilled_modules(
    module_manager: ModuleManager,
    model_root: str | Path,
    *,
    video_vae: Literal["diff", "conv"] = "diff",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> LTX25ModelPaths:
    """Load every LTX-2.5 component on CPU and register it with ``module_manager``."""
    if video_vae not in ("diff", "conv"):
        raise ValueError(f"video_vae must be 'diff' or 'conv', got {video_vae!r}")
    paths = LTX25ModelPaths.from_model_root(model_root)
    video_vae_path = paths.video_vae_path if video_vae == "diff" else paths.conv_video_vae_path

    def add(module: torch.nn.Module, name: str, path: str | Path) -> None:
        module_manager.add_module(module, name, path=str(path))

    add(
        LTX25Gemma4TextEncoder.from_checkpoint(paths.text_encoder_path, device="cpu", torch_dtype=torch_dtype),
        "ltx25_gemma4",
        paths.text_encoder_path,
    )
    add(
        LTX25EmbeddingsProcessor.from_checkpoints(
            paths.transformer_path,
            paths.text_encoder_path,
            device="cpu",
            torch_dtype=torch_dtype,
        ),
        "ltx25_embeddings_processor",
        paths.transformer_path,
    )
    add(
        LTX25DurationHead.from_checkpoint(paths.duration_head_path, device="cpu", torch_dtype=torch_dtype),
        "ltx25_duration_head",
        paths.duration_head_path,
    )
    add(
        LTX25VideoEncoder.from_checkpoint(video_vae_path, device="cpu", torch_dtype=torch_dtype),
        "ltx25_video_encoder",
        video_vae_path,
    )
    add(
        LTX25AVTransformer.from_checkpoint(paths.transformer_path, device="cpu", torch_dtype=torch_dtype),
        "ltx25_transformer",
        paths.transformer_path,
    )
    add(
        LTX25SpatialUpsampler.from_checkpoint(paths.spatial_upsampler_path, device="cpu", torch_dtype=torch_dtype),
        "ltx25_spatial_upsampler",
        paths.spatial_upsampler_path,
    )
    add(load_video_latent_statistics(video_vae_path), "ltx25_video_latent_statistics", video_vae_path)

    if video_vae == "diff":
        video_decoder = DiffusionVideoDecoder.from_checkpoint(video_vae_path, device="cpu", torch_dtype=torch_dtype)
    else:
        video_decoder = LTX25ConvVideoVAE.from_checkpoint(video_vae_path, device="cpu", torch_dtype=torch_dtype)
    add(video_decoder, "ltx25_video_decoder", video_vae_path)

    audio_decoder, vocoder = load_ltx25_audio_decoder_and_vocoder(
        paths.audio_vae_path, device="cpu", torch_dtype=torch_dtype
    )
    add(audio_decoder, "ltx25_audio_decoder", paths.audio_vae_path)
    add(vocoder, "ltx25_vocoder", paths.audio_vae_path)
    return paths


__all__ = ["load_ltx25_distilled_modules"]
