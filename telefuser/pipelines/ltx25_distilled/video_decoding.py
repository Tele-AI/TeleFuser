"""Video decoding for LTX-2.5."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx25 import DiffusionVideoDecoder, LTX25ConvVideoVAE


class LTX25VideoDecodingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig, *, video_vae: str) -> None:
        super().__init__("ltx25_video_decoding", config)
        self.video_decoder: DiffusionVideoDecoder | LTX25ConvVideoVAE = module_manager.fetch_module(
            "ltx25_video_decoder"
        )
        self.video_vae = video_vae
        self.model_names = ["video_decoder"]

    @with_model_offload(["video_decoder"])
    @torch.inference_mode()
    def decode(self, latent: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        if self.video_vae == "diff":
            return tuple(self.video_decoder.decode_video(latent, generator=generator))  # type: ignore[union-attr]
        return _conv_video_chunks_to_rgb(self.video_decoder.decode(latent, generator=generator))  # type: ignore[union-attr]


def _conv_video_chunks_to_rgb(chunks: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    output: list[torch.Tensor] = []
    for chunk in chunks:
        if chunk.ndim != 5 or chunk.shape[0] != 1 or chunk.shape[1] != 3:
            raise ValueError(f"LTX-2.5 ConvVAE decoder must return [1, 3, F, H, W], got {tuple(chunk.shape)}")
        output.append(chunk[0].permute(1, 2, 3, 0).add(1).mul(0.5).clamp(0, 1))
    return tuple(output)


__all__ = ["LTX25VideoDecodingStage"]
