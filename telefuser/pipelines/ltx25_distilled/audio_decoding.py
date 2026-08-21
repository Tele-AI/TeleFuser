"""Audio decoding and vocoding for LTX-2.5."""

from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager


class LTX25AudioDecodingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_audio_decoding", config)
        self.audio_decoder: torch.nn.Module = module_manager.fetch_module("ltx25_audio_decoder")
        self.vocoder: torch.nn.Module = module_manager.fetch_module("ltx25_vocoder")
        self.model_names = ["audio_decoder", "vocoder"]

    @with_model_offload(["audio_decoder", "vocoder"])
    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vocoder(self.audio_decoder(latent)).squeeze(0).float()


__all__ = ["LTX25AudioDecodingStage"]
