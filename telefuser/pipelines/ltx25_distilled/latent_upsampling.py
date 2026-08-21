"""Latent resolution bridge for LTX-2.5."""

from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx25 import LTX25SpatialUpsampler
from telefuser.models.ltx25.spatial_upsampler import LTX25PerChannelStatistics


class LTX25LatentUpsamplingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_latent_upsampling", config)
        self.spatial_upsampler: LTX25SpatialUpsampler = module_manager.fetch_module("ltx25_spatial_upsampler")
        self.latent_statistics: LTX25PerChannelStatistics = module_manager.fetch_module("ltx25_video_latent_statistics")
        self.model_names = ["spatial_upsampler", "latent_statistics"]

    @with_model_offload(["spatial_upsampler", "latent_statistics"])
    @torch.inference_mode()
    def process(self, latent: torch.Tensor) -> torch.Tensor:
        return self.latent_statistics.normalize(self.spatial_upsampler(self.latent_statistics.un_normalize(latent)))


__all__ = ["LTX25LatentUpsamplingStage"]
