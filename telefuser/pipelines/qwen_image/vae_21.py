"""Qwen-Image 2.1 VAE stage."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from diffusers.image_processor import VaeImageProcessor

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics


class VAE21Stage(BaseStage):
    """Decode the ModuleManager-owned Qwen-Image 2.1 VAE."""

    def __init__(self, name: str, module_manager: ModuleManager, config: ModelRuntimeConfig):
        super().__init__(name, config)
        self.vae = module_manager.fetch_module("vae")
        self.model_names = ["vae"]
        self.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)

    @with_model_offload(["vae"])
    @torch.inference_mode()
    @with_metrics
    def encode_conditions(
        self, images: list[Image.Image], batch_size: int
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        """Encode condition images into the DiT's normalized latent tokens."""

        mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.torch_dtype).view(
            1, 64, 1, 1, 1
        )
        std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.torch_dtype).view(1, 64, 1, 1, 1)
        all_latents = []
        shapes = []
        for image in images:
            pixels = self.image_processor.preprocess(image, width=image.width, height=image.height).unsqueeze(2)
            pixels = pixels.to(device=self.device, dtype=self.torch_dtype)
            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                encoded = self.vae.encode(pixels).latent_dist.mode()
            normalized = (encoded - mean) / std
            latent_height, latent_width = normalized.shape[-2:]
            shapes.append((1, latent_height, latent_width))
            packed = normalized.flatten(2).transpose(1, 2)
            all_latents.append(packed.repeat(batch_size, 1, 1))
        return torch.cat(all_latents, dim=1), shapes

    @with_model_offload(["vae"])
    @torch.inference_mode()
    @with_metrics
    def process(self, latents: torch.Tensor, latent_height: int, latent_width: int) -> list[Image.Image]:
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=latents.dtype).view(1, 64, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=latents.dtype).view(1, 64, 1, 1, 1)
        latents = latents.transpose(1, 2).reshape(latents.shape[0], 64, 1, latent_height, latent_width) * std + mean
        images = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = ((images.float() / 2 + 0.5).clip(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(item) for item in images]
