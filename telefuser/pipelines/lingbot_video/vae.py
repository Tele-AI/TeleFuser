"""VAE geometry and normalization helpers for LingBot-Video.

Latent normalization and condition-encoding behavior are adapted from the
Apache-2.0 licensed upstream LingBot-Video implementation.
"""

from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig


def latent_shape(
    num_frames: int, height: int, width: int, *, temporal_factor: int = 4, spatial_factor: int = 8
) -> tuple[int, int, int]:
    """Return ``(latent_frames, latent_height, latent_width)`` for video input."""
    if num_frames < 1 or (num_frames - 1) % temporal_factor:
        raise ValueError("num_frames must satisfy the VAE temporal contract")
    if height % spatial_factor or width % spatial_factor:
        raise ValueError("height and width must be divisible by VAE spatial factor")
    return ((num_frames - 1) // temporal_factor + 1, height // spatial_factor, width // spatial_factor)


def normalize_latent(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply checkpoint-specific latent normalization with channel broadcasting."""
    if latent.ndim < 2:
        raise ValueError("latent must include a channel dimension")
    mean = mean.to(device=latent.device, dtype=latent.dtype).reshape(1, -1, *([1] * (latent.ndim - 2)))
    std = std.to(device=latent.device, dtype=latent.dtype).reshape(1, -1, *([1] * (latent.ndim - 2)))
    if mean.shape[1] != latent.shape[1] or torch.any(std == 0):
        raise ValueError("latent normalization statistics do not match channels")
    return (latent - mean) * std.reciprocal()


def denormalize_latent(latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo checkpoint-specific latent normalization."""
    if latent.ndim < 2:
        raise ValueError("latent must include a channel dimension")
    mean = mean.to(device=latent.device, dtype=latent.dtype).reshape(1, -1, *([1] * (latent.ndim - 2)))
    std = std.to(device=latent.device, dtype=latent.dtype).reshape(1, -1, *([1] * (latent.ndim - 2)))
    if mean.shape[1] != latent.shape[1]:
        raise ValueError("latent normalization statistics do not match channels")
    # Preserve the upstream division-by-inverse operation order. Multiplying
    # by ``std`` is mathematically equivalent but does not have identical FP32
    # rounding before a temporally coupled VAE decode.
    return latent / std.reciprocal() + mean


def first_frame_condition_mask(latent_frames: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a temporal mask selecting only latent frame zero."""
    if latent_frames < 1:
        raise ValueError("latent_frames must be positive")
    mask = torch.zeros(latent_frames, dtype=torch.bool, device=device)
    mask[0] = True
    return mask


class LingBotVideoVAEEncodeStage(BaseStage):
    """Encode RGB video tensors using official LingBot latent normalization."""

    def __init__(self, name: str, vae: torch.nn.Module, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        self.vae = vae
        self.model_names = ["vae"]

    @with_model_offload(["vae"])
    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Encode ``[B,3,F,H,W]`` pixels in [0,1] to normalized diffusion latents."""
        if pixel_values.ndim != 5 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape [B,3,F,H,W]")
        autocast_enabled = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            encoded = self.vae.encode((pixel_values.to(self.device, torch.float32) - 0.5) / 0.5)
        posterior = encoded.latent_dist
        latent = posterior.sample(generator=generator) if generator is not None else posterior.sample()
        mean = torch.tensor(self.vae.config.latents_mean, device=latent.device)
        std = torch.tensor(self.vae.config.latents_std, device=latent.device)
        return normalize_latent(latent.float(), mean, std)


class LingBotVideoVAEDecodeStage(BaseStage):
    """Decode normalized LingBot diffusion latents to RGB video tensors."""

    def __init__(self, name: str, vae: torch.nn.Module, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        self.vae = vae
        self.model_names = ["vae"]

    @with_model_offload(["vae"])
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents and map output from [-1,1] to [0,1]."""
        mean = torch.tensor(self.vae.config.latents_mean, device=self.device)
        std = torch.tensor(self.vae.config.latents_std, device=self.device)
        raw_latent = denormalize_latent(latents.to(self.device, torch.float32), mean, std)
        if raw_latent.ndim == 5:
            raw_latent = raw_latent.contiguous(memory_format=torch.channels_last_3d)
        decoded = self.vae.decode(raw_latent).sample
        return decoded.float().add(1.0).div(2.0).clamp(0.0, 1.0)
