"""Session-safe VAE stage for the LingBot realtime pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.wan_video_vae import (
    WanVideoVAE,
    WanVideoVAEStreamingDecodeState,
    WanVideoVAEStreamingEncodeState,
)


@dataclass
class _VAEEncodeCacheState:
    """Worker-local condition image and causal encoder cache for one session."""

    condition_image: torch.Tensor | None
    encoder_state: WanVideoVAEStreamingEncodeState = field(default_factory=WanVideoVAEStreamingEncodeState)


class LingBotWorldFastVAEEncodeStage(BaseStage):
    """Run condition encoding in a VAE worker independent from video decoding."""

    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        self.vae: WanVideoVAE = module_manager.fetch_module("wan_video_vae")
        if self.vae is None:
            raise ValueError("LingBot VAE encode stage requires a loaded wan_video_vae module")
        self.model_names = ["vae"]
        self._cache_registry: dict[int, _VAEEncodeCacheState] = {}

    def initialize_cache(self, cache_handle: int, condition_image: torch.Tensor) -> bool:
        """Register the encoder state for one session."""
        if cache_handle in self._cache_registry:
            raise ValueError(f"VAE encode cache handle {cache_handle} is already registered")
        self._cache_registry[cache_handle] = _VAEEncodeCacheState(condition_image=condition_image)
        return True

    @with_model_offload(["vae"])
    def encode_condition_chunk(
        self, cache_handle: int, chunk_index: int, chunk_count: int, chunk_size: int, height: int, width: int
    ) -> torch.Tensor:
        """Encode one condition chunk and return CPU transport features."""
        state = self._cache_registry[cache_handle]
        is_first = chunk_index == 0
        video = torch.zeros(
            (3, 1 + 4 * (chunk_size - 1) if is_first else 4 * chunk_size, height, width),
            device=self.device,
            dtype=self.torch_dtype,
        )
        if is_first:
            if state.condition_image is None:
                raise RuntimeError("The first condition chunk requires the session image tensor")
            video[:, 0] = state.condition_image
        latent = self.vae.cached_encode_withflag(
            video,
            device=self.device,
            is_first_clip=is_first,
            is_last_clip=chunk_index == chunk_count - 1,
            encode_state=state.encoder_state,
        )
        if latent.shape[1] != chunk_size:
            raise RuntimeError(f"VAE condition chunk has {latent.shape[1]} latent frames, expected {chunk_size}")
        mask = torch.zeros((4, chunk_size, latent.shape[2], latent.shape[3]), device=latent.device, dtype=latent.dtype)
        if is_first:
            mask[:, 0] = 1
            state.condition_image = None
        return torch.cat([mask, latent], dim=0).unsqueeze(0).cpu()

    def release_cache(self, cache_handle: int) -> bool:
        """Release encoder state for one session."""
        return self._cache_registry.pop(cache_handle, None) is not None


@dataclass
class _VAEDecodeCacheState:
    """Worker-local causal decoder cache for one session."""

    decoder_state: WanVideoVAEStreamingDecodeState = field(default_factory=WanVideoVAEStreamingDecodeState)


class LingBotWorldFastVAEDecodeStage(BaseStage):
    """Run video decoding in a VAE worker independent from condition encoding."""

    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        self.vae: WanVideoVAE = module_manager.fetch_module("wan_video_vae")
        if self.vae is None:
            raise ValueError("LingBot VAE decode stage requires a loaded wan_video_vae module")
        self.model_names = ["vae"]
        self._cache_registry: dict[int, _VAEDecodeCacheState] = {}

    def initialize_cache(self, cache_handle: int) -> bool:
        """Register the decoder state for one session."""
        if cache_handle in self._cache_registry:
            raise ValueError(f"VAE decode cache handle {cache_handle} is already registered")
        self._cache_registry[cache_handle] = _VAEDecodeCacheState()
        return True

    @with_model_offload(["vae"])
    def decode_chunk(
        self, cache_handle: int, latents: torch.Tensor, is_first_clip: bool, is_last_clip: bool
    ) -> torch.Tensor:
        """Decode one latent chunk and return CPU frame tensors."""
        state = self._cache_registry[cache_handle]
        return self.vae.cached_decode_withflag(
            latents,
            device=self.device,
            is_first_clip=is_first_clip,
            is_last_clip=is_last_clip,
            decode_state=state.decoder_state,
        ).cpu()

    def release_cache(self, cache_handle: int) -> bool:
        """Release decoder state for one session."""
        return self._cache_registry.pop(cache_handle, None) is not None
