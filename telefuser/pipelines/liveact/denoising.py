"""LiveAct denoising stage with audio cross-attention and KV cache.

Handles the diffusion denoising process with:
- Audio cross-attention for talking head generation
- KV cache management for streaming generation
- Memory compression for efficient long video generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


@dataclass
class KVCacheConfig:
    """Configuration for KV cache management.

    Memory requirements for 480x832 video:
    - bfloat16 on GPU: ~200 GB
    - fp8 on GPU: ~100 GB
    - bfloat16 on CPU (offload): ~200 GB CPU RAM, minimal GPU
    - fp8 on CPU: ~100 GB CPU RAM, minimal GPU

    Recommended for single GPU: offload_cache=True
    """

    enabled: bool = True
    fp8_kv_cache: bool = False
    offload_cache: bool = True  # Default: offload to CPU for single GPU
    mean_memory: bool = False


# Pre-defined timesteps matching original generate.py (created once)
TIMESTEPS = [1000.0, 937.5, 833.33333333, 0.0]


class LiveActDenoisingStage(BaseStage):
    """Denoising stage for LiveAct with audio cross-attention.

    Supports:
    - Streaming video generation with KV cache
    - Audio CFG (classifier-free guidance for audio)
    - Memory-efficient KV cache with optional FP8 quantization
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(name, model_runtime_config)
        self.dit = module_manager.fetch_module("liveact_dit")
        self.dit.set_attention_config(model_runtime_config.attention_config)
        self.model_names = ["dit"]

        # Get model architecture params from dit
        self.num_layers = len(self.dit.blocks)
        self.num_heads = self.dit.num_heads
        self.head_dim = self.dit.dim // self.dit.num_heads

        self.scheduler = FlowMatchScheduler(template="Wan")

        self.kv_cache: dict[int, dict[int, dict[str, Any]]] = {}
        self.kv_cache_config = kv_cache_config or KVCacheConfig()

        self._timestep_tensors: list[torch.Tensor] | None = None
        self._kv_cache_frame_len: int | None = None
        self._kv_cache_null_audio: dict | None = None

    def set_kv_cache_config(self, config: KVCacheConfig) -> None:
        """Set KV cache configuration."""
        self.kv_cache_config = config

    def _get_timestep_tensors(self) -> list[torch.Tensor]:
        """Get or create timestep tensors on device.

        Matches original generate.py which creates timesteps once before all iterations.
        """
        if self._timestep_tensors is None or self._timestep_tensors[0].device != self.device:
            self._timestep_tensors = [torch.tensor([t], device=self.device, dtype=torch.float32) for t in TIMESTEPS]
        return self._timestep_tensors

    def init_kv_cache(
        self,
        frame_len: int,
        num_timesteps: int = 3,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        config: KVCacheConfig | None = None,
    ) -> dict:
        """Initialize KV cache for streaming generation.

        Args:
            frame_len: Number of tokens per frame (H/patch * W/patch)
            num_timesteps: Number of denoising timesteps
            device: Device for KV cache
            dtype: Data type for KV cache
            config: KV cache configuration

        Returns:
            Initialized KV cache dictionary
        """
        if config is not None:
            self.kv_cache_config = config

        kv_cache_dtype = torch.float8_e4m3fn if self.kv_cache_config.fp8_kv_cache else dtype
        kv_cache_device = "cpu" if self.kv_cache_config.offload_cache else device

        # Total KV cache tokens: 6 + 8 = 14 frames worth
        # First iteration: 6 frames, subsequent: 8 frames
        kv_cache_tokens = frame_len * 14
        kv_scale_shape = (1, kv_cache_tokens, self.num_heads, 1)

        kv_cache = {}
        for t_idx in range(num_timesteps):
            kv_cache[t_idx] = {}
            for layer_id in range(self.num_layers):
                kv_cache[t_idx][layer_id] = {
                    "k": torch.zeros(
                        [1, kv_cache_tokens, self.num_heads, self.head_dim],
                        dtype=kv_cache_dtype,
                        device=kv_cache_device,
                    ),
                    "v": torch.zeros(
                        [1, kv_cache_tokens, self.num_heads, self.head_dim],
                        dtype=kv_cache_dtype,
                        device=kv_cache_device,
                    ),
                    "k_scale": (
                        torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device)
                        if self.kv_cache_config.fp8_kv_cache
                        else None
                    ),
                    "v_scale": (
                        torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_cache_device)
                        if self.kv_cache_config.fp8_kv_cache
                        else None
                    ),
                    "mean_memory": self.kv_cache_config.mean_memory,
                    "offload_cache": self.kv_cache_config.offload_cache,
                    "fp8_kv_cache": self.kv_cache_config.fp8_kv_cache,
                }

        self.kv_cache = kv_cache
        return kv_cache

    def clear_kv_cache(self) -> None:
        """Clear KV cache and related state."""
        self.kv_cache = {}
        self._kv_cache_frame_len = None
        self._kv_cache_null_audio = None

    def predict_noise(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor,
        audio_embedding: torch.Tensor,
        y: torch.Tensor,
        ref_target_masks: torch.Tensor | None,
        kv_cache: dict,
        start_idx: int,
        end_idx: int,
        update_cache: bool = False,
        skip_audio: bool = False,
    ) -> torch.Tensor:
        """Predict noise with audio cross-attention.

        Args:
            latent: Current latent [B, C, T, H, W]
            timestep: Current timestep
            context: Text embedding
            clip_fea: CLIP visual features
            audio_embedding: Audio embedding for cross-attention
            y: VAE latent with mask
            ref_target_masks: Reference target masks
            kv_cache: KV cache dictionary
            start_idx: Start index for KV cache
            end_idx: End index for KV cache
            update_cache: Whether to update KV cache
            skip_audio: Whether to skip audio cross-attention

        Returns:
            Predicted noise
        """
        return self.dit(
            [latent],
            t=timestep,
            context=context,
            clip_fea=clip_fea,
            audio=audio_embedding,
            y=y,
            ref_target_masks=ref_target_masks,
            kv_cache=kv_cache,
            start_idx=start_idx,
            end_idx=end_idx,
            update_cache=update_cache,
            skip_audio=skip_audio,
        )[0]

    def predict_noise_with_audio_cfg(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor,
        audio_embedding: torch.Tensor,
        y: torch.Tensor,
        ref_target_masks: torch.Tensor | None,
        kv_cache: dict,
        kv_cache_null_audio: dict | None,
        start_idx: int,
        end_idx: int,
        update_cache: bool,
        audio_cfg: float = 1.0,
        step_idx: int = 0,
    ) -> torch.Tensor:
        """Predict noise with audio CFG.

        Args:
            latent: Current latent
            timestep: Current timestep
            context: Text embedding
            clip_fea: CLIP visual features
            audio_embedding: Audio embedding
            y: VAE latent with mask
            ref_target_masks: Reference target masks
            kv_cache: KV cache for positive audio
            kv_cache_null_audio: KV cache for null audio (CFG)
            start_idx: Start index for KV cache
            end_idx: End index for KV cache
            update_cache: Whether to update KV cache
            audio_cfg: Audio CFG scale
            step_idx: Current step index

        Returns:
            Predicted noise with audio CFG applied
        """
        # Run with audio
        skip_audio = False
        noise_pred = self.predict_noise(
            latent,
            timestep,
            context,
            clip_fea,
            audio_embedding,
            y,
            ref_target_masks,
            kv_cache,
            start_idx,
            end_idx,
            update_cache=update_cache,
            skip_audio=skip_audio,
        )

        # Apply audio CFG if enabled and in steps 1, 2
        if audio_cfg > 1.0 and step_idx in [1, 2] and kv_cache_null_audio is not None:
            # Run with null audio
            null_audio = torch.zeros_like(audio_embedding)
            noise_pred_drop_audio = self.predict_noise(
                latent,
                timestep,
                context,
                clip_fea,
                null_audio,
                y,
                ref_target_masks,
                kv_cache_null_audio,
                start_idx,
                end_idx,
                update_cache=update_cache,
                skip_audio=False,
            )
            # Apply CFG
            noise_pred = noise_pred_drop_audio + audio_cfg * (noise_pred - noise_pred_drop_audio)

        return noise_pred

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("liveact_denoise")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor,
        audio_embedding: torch.Tensor,
        y: torch.Tensor,
        ref_target_masks: torch.Tensor | None,
        height: int,
        width: int,
        audio_cfg: float = 1.0,
        num_inference_steps: int = 3,
        seed: int | None = None,
        iteration: int = 0,  # Outer iteration for update_cache logic
    ) -> torch.Tensor:
        """Run denoising with audio cross-attention and KV cache.

        Args:
            latent: Initial latent [16, blksz, H', W']
            context: Text embedding
            clip_fea: CLIP visual features
            audio_embedding: Audio embedding
            y: VAE latent with mask [1, 17, T', H', W']
            ref_target_masks: Reference target masks
            height: Video height
            width: Video width
            audio_cfg: Audio CFG scale
            num_inference_steps: Number of denoising steps
            seed: Random seed
            iteration: Outer iteration index (for update_cache: True if iteration > 1)

        Returns:
            Denoised latent
        """
        blksz_lst = [6, 8]

        latent_blksz = latent.shape[1]
        f = 0 if latent_blksz == blksz_lst[0] else 1

        vae_stride = (4, 8, 8)
        patch_size = (1, 2, 2)
        frame_len = (height // (patch_size[1] * vae_stride[1])) * (width // (patch_size[2] * vae_stride[2]))

        if self._kv_cache_frame_len != frame_len:
            self.init_kv_cache(
                frame_len,
                num_timesteps=num_inference_steps,
                device=self.device,
                dtype=self.torch_dtype,
            )
            self._kv_cache_frame_len = frame_len

        kv_cache_null_audio = None
        if audio_cfg > 1.0 and self._kv_cache_null_audio is None:
            self._kv_cache_null_audio = self.init_kv_cache(
                frame_len,
                num_timesteps=num_inference_steps,
                device=self.device,
                dtype=self.torch_dtype,
            )
            kv_cache_null_audio = self._kv_cache_null_audio
        elif audio_cfg > 1.0:
            kv_cache_null_audio = self._kv_cache_null_audio

        if seed is not None:
            torch.manual_seed(seed)

        y_slice = y[:, :, sum(blksz_lst[:f]) : sum(blksz_lst[: f + 1]), ...]

        update_cache = iteration > 1

        timestep_tensors = self._get_timestep_tensors()

        for i in tqdm(range(len(TIMESTEPS) - 1), desc="liveact denoise"):
            timestep = timestep_tensors[i]

            start_idx = sum(blksz_lst[:f]) * frame_len
            end_idx = sum(blksz_lst[: f + 1]) * frame_len

            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                noise_pred = self.predict_noise_with_audio_cfg(
                    latent=latent,
                    timestep=timestep,
                    context=context,
                    clip_fea=clip_fea,
                    audio_embedding=audio_embedding,
                    y=y_slice,
                    ref_target_masks=ref_target_masks,
                    kv_cache=self.kv_cache[i],
                    kv_cache_null_audio=kv_cache_null_audio[i] if kv_cache_null_audio else None,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    update_cache=update_cache,  # Based on outer iteration
                    audio_cfg=audio_cfg,
                    step_idx=i,
                )

            dt = (TIMESTEPS[i] - TIMESTEPS[i + 1]) / 1000
            latent = latent + (-noise_pred) * dt

        return latent
