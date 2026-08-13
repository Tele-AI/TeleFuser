"""Causal long-forcing denoising stage for ABot-World."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.abot_world_dit import ABotWorldDiT
from telefuser.schedulers.flow_match import FlowMatchScheduler


class ABotWorldDenoisingStage(BaseStage):
    """Run ABot's published four-step, x0-prediction causal sampler on one GPU."""

    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        dit = module_manager.fetch_module("abot_world_dit")
        if dit is None or not isinstance(dit, ABotWorldDiT):
            raise ValueError("ABot-World requires a loaded abot_world_dit module")
        self.dit = dit
        self.model_names = ["dit"]

    def parallel_models(self) -> None:
        if self.model_runtime_config.parallel_config.world_size != 1:
            raise ValueError("ABot-World initial integration supports exactly one DiT GPU")
        self.dit.set_attention_config(self.model_runtime_config.attention_config)

    def _new_cache(self, batch_size: int, height: int, width: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        frame_tokens = (height // self.dit.patch_size[1]) * (width // self.dit.patch_size[2])
        # The fixed window includes both the retained sink and rolling tail,
        # matching LingBot-World v2: 6 + 12 = 18 latent frames by default.
        kv_size = self.dit.local_attn_size * frame_tokens
        dtype = self.torch_dtype
        self_cache: list[dict[str, Any]] = []
        cross_cache: list[dict[str, Any]] = []
        head_dim = self.dit.dim // self.dit.num_heads
        for _ in range(self.dit.num_layers):
            self_cache.append(
                {
                    "k": torch.zeros(
                        (batch_size, kv_size, self.dit.num_heads, head_dim), dtype=dtype, device=self.device
                    ),
                    "v": torch.zeros(
                        (batch_size, kv_size, self.dit.num_heads, head_dim), dtype=dtype, device=self.device
                    ),
                    "global_end_index": torch.zeros(1, dtype=torch.long, device=self.device),
                    "local_end_index": torch.zeros(1, dtype=torch.long, device=self.device),
                }
            )
            cross_cache.append(
                {
                    "k": torch.zeros(
                        (batch_size, self.dit.text_len, self.dit.num_heads, head_dim), dtype=dtype, device=self.device
                    ),
                    "v": torch.zeros(
                        (batch_size, self.dit.text_len, self.dit.num_heads, head_dim), dtype=dtype, device=self.device
                    ),
                    "is_init": False,
                    "sequence_length": 0,
                }
            )
        return self_cache, cross_cache

    @staticmethod
    def _x0_prediction(
        flow_prediction: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        scheduler: FlowMatchScheduler,
    ) -> torch.Tensor:
        if timestep.shape != (latent.shape[0], latent.shape[2]):
            raise ValueError("ABot timestep must have one value per latent frame")
        flat_timestep = timestep.flatten().float()
        sigma_index = (
            (scheduler.timesteps.to(flat_timestep.device).unsqueeze(0) - flat_timestep.unsqueeze(1)).abs().argmin(dim=1)
        )
        sigma = scheduler.sigmas.to(device=latent.device, dtype=torch.float64)[sigma_index]
        sigma = sigma.view(latent.shape[0], 1, latent.shape[2], 1, 1)
        return (latent.double() - sigma * flow_prediction.double()).to(flow_prediction.dtype)

    @staticmethod
    def _scheduler() -> FlowMatchScheduler:
        scheduler = FlowMatchScheduler(template="Wan")
        # ABot's official wrapper creates the full Wan training schedule, then
        # uses these exact four training timesteps.
        scheduler.set_timesteps(1000, training=True, shift=5.0)
        return scheduler

    @staticmethod
    def _official_denoising_timesteps(scheduler: FlowMatchScheduler) -> torch.Tensor:
        """Return the four warped training times from ABot's published config."""
        # ``warp_denoising_step: true`` indexes the 1,000-step shifted Wan
        # schedule with ``1000 - [1000, 750, 500, 250]``.
        return scheduler.timesteps[torch.tensor((0, 250, 500, 750), dtype=torch.long)]

    def _denoise_block(
        self,
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        first_frame_latent: torch.Tensor | None,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        current_start: int,
        generator: torch.Generator | Sequence[torch.Generator],
        scheduler: FlowMatchScheduler,
    ) -> torch.Tensor:
        current = latent
        batch, _, frames, height, width = current.shape
        replace_first = current_start == 0 and first_frame_latent is not None
        if replace_first:
            current = current.clone()
            current[:, :, :1].copy_(first_frame_latent)
        frame_tokens = (height // self.dit.patch_size[1]) * (width // self.dit.patch_size[2])
        timesteps = self._official_denoising_timesteps(scheduler).to(device=self.device)
        for index, current_timestep in enumerate(timesteps):
            timestep = torch.full((batch, frames), current_timestep, dtype=timesteps.dtype, device=self.device)
            if replace_first:
                timestep[:, 0] = 0
            with torch.autocast(self.device.type, dtype=self.torch_dtype, enabled=self.device.type == "cuda"):
                flow_prediction = self.dit(
                    x=current.to(dtype=self.torch_dtype),
                    timestep=timestep,
                    context=prompt_emb,
                    act_context=action_context,
                    kv_cache=self_cache,
                    crossattn_cache=cross_cache,
                    current_start=current_start * frame_tokens,
                )
            x0 = self._x0_prediction(flow_prediction, current, timestep, scheduler)
            if index < len(timesteps) - 1:
                if isinstance(generator, Sequence):
                    if len(generator) != x0.shape[0]:
                        raise ValueError("ABot batched denoising requires one generator per session")
                    noise = torch.cat(
                        [
                            torch.randn(
                                (1, *x0.shape[1:]),
                                generator=item_generator,
                                dtype=x0.dtype,
                                device=self.device,
                            )
                            for item_generator in generator
                        ],
                        dim=0,
                    )
                else:
                    noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=self.device)
                current = scheduler.add_noise(x0, noise, timesteps[index + 1])
            else:
                current = x0
            if replace_first:
                current[:, :, :1].copy_(first_frame_latent)
        context_timestep = torch.zeros_like(timestep)
        self.dit(
            x=current.to(dtype=self.torch_dtype),
            timestep=context_timestep,
            context=prompt_emb,
            act_context=action_context,
            kv_cache=self_cache,
            crossattn_cache=cross_cache,
            current_start=current_start * frame_tokens,
        )
        return current

    @with_model_offload(["dit"])
    @torch.inference_mode()
    def process(
        self,
        noise: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        first_frame_latent: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """Generate a ``1 mod 3`` latent-frame video from a starting image."""
        if noise.ndim != 5 or noise.shape[2] < 1 or (noise.shape[2] - 1) % 3:
            raise ValueError("ABot latent frame count must be positive and equal to 1 mod 3")
        if action_context.shape[:3] != (noise.shape[0], 32, noise.shape[2]):
            raise ValueError("ABot action context must be [batch, 32, latent_frames, height, width]")
        if first_frame_latent.shape != noise[:, :, :1].shape:
            raise ValueError("ABot starting-image latent must be [batch, 48, 1, latent_height, latent_width]")
        self.dit.set_causal_attention_window(self.dit.local_attn_size, self.dit.sink_size)
        self_cache, cross_cache = self._new_cache(noise.shape[0], noise.shape[-2], noise.shape[-1])
        scheduler = self._scheduler()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output = []
        for start in range(0, noise.shape[2], 3):
            frames = 1 if start == 0 else 3
            block = self._denoise_block(
                noise[:, :, start : start + frames].to(device=self.device, dtype=self.torch_dtype),
                prompt_emb.to(device=self.device, dtype=self.torch_dtype),
                action_context[:, :, start : start + frames].to(device=self.device, dtype=self.torch_dtype),
                first_frame_latent.to(device=self.device, dtype=self.torch_dtype) if start == 0 else None,
                self_cache,
                cross_cache,
                start,
                generator,
                scheduler,
            )
            output.append(block)
        return torch.cat(output, dim=2)
