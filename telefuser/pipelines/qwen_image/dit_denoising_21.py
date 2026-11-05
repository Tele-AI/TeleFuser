"""Qwen-Image 2.1 native DiT denoising stage."""

from __future__ import annotations

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.qwen_image_21_dit import QwenImage21DiT
from telefuser.schedulers.flow_match import FlowMatchScheduler


class DitDenoising21Stage(BaseStage):
    """Denoise packed 64-channel latents with the native Qwen-Image 2.1 DiT."""

    def __init__(
        self, name: str, module_manager: ModuleManager, config: ModelRuntimeConfig, scheduler: FlowMatchScheduler
    ):
        super().__init__(name, config)
        self.dit: QwenImage21DiT = module_manager.fetch_module("dit")
        self.dit.set_attention_config(config.attention_config)
        self.model_names = ["dit"]
        self.scheduler = scheduler

    def _predict(
        self,
        latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        img_shapes: list[list[tuple[int, int, int]]],
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor | None,
        image_mask: torch.Tensor,
        negative_image_mask: torch.Tensor | None,
        cfg_scale: float,
        negative_embeds: torch.Tensor | None,
        negative_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        kwargs = {
            "hidden_states": torch.cat([condition_latents, latents], dim=1)
            if condition_latents is not None
            else latents,
            "img_shapes": img_shapes,
            "timestep": timestep / 1000,
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_mask": prompt_mask,
            "img_mask": image_mask,
            "return_dict": False,
        }
        positive = self.dit(**kwargs)[0][:, -latents.shape[1] :]
        if cfg_scale <= 1 or negative_embeds is None:
            return positive
        kwargs["encoder_hidden_states"] = negative_embeds
        kwargs["encoder_hidden_states_mask"] = negative_mask
        kwargs["img_mask"] = negative_image_mask
        negative = self.dit(**kwargs)[0][:, -latents.shape[1] :]
        return negative + cfg_scale * (positive - negative)

    @with_model_offload(["dit"])
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latents: torch.Tensor,
        condition_latents: torch.Tensor | None,
        img_shapes: list[list[tuple[int, int, int]]],
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor | None,
        image_mask: torch.Tensor,
        negative_image_mask: torch.Tensor | None,
        num_inference_steps: int,
        cfg_scale: float = 1.0,
        negative_embeds: torch.Tensor | None = None,
        negative_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.scheduler.set_timesteps(
            num_inference_steps,
            dynamic_shift_len=latents.shape[1],
            shift_terminal=0.02,
            max_shift=1.15,
        )
        for timestep in tqdm(self.scheduler.timesteps):
            timestep = timestep.expand(latents.shape[0]).to(self.device, dtype=self.torch_dtype)
            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                noise_pred = self._predict(
                    latents,
                    condition_latents,
                    img_shapes,
                    timestep,
                    prompt_embeds,
                    prompt_mask,
                    image_mask,
                    negative_image_mask,
                    cfg_scale,
                    negative_embeds,
                    negative_mask,
                )
            latents = self.scheduler.step(noise_pred, timestep, latents)
        return latents
