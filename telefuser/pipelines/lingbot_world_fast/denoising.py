from __future__ import annotations

from dataclasses import dataclass

import torch

from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.logging import logger


@dataclass(frozen=True)
class LingBotWorldFastTimesteps:
    indices: tuple[int, ...] = (0, 179, 358, 679)
    num_train_timesteps: int = 1000

    def select(self, scheduler: FlowUniPCMultistepScheduler, shift: float) -> torch.Tensor:
        scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        return scheduler.timesteps[list(self.indices)].clone()


class LingBotWorldFastDenoisingStage:
    """Chunk-level denoising for LingBot-World-Fast."""

    def __init__(self, dit_model, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        self.dit = dit_model
        self.torch_dtype = torch_dtype

    @staticmethod
    def _convert_flow_pred_to_x0(
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin((timesteps - timestep.double()).abs())
        sigma_t = sigmas[timestep_id].reshape(-1)
        while sigma_t.ndim < xt.ndim:
            sigma_t = sigma_t.unsqueeze(-1)
        x0 = xt - sigma_t * flow_pred
        return x0.to(original_dtype)

    def denoise_chunk(
        self,
        latent_chunk: torch.Tensor,
        condition_chunk: torch.Tensor,
        prompt_emb: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
        control_chunk: torch.Tensor | None,
        self_kv_cache: list[dict[str, torch.Tensor | int]],
        crossattn_cache: list[dict[str, torch.Tensor | bool]],
        current_start: int,
        max_attention_size: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        current_latent = latent_chunk
        for timestep_idx in range(len(timesteps)):
            schedule_timestep = timesteps[timestep_idx].view(1).to(device=current_latent.device)
            model_timestep = schedule_timestep.to(dtype=torch.float32)
            noise_pred = self.dit(
                x=current_latent.to(dtype=self.torch_dtype),
                timestep=model_timestep,
                context=prompt_emb,
                y=condition_chunk,
                control_tensor=control_chunk,
                kv_cache=self_kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                max_attention_size=max_attention_size,
            )
            x0 = self._convert_flow_pred_to_x0(noise_pred, current_latent, schedule_timestep[0], scheduler)
            if timestep_idx < len(timesteps) - 1:
                next_timestep = timesteps[timestep_idx + 1].view(1).to(device=x0.device)
                noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
                current_latent = scheduler.add_noise(x0, noise, next_timestep)
            else:
                current_latent = x0

        logger.debug("LingBotWorldFast chunk denoised")
        return current_latent
