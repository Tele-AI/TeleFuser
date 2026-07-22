"""Reference LingBot-Video denoising loop primitives.

Sampling order and TI2V condition behavior are adapted from the Apache-2.0
licensed upstream LingBot-Video implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_ulysses_group
from telefuser.distributed.fsdp import shard_model
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger


def reinject_ti2v_condition(latent: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace the masked temporal prefix with the clean VAE condition latent."""
    if latent.ndim != 5 or condition.ndim != 5 or mask.ndim not in {1, latent.ndim}:
        raise ValueError("latent, condition, and mask shapes are incompatible")
    if latent.shape[:2] != condition.shape[:2] or latent.shape[3:] != condition.shape[3:]:
        raise ValueError("condition must match latent batch, channel, and spatial dimensions")
    if condition.shape[2] > latent.shape[2]:
        raise ValueError("condition temporal prefix is longer than the latent")
    if condition.shape != latent.shape:
        if mask.ndim != 1 or not bool(mask[: condition.shape[2]].all()) or bool(mask[condition.shape[2] :].any()):
            raise ValueError("prefix condition requires a matching prefix-only temporal mask")
        output = latent.clone()
        output[:, :, : condition.shape[2]] = condition.to(dtype=latent.dtype)
        return output
    expanded = mask.to(device=latent.device, dtype=torch.bool)
    if expanded.ndim == 1 and latent.ndim >= 3:
        expanded = expanded.reshape(1, 1, -1, *([1] * (latent.ndim - 3)))
    return torch.where(expanded, condition, latent)


def transformer_timestep(timestep: torch.Tensor, transformer_dtype: torch.dtype) -> torch.Tensor:
    """Match the upstream BF16/FP16 sigma rounding before DiT time embedding."""
    sigma = timestep.float() / 1000.0
    if transformer_dtype in {torch.bfloat16, torch.float16}:
        sigma = sigma.to(transformer_dtype)
    return (sigma * 1000.0).float()


def denoise_lingbot_video(
    latent: torch.Tensor,
    timesteps: torch.Tensor,
    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    condition: torch.Tensor | None = None,
    condition_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a scheduler-agnostic denoising loop with optional TI2V reinjection."""
    current = latent
    for timestep in timesteps:
        if condition is not None:
            if condition_mask is None:
                raise ValueError("condition_mask is required when condition is provided")
            current = reinject_ti2v_condition(current, condition, condition_mask)
        prediction = predict(current, timestep)
        current = step(prediction, timestep, current)
    if condition is not None and condition_mask is not None:
        current = reinject_ti2v_condition(current, condition, condition_mask)
    return current


class LingBotVideoDenoisingStage(BaseStage):
    """Dense LingBot-Video denoising stage with source-equivalent CFG order."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        *,
        batch_cfg: bool = False,
    ) -> None:
        super().__init__(name, model_runtime_config)
        transformer = module_manager.fetch_module("transformer")
        self.transformer = transformer
        set_attention_config = getattr(transformer, "set_attention_config", None)
        if callable(set_attention_config):
            set_attention_config(model_runtime_config.attention_config)
        self.model_names = ["transformer"]
        self.batch_cfg = batch_cfg

    def parallel_models(self) -> None:
        """Attach Ulysses SP and optional per-block FSDP to the DiT stage."""
        parallel_config = self.model_runtime_config.parallel_config
        if parallel_config.world_size == 1:
            return
        self.transformer.device_mesh = create_device_mesh_from_config(parallel_config)
        if parallel_config.sp_ulysses_degree > 1:
            self.transformer.set_ulysses_group(get_ulysses_group(self.transformer.device_mesh))
            logger.info(f"enabled LingBot Ulysses SP degree={parallel_config.sp_ulysses_degree}")
        if parallel_config.enable_fsdp:
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                raise ValueError("LingBot FSDP inference cannot be combined with model CPU offload")
            # ParallelWorker receives a CPU checkpoint. Move all state first so
            # intentionally unsharded FP32 parameters do not remain on CPU.
            self.transformer.to(self.device)
            ignored_states = [
                parameter for parameter in self.transformer.parameters() if parameter.dtype != self.torch_dtype
            ]
            if ignored_states:
                logger.info(f"retaining {len(ignored_states)} LingBot parameters with a non-runtime dtype outside FSDP")
            logger.info(f"enabled LingBot block FSDP for {self.name}")
            self.transformer = shard_model(
                module=self.transformer,
                device_id=self.device.index if self.device.index is not None else torch.cuda.current_device(),
                wrap_module_names=self.transformer.get_fsdp_module_names(),
                ignored_states=ignored_states,
            )
            self.onload_models_flag = True
            current_platform.empty_cache()

    @with_model_offload(["transformer"])
    @torch.no_grad()
    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None = None,
        positive_attention_mask: torch.Tensor | None = None,
        negative_attention_mask: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run positive then negative CFG under transformer compute autocast."""
        transformer_dtype = self.transformer.patch_embedder.weight.dtype
        model_timestep = transformer_timestep(timestep, transformer_dtype)
        autocast_enabled = latents.device.type == "cuda" and transformer_dtype in {torch.bfloat16, torch.float16}
        with torch.autocast(device_type=latents.device.type, dtype=transformer_dtype, enabled=autocast_enabled):
            if guidance_scale == 1.0:
                return self.transformer(
                    latents, model_timestep, positive_prompt_embeds, positive_attention_mask
                ).float()
            if negative_prompt_embeds is None:
                raise ValueError("negative_prompt_embeds is required when guidance_scale is not 1")
            matching_masks = (positive_attention_mask is None) == (negative_attention_mask is None)
            matching_shapes = positive_prompt_embeds.shape == negative_prompt_embeds.shape
            if self.batch_cfg and matching_masks and matching_shapes:
                combined_latents = torch.cat((latents, latents))
                combined_timestep = torch.cat((model_timestep, model_timestep))
                combined_embeds = torch.cat((positive_prompt_embeds, negative_prompt_embeds))
                combined_mask = (
                    None
                    if positive_attention_mask is None
                    else torch.cat((positive_attention_mask, negative_attention_mask))
                )
                positive, negative = (
                    self.transformer(combined_latents, combined_timestep, combined_embeds, combined_mask)
                    .float()
                    .chunk(2)
                )
            else:
                positive = self.transformer(
                    latents, model_timestep, positive_prompt_embeds, positive_attention_mask
                ).float()
                negative = self.transformer(
                    latents, model_timestep, negative_prompt_embeds, negative_attention_mask
                ).float()
        return negative + guidance_scale * (positive - negative)
