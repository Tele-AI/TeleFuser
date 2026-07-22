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
from telefuser.distributed.fsdp import shard_model_fsdp2_inference
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


def _batch_cfg_prompt_inputs(
    positive_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    positive_mask: torch.Tensor | None,
    negative_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad positive and negative prompt conditions for one batched CFG forward."""
    if positive_embeds.ndim != 3 or negative_embeds.ndim != 3:
        raise ValueError("CFG prompt embeddings must have shape [batch, sequence, hidden_size]")
    if positive_embeds.shape[0] != negative_embeds.shape[0] or positive_embeds.shape[2] != negative_embeds.shape[2]:
        raise ValueError("positive and negative CFG embeddings must have matching batch and hidden dimensions")

    def pad(
        embeds: torch.Tensor,
        mask: torch.Tensor | None,
        target_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
        if mask.shape != embeds.shape[:2]:
            raise ValueError("CFG prompt attention mask must match the embedding batch and sequence dimensions")
        padding = target_length - embeds.shape[1]
        if padding:
            embeds = torch.nn.functional.pad(embeds, (0, 0, 0, padding))
            mask = torch.nn.functional.pad(mask, (0, padding), value=False)
        return embeds, mask

    target_length = max(positive_embeds.shape[1], negative_embeds.shape[1])
    positive_embeds, positive_mask = pad(positive_embeds, positive_mask, target_length)
    negative_embeds, negative_mask = pad(negative_embeds, negative_mask, target_length)
    return torch.cat((positive_embeds, negative_embeds)), torch.cat((positive_mask, negative_mask))


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
        self.scheduler = module_manager.fetch_module("scheduler")
        set_attention_config = getattr(transformer, "set_attention_config", None)
        if callable(set_attention_config):
            set_attention_config(model_runtime_config.attention_config)
        self.model_names = ["transformer"]
        self.batch_cfg = batch_cfg
        # Denoising repeatedly invokes the same FSDP graph. Releasing the CUDA
        # allocator cache after every step forces costly weight-buffer
        # reallocations that the source runner keeps cached for the full loop.
        self.empty_cache_after_call = False

    def parallel_models(self) -> None:
        """Attach Ulysses SP and optional per-block FSDP to the DiT stage."""
        if self.device.type == "cuda":
            # Match the source runner's FP32 matmul policy. This primarily
            # affects the MoE router projections, which intentionally run in
            # FP32 and are otherwise much slower on Ampere-and-newer GPUs.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

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
            ignored_states = [
                parameter for parameter in self.transformer.parameters() if parameter.dtype != self.torch_dtype
            ]
            if ignored_states:
                logger.info(f"retaining {len(ignored_states)} LingBot parameters with a non-runtime dtype outside FSDP")
            logger.info(f"enabled LingBot block FSDP2 for {self.name}")
            self.transformer = shard_model_fsdp2_inference(
                module=self.transformer,
                device_mesh=self.transformer.device_mesh,
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
        return self._predict_noise_with_cfg(
            latents,
            timestep,
            positive_prompt_embeds,
            negative_prompt_embeds,
            positive_attention_mask,
            negative_attention_mask,
            guidance_scale,
        )

    def _predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        positive_attention_mask: torch.Tensor | None,
        negative_attention_mask: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run CFG without crossing the stage lifecycle boundary."""
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
            if self.batch_cfg:
                combined_latents = torch.cat((latents, latents))
                combined_timestep = torch.cat((model_timestep, model_timestep))
                combined_embeds, combined_mask = _batch_cfg_prompt_inputs(
                    positive_prompt_embeds,
                    negative_prompt_embeds,
                    positive_attention_mask,
                    negative_attention_mask,
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

    @with_model_offload(["transformer"])
    @torch.no_grad()
    def denoise(
        self,
        latent: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        positive_attention_mask: torch.Tensor | None,
        negative_attention_mask: torch.Tensor | None,
        guidance_scale: float,
        num_inference_steps: int,
        shift: float,
        condition: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the complete sampling loop in one worker invocation."""
        self.scheduler.set_timesteps(num_inference_steps, device=latent.device, shift=shift)
        for timestep in self.scheduler.timesteps:
            if condition is not None:
                if condition_mask is None:
                    raise ValueError("condition_mask is required when condition is provided")
                latent = reinject_ti2v_condition(latent, condition, condition_mask)
            prediction = self._predict_noise_with_cfg(
                latent,
                timestep.expand(latent.shape[0]),
                positive_prompt_embeds,
                negative_prompt_embeds,
                positive_attention_mask,
                negative_attention_mask,
                guidance_scale,
            )
            latent = self.scheduler.step(prediction, timestep, latent)
        if condition is not None and condition_mask is not None:
            latent = reinject_ti2v_condition(latent, condition, condition_mask)
        return latent
