from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.distributed.fsdp import shard_model
from telefuser.metrics import with_metrics
from telefuser.models.longcat_video_dit import LongCatVideoTransformer3DModel
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.lora_loader import LoRALoader
from telefuser.utils.profiler import ProfilingContext4Debug
from telefuser.utils.torch_compile import set_compile_configs


class LongCatDitDenoisingStage(BaseStage):
    """DiT denoising stage for LongCat video generation with CFG and KV-cache support."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        scheduler: Any,
    ):
        super().__init__(name, model_runtime_config)
        self.dit: LongCatVideoTransformer3DModel = module_manager.fetch_module("wan_video_dit")
        self.load_loras()
        self.model_names = ["dit"]
        self.batch_cfg = False
        self.scheduler = scheduler
        self.num_timesteps = 1000
        self.num_distill_sample_steps = 50
        self._pending_async_offload = False

        if model_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            if model_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and sequential cpu offload can not be used together")
            else:
                logger.info("enable sequential cpu offload for longcat dit")
                self.dit.enable_sequential_cpu_offload(self.device, self.torch_dtype)

        if model_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            if model_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and async cpu offload can not be used together")
            else:
                logger.info("enable async cpu offload for dit")
                self._pending_async_offload = True

        # Handle torch.compile - only compile in __init__ if single GPU mode
        parallel_cfg = model_runtime_config.parallel_config
        if model_runtime_config.compile and parallel_cfg.world_size == 1:
            set_compile_configs(descent_tuning=True, compute_comm_overlap=False)
            logger.info("enable torch.compile for longcat dit (single GPU mode)")
            self.dit.compile()

    def load_loras(self):
        lora_configs = self.model_runtime_config.lora_configs
        lora_loader = LoRALoader()
        for lora_config in lora_configs:
            lora_path = lora_config.path
            strength = lora_config.strength
            lora_loader.apply_lora(self.dit, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength} for longcat dit")

    def optimized_scale(self, positive_flat: torch.Tensor, negative_flat: torch.Tensor):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        st_star = dot_product / squared_norm
        return st_star

    def get_timesteps_sigmas(self, sampling_steps: int, use_distill: bool = False):
        if use_distill:
            distill_indices = torch.arange(1, self.num_distill_sample_steps + 1, dtype=torch.float32)
            distill_indices = (distill_indices * (self.num_timesteps // self.num_distill_sample_steps)).round().long()

            inference_indices = np.linspace(0, self.num_distill_sample_steps, num=sampling_steps, endpoint=False)
            inference_indices = np.floor(inference_indices).astype(np.int64)

            sigmas = torch.flip(distill_indices, [0])[inference_indices].float() / self.num_timesteps
        else:
            sigmas = torch.linspace(1, 0.001, sampling_steps)
        sigmas = sigmas.to(torch.float32)
        return sigmas

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        attn_impl: str,
        cfg_scale: float = 1.0,
        use_cfg_zero_star: bool = False,
        num_cond_latents: int = 0,
    ):
        if cfg_scale == 1.0:
            return self.dit.forward(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                attn_impl=attn_impl,
                num_cond_latents=num_cond_latents,
            )

        if not self.batch_cfg:
            noise_pred_posi = self.dit.forward(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                attn_impl=attn_impl,
                num_cond_latents=num_cond_latents,
            )
            noise_pred_nega = self.dit.forward(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                encoder_attention_mask=negative_prompt_attention_mask,
                attn_impl=attn_impl,
                num_cond_latents=num_cond_latents,
            )
        else:
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([prompt_attention_mask, negative_prompt_attention_mask], dim=0)
            latents = torch.cat([latents, latents], dim=0)
            timestep = torch.cat([timestep, timestep], dim=0)

            noise_pred_posi, noise_pred_nega = self.dit.forward(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                attn_impl=attn_impl,
                num_cond_latents=num_cond_latents,
            )
            noise_pred_posi = noise_pred_posi.unsqueeze(0)
            noise_pred_nega = noise_pred_nega.unsqueeze(0)

        if use_cfg_zero_star:
            positive_flat = noise_pred_posi.view(1, -1)
            negative_flat = noise_pred_nega.view(1, -1)
            alpha = self.optimized_scale(positive_flat, negative_flat)
            alpha = alpha.view(1, *([1] * (len(noise_pred_posi.shape) - 1)))
            alpha = alpha.to(noise_pred_posi.dtype)
            noise_pred = noise_pred_nega * alpha + cfg_scale * (noise_pred_posi - noise_pred_nega * alpha)
        else:
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        return noise_pred

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("dit_denoise")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        num_cond_latents: int,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_attention_mask: torch.Tensor | None,
        cfg_scale: float,
        sigma_shift: float,
        use_kv_cache: bool = False,
        use_cfg_zero_star: bool = True,
        attn_impl: str = "sdpa",
        use_distill: bool = False,
        enhance_hf: bool = False,
        progress_bar_cmd: Callable = tqdm,
    ):
        if self._pending_async_offload:
            self.dit.enable_async_offload(self.device, self.model_runtime_config.offload_config)
            self._pending_async_offload = False
        sigmas = self.get_timesteps_sigmas(num_inference_steps, use_distill=use_distill)
        if hasattr(self.scheduler, "config") and hasattr(self.scheduler.config, "shift"):
            self.scheduler.config.shift = sigma_shift

        self.scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=self.device)
        if enhance_hf:
            timesteps = self.scheduler.timesteps
            tail_uniform_start = 500
            tail_uniform_end = 0
            num_tail_uniform_steps = 10
            timesteps_uniform_tail = list(
                np.linspace(
                    tail_uniform_start,
                    tail_uniform_end,
                    num_tail_uniform_steps,
                    dtype=np.float32,
                    endpoint=(tail_uniform_end != 0),
                )
            )
            timesteps_uniform_tail = [torch.tensor(t, device=self.device).unsqueeze(0) for t in timesteps_uniform_tail]
            filtered_timesteps = [timestep.unsqueeze(0) for timestep in timesteps if timestep > tail_uniform_start]
            timesteps = torch.cat(filtered_timesteps + timesteps_uniform_tail)
            self.scheduler.timesteps = timesteps
            self.scheduler.sigmas = torch.cat([timesteps / 1000, torch.zeros(1, device=timesteps.device)])

        if use_kv_cache and num_cond_latents > 0:
            cond_latents = latents[:, :, :num_cond_latents]
            if self.dit.use_cfgp:
                cond_latents = torch.cat([cond_latents, cond_latents], dim=0)
            # Pass self.dit as fsdp_forward_fn so that under FSDP, the forward call goes
            # through the FSDP wrapper (triggering allgather) instead of the inner module
            self.dit.cache_clean_latents(cond_latents=cond_latents, offload_kv_cache=False, fsdp_forward_fn=self.dit)
            latents = latents[:, :, num_cond_latents:]

        for _, t in enumerate(progress_bar_cmd(self.scheduler.timesteps, desc="longcat denoise")):
            timestep = t.expand(latents.shape[0]).to(self.torch_dtype)
            timestep = timestep.unsqueeze(-1).repeat(1, latents.shape[2])

            if not use_kv_cache:
                timestep[:, :num_cond_latents] = 0

            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                latents = latents.to(self.torch_dtype)
                noise_pred = self.predict_noise_with_cfg(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_attention_mask=negative_prompt_attention_mask,
                    attn_impl=attn_impl,
                    cfg_scale=cfg_scale,
                    use_cfg_zero_star=use_cfg_zero_star,
                    num_cond_latents=num_cond_latents,
                ).to(latents.device)

            noise_pred = -noise_pred

            # Scheduler step
            if use_kv_cache:
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            else:
                latents[:, :, num_cond_latents:] = self.scheduler.step(
                    noise_pred[:, :, num_cond_latents:],
                    t,
                    latents[:, :, num_cond_latents:],
                    return_dict=False,
                )[0]

        # clear KV Cache
        self.dit.clear_cache()

        return latents

    def parallel_models(self):
        parallel_cfg = self.model_runtime_config.parallel_config
        self.dit.device_mesh = create_device_mesh_from_config(parallel_cfg)

        if parallel_cfg.cfg_degree > 1:
            self.batch_cfg = True
            self.dit.enable_cfgp()
            logger.info("longcat dit enabled cfgp")

        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit.enable_usp()
            logger.info("longcat dit enabled usp")

        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name}")
            if hasattr(self.dit, "quant_type") and self.dit.quant_type is not None:
                from telefuser.ops.quantized_linear import convert_params_to_buffers

                self.dit = convert_params_to_buffers(self.dit, self.dit.quant_type)
            shard_fn = partial(shard_model, wrap_module_names=self.dit.get_fsdp_module_names())
            self.dit = shard_fn(module=self.dit, device_id=self.device)
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit.cpu()
                current_platform.empty_cache()

        # Handle torch.compile after parallel setup
        if self.model_runtime_config.compile and parallel_cfg.world_size > 1:
            set_compile_configs(descent_tuning=True, compute_comm_overlap=True)
            logger.info("enable torch.compile for longcat dit (parallel mode)")
            self.dit.compile()
