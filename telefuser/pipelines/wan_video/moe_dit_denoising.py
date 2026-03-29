from __future__ import annotations

from functools import partial
from typing import Any

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_cfg_rank, get_cfg_world_size
from telefuser.distributed.fsdp import shard_model
from telefuser.distributed.parallel_shard import cfg_parallel_unshard
from telefuser.metrics import with_metrics
from telefuser.ops.quantized_linear import convert_params_to_buffers
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.lora_loader import LoRALoader
from telefuser.utils.profiler import ProfilingContext4Debug
from telefuser.utils.torch_compile import apply_compile_config


class MoeDitDenoisingStage(BaseStage):
    """MoE (Mixture of Experts) DiT denoising stage for Wan2.2 video generation.

    Uses two separate DiT models: high-quality expert (dit_high) for early timesteps
    and low-quality expert (dit_low) for later timesteps. Models switch at a
    boundary timestep (default 0.875) for efficient inference.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        dit_high_runtime_config: ModelRuntimeConfig,
        dit_low_runtime_config: ModelRuntimeConfig,
        scheduler: Any,
    ) -> None:
        super().__init__(name, dit_high_runtime_config)
        self.num_train_timesteps = 1000
        self.dit_high, self.dit_low = module_manager.fetch_module(
            "wan_video_dit",
            index=2,
        )
        self.dit_high.set_attention_config(dit_high_runtime_config.attention_config)
        self.dit_low.set_attention_config(dit_low_runtime_config.attention_config)
        self.model_names = ["dit_high", "dit_low"]
        self.scheduler = scheduler
        self.dit_high_runtime_config = dit_high_runtime_config
        self.dit_low_runtime_config = dit_low_runtime_config
        self.load_loras()
        self.dit_high_onload_flag = False
        self.dit_low_onload_flag = False
        self.dit_high_cpu_offload = (
            dit_high_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD
        )
        self.dit_low_cpu_offload = (
            dit_low_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD
        )
        # Configure CPU offloading for dit_high
        if dit_high_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            if dit_high_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and sequential cpu offload can not be used together")
            else:
                logger.info("enable sequential cpu offload for dit high")
                self.dit_high.enable_sequential_cpu_offload(device=self.device, torch_dtype=self.torch_dtype)
        if dit_high_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            if dit_high_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and async cpu offload can not be used together")
            else:
                logger.info("enable async cpu offload for dit high")
                self.dit_high.enable_async_offload(self.device, dit_high_runtime_config.offload_config)

        # Configure CPU offloading for dit_low
        if dit_low_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            if dit_low_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and sequential cpu offload can not be used together")
            else:
                logger.info("enable sequential cpu offload for dit low")
                self.dit_low.enable_sequential_cpu_offload(device=self.device, torch_dtype=self.torch_dtype)
        if dit_low_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            if dit_low_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and async cpu offload can not be used together")
            else:
                logger.info("enable async cpu offload for dit low")
                self.dit_low.enable_async_offload(self.device, dit_low_runtime_config.offload_config)

        # Handle torch.compile for single GPU mode
        parallel_cfg = dit_high_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and (
            dit_high_runtime_config.compile_config.enabled or dit_low_runtime_config.compile_config.enabled
        ):
            apply_compile_config(dit_high_runtime_config.compile_config)
            if dit_high_runtime_config.compile_config.enabled:
                logger.info("enable torch.compile for dit_high")
                self.dit_high.compile()
            if dit_low_runtime_config.compile_config.enabled:
                logger.info("enable torch.compile for dit_low")
                self.dit_low.compile()

    def load_loras(self):
        """Load LoRA weights into both DiT models."""
        lora_configs_high = self.dit_high_runtime_config.lora_configs
        lora_loader = LoRALoader()
        for lora_config in lora_configs_high:
            lora_path = lora_config.path
            strength = lora_config.strength
            lora_loader.apply_lora(self.dit_high, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength} for dit high")
        lora_configs_low = self.dit_low_runtime_config.lora_configs
        for lora_config in lora_configs_low:
            lora_path = lora_config.path
            strength = lora_config.strength
            lora_loader.apply_lora(self.dit_low, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength} for dit low")

    def predict_noise_with_cfg(
        self,
        dit: Any,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predict noise with classifier-free guidance."""
        if cfg_scale == 1.0:
            return dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                context=prompt_emb_posi,
            )

        # Check if CFGP is enabled
        cfgp_enabled = get_cfg_world_size(dit.device_mesh) > 1

        if cfgp_enabled:
            # CFGP mode: each rank processes one branch (posi or nega)
            cfg_rank = get_cfg_rank(dit.device_mesh)
            is_posi = cfg_rank == 0

            context = prompt_emb_posi if is_posi else prompt_emb_nega
            cond_flag = is_posi

            noise_pred = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=cond_flag,
                context=context,
            )

            # Unshard and compute CFG
            noise_pred = cfg_parallel_unshard(dit.device_mesh, [noise_pred])[0]
            noise_pred_posi = noise_pred[0:1]
            noise_pred_nega = noise_pred[1:2]
        else:
            # Non-CFGP mode: sequential forward passes
            noise_pred_posi = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                context=prompt_emb_posi,
            )
            noise_pred_nega = dit.forward(
                x=latents,
                timestep=timestep,
                cond_flag=False,
                context=prompt_emb_nega,
            )

        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        return noise_pred

    @ProfilingContext4Debug("moe_dit_denosing")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        ref_latent: torch.Tensor | None,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor | None,
        cfg_scale_high: float,
        cfg_scale_low: float,
        sigma_shift: float,
        boundary: float,
    ) -> torch.Tensor:
        """Run denoising with MoE model switching at boundary timestep.

        Uses dit_high for early timesteps (> boundary * 1000) and
        dit_low for later timesteps for efficiency.
        """
        # set mix euler scheduler
        if not isinstance(num_inference_steps, int):
            self.scheduler.set_timesteps(
                num_inference_steps,
                shift=sigma_shift,
                boundary=boundary,
            )
            num_inference_steps = num_inference_steps[0]
        else:
            self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        # Set up feature cache for dit_high from runtime config
        cache_config_dit_high = self.dit_high_runtime_config.feature_cache_config
        self.setup_feature_cache(self.dit_high, cache_config_dit_high, num_inference_steps)

        if self.dit_high_cpu_offload or (not self.dit_high_onload_flag):
            logger.info(f"onload dit high to {self.device}")
            self.dit_high.onload_device(self.device)
            self.dit_high_onload_flag = True
        dit = self.dit_high
        cfg_scale = cfg_scale_high
        for progress_id, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="dit denoise")):
            # Switch to dit_low at boundary timestep
            if timestep < boundary * self.num_train_timesteps and dit is self.dit_high:
                if self.dit_high_cpu_offload:
                    logger.info("offload dit high")
                    self.dit_high.offload_device()
                if self.dit_low_cpu_offload or (not self.dit_low_onload_flag):
                    logger.info(f"onload dit low to {self.device}")
                    self.dit_low.onload_device(self.device)
                    self.dit_low_onload_flag = True
                dit = self.dit_low
                cfg_scale = cfg_scale_low
                # Set up feature cache for dit_low with init_step
                self.setup_feature_cache(
                    self.dit_low,
                    self.dit_low_runtime_config.feature_cache_config,
                    num_inference_steps,
                    init_step=progress_id,
                )
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)
            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                input_latent = input_latent.to(self.torch_dtype)
                noise_pred = self.predict_noise_with_cfg(
                    dit=dit,
                    latents=input_latent,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    cfg_scale=cfg_scale,
                )
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
        if self.dit_low_cpu_offload:
            logger.info("offload dit low")
            self.dit_low.offload_device()
        return latents

    def parallel_models(self):
        """Configure parallel processing for both DiT models."""
        # Configure dit_high parallelism
        parallel_cfg = self.dit_high_runtime_config.parallel_config
        self.dit_high.device_mesh = create_device_mesh_from_config(parallel_cfg)
        self.dit_high.set_attention_config(self.dit_high_runtime_config.attention_config)
        print(f"dit high device mesh: {self.dit_high.device_mesh}")
        # Note: CFGP is handled at stage level, no need to configure on model
        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit_high.enable_usp()
        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name} dit high")
            if self.dit_high.quant_type is not None:
                self.dit_high = convert_params_to_buffers(self.dit_high, self.dit_high.quant_type)
            shard_fn = partial(
                shard_model,
                wrap_module_names=self.dit_high.get_fsdp_module_names(),
                param_dtype=self.dit_high.quant_type if self.dit_high.quant_type is not None else self.torch_dtype,
            )
            self.dit_high = shard_fn(module=self.dit_high, device_id=self.device)
            if self.dit_high_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit_high.cpu()
                current_platform.empty_cache()

        # Configure dit_low parallelism
        parallel_cfg = self.dit_low_runtime_config.parallel_config
        self.dit_low.device_mesh = create_device_mesh_from_config(parallel_cfg)
        self.dit_low.set_attention_config(self.dit_low_runtime_config.attention_config)
        # Note: CFGP is handled at stage level, no need to configure on model
        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit_low.enable_usp()
        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name} dit low")
            if self.dit_low.quant_type is not None:
                self.dit_low = convert_params_to_buffers(self.dit_low, self.dit_low.quant_type)
            shard_fn = partial(
                shard_model,
                wrap_module_names=self.dit_low.get_fsdp_module_names(),
                param_dtype=self.dit_low.quant_type if self.dit_low.quant_type is not None else self.torch_dtype,
            )
            self.dit_low = shard_fn(module=self.dit_low, device_id=self.device)
            if self.dit_low_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit_low.cpu()
                current_platform.empty_cache()

        # Handle torch.compile for distributed mode
        if self.dit_high_runtime_config.compile_config.enabled or self.dit_low_runtime_config.compile_config.enabled:
            apply_compile_config(self.dit_high_runtime_config.compile_config)
            if self.dit_high_runtime_config.compile_config.enabled:
                logger.info("enable torch.compile for dit_high")
                self.dit_high.compile()
            if self.dit_low_runtime_config.compile_config.enabled:
                logger.info("enable torch.compile for dit_low")
                self.dit_low.compile()
