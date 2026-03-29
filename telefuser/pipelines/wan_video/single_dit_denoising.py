from __future__ import annotations

from functools import partial
from typing import Any

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_cfg_rank, get_cfg_world_size
from telefuser.distributed.fsdp import shard_model
from telefuser.distributed.parallel_shard import cfg_parallel_unshard
from telefuser.metrics import with_metrics
from telefuser.models.wan_video_dit import WanModel
from telefuser.ops.quantized_linear import convert_params_to_buffers
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.lora_loader import LoRALoader
from telefuser.utils.profiler import ProfilingContext4Debug
from telefuser.utils.torch_compile import apply_compile_config


class SingleDitDenoisingStage(BaseStage):
    """Single DiT denoising stage for Wan2.1 video generation.

    Supports classifier-free guidance, sparse attention (radial attention),
    AdaTaylorCache for efficient inference, and various LoRA weights.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        scheduler: Any,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.dit: WanModel = module_manager.fetch_module("wan_video_dit")
        self.dit.set_attention_config(model_runtime_config.attention_config)
        self.load_loras()
        self.model_names = ["dit"]
        self.scheduler = scheduler
        if model_runtime_config.offload_config.offload_type == WeightOffloadType.SEQUENTIAL_CPU_OFFLOAD:
            if model_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and sequential cpu offload can not be used together")
            else:
                logger.info("enable sequential cpu offload for dit")
                self.dit.enable_sequential_cpu_offload(self.device, self.torch_dtype)
        if model_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            if model_runtime_config.parallel_config.enable_fsdp:
                logger.warning("fsdp and async cpu offload can not be used together")
            else:
                logger.info("enable async cpu offload for dit")
                self.dit.enable_async_offload(self.device, model_runtime_config.offload_config)

        # Handle torch.compile for single GPU mode
        parallel_cfg = model_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and model_runtime_config.compile_config.enabled:
            apply_compile_config(model_runtime_config.compile_config)
            logger.info("enable torch.compile for dit")
            self.dit.compile()

    def load_loras(self):
        """Load LoRA weights into the DiT model."""
        lora_configs = self.model_runtime_config.lora_configs
        lora_loader = LoRALoader()
        for lora_config in lora_configs:
            lora_path = lora_config.path
            strength = lora_config.strength
            lora_loader.apply_lora(self.dit, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength} for dit high")

    def _get_forward_fn(self):
        """Get the appropriate forward function based on PP status."""
        if hasattr(self.dit, "pp_flag") and self.dit.pp_flag:
            return self.dit.pp_forward
        return self.dit.forward

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor,
        clip_feature: torch.Tensor | None,
        cfg_scale: float = 1.0,
        sparse_state: dict[str, Any] | None = None,
    ) -> torch.Tensor | None:
        """Predict noise with classifier-free guidance and optional sparse attention.

        Returns:
            Noise prediction tensor (only valid on last PP stage, None on other stages)
        """
        forward_fn = self._get_forward_fn()

        if cfg_scale == 1.0:
            return forward_fn(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                clip_feature=clip_feature,
                context=prompt_emb_posi,
                sparse_state=sparse_state,
            )

        # Check if CFGP is enabled
        cfgp_enabled = get_cfg_world_size(self.dit.device_mesh) > 1

        if cfgp_enabled:
            # CFGP mode: each rank processes one branch (posi or nega)
            cfg_rank = get_cfg_rank(self.dit.device_mesh)
            is_posi = cfg_rank == 0

            context = prompt_emb_posi if is_posi else prompt_emb_nega
            cond_flag = is_posi

            noise_pred = forward_fn(
                x=latents,
                timestep=timestep,
                cond_flag=cond_flag,
                clip_feature=clip_feature,
                context=context,
                sparse_state=sparse_state,
            )

            # Handle PP case: only last stage computes final noise prediction
            if noise_pred is None:
                return None

            # Unshard and compute CFG
            noise_pred = cfg_parallel_unshard(self.dit.device_mesh, [noise_pred])[0]
            noise_pred_posi = noise_pred[0:1]
            noise_pred_nega = noise_pred[1:2]
        else:
            # Non-CFGP mode: sequential forward passes
            noise_pred_posi = forward_fn(
                x=latents,
                timestep=timestep,
                cond_flag=True,
                clip_feature=clip_feature,
                context=prompt_emb_posi,
                sparse_state=sparse_state,
            )
            noise_pred_nega = forward_fn(
                x=latents,
                timestep=timestep,
                cond_flag=False,
                clip_feature=clip_feature,
                context=prompt_emb_nega,
                sparse_state=sparse_state,
            )

            # Handle PP case: only last stage computes final noise prediction
            if noise_pred_posi is None or noise_pred_nega is None:
                return None

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
        ref_latent: torch.Tensor | None,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor | None,
        clip_feature: torch.Tensor | None,
        cfg_scale: float,
        sigma_shift: float,
    ) -> torch.Tensor:
        """Run denoising with optional AdaTaylorCache and sparse attention."""
        dist = torch.distributed

        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        # Set up feature cache from runtime config
        cache_config = self.model_runtime_config.feature_cache_config
        self.setup_feature_cache(self.dit, cache_config, num_inference_steps)

        has_sparse_attention = hasattr(self.dit, "sparse_attention_state")

        # Check if PP is enabled
        is_pp_enabled = hasattr(self.dit, "pp_flag") and self.dit.pp_flag
        is_pp_last_stage = hasattr(self.dit, "is_pp_last_stage") and self.dit.is_pp_last_stage

        for progress_id, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="dit denoise")):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            input_latent = latents
            if ref_latent is not None:
                input_latent = torch.cat([input_latent, ref_latent], dim=1)

            # Create sparse_state if radial attention is enabled
            sparse_state = None
            if has_sparse_attention:
                numeral_timestep = num_inference_steps - progress_id - 1
                sparse_state = self.dit.create_sparse_state(
                    numeral_timestep=numeral_timestep,
                    layer_idx=0,  # Updated per layer in forward_blocks
                )

            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                input_latent = input_latent.to(self.torch_dtype)
                noise_pred = self.predict_noise_with_cfg(
                    latents=input_latent,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    clip_feature=clip_feature,
                    cfg_scale=cfg_scale,
                    sparse_state=sparse_state,
                )

            # In PP mode, only last stage computes scheduler.step
            # Then broadcast updated latents to all stages
            if is_pp_enabled:
                pp_group = self.dit.device_mesh.get_group("pp")
                pp_world_size = dist.get_world_size(pp_group)
                last_stage_rank = pp_world_size - 1  # Last stage's rank in PP group

                if is_pp_last_stage:
                    # Last stage: compute scheduler.step and broadcast
                    latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                    # Broadcast updated latents to all stages (last stage is the source)
                    dist.broadcast(latents, src=last_stage_rank, group=pp_group)
                else:
                    # Non-last stages: receive broadcasted latents from last stage
                    dist.broadcast(latents, src=last_stage_rank, group=pp_group)
            else:
                # Non-PP mode: all ranks compute scheduler.step
                if noise_pred is not None:
                    latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)

        return latents

    def parallel_models(self):
        """Configure parallel processing for the DiT model."""
        parallel_cfg = self.model_runtime_config.parallel_config
        self.dit.device_mesh = create_device_mesh_from_config(parallel_cfg)
        self.dit.set_attention_config(self.model_runtime_config.attention_config)
        # Note: CFGP is handled at stage level, no need to configure on model
        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit.enable_usp()
        if parallel_cfg.pp_degree > 1:
            self.dit.enable_pp()
        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name}")
            if self.dit.quant_type is not None:
                self.dit = convert_params_to_buffers(self.dit, self.dit.quant_type)
            shard_fn = partial(shard_model, wrap_module_names=self.dit.get_fsdp_module_names())
            self.dit = shard_fn(module=self.dit, device_id=self.device)
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit.cpu()
                current_platform.empty_cache()

        # Handle torch.compile for distributed mode
        if self.model_runtime_config.compile_config.enabled:
            apply_compile_config(self.model_runtime_config.compile_config)
            logger.info("enable torch.compile for dit")
            self.dit.compile()
