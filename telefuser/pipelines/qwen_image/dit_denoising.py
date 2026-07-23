from __future__ import annotations

from functools import partial

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_cfg_rank, get_cfg_world_size
from telefuser.distributed.fsdp import shard_model
from telefuser.distributed.parallel_shard import cfg_parallel_unshard
from telefuser.metrics import with_metrics
from telefuser.models.qwen_image_dit import QwenImageDiT
from telefuser.platforms import current_platform
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.logging import logger
from telefuser.utils.lora_loader import LoRALoader
from telefuser.utils.profiler import ProfilingContext4Debug


class DitDenoisingStage(BaseStage):
    """Diffusion denoising stage for Qwen-Image generation.

    Supports classifier-free guidance (CFG) with both separate and batched modes,
    LoRA weight loading, and various parallel processing strategies.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        scheduler: FlowMatchScheduler,
    ):
        super().__init__(name, model_runtime_config)
        self.dit: QwenImageDiT = module_manager.fetch_module("qwen_image_dit")
        self.dit.set_attention_config(model_runtime_config.attention_config)
        if self.model_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            self.dit.enable_async_offload(self.device, offload_config=self.model_runtime_config.offload_config)

        # Handle torch.compile for single GPU mode
        parallel_cfg = model_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

        self.model_names = ["dit"]
        self.load_loras()
        self.scheduler = scheduler

    def load_loras(self):
        """Load LoRA weights into the DiT model."""
        lora_configs = self.model_runtime_config.lora_configs
        lora_loader = LoRALoader()
        for lora_config in lora_configs:
            lora_path = lora_config.path
            strength = lora_config.strength
            lora_loader.apply_lora(self.dit, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength} for dit")

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_mask_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor | None = None,
        prompt_emb_mask_nega: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        edit_latents: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Predict noise with classifier-free guidance.

        Supports CFGP (cfg_degree > 1) for parallel processing efficiency.
        Applies norm preservation to maintain generation quality.
        """
        if cfg_scale == 1.0:
            return self.dit.forward(
                latents=latents,
                timestep=timestep,
                prompt_emb=prompt_emb_posi,
                prompt_emb_mask=prompt_emb_mask_posi,
                edit_latents=edit_latents,
                cond_flag=True,
            )

        # Check if CFGP is enabled
        cfgp_enabled = get_cfg_world_size(self.dit.device_mesh) > 1

        if cfgp_enabled:
            # CFGP mode: each rank processes one branch (posi or nega)
            cfg_rank = get_cfg_rank(self.dit.device_mesh)
            is_posi = cfg_rank == 0

            if is_posi:
                prompt_emb = prompt_emb_posi
                prompt_emb_mask = prompt_emb_mask_posi
                cond_flag = True
            else:
                prompt_emb = prompt_emb_nega
                prompt_emb_mask = prompt_emb_mask_nega
                cond_flag = False

            noise_pred = self.dit.forward(
                latents=latents,
                timestep=timestep,
                prompt_emb=prompt_emb,
                prompt_emb_mask=prompt_emb_mask,
                edit_latents=edit_latents,
                cond_flag=cond_flag,
            )

            # Unshard and compute CFG
            noise_pred = cfg_parallel_unshard(self.dit.device_mesh, [noise_pred])[0]
            noise_pred_posi = noise_pred[0:1]
            noise_pred_nega = noise_pred[1:2]
        else:
            # Non-CFGP mode: sequential forward passes
            noise_pred_posi = self.dit.forward(
                latents=latents,
                timestep=timestep,
                prompt_emb=prompt_emb_posi,
                prompt_emb_mask=prompt_emb_mask_posi,
                edit_latents=edit_latents,
                cond_flag=True,
            )
            noise_pred_nega = self.dit.forward(
                latents=latents,
                timestep=timestep,
                prompt_emb=prompt_emb_nega,
                prompt_emb_mask=prompt_emb_mask_nega,
                edit_latents=edit_latents,
                cond_flag=False,
            )

        # Apply CFG formula with norm preservation
        comb_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        cond_norm = torch.norm(noise_pred_posi, dim=-1, keepdim=True, dtype=latents.dtype)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True, dtype=latents.dtype)
        noise_pred = comb_pred * (cond_norm / noise_norm)
        return noise_pred

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("dit_denosing")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latents: torch.Tensor,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_mask_posi: torch.Tensor,
        cfg_scale: float,
        num_inference_steps: int,
        edit_latents: list[torch.Tensor] | None = None,
        prompt_emb_nega: torch.Tensor | None = None,
        prompt_emb_mask_nega: torch.Tensor | None = None,
        denoising_strength: float = 1.0,
        exponential_shift_mu: float | None = None,
        shift_terminal: float | None = 0.02,
    ) -> torch.Tensor:
        """Run denoising process for specified number of steps.

        Args:
            latents: Input latent tensor.
            prompt_emb_posi: Positive prompt embeddings.
            prompt_emb_mask_posi: Positive prompt embedding mask.
            cfg_scale: CFG scale.
            num_inference_steps: Number of inference steps.
            edit_latents: Optional edit latents for image editing.
            prompt_emb_nega: Negative prompt embeddings.
            prompt_emb_mask_nega: Negative prompt embedding mask.
            denoising_strength: Denoising strength.
            exponential_shift_mu: Exponential shift mu.
            shift_terminal: Shift terminal.

        Returns:
            Denoised latent tensor.
        """
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            dynamic_shift_len=latents.shape[1],
            exponential_shift_mu=exponential_shift_mu,
            shift_terminal=shift_terminal,
        )

        # Set up feature cache from runtime config
        self.setup_feature_cache(self.dit, self.model_runtime_config.feature_cache_config, num_inference_steps)

        for timestep in tqdm(self.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                noise_pred = self.predict_noise_with_cfg(
                    latents=latents,
                    timestep=timestep,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    prompt_emb_mask_posi=prompt_emb_mask_posi,
                    prompt_emb_mask_nega=prompt_emb_mask_nega,
                    cfg_scale=cfg_scale,
                    edit_latents=edit_latents,
                )
            latents = self.scheduler.step(noise_pred, timestep, latents)
        return latents

    def parallel_models(self):
        """Configure parallel processing for the DiT model."""
        parallel_cfg = self.model_runtime_config.parallel_config
        device_mesh = create_device_mesh_from_config(parallel_cfg)
        self.dit.set_attention_config(self.model_runtime_config.attention_config)
        self.dit.device_mesh = device_mesh
        # Note: CFGP is handled at stage level, no need to configure on model
        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit.enable_usp()
        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name}")
            shard_fn = partial(shard_model, wrap_module_names=self.dit.get_fsdp_module_names())
            self.dit = shard_fn(module=self.dit, device_id=self.device)
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit.cpu()
                current_platform.empty_cache()

        # Handle torch.compile for distributed mode
        if self.model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **self.model_runtime_config.compile_config.get_compile_kwargs())
