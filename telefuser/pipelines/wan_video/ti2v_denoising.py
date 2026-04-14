"""TI2V Denoising Stage for Wan2.2 Text-Image-to-Video.

Supports both T2V and I2V generation with:
- Classifier-free guidance
- Blended latent approach for image conditioning (Wan2.2 style)
- Feature caching for efficiency
"""

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


class TI2VDenoisingStage(BaseStage):
    """TI2V denoising stage for Wan2.2 unified T2V/I2V generation.

    Uses Wan2.2's blended latent approach for I2V:
    - T2V: noise latent [48, T, H, W]
    - I2V: blended latent = (1 - mask) * image_latent + mask * noise

    The mask has value 0 for conditioned frames (image latent) and 1 for
    generated frames (noise). This allows the model to use in_dim=48 for
    both T2V and I2V modes.

    For I2V, per-token timestep modification is applied where conditioned
    frames get timestep 0 (clean) and noise frames get the current timestep.
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
        self.num_train_timesteps = 1000
        self.patch_size = self.dit.patch_size  # (1, 2, 2) for Wan2.2

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
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

    def load_loras(self):
        """Load LoRA weights into the DiT model."""
        lora_configs = self.model_runtime_config.lora_configs
        lora_loader = LoRALoader()
        for lora_config in lora_configs:
            lora_path = lora_config.path
            strength = lora_config.strength
            lora_loader.apply_lora(self.dit, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength}")

    def _get_forward_fn(self):
        """Get the appropriate forward function based on PP status."""
        if hasattr(self.dit, "pp_flag") and self.dit.pp_flag:
            return self.dit.pp_forward
        return self.dit.forward

    def _create_per_token_timestep(
        self,
        timestep: torch.Tensor,
        mask: torch.Tensor,
        latent_t: int,
        latent_h: int,
        latent_w: int,
    ) -> torch.Tensor:
        """Create per-token timestep for I2V conditioning.

        In Wan2.2 TI2V, conditioned frames (mask=0) should have timestep 0
        (indicating clean latent), while noise frames (mask=1) have the
        current timestep. This is implemented by multiplying the mask with
        the timestep and expanding to the sequence length.

        Args:
            timestep: Scalar timestep tensor [1]
            mask: Mask tensor [z_dim, T, H, W] where 0=conditioned, 1=noise
            latent_t: Temporal dimension of latent
            latent_h: Height dimension of latent
            latent_w: Width dimension of latent

        Returns:
            Per-token timestep tensor [1, seq_len]
        """
        # Get patch size for spatial downsampling
        # patch_size is (temporal, height, width), typically (1, 2, 2)
        patch_h = self.patch_size[1]
        patch_w = self.patch_size[2]

        # Compute sequence length after patchifying
        seq_len = latent_t * (latent_h // patch_h) * (latent_w // patch_w)

        # Take first channel of mask and downsample spatially by patch size
        # mask shape: [z_dim, T, H, W] -> [T, H, W] (first channel)
        # Then downsample: [T, H::patch_h, W::patch_w]
        mask_2d = mask[0, :, ::patch_h, ::patch_w]  # [T, H//patch_h, W//patch_w]

        # Multiply by timestep and flatten
        # Conditioned frames (mask=0) get timestep 0
        # Noise frames (mask=1) get the current timestep
        per_token_ts = (mask_2d * timestep).flatten()  # [T * H//patch_h * W//patch_w]

        # Pad to full sequence length if needed
        if per_token_ts.size(0) < seq_len:
            padding = per_token_ts.new_ones(seq_len - per_token_ts.size(0)) * timestep
            per_token_ts = torch.cat([per_token_ts, padding])

        return per_token_ts.unsqueeze(0)  # [1, seq_len]

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor | None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor | None:
        """Predict noise with classifier-free guidance.

        Args:
            latents: Input latent tensor [B, C, T, H, W] (already blended for I2V)
            timestep: Timestep tensor
            prompt_emb_posi: Positive prompt embedding
            prompt_emb_nega: Negative prompt embedding (None if cfg_scale=1)
            cfg_scale: CFG scale (1.0 means no CFG)

        Returns:
            Noise prediction tensor (only valid on last PP stage, None on other stages)
        """
        forward_fn = self._get_forward_fn()

        if cfg_scale == 1.0:
            return forward_fn(
                x=latents,
                timestep=timestep,
                context=prompt_emb_posi,
                cond_flag=True,
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
                context=context,
                cond_flag=cond_flag,
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
                context=prompt_emb_posi,
                cond_flag=True,
            )
            noise_pred_nega = forward_fn(
                x=latents,
                timestep=timestep,
                context=prompt_emb_nega,
                cond_flag=False,
            )

            # Handle PP case: only last stage computes final noise prediction
            if noise_pred_posi is None or noise_pred_nega is None:
                return None

        noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        return noise_pred

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("ti2v_denoise")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        ref_latent: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        ref_mask: int | None,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: torch.Tensor | None,
        cfg_scale: float,
        sigma_shift: float,
    ) -> torch.Tensor:
        """Run denoising loop for T2V or I2V.

        Uses Wan2.2's blended latent approach for I2V:
        - Initial latent: (1 - mask) * ref_latent + mask * noise
        - After each step: re-blend to keep conditioned frames fixed

        Args:
            latents: Initial noise latent [B, C, T, H, W]
            num_inference_steps: Number of denoising steps
            ref_latent: Reference image latent for I2V.
                        For start image only: [C, 1, H, W] (single frame)
                        For start + end images: tuple of (start_latent, end_latent), each [C, 1, H, W]
                        None for T2V.
            ref_mask: Number of frames for the output video. Used to create the blend mask.
            prompt_emb_posi: Positive prompt embedding
            prompt_emb_nega: Negative prompt embedding
            cfg_scale: CFG scale
            sigma_shift: Noise schedule shift parameter

        Returns:
            Denoised latent tensor
        """
        dist = torch.distributed

        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)

        # Set up feature cache from runtime config
        cache_config = self.model_runtime_config.feature_cache_config
        self.setup_feature_cache(self.dit, cache_config, num_inference_steps)

        # Check if PP is enabled
        is_pp_enabled = hasattr(self.dit, "pp_flag") and self.dit.pp_flag
        is_pp_last_stage = hasattr(self.dit, "is_pp_last_stage") and self.dit.is_pp_last_stage

        # For I2V: create mask and blend noise with image latent
        # Wan2.2 TI2V style:
        # - ref_latent is [z_dim, 1, H, W] (single frame latent)
        # - mask is [z_dim, T, H, W] where first frame = 0, rest = 1
        # - Blend: first frame uses image latent, rest uses noise
        mask = None
        start_latent = None
        end_latent = None

        if ref_latent is not None and ref_mask is not None:
            z_dim = latents.shape[1]
            latent_t = latents.shape[2]
            latent_h = latents.shape[3]
            latent_w = latents.shape[4]

            # Create mask: 0 for conditioned frames (first frame), 1 for noise frames
            # Shape: [z_dim, T, H, W]
            mask = torch.ones(z_dim, latent_t, latent_h, latent_w, device=latents.device, dtype=latents.dtype)
            mask[:, 0] = 0  # First frame is conditioned (keep image latent)

            # Handle start and end images
            if isinstance(ref_latent, tuple):
                # Both start and end images provided
                start_latent, end_latent = ref_latent
                mask[:, -1] = 0  # Last frame is also conditioned
            else:
                # Only start image
                start_latent = ref_latent

            # Blend: (1 - mask) * image_latent + mask * noise
            # start_latent shape: [z_dim, 1, H, W] -> broadcast to [z_dim, T, H, W]
            # This matches the original Wan2.2 implementation where z[0] (single frame) is broadcast
            latents = (1.0 - mask) * start_latent + mask * latents[0]
            latents = latents.unsqueeze(0)  # Add batch dimension back

            if end_latent is not None:
                # Apply end image latent to the last frame
                # The mask already has last frame = 0, so we need to manually set it
                latents[0, :, -1:] = end_latent

        for progress_id, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="TI2V denoise")):
            # For I2V, create per-token timestep where conditioned frames get 0
            if mask is not None:
                timestep_tensor = self._create_per_token_timestep(
                    timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device),
                    mask,
                    latent_t,
                    latent_h,
                    latent_w,
                )
            else:
                timestep_tensor = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                latents_input = latents.to(self.torch_dtype)
                noise_pred = self.predict_noise_with_cfg(
                    latents=latents_input,
                    timestep=timestep_tensor,
                    prompt_emb_posi=prompt_emb_posi,
                    prompt_emb_nega=prompt_emb_nega,
                    cfg_scale=cfg_scale,
                )

            # In PP mode, only last stage computes scheduler.step
            if is_pp_enabled:
                pp_group = self.dit.device_mesh.get_group("pp")
                pp_world_size = dist.get_world_size(pp_group)
                last_stage_rank = pp_world_size - 1

                if is_pp_last_stage:
                    latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                    # Re-blend to keep conditioned frames fixed (I2V)
                    if mask is not None and start_latent is not None:
                        latents = (1.0 - mask) * start_latent + mask * latents[0]
                        latents = latents.unsqueeze(0)
                        if end_latent is not None:
                            latents[0, :, -1:] = end_latent
                    dist.broadcast(latents, src=last_stage_rank, group=pp_group)
                else:
                    dist.broadcast(latents, src=last_stage_rank, group=pp_group)
            else:
                # Non-PP mode: all ranks compute scheduler.step
                if noise_pred is not None:
                    latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                    # Re-blend to keep conditioned frames fixed (I2V)
                    if mask is not None and start_latent is not None:
                        latents = (1.0 - mask) * start_latent + mask * latents[0]
                        latents = latents.unsqueeze(0)
                        if end_latent is not None:
                            latents[0, :, -1:] = end_latent

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
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **self.model_runtime_config.compile_config.get_compile_kwargs())
