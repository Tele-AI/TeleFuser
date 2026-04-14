"""SR DiT denoising stage for HunyuanVideo Super-Resolution pipeline.

This stage works with HunyuanVideo_1_5_DiffusionTransformer for super-resolution:
    from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer

The SR pipeline takes low-quality video latents (already upsampled by UpsamplerStage),
adds noise, and then denoises with the DiT transformer using dual conditioning.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.utils.logging import logger


def _expand_dims(tensor: torch.Tensor, ndim: int) -> torch.Tensor:
    """Expand tensor to target dimensions by adding trailing dimensions."""
    shape = tensor.shape + (1,) * (ndim - tensor.ndim)
    return tensor.reshape(shape)


class HunyuanVideoSRDenoisingStage(BaseStage):
    """Super-Resolution denoising stage for HunyuanVideo using DiT.

    This stage wraps the SR pipeline from HunyuanVideo-1.5 which uses:
    - An upsampler (SRTo720pUpsampler) to upsample low-quality latents
    - The same HunyuanVideo_1_5_DiffusionTransformer for denoising
    - Dual conditioning: task conditioning + LQ conditioning

    Reference: hyvideo/pipelines/hunyuan_video_sr_pipeline.py
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        scheduler: Any,
        lq_noise_strength: float = 0.7,
        sigma_shift: float = 2.0,
    ) -> None:
        """Initialize SR denoising stage.

        Args:
            name: Stage name
            module_manager: Module manager with loaded models
            model_runtime_config: Runtime configuration
            scheduler: Diffusion scheduler
            lq_noise_strength: Noise strength for LQ latents (0.0-1.0)
            sigma_shift: Sigma shift for SR scheduler (default: 2.0 for SR)
        """
        super().__init__(name, model_runtime_config)
        # Try to fetch SR-specific transformer, fall back to base transformer
        try:
            self.dit = module_manager.fetch_module("sr_dit")
        except KeyError:
            self.dit = module_manager.fetch_module("hunyuan_video_dit")
        self.dit.set_attention_config(model_runtime_config.attention_config)
        self.scheduler = scheduler
        self.lq_noise_strength = lq_noise_strength
        self.sigma_shift = sigma_shift
        # Auto-detect use_meanflow from model's time_r_in module
        self.use_meanflow = hasattr(self.dit, "time_r_in") and self.dit.time_r_in is not None
        self.model_names = ["dit"]
        # Note: upsampler is handled by separate HunyuanVideoUpsamplerStage in pipeline

        # Handle torch.compile for single GPU mode
        parallel_cfg = model_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

    def _add_noise_to_lq(self, lq_latents: torch.Tensor) -> torch.Tensor:
        """Add noise to low-quality latents.

        Args:
            lq_latents: Low-quality latent tensor (B, C, T, H, W)

        Returns:
            Noisy LQ latents
        """
        noise = torch.randn_like(lq_latents)
        timestep = torch.tensor([1000.0], device=self.device) * self.lq_noise_strength
        t = _expand_dims(timestep, lq_latents.ndim)
        return (1 - t / 1000.0) * lq_latents + (t / 1000.0) * noise

    def _prepare_lq_cond_latents(self, lq_latents: torch.Tensor) -> torch.Tensor:
        """Prepare conditional latents for LQ input.

        Args:
            lq_latents: Low-quality latent tensor (B, C, T, H, W)

        Returns:
            Conditional latents [lq_latents, mask_ones]
        """
        b, _, f, h, w = lq_latents.shape
        mask_ones = torch.ones(b, 1, f, h, w, device=lq_latents.device, dtype=lq_latents.dtype)
        return torch.cat([lq_latents, mask_ones], dim=1)

    def _prepare_task_cond_latents(
        self,
        latents: torch.Tensor,
        image_cond: torch.Tensor | None,
        task_type: str,
        multitask_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Prepare conditional latents for task conditioning.

        Args:
            latents: Target latent tensor (B, C, T, H, W)
            image_cond: Image conditioning for I2V
            task_type: "t2v" or "i2v"
            multitask_mask: Task mask tensor

        Returns:
            Task conditional latents
        """
        if image_cond is not None and task_type == "i2v":
            # I2V: image cond latents + mask
            latents_concat = image_cond.repeat(1, 1, latents.shape[2], 1, 1)
            latents_concat[:, :, 1:, :, :] = 0.0

            mask = torch.zeros(
                latents.shape[0],
                1,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
                device=latents.device,
                dtype=latents.dtype,
            )
            mask[:, :, 0, :, :] = 1.0
            cond_latents = torch.cat([latents_concat, mask], dim=1)
        else:
            # T2V: zeros + mask
            latents_concat = torch.zeros_like(latents)
            mask = torch.zeros(
                latents.shape[0],
                1,
                latents.shape[2],
                latents.shape[3],
                latents.shape[4],
                device=latents.device,
                dtype=latents.dtype,
            )
            cond_latents = torch.cat([latents_concat, mask], dim=1)

        if multitask_mask is not None:
            cond_latents = torch.cat([cond_latents, multitask_mask], dim=1)

        return cond_latents

    @with_model_offload(["dit"])
    @torch.inference_mode()
    def process(
        self,
        latents: torch.Tensor,
        lq_latents: torch.Tensor,
        num_inference_steps: int,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: Optional[torch.Tensor] = None,
        text_states_2: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 6.0,
        embedded_guidance_scale: Optional[float] = None,
        image_cond: Optional[torch.Tensor] = None,
        vision_states: Optional[torch.Tensor] = None,
        task_type: str = "t2v",
        byt5_text_states: Optional[torch.Tensor] = None,
        byt5_text_mask: Optional[torch.Tensor] = None,
        byt5_text_states_nega: Optional[torch.Tensor] = None,
        byt5_text_mask_nega: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run SR denoising process.

        Args:
            latents: Input latent tensor (B, C, T, H, W) - noise to denoise
            lq_latents: Upsampled low-quality video latents (B, C, F, H, W) - already at target resolution
            num_inference_steps: Number of denoising steps
            prompt_emb_posi: Positive prompt embeddings
            prompt_emb_nega: Negative prompt embeddings
            text_states_2: Secondary text embeddings (pooled)
            attention_mask: Attention mask for text
            cfg_scale: CFG scale
            embedded_guidance_scale: Embedded guidance scale for transformer
            image_cond: Image conditioning for I2V
            vision_states: Vision encoder output for I2V
            task_type: "t2v" or "i2v"
            byt5_text_states: Positive ByT5 embeddings for glyph text rendering (B, L, D)
            byt5_text_mask: Positive ByT5 attention mask (B, L)
            byt5_text_states_nega: None for CFG unconditional (DiT skips ByT5 processing)
            byt5_text_mask_nega: None for CFG unconditional

        Returns:
            Denoised latent tensor
        """
        # Set timesteps with SR-specific shift
        # Original HunyuanVideo: self.scheduler = self._create_scheduler(self.config.flow_shift)
        # SR uses flow_shift=2.0 instead of base shift (7.0)
        from telefuser.schedulers.flow_match_discrete import FlowMatchDiscreteScheduler

        sr_scheduler = FlowMatchDiscreteScheduler(
            num_train_timesteps=1000,
            shift=self.sigma_shift,
            reverse=True,
            solver="euler",
        )
        sr_scheduler.set_timesteps(num_inference_steps)
        timesteps = sr_scheduler.timesteps

        # Get grid size for RoPE
        tt = latents.shape[2]
        th = latents.shape[3]
        tw = latents.shape[4]

        # Compute RoPE frequencies using transformer's method (pre-compute once for efficiency)
        freqs_cos, freqs_sin = self.dit.get_rotary_pos_embed((tt, th, tw))
        freqs_cos = freqs_cos.to(self.device, self.torch_dtype)
        freqs_sin = freqs_sin.to(self.device, self.torch_dtype)

        # lq_latents is already upsampled by HunyuanVideoUpsamplerStage
        # Add noise to upsampled LQ latents
        lq_latents = self._add_noise_to_lq(lq_latents.to(device=self.device, dtype=self.torch_dtype))

        # Prepare conditioning
        # Task conditioning (from base pipeline)
        task_cond = self._prepare_task_cond_latents(latents, image_cond, task_type, None)

        # LQ conditioning
        lq_cond = self._prepare_lq_cond_latents(lq_latents)

        # Combined conditioning
        condition = torch.cat([task_cond, lq_cond], dim=1)

        # Prepare zero LQ condition for later timesteps
        # When t < 1000 * noise_scale, we zero out the LQ condition
        c = lq_latents.shape[1]
        zero_lq_condition = condition.clone()
        zero_lq_condition[:, c + 1 : 2 * c + 1] = torch.zeros_like(lq_latents)
        zero_lq_condition[:, 2 * c + 1] = 0

        # Prepare guidance
        guidance = None
        if getattr(self.dit, "guidance_embed", False):
            if embedded_guidance_scale is not None:
                guidance = (
                    torch.tensor(
                        [embedded_guidance_scale],
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                    * 1000.0
                )

        # Denoising loop
        for i, timestep in enumerate(tqdm(timesteps, desc="SR Denoising")):
            # Switch to zero LQ condition after certain timestep
            current_condition = condition
            if timestep < 1000 * self.lq_noise_strength:
                current_condition = zero_lq_condition

            # Concat latents with condition
            latents_concat = torch.cat([latents, current_condition], dim=1)
            timestep_batch = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Compute timestep_r for mean flow mode (distilled models)
            # In mean flow: timestep_r is the target timestep to reach
            # For last step, target is 0; otherwise it's the next timestep
            timestep_r = None
            if self.use_meanflow:
                if i == len(timesteps) - 1:
                    timestep_r = torch.tensor([0.0], device=self.device, dtype=self.torch_dtype)
                else:
                    timestep_r = timesteps[i + 1].unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            if cfg_scale > 1.0:
                # Sequential CFG: run two forward passes instead of batch=2
                text_states_cond = prompt_emb_posi
                text_states_uncond = prompt_emb_nega if prompt_emb_nega is not None else prompt_emb_posi

                with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                    # Conditional forward: use positive ByT5 embeddings
                    noise_pred_cond = self.dit(
                        hidden_states=latents_concat.to(self.torch_dtype),
                        timestep=timestep_batch,
                        text_states=text_states_cond,
                        text_states_2=text_states_2,
                        encoder_attention_mask=attention_mask,
                        guidance=guidance,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        vision_states=vision_states,
                        mask_type=task_type,
                        byt5_text_states=byt5_text_states,
                        byt5_text_mask=byt5_text_mask,
                        timestep_r=timestep_r,
                        return_dict=False,
                    )

                    # Unconditional forward: use zero ByT5 embeddings
                    noise_pred_uncond = self.dit(
                        hidden_states=latents_concat.to(self.torch_dtype),
                        timestep=timestep_batch,
                        text_states=text_states_uncond,
                        text_states_2=text_states_2,
                        encoder_attention_mask=attention_mask,
                        guidance=guidance,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        vision_states=vision_states,
                        mask_type=task_type,
                        byt5_text_states=byt5_text_states_nega,
                        byt5_text_mask=byt5_text_mask_nega,
                        timestep_r=timestep_r,
                        return_dict=False,
                    )

                # Apply CFG
                noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # No CFG: single forward pass
                with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                    noise_pred = self.dit(
                        hidden_states=latents_concat.to(self.torch_dtype),
                        timestep=timestep_batch,
                        text_states=prompt_emb_posi,
                        text_states_2=text_states_2,
                        encoder_attention_mask=attention_mask,
                        guidance=guidance,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        vision_states=vision_states,
                        mask_type=task_type,
                        byt5_text_states=byt5_text_states,
                        byt5_text_mask=byt5_text_mask,
                        timestep_r=timestep_r,
                        return_dict=False,
                    )

            # Scheduler step
            if noise_pred is not None:
                scheduler_output = sr_scheduler.step(noise_pred, timestep, latents, return_dict=False)
                latents = scheduler_output[0] if isinstance(scheduler_output, tuple) else scheduler_output

        return latents

    def parallel_models(self):
        """Configure parallel processing for the DiT model.

        Note: HunyuanVideoDiT does not currently support USP or FSDP.
        This method is provided for future expansion and torch.compile support.
        """
        # Handle torch.compile for distributed mode
        if self.model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **self.model_runtime_config.compile_config.get_compile_kwargs())
