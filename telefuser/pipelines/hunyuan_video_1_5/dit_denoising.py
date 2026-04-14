"""DiT denoising stage for HunyuanVideo pipeline.

This stage works with HunyuanVideoDiT from telefuser.models.hunyuan_video_dit.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.utils.logging import logger


class HunyuanVideoDenoisingStage(BaseStage):
    """Denoising stage for HunyuanVideo using DiT from HunyuanVideo repository.

    This stage wraps HunyuanVideo_1_5_DiffusionTransformer from
    hyvideo.models.transformers.hunyuanvideo_1_5_transformer.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        scheduler: Any,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.dit = module_manager.fetch_module("hunyuan_video_dit")
        self.dit.set_attention_config(model_runtime_config.attention_config)
        self.scheduler = scheduler
        # Auto-detect use_meanflow from model's time_r_in module
        self.use_meanflow = hasattr(self.dit, "time_r_in") and self.dit.time_r_in is not None
        self.model_names = ["dit"]

        # Handle torch.compile for single GPU mode
        parallel_cfg = model_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

    @with_model_offload(["dit"])
    @torch.inference_mode()
    def process(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        prompt_emb_posi: torch.Tensor,
        prompt_emb_nega: Optional[torch.Tensor] = None,
        text_states_2: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        image_latents: Optional[torch.Tensor] = None,
        vision_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        nega_attention_mask: Optional[torch.Tensor] = None,
        task_type: str = "t2v",
        byt5_text_states: Optional[torch.Tensor] = None,
        byt5_text_mask: Optional[torch.Tensor] = None,
        byt5_text_states_nega: Optional[torch.Tensor] = None,
        byt5_text_mask_nega: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run denoising process using HunyuanVideo Transformer.

        Args:
            latents: Input latent tensor (B, C, T, H, W)
            num_inference_steps: Number of denoising steps
            prompt_emb_posi: Positive prompt embeddings
            prompt_emb_nega: Negative prompt embeddings
            text_states_2: Secondary text embeddings (pooled)
            cfg_scale: CFG scale
            sigma_shift: Sigma shift for scheduler (note: shift is configured at scheduler init)
            image_latents: Image latents for I2V (B, C, 1, H, W)
            vision_states: Vision encoder output for I2V
            attention_mask: Attention mask for text (B, L), passed to DiT for extracting valid tokens.
            nega_attention_mask: nega prompt Attention mask for text (B, L), passed to DiT for extracting valid tokens.
            task_type: "t2v" or "i2v"
            byt5_text_states: Positive ByT5 embeddings for glyph text rendering (B, L, D)
            byt5_text_mask: Positive ByT5 attention mask (B, L)
            byt5_text_states_nega: None for CFG unconditional (DiT skips ByT5 processing)
            byt5_text_mask_nega: None for CFG unconditional

        Returns:
            Denoised latent tensor
        """
        # Set timesteps
        # Note: shift is configured at scheduler initialization time via config.shift
        # HunyuanVideo scheduler uses sd3_time_shift if config.shift != 1.0
        self.scheduler.set_timesteps(num_inference_steps)

        cache_config = self.model_runtime_config.feature_cache_config
        self.setup_feature_cache(self.dit, cache_config, num_inference_steps)

        # Get grid size for RoPE
        tt = latents.shape[2]
        th = latents.shape[3]
        tw = latents.shape[4]

        # Compute RoPE frequencies using transformer's method
        freqs_cos, freqs_sin = self.dit.get_rotary_pos_embed((tt, th, tw))
        freqs_cos = freqs_cos.to(self.device, self.torch_dtype)
        freqs_sin = freqs_sin.to(self.device, self.torch_dtype)

        # Prepare guidance
        guidance = None
        if hasattr(self.dit, "guidance_embed") and self.dit.guidance_embed:
            guidance = torch.tensor([cfg_scale], device=self.device, dtype=self.torch_dtype)
            guidance = guidance.expand(latents.shape[0])

        # Prepare conditional latents
        # For T2V: condition is zeros with same shape as latents + mask
        # For I2V: condition is image latents repeated + mask
        if image_latents is not None and task_type == "i2v":
            cond_latents = self._prepare_cond_latents(image_latents, latents)
        else:
            # T2V: zeros for latents_concat + zeros for mask
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

        # Denoising loop
        for progress_id, timestep in enumerate(tqdm(self.scheduler.timesteps, desc="Denoising")):
            timestep_batch = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Compute timestep_r for mean flow mode (distilled models)
            # In mean flow: timestep_r is the target timestep to reach
            # For last step, target is 0; otherwise it's the next timestep
            timestep_r = None
            if self.use_meanflow:
                if progress_id == len(self.scheduler.timesteps) - 1:
                    timestep_r = torch.tensor([0.0], device=self.device, dtype=self.torch_dtype)
                else:
                    timestep_r = (
                        self.scheduler.timesteps[progress_id + 1]
                        .unsqueeze(0)
                        .to(dtype=self.torch_dtype, device=self.device)
                    )

            # Concat latents with condition
            latents_concat = torch.cat([latents, cond_latents], dim=1)

            if cfg_scale > 1.0:
                # Sequential CFG: run two forward passes instead of batch=2

                with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                    # Conditional forward: use positive ByT5 embeddings
                    noise_pred_cond = self.dit(
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
                        cond_flag=True,
                    )

                    # Unconditional forward: use zero ByT5 embeddings
                    noise_pred_uncond = self.dit(
                        hidden_states=latents_concat.to(self.torch_dtype),
                        timestep=timestep_batch,
                        text_states=prompt_emb_nega,
                        text_states_2=text_states_2,
                        encoder_attention_mask=nega_attention_mask,
                        guidance=guidance,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin,
                        vision_states=vision_states,
                        mask_type=task_type,
                        byt5_text_states=byt5_text_states_nega,
                        byt5_text_mask=byt5_text_mask_nega,
                        timestep_r=timestep_r,
                        cond_flag=False,
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
                        cond_flag=True,
                    )

            if noise_pred is not None:
                scheduler_output = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
                # Handle scheduler output (may be a SchedulerOutput object or tensor)
                if hasattr(scheduler_output, "prev_sample"):
                    latents = scheduler_output.prev_sample
                else:
                    latents = scheduler_output

        return latents

    def _prepare_cond_latents(
        self,
        image_latents: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare conditional latents for I2V.

        Args:
            image_latents: Image latent tensor (B, C, 1, H, W)
            latents: Main latents tensor (B, C, T, H, W)

        Returns:
            Concatenated [latents_concat, mask] tensor for I2V conditioning
        """
        # Repeat image latents along time dimension
        latents_concat = image_latents.repeat(1, 1, latents.shape[2], 1, 1)
        # Zero out all frames except first
        latents_concat[:, :, 1:, :, :] = 0.0

        # Create mask: 1 for first frame, 0 for others
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

        # Concatenate latents and mask
        cond_latents = torch.cat([latents_concat, mask], dim=1)
        return cond_latents

    def parallel_models(self):
        """Configure parallel processing for the DiT model.

        Note: HunyuanVideoDiT does not currently support USP or FSDP.
        This method is provided for future expansion and torch.compile support.
        """
        # Handle torch.compile for distributed mode
        if self.model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **self.model_runtime_config.compile_config.get_compile_kwargs())
