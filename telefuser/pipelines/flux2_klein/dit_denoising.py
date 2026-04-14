"""DiT denoising stage for Flux2 Klein."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.fsdp import shard_model
from telefuser.metrics import with_metrics
from telefuser.models.flux2_dit import Flux2DiT
from telefuser.platforms import current_platform
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute empirical mu for adaptive timestep scheduling.

    Args:
        image_seq_len: Image sequence length (H*W/4)
        num_steps: Number of inference steps

    Returns:
        Mu value for scheduler
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


class DitDenoisingStage(BaseStage):
    """Diffusion denoising stage for Flux2 Klein.

    Supports classifier-free guidance (CFG), KV caching for reference images,
    and various parallel processing strategies.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        scheduler: FlowMatchScheduler,
    ):
        super().__init__(name, model_runtime_config)
        self.transformer: Flux2DiT = module_manager.fetch_module("transformer")

        # Set attention config if supported (Flux2DiT has set_attention_config)
        if hasattr(self.transformer, "set_attention_config"):
            self.transformer.set_attention_config(model_runtime_config.attention_config)

        # Enable async offload if configured
        if hasattr(self.transformer, "enable_async_offload"):
            if self.model_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
                self.transformer.enable_async_offload(
                    self.device, offload_config=self.model_runtime_config.offload_config
                )

        self.model_names = ["transformer"]
        self.scheduler = scheduler
        self.batch_cfg = False

        # Handle torch.compile for single GPU mode
        parallel_cfg = model_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for transformer")
            self.transformer = torch.compile(
                self.transformer, **model_runtime_config.compile_config.get_compile_kwargs()
            )

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        """Generate 4D position coordinates (T, H, W, L) for latent tokens.

        Image uses T=0, H=[0..H-1], W=[0..W-1], L=0.

        Args:
            latents: Latent tensor of shape (B, C, H, W)

        Returns:
            Position IDs of shape (B, H*W, 4)
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0]
        h = torch.arange(height)
        w = torch.arange(width)
        layer_dim = torch.arange(1)  # [0]

        latent_ids = torch.cartesian_prod(t, h, w, layer_dim)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _prepare_image_ids(image_latents: list[torch.Tensor], scale: int = 10) -> torch.Tensor:
        """Generate 4D position coordinates for reference image tokens.

        Reference images use scaled T coords: T = scale + scale*i for i-th image.

        Args:
            image_latents: List of latent tensors [(1, C, H, W), ...]
            scale: Time coordinate scale factor

        Returns:
            Position IDs of shape (1, N_total, 4)
        """
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape
            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_ids: torch.Tensor,
        cfg_scale: float,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_text_ids: torch.Tensor | None = None,
        image_latents: torch.Tensor | None = None,
        image_latent_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict noise with classifier-free guidance.

        Args:
            latents: Noise latents of shape (B, seq, C)
            timestep: Timestep tensor
            prompt_embeds: Positive prompt embeddings
            text_ids: Position IDs for text
            latent_ids: Position IDs for latents
            cfg_scale: CFG scale
            negative_prompt_embeds: Negative prompt embeddings
            negative_text_ids: Position IDs for negative text
            image_latents: Optional reference image latents
            image_latent_ids: Position IDs for reference images

        Returns:
            Predicted noise tensor
        """
        # Prepare input
        hidden_states = latents
        img_ids = latent_ids

        if image_latents is not None:
            hidden_states = torch.cat([latents, image_latents], dim=1)
            img_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

        # Remove batch dim from position IDs if needed
        if img_ids.ndim == 3:
            img_ids_0 = img_ids[0]
        else:
            img_ids_0 = img_ids
        if text_ids.ndim == 3:
            text_ids_0 = text_ids[0]
        else:
            text_ids_0 = text_ids

        # Conditional forward
        noise_pred = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids_0,
            img_ids=img_ids_0,
            return_dict=False,
        )[0]

        # Remove reference tokens from output
        noise_pred = noise_pred[:, : latents.size(1)]

        if cfg_scale > 1.0 and negative_prompt_embeds is not None:
            # Unconditional forward
            if negative_text_ids.ndim == 3:
                neg_text_ids_0 = negative_text_ids[0]
            else:
                neg_text_ids_0 = negative_text_ids

            neg_noise_pred = self.transformer(
                hidden_states=hidden_states,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=neg_text_ids_0,
                img_ids=img_ids_0,
                return_dict=False,
            )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

            # CFG formula
            noise_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

        return noise_pred

    @with_model_offload(["transformer"])
    @ProfilingContext4Debug("dit_denoising")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        cfg_scale: float,
        num_inference_steps: int,
        latent_ids: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_text_ids: torch.Tensor | None = None,
        image_latents: torch.Tensor | None = None,
        image_latent_ids: torch.Tensor | None = None,
        sigmas: list[float] | None = None,
    ) -> torch.Tensor:
        """Run denoising process for specified number of steps.

        Args:
            latents: Initial noise latents of shape (B, seq, C)
            prompt_embeds: Positive prompt embeddings
            text_ids: Position IDs for text
            cfg_scale: CFG scale
            num_inference_steps: Number of inference steps
            latent_ids: Position IDs for latents
            negative_prompt_embeds: Negative prompt embeddings
            negative_text_ids: Position IDs for negative text
            image_latents: Optional reference image latents
            image_latent_ids: Position IDs for reference images
            sigmas: Custom sigmas for scheduling

        Returns:
            Denoised latent tensor
        """
        # Compute empirical mu for adaptive scheduling
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)

        # Set timesteps
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        self.scheduler.set_timesteps(num_inference_steps)
        # Override with empirical mu if supported
        if hasattr(self.scheduler, "set_timesteps_with_mu"):
            self.scheduler.set_timesteps_with_mu(sigmas=sigmas, mu=mu)
        else:
            self.scheduler.set_timesteps(num_inference_steps)

        # Set up feature cache from runtime config (only if model supports it)
        if hasattr(self.transformer, "set_ada_taylor_cache"):
            self.setup_feature_cache(
                self.transformer, self.model_runtime_config.feature_cache_config, num_inference_steps
            )

        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Denoising"):
            timestep = t.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            if latents.shape[0] > 1:
                timestep = timestep.expand(latents.shape[0])

            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                noise_pred = self.predict_noise_with_cfg(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    latent_ids=latent_ids,
                    cfg_scale=cfg_scale,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    image_latents=image_latents,
                    image_latent_ids=image_latent_ids,
                )

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)

        return latents

    def parallel_models(self):
        """Configure parallel processing for the DiT model."""
        parallel_cfg = self.model_runtime_config.parallel_config

        # Set attention config for parallel processing
        if hasattr(self.transformer, "set_attention_config"):
            self.transformer.set_attention_config(self.model_runtime_config.attention_config)

        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name}")
            shard_fn = partial(shard_model, wrap_module_names=["Flux2TransformerBlock", "Flux2SingleTransformerBlock"])
            self.transformer = shard_fn(module=self.transformer, device_id=self.device)
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.transformer.cpu()
                current_platform.empty_cache()

        # Handle torch.compile for distributed mode
        if self.model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for transformer")
            self.transformer = torch.compile(
                self.transformer, **self.model_runtime_config.compile_config.get_compile_kwargs()
            )
