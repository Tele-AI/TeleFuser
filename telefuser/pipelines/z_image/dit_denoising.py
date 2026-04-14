from __future__ import annotations

import inspect
from functools import partial
from typing import Any, List

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.distributed.fsdp import shard_model
from telefuser.metrics import with_metrics
from telefuser.models.z_image_dit import ZImageTransformer2DModel
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate timestep shift based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler: Any,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: List[int] | None = None,
    sigmas: List[float] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, int]:
    """Retrieve timesteps from scheduler with custom timestep support.

    Args:
        scheduler: The scheduler to get timesteps from
        num_inference_steps: Number of diffusion steps
        device: Device to move timesteps to
        timesteps: Custom timesteps to override scheduler
        sigmas: Custom sigmas to override scheduler
        **kwargs: Additional arguments for scheduler.set_timesteps

    Returns:
        Tuple of (timesteps, num_inference_steps)
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class DitDenoisingStage(BaseStage):
    """DiT denoising stage for Z-Image generation.

    Supports classifier-free guidance with truncation and normalization,
    dynamic timestep shift based on image size, and various parallel strategies.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        scheduler: Any,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.dit: ZImageTransformer2DModel = module_manager.fetch_module("zimage_dit")
        self.model_names = ["dit"]
        if self.model_runtime_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            self.dit.enable_async_offload(self.device, self.model_runtime_config.offload_config)
        self.batch_cfg = False
        self.scheduler = scheduler

        # Handle torch.compile for single GPU mode
        parallel_cfg = model_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

    def load_loras(self):
        """Load LoRA weights - not implemented for Z-Image."""
        raise NotImplementedError()

    def predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: List[torch.Tensor],
        cond_flag: bool = True,
    ):
        """Predict noise for given latents and timestep.

        Args:
            latents: Input latent tensor.
            timestep: Timestep tensor.
            prompt_emb: List of prompt embeddings.
            cond_flag: True for conditional path, False for unconditional path.

        Returns:
            Predicted noise tensor.
        """
        latents_list = list(latents.unsqueeze(2).unbind(dim=0))
        noise_pred = self.dit(
            x=latents_list,
            t=timestep,
            cap_feats=prompt_emb,
            cond_flag=cond_flag,
        )[0]
        return noise_pred

    def predict_noise_with_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: List[torch.Tensor],
        negative_prompt_emb: List[torch.Tensor],
        cfg_scale: float = 5.0,
        cfg_truncation: float = 1.0,
        cfg_normalization: float = 0.0,  # 0.0 means disabled
        batch_cfg: bool = False,
    ):
        """Predict noise with classifier-free guidance and optional truncation/normalization."""
        t = timestep.expand(latents.shape[0])
        t = (1000 - t) / 1000
        progress = t[0].item()

        # Apply CFG truncation at high progress values
        current_cfg_scale = cfg_scale
        if cfg_truncation <= 1.0 and progress > cfg_truncation:
            current_cfg_scale = 0.0

        do_cfg = current_cfg_scale > 0 and negative_prompt_emb is not None

        if not do_cfg:
            comb_pred = self.predict_noise(latents, t, prompt_emb, cond_flag=True)
        else:
            if not batch_cfg:
                positive_noise_pred = self.predict_noise(latents, t, prompt_emb, cond_flag=True)
                negative_noise_pred = self.predict_noise(latents, t, negative_prompt_emb, cond_flag=False)
            else:
                latents_input = torch.cat([latents, latents], dim=0)
                t = torch.cat([t, t], dim=0)
                prompt_input = prompt_emb + negative_prompt_emb

                noise_pred = self.predict_noise(latents_input, t, prompt_input)

                positive_noise_pred, negative_noise_pred = noise_pred[0], noise_pred[1]

            comb_pred = positive_noise_pred + current_cfg_scale * (positive_noise_pred - negative_noise_pred)

            # Apply CFG normalization to prevent over-saturation
            if cfg_normalization is not None and cfg_normalization > 0:
                cond_norm = torch.linalg.vector_norm(positive_noise_pred)
                new_norm = torch.linalg.vector_norm(comb_pred)
                max_allowed_norm = cond_norm * cfg_normalization
                new_norm = torch.where(new_norm < 1e-6, torch.ones_like(new_norm), new_norm)
                scale_factor = max_allowed_norm / new_norm
                scale_factor = torch.clamp(scale_factor, max=1.0)
                comb_pred = comb_pred * scale_factor

        comb_pred = -comb_pred.squeeze(1).unsqueeze(0)
        return comb_pred

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("dit_denosing")
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        latents: torch.Tensor,
        prompt_embeds: List[torch.Tensor],
        cfg_scale: float,
        num_inference_steps: int,
        negative_prompt_embeds: List[torch.Tensor] | None = None,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
    ) -> torch.Tensor:
        """Run denoising with dynamic shift based on image sequence length.

        Args:
            latents: Input latent tensor.
            prompt_embeds: List of prompt embeddings.
            cfg_scale: CFG scale.
            num_inference_steps: Number of inference steps.
            negative_prompt_embeds: List of negative prompt embeddings.
            cfg_normalization: CFG normalization factor.
            cfg_truncation: CFG truncation threshold.

        Returns:
            Denoised latent tensor.
        """
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            **scheduler_kwargs,
        )

        # Set up feature cache from runtime config
        self.setup_feature_cache(self.dit, self.model_runtime_config.feature_cache_config, num_inference_steps)

        for timestep in tqdm(timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
                noise_pred = self.predict_noise_with_cfg(
                    latents=latents,
                    timestep=timestep,
                    prompt_emb=prompt_embeds,
                    negative_prompt_emb=negative_prompt_embeds,
                    batch_cfg=self.batch_cfg,
                    cfg_scale=cfg_scale,
                    cfg_truncation=cfg_truncation,
                    cfg_normalization=cfg_normalization,
                )
            latents = self.scheduler.step(noise_pred.to(torch.float32), timestep, latents, return_dict=False)[0]
        return latents

    def parallel_models(self):
        """Configure parallel processing for the DiT model."""
        parallel_cfg = self.model_runtime_config.parallel_config
        device_mesh = create_device_mesh_from_config(parallel_cfg)
        self.dit.device_mesh = device_mesh
        if parallel_cfg.cfg_degree > 1:
            logger.warning("Z-Image DiT does not support CFG parallelism (cfgp), skipping")
        if parallel_cfg.sp_ulysses_degree > 1:
            logger.warning("Z-Image DiT does not support Ulysses sequence parallelism (usp), skipping")
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
