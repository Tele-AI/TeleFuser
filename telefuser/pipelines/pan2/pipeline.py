# SPDX-License-Identifier: Apache-2.0
"""TeleFuser PAN2 text-to-video and first-frame image-to-video pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from transformers import AutoTokenizer

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.worker import ParallelWorker

from .data import get_pan2_default_negative_prompt
from .stages import PAN2DenoisingStage, PAN2TextStage, PAN2VAEStage


@dataclass
class PAN2PipelineConfig:
    tokenizer_path: str
    scheduler_config_path: str
    text_encoder_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)

    def __post_init__(self) -> None:
        for name in ("text_encoder_config", "dit_config", "vae_config"):
            runtime = getattr(self, name)
            runtime.parallel_config.validate()
            if runtime.parallel_config.enable_fsdp:
                raise NotImplementedError("This pipeline supports Ulysses SP and TP, not FSDP")
            if runtime.offload_config.offload_type not in {
                WeightOffloadType.NO_CPU_OFFLOAD,
                WeightOffloadType.MODEL_CPU_OFFLOAD,
            }:
                raise NotImplementedError("This pipeline supports no offload or model CPU offload")
            if runtime.lora_configs or runtime.feature_cache_config.enabled or runtime.compile_config.enabled:
                raise NotImplementedError("LoRA, feature caching and compilation are not configured by this pipeline")
            if name != "dit_config" and runtime.quant_config.enabled:
                raise NotImplementedError("Online quantization is supported on the DiT stage only")


class PAN2Pipeline(BasePipeline):
    def __init__(self, device: str | torch.device = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.config: PAN2PipelineConfig | None = None
        self.text_stage: PAN2TextStage | None = None
        self.vae_stage: PAN2VAEStage | None = None
        self.denoising_stage: PAN2DenoisingStage | ParallelWorker | None = None

    def init(self, module_manager: ModuleManager, config: PAN2PipelineConfig) -> None:
        self.config = config
        if (
            config.text_encoder_config.parallel_config.world_size != 1
            or config.vae_config.parallel_config.world_size != 1
        ):
            raise ValueError("PAN2 conditioning and VAE currently require single-device stage configs")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, local_files_only=True)
        self.text_stage = PAN2TextStage(module_manager, config.text_encoder_config, tokenizer)
        self.vae_stage = PAN2VAEStage(module_manager, config.vae_config)
        stage = PAN2DenoisingStage(module_manager, config.dit_config, config.scheduler_config_path)
        if config.dit_config.parallel_config.world_size > 1 and not dist.is_initialized():
            self.denoising_stage = ParallelWorker(stage)
        else:
            self.denoising_stage = stage
            if dist.is_initialized():
                stage.parallel_models()
        self._model_info = module_manager.get_model_info()

    def _get_stages(self) -> list[Any]:
        return [stage for stage in (self.text_stage, self.vae_stage, self.denoising_stage) if stage is not None]

    def stop(self) -> None:
        if isinstance(self.denoising_stage, ParallelWorker):
            self.denoising_stage.close()

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        height: int = 704,
        width: int = 1248,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        seed: int = 0,
        negative_prompt: str | None = None,
        image: Image.Image | None = None,
        fps: float = 24.0,
        max_sequence_length: int = 1536,
    ) -> torch.Tensor:
        if any(stage is None for stage in (self.text_stage, self.vae_stage, self.denoising_stage)):
            raise RuntimeError("PAN2Pipeline.init must be called before generation")
        if not isinstance(prompt, str) or (negative_prompt is not None and not isinstance(negative_prompt, str)):
            raise ValueError("prompt and negative_prompt must be strings")
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("PAN2 height and width must be positive multiples of 32")
        if num_frames < 1 or (num_frames - 1) % 4:
            raise ValueError("PAN2 num_frames must be 4*k+1")
        if num_inference_steps < 1 or fps <= 0 or max_sequence_length < 1 or guidance_scale < 0:
            raise ValueError("Invalid steps, fps, text length or guidance scale")
        negative = (
            (negative_prompt or get_pan2_default_negative_prompt(num_frames, fps)) if guidance_scale > 1 else None
        )
        text, uncond = self.text_stage.encode(prompt, negative, max_sequence_length)
        first = self.vae_stage.encode_image(image, height, width) if image is not None else None
        latent = self.denoising_stage.denoise(
            text, uncond, first, height, width, num_frames, num_inference_steps, guidance_scale, seed
        )
        if callable(latent):
            latent = latent()
        # Follow the existing video example convention: RGB frames in [T,H,W,C], range [0,1].
        return self.vae_stage.decode_video(latent)[0].permute(1, 2, 3, 0)
