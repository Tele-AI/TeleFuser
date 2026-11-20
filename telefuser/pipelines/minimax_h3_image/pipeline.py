# SPDX-License-Identifier: Apache-2.0
"""TeleFuser text-to-image and reference-image editing with MiniMax-H3-Image."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from transformers import AutoProcessor

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.worker import ParallelWorker

from .data import bound_short_edge, get_image_canvas_size
from .stages import MiniMaxH3ImageDenoisingStage, MiniMaxH3ImageTextStage, MiniMaxH3ImageVAEStage


@dataclass
class MiniMaxH3ImagePipelineConfig:
    processor_path: str
    pdd_config_path: str
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


class MiniMaxH3ImagePipeline(BasePipeline):
    def __init__(self, device: str | torch.device = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.config: MiniMaxH3ImagePipelineConfig | None = None
        self.text_stage: MiniMaxH3ImageTextStage | None = None
        self.vae_stage: MiniMaxH3ImageVAEStage | None = None
        self.denoising_stage: MiniMaxH3ImageDenoisingStage | ParallelWorker | None = None

    def init(self, module_manager: ModuleManager, config: MiniMaxH3ImagePipelineConfig) -> None:
        self.config = config
        scheduler = json.loads(Path(config.scheduler_config_path).read_text())
        if scheduler.get("euler_update_form") != "velocity" or scheduler.get("shift_provided_sigmas", False):
            raise ValueError("H3-Image requires the released unshifted velocity scheduler")
        pdd = json.loads(Path(config.pdd_config_path).read_text())
        if pdd["num_inference_steps"] != len(pdd["sigmas"]) - 1:
            raise ValueError("PDD config has inconsistent step count and sigmas")
        processor = AutoProcessor.from_pretrained(config.processor_path, local_files_only=True)
        self.text_stage = MiniMaxH3ImageTextStage(module_manager, config.text_encoder_config, processor=processor)
        self.vae_stage = MiniMaxH3ImageVAEStage(module_manager, config.vae_config)
        stage = MiniMaxH3ImageDenoisingStage(module_manager, config.dit_config, pdd["sigmas"])
        if (
            config.text_encoder_config.parallel_config.world_size != 1
            or config.vae_config.parallel_config.world_size != 1
        ):
            raise ValueError("H3-Image conditioning and VAE currently require single-device stage configs")
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
        height: int = 2048,
        width: int = 2048,
        image: Image.Image | list[Image.Image] | None = None,
        seed: int = 0,
        min_generation_area: int = 2048 * 2048,
    ) -> Image.Image:
        if any(stage is None for stage in (self.text_stage, self.vae_stage, self.denoising_stage)):
            raise RuntimeError("MiniMaxH3ImagePipeline.init must be called before generation")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if height <= 0 or width <= 0 or not 1024 <= min_generation_area <= 4096 * 4096:
            raise ValueError("Positive dimensions and min_generation_area in [1024, 4096**2] are required")
        images = [] if image is None else (image if isinstance(image, list) else [image])
        if len(images) > 16:
            raise ValueError("H3-Image accepts at most 16 reference images")
        refs, vision = [], []
        for item in images:
            upright = ImageOps.exif_transpose(item).convert("RGB")
            if min(upright.size) < 32:
                raise ValueError("Reference images must have at least 32 pixels on each side")
            for target, bounds in ((refs, (1024, 2048)), (vision, (1024, 1024))):
                size = bound_short_edge(*upright.size, *bounds, 32)
                target.append(upright if size == upright.size else upright.resize(size, Image.Resampling.LANCZOS))
        canvas_h, canvas_w = get_image_canvas_size(height, width, min_generation_area, 32)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        text = self.text_stage.encode_image_prompt(prompt, vision)
        encoded = (
            self.vae_stage.encode_references(refs, generator.get_state())
            if refs
            else {"latents": [], "rng_state": generator.get_state()}
        )
        latent = self.denoising_stage.denoise(text, encoded["latents"], canvas_h, canvas_w, encoded["rng_state"])
        if callable(latent):
            latent = latent()
        pixels = self.vae_stage.decode_image(latent)[0].permute(1, 2, 0).numpy()
        result = Image.fromarray((pixels * 255).clip(0, 255).astype("uint8"))
        if (canvas_h, canvas_w) != (height, width):
            result = result.resize((width, height), Image.Resampling.LANCZOS)
        return result
