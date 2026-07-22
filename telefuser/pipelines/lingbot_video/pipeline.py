"""Precision-first LingBot-Video sampling pipeline.

Sampling order and TI2V conditioning behavior are adapted from the
Apache-2.0 licensed upstream LingBot-Video implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.worker.parallel_worker import ParallelWorker

from .data import (
    LingBotVideoModelConfig,
    LingBotVideoRequest,
    default_negative_caption,
    preprocess_ti2v_image,
    validate_frame_count,
)
from .denoising import LingBotVideoDenoisingStage, reinject_ti2v_condition
from .text_encoding import LingBotVideoTextEncodingStage
from .vae import (
    LingBotVideoVAEDecodeStage,
    LingBotVideoVAEEncodeStage,
    first_frame_condition_mask,
    latent_shape,
)


@dataclass
class LingBotVideoPipelineConfig:
    """Configuration for LingBot-Video model stages and sampling."""

    model: LingBotVideoModelConfig = field(default_factory=LingBotVideoModelConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    guidance_scale: float = 3.0
    num_inference_steps: int = 50
    shift: float = 3.0
    batch_cfg: bool = False
    enable_denoising_parallel: bool = False

    def __post_init__(self) -> None:
        if self.guidance_scale < 0 or self.num_inference_steps <= 0 or self.shift <= 0:
            raise ValueError("guidance_scale must be non-negative and steps must be positive")


@dataclass(frozen=True)
class LingBotVideoPromptConditions:
    """Text conditions produced for one source-order CFG generation."""

    positive_prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    positive_attention_mask: torch.Tensor
    negative_attention_mask: torch.Tensor
    has_visual_condition: bool


@dataclass(frozen=True)
class LingBotVideoGeneration:
    """Generated output together with prompt conditions that produced it."""

    output: torch.Tensor
    prompt_conditions: LingBotVideoPromptConditions


class LingBotVideoPipeline(BasePipeline):
    """LingBot-Video pipeline initialized from ModuleManager-owned components."""

    def __init__(self, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.config: LingBotVideoPipelineConfig | None = None
        self.text_stage: LingBotVideoTextEncodingStage | None = None
        self.denoising_stage: LingBotVideoDenoisingStage | None = None
        self.vae_encode_stage: LingBotVideoVAEEncodeStage | None = None
        self.vae_decode_stage: LingBotVideoVAEDecodeStage | None = None
        self.scheduler: FlowUniPCMultistepScheduler | None = None
        self.variant: str = "dense"

    def init(self, module_manager: ModuleManager, config: LingBotVideoPipelineConfig) -> None:
        """Initialize all LingBot-Video stages from a module manager."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        self.variant = config.model.variant

        self.text_stage = LingBotVideoTextEncodingStage("text_encoder", module_manager, config.text_encoding_config)
        denoising_stage = LingBotVideoDenoisingStage(
            "transformer", module_manager, config.dit_config, batch_cfg=config.batch_cfg
        )
        self.denoising_stage = denoising_stage
        self.vae_encode_stage = LingBotVideoVAEEncodeStage("vae_encode", module_manager, config.vae_config)
        self.vae_decode_stage = LingBotVideoVAEDecodeStage("vae_decode", module_manager, config.vae_config)
        self.scheduler = module_manager.fetch_module("scheduler")

        if config.enable_denoising_parallel and not dist.is_initialized():
            self.denoising_stage = ParallelWorker(denoising_stage)
        else:
            denoising_stage.parallel_models()

    def _get_stages(self) -> list[object]:
        """Return independently managed stages, including a distributed DiT worker."""
        return [
            stage
            for stage in (self.text_stage, self.denoising_stage, self.vae_encode_stage, self.vae_decode_stage)
            if stage is not None
        ]

    def stop(self) -> None:
        """Close any owned distributed stage workers during service shutdown."""
        for stage in self._get_stages():
            close = getattr(stage, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _resolve_stage_result(value: object) -> torch.Tensor:
        """Resolve a ParallelWorker deferred result while preserving direct-stage outputs."""
        result = value() if callable(value) else value
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"LingBot stage returned {type(result).__name__}, expected a tensor")
        return result

    @staticmethod
    def validate_request(request: LingBotVideoRequest) -> LingBotVideoRequest:
        """Validate and return a request before allocating model resources."""
        validate_frame_count(request.num_frames)
        return request

    def release_gpu_resources(self) -> None:
        """Offload attached stages so a separately loaded refiner can use the GPU."""
        for stage in (self.text_stage, self.denoising_stage, self.vae_encode_stage, self.vae_decode_stage):
            close = getattr(stage, "close", None)
            if callable(close):
                close()
                continue
            offload_models = getattr(stage, "offload_models", None)
            if callable(offload_models):
                offload_models()
                stage.onload_models_flag = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(
        self,
        request: LingBotVideoRequest,
        *,
        negative_caption: str | None = None,
        generator: torch.Generator | None = None,
        decode: bool = True,
    ) -> torch.Tensor:
        """Run source-order sampling and return only the generated output."""
        return self.generate(request, negative_caption=negative_caption, generator=generator, decode=decode).output

    def generate(
        self,
        request: LingBotVideoRequest,
        *,
        negative_caption: str | None = None,
        generator: torch.Generator | None = None,
        decode: bool = True,
    ) -> LingBotVideoGeneration:
        """Run source-order sampling and retain its exact CFG prompt conditions."""
        self.validate_request(request)
        if self.config is None or self.text_stage is None or self.denoising_stage is None or self.scheduler is None:
            raise RuntimeError("LingBot-Video runtime has not been configured")
        if negative_caption is None:
            negative_caption = default_negative_caption(request.num_frames)
        condition_pixels: torch.Tensor | None = None
        vision_images: list[object] | None = None
        if request.image is not None:
            if self.vae_encode_stage is None or not isinstance(request.image, torch.Tensor):
                raise ValueError("TI2V requires a VAE encode stage and an RGB tensor image")
            source_image = request.image
            if source_image.ndim == 5:
                if source_image.shape[2] < 1:
                    raise ValueError("TI2V source image must include a frame")
                source_image = source_image[:, :, 0]
            condition_pixels = preprocess_ti2v_image(source_image, height=request.height, width=request.width)
            vision_images = [self.text_stage.prepare_ti2v_vlm_image(condition_pixels)]
        positive, positive_mask = self.text_stage.encode(request.caption, images=vision_images)
        negative, negative_mask = self.text_stage.encode(negative_caption, images=vision_images)
        prompt_conditions = LingBotVideoPromptConditions(
            positive_prompt_embeds=positive,
            negative_prompt_embeds=negative,
            positive_attention_mask=positive_mask,
            negative_attention_mask=negative_mask,
            has_visual_condition=condition_pixels is not None,
        )
        frames, latent_height, latent_width = latent_shape(request.num_frames, request.height, request.width)
        condition = None
        condition_mask = None
        if condition_pixels is not None:
            condition = self.vae_encode_stage.encode(condition_pixels, generator=generator)
            condition_mask = first_frame_condition_mask(frames, device=self.denoising_stage.device)
        latent = torch.randn(
            1,
            self.config.model.in_channels,
            frames,
            latent_height,
            latent_width,
            device=self.denoising_stage.device,
            dtype=torch.float32,
            generator=generator,
        )
        if self.config.enable_denoising_parallel:
            latent = self._resolve_stage_result(
                self.denoising_stage.denoise(
                    latent,
                    positive,
                    negative,
                    positive_mask,
                    negative_mask,
                    self.config.guidance_scale,
                    self.config.num_inference_steps,
                    self.config.shift,
                    condition,
                    condition_mask,
                )
            )
        else:
            self.scheduler.set_timesteps(self.config.num_inference_steps, device=latent.device, shift=self.config.shift)
            for step in self.scheduler.timesteps:
                if condition is not None:
                    latent = reinject_ti2v_condition(latent, condition, condition_mask)
                prediction = self._resolve_stage_result(
                    self.denoising_stage.predict_noise_with_cfg(
                        latent,
                        step.expand(latent.shape[0]),
                        positive,
                        negative,
                        positive_mask,
                        negative_mask,
                        self.config.guidance_scale,
                    )
                )
                latent = self.scheduler.step(prediction, step, latent)
            if condition is not None:
                latent = reinject_ti2v_condition(latent, condition, condition_mask)
        if not decode:
            return LingBotVideoGeneration(latent, prompt_conditions)
        if self.vae_decode_stage is None:
            raise RuntimeError("decode=True requires a configured VAE decode stage")
        output = self._resolve_stage_result(self.vae_decode_stage.decode(latent))
        return LingBotVideoGeneration(output, prompt_conditions)
