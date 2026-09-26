"""Stage-composed TeleFuser pipeline for Qwen-Image 2.1."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.qwen_image_21_dit import QwenImage21DiT
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.hf_model_utils import resolve_hf_path
from telefuser.utils.logging import logger

from .dit_denoising_21 import DitDenoising21Stage
from .text_encoding_21 import TextEncoding21Stage
from .vae_21 import VAE21Stage


def _diffusers_components() -> tuple[type, type, type]:
    """Import only the standalone VAE, text encoder, and processor classes."""
    try:
        from diffusers.utils import import_utils

        if getattr(import_utils, "_xformers_available", False):
            import_utils._xformers_available = False
        from diffusers import AutoencoderKLQwenImage21
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            "Qwen-Image 2.1 requires Diffusers with AutoencoderKLQwenImage21 and "
            "Transformers with Qwen3VLForConditionalGeneration."
        ) from exc
    return AutoencoderKLQwenImage21, Qwen3VLForConditionalGeneration, AutoProcessor


@dataclass
class QwenImage21PipelineConfig:
    """Runtime configuration for the stage-composed Qwen-Image 2.1 pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_text_encoding_parallel: bool = False
    enable_metrics: bool = False


class QwenImage21Pipeline(BasePipeline):
    """Qwen-Image 2.1 pipeline assembled from text, DiT, and VAE stages."""

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 32
        self.width_division_factor = 32

    def _get_stages(self) -> list[Any]:
        return [self.text_encoding_stage, self.denoise_stage, self.vae_stage]

    def init(self, module_manager: ModuleManager, config: QwenImage21PipelineConfig) -> None:
        self._model_info = module_manager.get_model_info()
        self.config = config
        if config.sample_solver != "euler":
            raise NotImplementedError(f"solver {config.sample_solver} is not supported")
        scheduler = FlowMatchScheduler("Qwen-Image")
        self.text_encoding_stage = TextEncoding21Stage("text_encoding", module_manager, config.text_encoding_config)
        self.denoise_stage = DitDenoising21Stage("denoise", module_manager, config.dit_config, scheduler)
        self.vae_stage = VAE21Stage("vae", module_manager, config.vae_config)
        if config.enable_metrics:
            self.enable_metrics()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        image: Image.Image | list[Image.Image] | None = None,
        seed: int | None = None,
        height: int | None = None,
        width: int | None = None,
        cfg_scale: float = 1.0,
        true_cfg_scale: float | None = None,
        num_inference_steps: int = 40,
        num_images_per_prompt: int = 1,
        output_resolution: int = 1024,
        use_kv_cache: bool = False,
        **_: Any,
    ) -> list[Image.Image]:
        """Generate an image from text and optional condition images."""
        del use_kv_cache
        if true_cfg_scale is not None:
            cfg_scale = true_cfg_scale
        condition_images = None
        if image is not None:
            condition_images = image if isinstance(image, list) else [image]
            if not condition_images or len(condition_images) > 10:
                raise ValueError("Qwen-Image 2.1 accepts one to ten condition images")
            if not all(isinstance(item, Image.Image) for item in condition_images):
                raise TypeError("Condition images must be PIL images")
            ratio = condition_images[-1].width / condition_images[-1].height
            width_at_area = math.sqrt(output_resolution**2 * ratio)
            calculated_width = round(width_at_area / 32) * 32
            calculated_height = round((width_at_area / ratio) / 32) * 32
            height = height or calculated_height
            width = width or calculated_width
        height = height or output_resolution
        width = width or output_resolution
        height, width = self.check_resize_height_width(height, width)
        resized_conditions = None
        if condition_images is not None:
            resized_conditions = []
            for condition in condition_images:
                ratio = condition.width / condition.height
                width_at_area = math.sqrt(output_resolution**2 * ratio)
                input_width = round(width_at_area / 32) * 32
                input_height = round((width_at_area / ratio) / 32) * 32
                resized_conditions.append(
                    self.vae_stage.image_processor.resize(
                        condition.convert("RGBA"), width=input_width, height=input_height
                    )
                )
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        batch_size *= num_images_per_prompt
        latent_height, latent_width = height // 16, width // 16
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        latents = torch.randn(
            batch_size,
            latent_height * latent_width,
            64,
            generator=generator,
            device=self.device,
            dtype=self.torch_dtype,
        )
        negative = negative_prompt if cfg_scale > 1 else None
        prompt_embeds, prompt_mask, image_mask, negative_embeds, negative_mask, negative_image_mask = (
            self.text_encoding_stage.process(prompt, negative, num_images_per_prompt, resized_conditions)
        )
        condition_latents = None
        condition_shapes = []
        if resized_conditions is not None:
            condition_latents, condition_shapes = self.vae_stage.encode_conditions(resized_conditions, batch_size)
        latents = self.denoise_stage.process(
            latents=latents,
            condition_latents=condition_latents,
            img_shapes=[[*condition_shapes, (1, latent_height, latent_width)]] * batch_size,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            image_mask=image_mask,
            negative_image_mask=negative_image_mask,
            negative_embeds=negative_embeds,
            negative_mask=negative_mask,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
        )
        return self.vae_stage.process(latents, latent_height, latent_width)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | None = None,
        attention_config: AttentionConfig | None = None,
        enable_metrics: bool = False,
        **kwargs: Any,
    ) -> "QwenImage21Pipeline":
        """Load standalone modules into ModuleManager and compose stages."""
        model_root = resolve_hf_path(model_id_or_path, cache_dir)
        component_paths = {
            "transformer": os.path.join(model_root, "transformer"),
            "vae": os.path.join(model_root, "vae"),
            "text_encoder": os.path.join(model_root, "text_encoder"),
            "processor": os.path.join(model_root, "processor"),
        }
        for name, path in component_paths.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Qwen-Image 2.1 component directory not found: {name} ({path})")

        vae_cls, text_encoder_cls, processor_cls = _diffusers_components()
        manager = ModuleManager(torch_dtype=torch_dtype, device="cpu")
        manager.load_model(
            component_paths["transformer"],
            device="cpu",
            torch_dtype=torch_dtype,
            name="dit",
            model_class=QwenImage21DiT,
            model_resource="diffusers",
        )
        vae = vae_cls.from_pretrained(component_paths["vae"], torch_dtype=torch_dtype)
        text_encoder = text_encoder_cls.from_pretrained(component_paths["text_encoder"], torch_dtype=torch_dtype)
        processor = processor_cls.from_pretrained(component_paths["processor"])
        manager.add_module(vae, "vae", component_paths["vae"])
        manager.add_module(text_encoder, "text_encoder", component_paths["text_encoder"])
        manager.add_module(processor, "processor", component_paths["processor"])

        pipeline = cls(device=device, torch_dtype=torch_dtype)
        config = QwenImage21PipelineConfig(enable_metrics=enable_metrics)
        config.sample_solver = kwargs.pop("sample_solver", "euler")
        config.dit_config.attention_config = attention_config or AttentionConfig.dense_attention(
            AttnImplType.TORCH_SDPA
        )
        pipeline.init(manager, config)
        logger.info("Successfully loaded native Qwen-Image 2.1 stages from %s", model_root)
        return pipeline
