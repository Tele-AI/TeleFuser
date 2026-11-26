"""TeleFuser integration for Lightricks LTX-Video 2B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from diffusers import LTXPipeline

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig, QuantConfig, QuantKernelBackend, QuantType
from telefuser.core.module_manager import ModuleManager

from .stage import LTXVideo2BGenerationStage


def _nvfp4_runtime(profile: str) -> ModelRuntimeConfig:
    if profile not in {"quality", "full"}:
        raise ValueError("nvfp4_profile must be 'quality' or 'full'")
    return ModelRuntimeConfig(
        torch_dtype=torch.bfloat16,
        quant_config=QuantConfig(
            enabled=True,
            quant_type=QuantType.NVFP4,
            kernel_backend=QuantKernelBackend.TORCHAO,
            quantize_modules=(".ff.",) if profile == "quality" else None,
            skip_modules=(),
        ),
    )


@dataclass
class LTXVideo2BPipelineConfig:
    """Runtime configuration for LTX-Video 2B generation."""

    generation_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    nvfp4_profile: str = "quality"

    def __post_init__(self) -> None:
        quant = self.generation_config.quant_config
        if quant.enabled and quant.quant_type != QuantType.NVFP4:
            raise NotImplementedError("LTX-Video 2B supports NVFP4 or BF16")
        if self.nvfp4_profile not in {"quality", "full"}:
            raise ValueError("nvfp4_profile must be 'quality' or 'full'")


class LTXVideo2BPipeline(BasePipeline):
    """Run the official LTX-Video workload with TeleFuser NVFP4 lifecycle."""

    def __init__(self, device: str | torch.device = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.config: LTXVideo2BPipelineConfig | None = None
        self.generation_stage: LTXVideo2BGenerationStage | None = None

    def init(self, module_manager: ModuleManager, config: LTXVideo2BPipelineConfig) -> None:
        self.config = config
        self.generation_stage = LTXVideo2BGenerationStage(module_manager, config.generation_config)
        self._model_info = module_manager.get_model_info()

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        quantization: str = "nvfp4",
        nvfp4_profile: str = "quality",
        **kwargs: Any,
    ) -> "LTXVideo2BPipeline":
        """Load a local directory or Hugging Face Diffusers checkpoint."""
        if quantization not in {"nvfp4", "bf16"}:
            raise ValueError("quantization must be 'nvfp4' or 'bf16'")
        upstream = LTXPipeline.from_pretrained(model_id_or_path, torch_dtype=torch_dtype, **kwargs)
        manager = ModuleManager(torch_dtype=torch_dtype, device="cpu")
        manager.add_module(upstream, "ltx_video_2b_diffusers_pipeline", model_id_or_path)
        config = LTXVideo2BPipelineConfig(
            generation_config=_nvfp4_runtime(nvfp4_profile)
            if quantization == "nvfp4"
            else ModelRuntimeConfig(torch_dtype=torch_dtype),
            nvfp4_profile=nvfp4_profile,
        )
        pipeline = cls(device=device, torch_dtype=torch_dtype)
        pipeline.init(manager, config)
        return pipeline

    def _get_stages(self) -> list[LTXVideo2BGenerationStage]:
        return [self.generation_stage] if self.generation_stage is not None else []

    @property
    def nvfp4_linear_count(self) -> int:
        return self.generation_stage.nvfp4_linear_count if self.generation_stage is not None else 0

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 161,
        frame_rate: int = 24,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        decode_timestep: float = 0.03,
        decode_noise_scale: float | None = 0.025,
        seed: int = 123,
        max_sequence_length: int = 128,
    ) -> np.ndarray:
        if self.generation_stage is None:
            raise RuntimeError("LTXVideo2BPipeline.init must be called before generation")
        if height <= 0 or width <= 0 or height % 32 or width % 32:
            raise ValueError("LTX height and width must be positive multiples of 32")
        if num_frames < 1 or (num_frames - 1) % 8:
            raise ValueError("LTX num_frames must be 8*k+1")
        return self.generation_stage.process(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            seed=seed,
            max_sequence_length=max_sequence_length,
        )
