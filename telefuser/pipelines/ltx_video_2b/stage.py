"""Generation stage for the Diffusers-format LTX-Video 2B checkpoint."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, QuantKernelBackend, QuantType
from telefuser.core.module_manager import ModuleManager
from telefuser.ops.torchao_nvfp4_linear import (
    enable_torchao_nvfp4_triton,
    replace_linear_layers_with_torchao_nvfp4,
)
from telefuser.platforms import current_platform


class LTXVideo2BGenerationStage(BaseStage):
    """Own official LTX sampling while TeleFuser controls quantization and residency."""

    def __init__(self, manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx_video_2b_generation", config)
        self.pipeline = manager.fetch_module("ltx_video_2b_diffusers_pipeline")
        if self.pipeline is None:
            raise ValueError("ModuleManager must contain 'ltx_video_2b_diffusers_pipeline'")
        self.nvfp4_linear_count = 0
        quant = config.quant_config
        if quant.enabled:
            if quant.quant_type != QuantType.NVFP4:
                raise NotImplementedError("LTX-Video 2B supports NVFP4 or BF16")
            if quant.kernel_backend not in {QuantKernelBackend.AUTO, QuantKernelBackend.TORCHAO}:
                raise NotImplementedError("LTX-Video 2B NVFP4 currently uses the TorchAO backend")
            self.nvfp4_linear_count = replace_linear_layers_with_torchao_nvfp4(
                self.pipeline.transformer,
                include_names=quant.quantize_modules,
                exclude_names=quant.skip_modules,
            )
            self.pipeline.transformer.to(self.device)
            self.pipeline.vae.to(self.device)
            enable_torchao_nvfp4_triton(self.pipeline.transformer)
            self.pipeline.vae.use_framewise_decoding = True
        else:
            # Official Diffusers offload order: text_encoder -> transformer -> VAE.
            self.pipeline.enable_model_cpu_offload(device=self.device)
        self.pipeline.transformer.eval()

    def _encode_prompt(
        self, prompt: str, negative_prompt: str | None, max_sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.pipeline.text_encoder.to(self.device)
        encoded = self.pipeline.encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            device=self.device,
            dtype=self.torch_dtype,
            max_sequence_length=max_sequence_length,
        )
        self.pipeline.text_encoder.to("cpu")
        current_platform.empty_cache()
        return encoded

    @torch.inference_mode()
    def process(
        self,
        prompt: str,
        *,
        negative_prompt: str | None,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: int,
        num_inference_steps: int,
        guidance_scale: float,
        decode_timestep: float,
        decode_noise_scale: float | None,
        seed: int,
        max_sequence_length: int,
    ) -> np.ndarray:
        kwargs: dict[str, Any] = {}
        if self.model_runtime_config.quant_config.enabled:
            prompt_embeds, prompt_mask, negative_embeds, negative_mask = self._encode_prompt(
                prompt, negative_prompt, max_sequence_length
            )
            kwargs.update(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=negative_embeds,
                negative_prompt_attention_mask=negative_mask,
            )
        else:
            kwargs.update(prompt=prompt, negative_prompt=negative_prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipeline(
            **kwargs,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,
            output_type="np",
        )
        return result.frames[0]
