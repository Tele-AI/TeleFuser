# SPDX-License-Identifier: Apache-2.0
"""PAN2 text conditioning, flow matching and Wan2.2 decoding stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config

from .data import resize_and_center_crop


class PAN2TextStage(BaseStage):
    def __init__(self, manager: ModuleManager, config: ModelRuntimeConfig, tokenizer: object) -> None:
        super().__init__("pan2_text_encoding", config)
        self.text_encoder = manager.fetch_module("pan2_text_encoder")
        if self.text_encoder is None:
            raise ValueError("ModuleManager must contain 'pan2_text_encoder'")
        self.tokenizer = tokenizer
        self.model_names = ["text_encoder"]

    def _encode(self, prompt: str, max_sequence_length: int) -> torch.Tensor:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Generate the video with the following instructions.",
            },
            {"role": "user", "content": prompt or " "},
        ]
        header = self.tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True
        ).input_ids
        starts = [i + len(header) for i in range(len(ids)) if ids[i : i + len(header)] == header]
        if not starts:
            raise ValueError("PAN2 tokenizer chat template did not contain the user header")
        start = starts[0]
        ids = torch.tensor([ids[: start + max_sequence_length]], device=self.device)
        return self.text_encoder(ids)[:, start:].cpu()

    @with_model_offload(["text_encoder"])
    @torch.inference_mode()
    def encode(
        self, prompt: str, negative_prompt: str | None, max_sequence_length: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (
            self._encode(prompt, max_sequence_length),
            self._encode(negative_prompt, max_sequence_length) if negative_prompt is not None else None,
        )


class PAN2VAEStage(BaseStage):
    def __init__(self, manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("pan2_vae", config)
        self.vae = manager.fetch_module("pan2_vae")
        if self.vae is None:
            raise ValueError("ModuleManager must contain 'pan2_vae'")
        self.model_names = ["vae"]

    def _stats(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            torch.tensor(v, device=self.device, dtype=dtype).view(1, -1, 1, 1, 1)
            for v in (self.vae.config.latents_mean, self.vae.config.latents_std)
        )

    @with_model_offload(["vae"])
    @torch.inference_mode()
    def encode_image(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        image = resize_and_center_crop(image.convert("RGB"), height, width)
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None]
        pixels = (pixels.float() / 255 * 2 - 1).to(device=self.device, dtype=self.torch_dtype)
        latents = self.vae.encode(pixels)
        mean, std = self._stats(latents.dtype)
        return ((latents - mean) / std).cpu()

    @with_model_offload(["vae"])
    @torch.inference_mode()
    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats(torch.float32)
        latents = (latents.to(self.device, dtype=torch.float32) * std + mean).to(self.torch_dtype)
        pixels = self.vae.decode(latents)
        return (pixels * 0.5 + 0.5).clamp(0, 1).float().cpu()


class PAN2DenoisingStage(BaseStage):
    def __init__(self, manager: ModuleManager, config: ModelRuntimeConfig, scheduler_path: str) -> None:
        super().__init__("pan2_denoising", config)
        self.transformer = manager.fetch_module("pan2_transformer")
        if self.transformer is None:
            raise ValueError("ModuleManager must contain 'pan2_transformer'")
        self.transformer.set_attention_config(config.attention_config)
        self.scheduler_config = FlowMatchEulerDiscreteScheduler.load_config(Path(scheduler_path).parent)
        self.model_names = ["transformer"]

    def parallel_models(self) -> None:
        config = self.model_runtime_config.parallel_config
        if any(v != 1 for v in (config.dp_degree, config.cfg_degree, config.sp_ring_degree, config.pp_degree)):
            raise ValueError("PAN2 supports Ulysses SP and TP; other parallel degrees must be one")
        if config.enable_fsdp:
            raise NotImplementedError("PAN2 FSDP is not implemented")
        mesh = create_device_mesh_from_config(config)
        if config.tp_degree > 1:
            self.transformer.enable_tp(mesh)
        if config.sp_ulysses_degree > 1:
            self.transformer.enable_usp(mesh)

    @with_model_offload(["transformer"])
    @torch.inference_mode()
    def denoise(
        self,
        text: torch.Tensor,
        negative: torch.Tensor | None,
        image: torch.Tensor | None,
        height: int,
        width: int,
        num_frames: int,
        steps: int,
        guidance: float,
        seed: int,
    ) -> torch.Tensor:
        quant = self.model_runtime_config.quant_config
        if quant.enabled and self.transformer.quant_type != quant.quant_type:
            self.transformer.enable_quant(quant)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 0.0, steps + 1)[:-1], device=self.device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        channels = self.transformer.config.out_channels
        latent = torch.randn(
            (1, channels, (num_frames - 1) // 4 + 1, height // 16, width // 16), generator=generator
        ).to(self.device, dtype=torch.float32)
        condition = latent.new_zeros((1, channels + 1, *latent.shape[2:]))
        if image is not None:
            condition[:, :channels, :1] = image.to(condition)
            condition[:, channels:, :1] = 1
        text = text.to(self.device, dtype=self.torch_dtype)
        negative = negative.to(self.device, dtype=self.torch_dtype) if negative is not None else None
        for i, timestep in enumerate(scheduler.timesteps):
            inputs = torch.cat([latent, condition], dim=1).to(self.torch_dtype)
            pred = self.transformer(
                hidden_states=inputs, timestep=timestep.expand(1), encoder_hidden_states=text, sparse_step_index=i
            ).sample
            if negative is not None:
                uncond = self.transformer(
                    hidden_states=inputs,
                    timestep=timestep.expand(1),
                    encoder_hidden_states=negative,
                    sparse_step_index=i,
                ).sample
                guided = uncond + guidance * (pred - uncond)
                if image is not None:
                    guided[:, :, :1] = pred[:, :, :1]
                pred = guided
            latent = scheduler.step(pred.float(), timestep, latent, return_dict=False)[0]
        return latent.cpu()
