# SPDX-License-Identifier: Apache-2.0
"""Native H3-Image conditioning, six-head PDD sampling and image decoding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import AttnImplType, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.pipelines.minimax_h3.packed_tokens import minimax_h3_unpatchify_video_tokens
from telefuser.pipelines.minimax_h3.text_encoding import MiniMaxH3TextEncodingStage

from .data import build_image_packed_sequence, patchify_video_latents


class MiniMaxH3ImageTextStage(MiniMaxH3TextEncodingStage):
    @with_model_offload(["text_encoder"])
    @torch.inference_mode()
    def encode_image_prompt(self, prompt: str, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        ids: list[int] = []
        tags: list[int] = []
        vision: dict[str, Any] = {}
        if images:
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            merge = self.processor.image_processor.merge_size**2
            for i, grid in enumerate(vision["image_grid_thw"]):
                label = self.tokenizer(f"Picture {i + 1}: ", add_special_tokens=False)["input_ids"][:4096]
                block = (
                    [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * (int(grid.prod()) // merge)
                    + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                ids.extend(label + block)
                tags.extend([1] * len(label) + [0] * len(block))
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"][:4096]
        ids.extend(prompt_ids)
        tags.extend([1] * len(prompt_ids))
        hidden = self.text_encoder.encode_ids(
            torch.tensor(ids, dtype=torch.long),
            **{k: vision[k] for k in ("pixel_values", "image_grid_thw") if k in vision},
        )
        return {"hidden_states": hidden.cpu(), "token_tags": torch.tensor(tags, dtype=torch.long)}


class MiniMaxH3ImageVAEStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("minimax_h3_image_vae", config)
        self.vae = module_manager.fetch_module("minimax_h3_image_vae")
        if self.vae is None:
            raise ValueError("ModuleManager must contain 'minimax_h3_image_vae'")
        self.model_names = ["vae"]

    def _stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            torch.tensor(v, device=self.device).view(1, -1, 1, 1)
            for v in (self.vae.config.latents_mean, self.vae.config.latents_std)
        )

    @with_model_offload(["vae"])
    @torch.inference_mode()
    def encode_references(self, images: list[Image.Image], rng_state: torch.Tensor) -> dict[str, Any]:
        generator = torch.Generator(device="cpu").set_state(rng_state.cpu())
        mean, std = self._stats()
        latents = []
        for image in images:
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].float() / 255
            pixels = pixels[..., : pixels.shape[-2] // 16 * 16, : pixels.shape[-1] // 16 * 16]
            pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            posterior = self.vae.encode(((pixels - pixel_mean) / pixel_std).to(self.device)).latent_dist
            latent = posterior.sample(generator=generator).half().float()
            latent = latent[..., : latent.shape[-2] // 2 * 2, : latent.shape[-1] // 2 * 2]
            latents.append(((latent - mean) / std).cpu())
        return {"latents": latents, "rng_state": generator.get_state()}

    @with_model_offload(["vae"])
    @torch.inference_mode()
    def decode_image(self, latent: torch.Tensor) -> torch.Tensor:
        mean, std = self._stats()
        pixels = self.vae.decode(latent.to(self.device) * std + mean).sample.float()
        pixel_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (pixels * pixel_std + pixel_mean).clamp(0, 1).cpu()


class MiniMaxH3ImageDenoisingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig, sigmas: list[float]) -> None:
        super().__init__("minimax_h3_image_denoising", config)
        self.transformer = module_manager.fetch_module("minimax_h3_image_transformer")
        if self.transformer is None:
            raise ValueError("ModuleManager must contain 'minimax_h3_image_transformer'")
        if len(sigmas) != self.transformer.final_layer.video_out.weight.shape[0] + 1:
            raise ValueError("The PDD schedule must have one interval per output head")
        if sigmas[0] != 1 or sigmas[-1] != 0 or any(a <= b for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError("PDD sigmas must decrease strictly from one to zero")
        self.timesteps = torch.tensor([1.0 - sigma for sigma in sigmas[:-1]], dtype=torch.float32)
        self.sigmas = torch.cat([1 - self.timesteps, torch.zeros(1)])
        self.transformer.set_attention_config(config.attention_config)
        self.model_names = ["transformer"]
        self._request_serial = 0

    def parallel_models(self) -> None:
        config = self.model_runtime_config.parallel_config
        if any(v != 1 for v in (config.dp_degree, config.cfg_degree, config.sp_ring_degree, config.pp_degree)):
            raise ValueError("H3-Image supports Ulysses SP and TP; other parallel degrees must be one")
        if config.enable_fsdp:
            raise NotImplementedError("H3-Image FSDP is not implemented")
        mesh = create_device_mesh_from_config(config)
        if config.tp_degree > 1:
            self.transformer.enable_tp(mesh)
        if config.sp_ulysses_degree > 1:
            self.transformer.enable_usp(mesh)
        # Keep shards on CPU until conditioning has released its device memory.

    @with_model_offload(["transformer"])
    @torch.inference_mode()
    def denoise(
        self,
        text: dict[str, torch.Tensor],
        references: list[torch.Tensor],
        height: int,
        width: int,
        rng_state: torch.Tensor,
    ) -> torch.Tensor:
        quant = self.model_runtime_config.quant_config
        if quant.enabled and self.transformer.quant_type != quant.quant_type:
            self.transformer.enable_quant(quant)
        generator = torch.Generator(device="cpu").set_state(rng_state.cpu())
        references = [reference.cpu() for reference in references]
        channels = self.transformer.config.latents_dim
        patch = tuple(self.transformer.config.patch_size)
        latent = torch.randn((1, channels, 1, height // 16, width // 16), generator=generator)
        target = patchify_video_latents(latent, patch).to(self.device)
        ref_rows = []
        for ref in references:
            noise = torch.randn(ref.shape, generator=generator)
            ref_rows.append(patchify_video_latents((0.999 * ref + (1.0 - 0.999) * noise).unsqueeze(2), patch))
        positions, tags, image_ids, text_ids, nref = build_image_packed_sequence(
            text["token_tags"].cpu(), [tuple(r.shape[-2:]) for r in references], height // 16, width // 16, patch, 0
        )
        actual = len(tags)
        degree = self.model_runtime_config.parallel_config.sp_ulysses_degree
        alignment = degree
        if self.model_runtime_config.attention_config.attn_impl in {
            AttnImplType.SOL_ATTN,
            AttnImplType.SAGE_ATTN_2_8_8_SM90,
        }:
            alignment = math.lcm(64, degree)
        padded = (actual + alignment - 1) // alignment * alignment
        if alignment > degree and padded - actual >= 64:
            raise ValueError("H3 optimized attention requires fewer than 64 padding rows; use a power-of-two SP degree")
        positions = torch.nn.functional.pad(positions, (0, 0, 0, padded - actual)).to(self.device)
        tags = torch.nn.functional.pad(tags, (0, padded - actual)).to(self.device)
        image_ids, text_ids = image_ids.to(self.device), text_ids.to(self.device)
        state = torch.zeros((1, padded, target.shape[-1]), device=self.device, dtype=torch.float32)
        if ref_rows:
            state[0, image_ids[:nref]] = torch.cat(ref_rows).to(self.device)
        target_ids = image_ids[nref:]
        state[0, target_ids] = target
        cu = torch.tensor([0, actual] + ([padded] if padded > actual else []), device=self.device, dtype=torch.int32)
        self._request_serial += 1
        try:
            for step, timestep in enumerate(self.timesteps):
                row_t = torch.full((padded,), float(timestep), dtype=torch.float32)
                row_t[image_ids[:nref].cpu()] = 0.999
                unique, inverse = torch.unique(row_t, sorted=True, return_inverse=True)
                velocity = self.transformer(
                    x=state,
                    img_position_ids=positions[None],
                    unique_timesteps=unique.to(self.device),
                    inverse_indices=inverse.to(self.device),
                    token_tags=tags,
                    update_mask=torch.ones(len(image_ids), device=self.device, dtype=torch.bool),
                    prompt_embeds=text["hidden_states"].to(self.device),
                    img_pos_info={"position_ids": image_ids},
                    text_pos_info={"position_ids": text_ids},
                    img_pos_for_infer_output_info={"position_ids": image_ids},
                    packed_seq_params={"cu_seqlens_q": cu},
                    pdd_step_index=step,
                    sparse_step_index=step,
                    static_cache_key=self._request_serial,
                )
                state[0, target_ids] += (self.sigmas[step] - self.sigmas[step + 1]).to(self.device) * velocity[
                    nref:
                ].float()
            rows = state[0, target_ids].cpu()
        finally:
            # Prompt/RoPE caches belong to a request, including failed requests.
            for name in (
                "_static_cache_key",
                "_static_prompt",
                "_static_rope_cos_sin",
                "_static_sequence_lengths",
                "_static_cu_seqlens",
            ):
                setattr(self.transformer, name, None)
        return minimax_h3_unpatchify_video_tokens(
            rows, latent_shape=(1, height // 16 // patch[1], width // 16 // patch[2], channels), patch_size=patch
        )[:, :, 0]
