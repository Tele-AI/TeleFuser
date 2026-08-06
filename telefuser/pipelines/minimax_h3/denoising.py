# SPDX-License-Identifier: Apache-2.0
"""Packed single-branch MiniMax H3 denoising stage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_ulysses_rank, get_ulysses_world_size
from telefuser.distributed.fsdp import shard_model_fsdp2_inference
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger

from .condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from .packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .resolved_plan import MiniMaxH3ResolvedPlan
from .scheduler import MiniMaxH3EulerAncestralEta0SchedulerAdapter
from .text_encoding import MiniMaxH3TextCondition
from .time_request import minimax_h3_time_shift_sigmas
from .vae import MiniMaxH3PreparedCondition

MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0


@torch.inference_mode()
def _minimax_h3_update_target_rows_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_t: torch.Tensor,
    sigma_curr: float,
    sigma_ratio: torch.Tensor,
    one_minus_sigma_ratio: torch.Tensor,
    denoised_scratch: torch.Tensor,
) -> None:
    """Apply the Euler eta=0 update while reusing output and scratch storage."""
    torch.mul(sigma_t, velocity, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)
    if sigma_curr == 0.0:
        return
    torch.mul(one_minus_sigma_ratio, denoised_scratch, out=velocity)
    torch.mul(sigma_ratio, state, out=state)
    torch.add(state, velocity, out=state)


def _build_local_embedding_layout(
    *,
    seq_len: int,
    text_pos: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, int | torch.Tensor]:
    """Resolve request-static packed rows owned by one Ulysses rank."""
    if seq_len % world_size:
        raise ValueError(f"packed seq_len {seq_len} must divide Ulysses degree {world_size}")
    local_seq_len = seq_len // world_size
    row_start = rank * local_seq_len
    row_stop = row_start + local_seq_len

    def local_ids(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = torch.nonzero((positions >= row_start) & (positions < row_stop), as_tuple=False).view(-1)
        global_ids = positions.index_select(0, source_ids)
        return source_ids.to(device), global_ids.to(device)

    text_source_ids, text_global_ids = local_ids(text_pos)
    _, img_global_ids = local_ids(img_pos)
    _, audio_global_ids = local_ids(audio_pos)
    return {
        "row_start": row_start,
        "row_stop": row_stop,
        "text_source_ids": text_source_ids,
        "text_row_ids": text_global_ids - row_start,
        "img_global_ids": img_global_ids,
        "img_row_ids": img_global_ids - row_start,
        "audio_global_ids": audio_global_ids,
        "audio_row_ids": audio_global_ids - row_start,
    }


@dataclass(frozen=True)
class MiniMaxH3DenoiseResult:
    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    packed: dict[str, torch.Tensor]
    runtime_metrics: dict[str, float | int]


@dataclass(frozen=True)
class MiniMaxH3DenoiseRemainder:
    """Parent-consumed denoising outputs that do not enter video decoding."""

    audio_latent: torch.Tensor
    packed: dict[str, torch.Tensor]
    runtime_metrics: dict[str, float | int]


class MiniMaxH3DenoisingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__("minimax_h3_denoising", model_runtime_config)
        self.transformer = module_manager.fetch_module("minimax_h3_transformer")
        if self.transformer is None:
            raise ValueError("ModuleManager must contain 'minimax_h3_transformer'")
        self.scheduler = MiniMaxH3EulerAncestralEta0SchedulerAdapter()
        self.model_names = ["transformer"]
        self._request_serial = 0

    def _ensure_online_quantized(self) -> None:
        quant_config = self.model_runtime_config.quant_config
        if not quant_config.enabled:
            return
        if self.transformer.quant_type == quant_config.quant_type:
            return
        if self.transformer.quant_type is not None:
            raise RuntimeError(
                f"MiniMax H3 DiT is already quantized as {self.transformer.quant_type}, "
                f"cannot apply {quant_config.quant_type}"
            )
        self.transformer.enable_quant(quant_config)
        current_platform.empty_cache()

    def parallel_models(self) -> None:
        parallel_config = self.model_runtime_config.parallel_config
        unsupported = {
            "cfg_degree": parallel_config.cfg_degree,
            "sp_ring_degree": parallel_config.sp_ring_degree,
            "pp_degree": parallel_config.pp_degree,
        }
        invalid = {name: degree for name, degree in unsupported.items() if degree != 1}
        if invalid:
            raise NotImplementedError(f"MiniMax H3 does not support these parallel degrees yet: {invalid}")
        device_mesh = create_device_mesh_from_config(parallel_config)
        self.transformer.device_mesh = device_mesh
        self.transformer.set_attention_config(self.model_runtime_config.attention_config)
        if parallel_config.tp_degree > 1:
            if parallel_config.enable_fsdp:
                raise ValueError("MiniMax H3 DiT tensor parallelism cannot be combined with FSDP")
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                raise ValueError("MiniMax H3 DiT tensor parallelism cannot be combined with model CPU offload")
            logger.info(f"Enabling tensor parallelism for {self.name}")
            self.transformer.enable_tp(device_mesh)
            self.transformer.to(self.device)
            self.onload_models_flag = True
            current_platform.empty_cache()
        if parallel_config.sp_ulysses_degree > 1:
            self.transformer.enable_usp(device_mesh)
        if parallel_config.enable_fsdp:
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                raise ValueError("MiniMax H3 FSDP inference cannot be combined with model CPU offload")
            logger.info(f"Enabling block FSDP2 for {self.name}")
            fp32_parameters = [
                parameter for parameter in self.transformer.parameters() if parameter.dtype == torch.float32
            ]
            self.transformer = shard_model_fsdp2_inference(
                module=self.transformer,
                device_mesh=device_mesh,
                wrap_module_names=self.transformer.get_fsdp_module_names(),
                ignored_states=fp32_parameters,
            )
            self.onload_models_flag = True
            current_platform.empty_cache()

    @staticmethod
    def _reference_blocks(
        conditions: list[MiniMaxH3PreparedCondition],
    ) -> tuple[list[dict[str, object]], torch.Tensor | None, torch.Tensor | None]:
        blocks: list[dict[str, object]] = []
        visual: list[torch.Tensor] = []
        audio: list[torch.Tensor] = []
        for condition in conditions:
            if condition.kind == "image":
                if condition.visual_rows is None:
                    raise ValueError("reference image is missing visual VAE rows")
                blocks.append(
                    {
                        "kind": "image",
                        "latent_h": condition.latent_h,
                        "latent_w": condition.latent_w,
                    }
                )
                visual.append(condition.visual_rows)
            elif condition.kind == "audio":
                if condition.audio_rows is None:
                    raise ValueError("reference audio is missing audio VAE rows")
                blocks.append({"kind": "audio", "ref_audio_t": condition.ref_audio_t})
                audio.append(condition.audio_rows)
            elif condition.kind in {"video", "video_audio"}:
                if condition.visual_rows is None:
                    raise ValueError("reference video is missing visual VAE rows")
                blocks.append(
                    {
                        "kind": condition.kind,
                        "ref_audio_t": condition.ref_audio_t,
                        "latent_t": condition.latent_t,
                        "latent_h": condition.latent_h,
                        "latent_w": condition.latent_w,
                    }
                )
                visual.append(condition.visual_rows)
                if condition.audio_rows is not None:
                    audio.append(condition.audio_rows)
            else:
                raise ValueError(f"unsupported reference block kind {condition.kind!r}")
        visual_rows = None if not visual else torch.cat(visual, dim=0)
        audio_rows = None if not audio else torch.cat(audio, dim=0)
        return blocks, visual_rows, audio_rows

    @with_model_offload(["transformer"])
    @torch.inference_mode()
    def denoise(
        self,
        *,
        plan: MiniMaxH3ResolvedPlan,
        text: MiniMaxH3TextCondition | dict[str, torch.Tensor],
        conditions: list[MiniMaxH3PreparedCondition] | list[dict[str, Any]],
        num_inference_steps: int,
        _transport_video: bool = False,
    ) -> MiniMaxH3DenoiseResult:
        self._ensure_online_quantized()
        if isinstance(text, dict):
            text = MiniMaxH3TextCondition(**text)
        conditions = [
            MiniMaxH3PreparedCondition(**condition) if isinstance(condition, dict) else condition
            for condition in conditions
        ]
        shape = plan.shape
        latent_t = int(shape["video_latent_t"])
        latent_h = int(shape["height"]) // 16
        latent_w = int(shape["width"]) // 16
        audio_t = int(shape["audio_latent_t"])
        seed = 42 if plan.seed is None else int(plan.seed)
        if num_inference_steps < 2:
            raise ValueError("num_inference_steps must be at least 2")

        ref_blocks: list[dict[str, object]] | None = None
        if plan.task == "ref2va":
            ref_blocks, visual_cond, audio_cond = self._reference_blocks(conditions)
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=text.text_len,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                ref_blocks=ref_blocks,
            )
        else:
            visual_conditions = [condition for condition in conditions if condition.visual_rows is not None]
            visual_cond = (
                None
                if not visual_conditions
                else torch.cat([condition.visual_rows for condition in visual_conditions], dim=0)
            )
            audio_cond = None
            semantic_indices = tuple(
                int(condition.material.frame_index)
                for condition in visual_conditions
                if condition.material.frame_index is not None
            )
            packed = minimax_h3_packed_sequence(
                text_len=text.text_len,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                include_keyframe_cond=bool(visual_conditions),
                keyframe_frame_indices=semantic_indices if visual_conditions else None,
                frame_count=int(shape["frame_count"]) if visual_conditions else None,
            )

        token_tags = packed["token_tags"].clone()
        token_tags[: text.text_len] = text.token_tags
        if bool(((token_tags < -1) | (token_tags >= 3)).any().item()):
            raise ValueError(
                "MiniMax H3 token_tags must contain only padding (-1), video (0), text (1), or audio (2) values"
            )
        condition_shapes = [
            (condition.latent_t, condition.latent_h, condition.latent_w)
            for condition in conditions
            if condition.visual_rows is not None
        ]
        if visual_cond is not None:
            visual_cond = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_cond,
                condition_shapes=condition_shapes,
                target_latent_t=latent_t,
                imgvid_cond_num_frames=len(condition_shapes),
                seed=seed,
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            )
        audio_lengths = [condition.ref_audio_t for condition in conditions if condition.audio_rows is not None]
        if audio_cond is not None:
            audio_cond = minimax_h3_audio_cond_noise_aug_rows(
                audio_cond,
                condition_audio_t=audio_lengths,
                seed=seed,
                noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            )

        video_generator = torch.Generator(device="cpu").manual_seed(seed)
        video_native = torch.randn(
            1,
            24,
            latent_t,
            latent_h,
            latent_w,
            generator=video_generator,
            dtype=torch.float32,
        )
        video_target = minimax_h3_patchify_video_latent(video_native, patch_size=(1, 2, 2))
        audio_generator = torch.Generator(device="cpu").manual_seed(seed)
        audio_target = torch.randn(audio_t * 2, 32, generator=audio_generator, dtype=torch.float32)

        device = next(self.transformer.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        denoising_started = time.perf_counter()
        video_rows = torch.zeros(len(packed["img_pos"]), 96, dtype=torch.float32, device=device)
        audio_rows = torch.zeros(len(packed["audio_pos"]), 32, dtype=torch.float32, device=device)
        video_update_cpu = packed["update_mask"]
        audio_update_cpu = packed.get("audio_update_mask", torch.ones(len(packed["audio_pos"]), dtype=torch.bool))
        video_update = video_update_cpu.to(device)
        audio_update = audio_update_cpu.to(device)
        video_rows[video_update] = video_target.to(device)
        audio_rows[audio_update] = audio_target.to(device)
        if visual_cond is not None:
            video_rows[~video_update] = visual_cond.to(device)
        if audio_cond is not None:
            audio_rows[~audio_update] = audio_cond.to(device)

        video_shift = plan.flow_shift or plan.default_flow_shift
        audio_shift = plan.audio_flow_shift or plan.default_audio_flow_shift
        video_sigmas = minimax_h3_time_shift_sigmas(num_steps=num_inference_steps, shift_scale=video_shift)
        audio_sigmas = minimax_h3_time_shift_sigmas(num_steps=num_inference_steps, shift_scale=audio_shift)
        denoising_steps = len(video_sigmas) - 1
        if len(audio_sigmas) - 1 != denoising_steps:
            raise ValueError("MiniMax H3 video and audio schedules must have the same number of denoising steps")
        self.setup_feature_cache(
            self.transformer,
            self.model_runtime_config.feature_cache_config,
            denoising_steps,
        )
        img_pos_cpu = packed["img_pos"]
        audio_pos_cpu = packed["audio_pos"]
        img_pos = img_pos_cpu.to(device)
        audio_pos = audio_pos_cpu.to(device)
        text_pos_cpu = packed["text_pos"]
        text_pos = text_pos_cpu.to(device)
        target_img_pos = img_pos[video_update]
        target_video_row_start = int((~video_update_cpu).sum())
        target_audio_row_start = int((~audio_update_cpu).sum())
        condition_img_pos = img_pos_cpu[~video_update_cpu]
        target_audio_pos = audio_pos_cpu[audio_update_cpu]
        condition_audio_pos = audio_pos_cpu[~audio_update_cpu]

        seq_len = int(packed["seq_len"])
        x = torch.zeros(1, seq_len, 96, dtype=torch.float32, device=device)
        audio_x = torch.zeros(1, seq_len, 32, dtype=torch.float32, device=device)
        img_position_ids = packed["img_position_ids"].unsqueeze(0).float().to(device)
        prompt_embeds = text.hidden_states.to(device)
        cu_seqlens = packed["cu_seqlens"].to(device)
        block_token_tags_full = token_tags.to(device).clamp_min(0)
        use_ulysses = bool(getattr(self.transformer, "usp_flag", False))
        device_mesh = getattr(self.transformer, "device_mesh", None)
        ulysses_world_size = get_ulysses_world_size(device_mesh) if use_ulysses else 1
        ulysses_rank = get_ulysses_rank(device_mesh) if use_ulysses else 0
        local_seq_len = seq_len // ulysses_world_size
        local_row_start = ulysses_rank * local_seq_len
        local_row_stop = local_row_start + local_seq_len
        block_token_tags = block_token_tags_full[local_row_start:local_row_stop]
        local_embedding_layout = None
        if ulysses_world_size > 1:
            local_embedding_layout = _build_local_embedding_layout(
                seq_len=seq_len,
                text_pos=text_pos_cpu,
                img_pos=img_pos_cpu,
                audio_pos=audio_pos_cpu,
                world_size=ulysses_world_size,
                rank=ulysses_rank,
                device=device,
            )

        timestep_plan: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        row_timesteps = torch.empty(seq_len, dtype=torch.float32)
        for step in range(len(video_sigmas) - 1):
            t_video = float(1.0 - video_sigmas[step])
            t_audio = float(1.0 - audio_sigmas[step])
            row_timesteps.fill_(t_video)
            row_timesteps[condition_img_pos] = max(t_video, MINIMAX_H3_IMGVID_COND_TIMESTEP)
            row_timesteps[target_audio_pos] = t_audio
            row_timesteps[condition_audio_pos] = max(t_audio, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP)
            unique_timesteps_cpu, inverse_indices_cpu = torch.unique(
                row_timesteps,
                sorted=True,
                return_inverse=True,
            )
            inverse_indices = inverse_indices_cpu.to(device)
            block_combined_indices = block_token_tags + (inverse_indices[local_row_start:local_row_stop] * 3)
            timestep_plan.append(
                (
                    unique_timesteps_cpu.to(device),
                    inverse_indices,
                    block_combined_indices,
                )
            )

        video_step_t = torch.tensor(
            [float(1.0 - sigma) for sigma in video_sigmas[:-1]],
            dtype=torch.float32,
            device=device,
        )
        audio_step_t = torch.tensor(
            [float(1.0 - sigma) for sigma in audio_sigmas[:-1]],
            dtype=torch.float32,
            device=device,
        )
        video_sigmas_device = torch.tensor(video_sigmas, dtype=torch.float32, device=device)
        audio_sigmas_device = torch.tensor(audio_sigmas, dtype=torch.float32, device=device)
        video_sigma_ratios = video_sigmas_device[1:] / video_sigmas_device[:-1]
        audio_sigma_ratios = audio_sigmas_device[1:] / audio_sigmas_device[:-1]
        video_one_minus_sigma_ratios = 1.0 - video_sigma_ratios
        audio_one_minus_sigma_ratios = 1.0 - audio_sigma_ratios
        video_target_slice = slice(target_video_row_start, None)
        audio_target_slice = slice(target_audio_row_start, None)
        video_mask_is_suffix = torch.equal(
            video_update_cpu,
            torch.arange(video_update_cpu.numel()) >= target_video_row_start,
        )
        audio_mask_is_suffix = torch.equal(
            audio_update_cpu,
            torch.arange(audio_update_cpu.numel()) >= target_audio_row_start,
        )
        optimized_update = (
            video_mask_is_suffix and audio_mask_is_suffix and "step_denoising" not in self.scheduler.__dict__
        )
        video_denoised_scratch = torch.empty_like(video_rows[video_target_slice])
        audio_denoised_scratch = torch.empty_like(audio_rows[audio_target_slice])
        self._request_serial += 1
        static_cache_key = self._request_serial

        for step in range(len(video_sigmas) - 1):
            t_video = float(1.0 - video_sigmas[step])
            t_audio = float(1.0 - audio_sigmas[step])
            unique_timesteps, inverse_indices, block_combined_indices = timestep_plan[step]
            if step == 0 or not (video_mask_is_suffix and audio_mask_is_suffix):
                x[0].index_copy_(0, img_pos, video_rows)
                audio_x[0].index_copy_(0, audio_pos, audio_rows)
            else:
                x[0].index_copy_(0, target_img_pos, video_rows[video_target_slice])
                audio_x[0].index_copy_(0, audio_pos[audio_update], audio_rows[audio_target_slice])
            video_velocity, audio_velocity = self.transformer(
                x=x,
                audio_x=audio_x,
                img_position_ids=img_position_ids,
                unique_timesteps=unique_timesteps,
                inverse_indices=inverse_indices,
                update_mask=video_update,
                update_audio_mask=audio_update,
                prompt_embeds=prompt_embeds,
                img_pos_info={"position_ids": img_pos},
                audio_pos_info={"position_ids": audio_pos},
                text_pos_info={"position_ids": text_pos},
                img_pos_for_infer_output_info={"position_ids": target_img_pos},
                packed_seq_params={"cu_seqlens_q": cu_seqlens},
                block_token_tags=block_token_tags,
                block_combined_indices=block_combined_indices,
                local_embedding_layout=local_embedding_layout,
                static_cache_key=static_cache_key,
                skip_mask_out_condition=True,
            )
            audio_target_velocity = audio_velocity[audio_target_slice]
            if optimized_update:
                _minimax_h3_update_target_rows_(
                    video_rows[video_target_slice],
                    video_velocity.float(),
                    sigma_t=video_sigmas_device[step],
                    sigma_curr=video_sigmas[step],
                    sigma_ratio=video_sigma_ratios[step],
                    one_minus_sigma_ratio=video_one_minus_sigma_ratios[step],
                    denoised_scratch=video_denoised_scratch,
                )
                _minimax_h3_update_target_rows_(
                    audio_rows[audio_target_slice],
                    audio_target_velocity.float(),
                    sigma_t=audio_sigmas_device[step],
                    sigma_curr=audio_sigmas[step],
                    sigma_ratio=audio_sigma_ratios[step],
                    one_minus_sigma_ratio=audio_one_minus_sigma_ratios[step],
                    denoised_scratch=audio_denoised_scratch,
                )
            else:
                stepped = self.scheduler.step_denoising(
                    input_visual_latent=video_rows[video_update],
                    input_audio_latent=audio_rows[audio_update],
                    timestep=video_step_t[step],
                    video_timestep=video_step_t[step],
                    audio_timestep=audio_step_t[step],
                    noise_pred_visual=video_velocity,
                    noise_pred_audio=audio_target_velocity,
                    sigma_curr=video_sigmas[step],
                    sigma_next=video_sigmas[step + 1],
                    video_sigma_curr=video_sigmas[step],
                    video_sigma_next=video_sigmas[step + 1],
                    audio_sigma_curr=audio_sigmas[step],
                    audio_sigma_next=audio_sigmas[step + 1],
                )
                video_rows[video_update] = stepped["output_visual_latent"]
                audio_rows[audio_update] = stepped["output_audio_latent"]

        video_tokens = video_rows[video_update]
        if not _transport_video:
            video_tokens = video_tokens.cpu()
        video_latent = minimax_h3_unpatchify_video_tokens(
            video_tokens,
            latent_shape=(latent_t, latent_h // 2, latent_w // 2, 24),
            patch_size=(1, 2, 2),
        )
        audio_latent = minimax_h3_unpack_audio_tokens(
            audio_rows[audio_update].cpu(),
            audio_t=audio_t * 2,
            audio_channel=2,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_allocated = 0
            peak_reserved = 0
        runtime_metrics: dict[str, float | int] = {
            "denoising_seconds": time.perf_counter() - denoising_started,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        }
        feature_cache = getattr(self.transformer, "feature_cache", None)
        get_compute_steps = getattr(feature_cache, "get_compute_steps", None)
        computed_steps = len(get_compute_steps()) if callable(get_compute_steps) else denoising_steps
        runtime_metrics["feature_cache_computed_steps"] = computed_steps
        runtime_metrics["feature_cache_skipped_steps"] = denoising_steps - computed_steps
        finalize_online_adaln_cache = getattr(self.transformer, "finalize_online_adaln_cache", None)
        if callable(finalize_online_adaln_cache):
            finalize_online_adaln_cache()
        return MiniMaxH3DenoiseResult(video_latent, audio_latent, packed, runtime_metrics)

    def denoise_for_video_vae(
        self,
        *,
        plan: MiniMaxH3ResolvedPlan,
        text: MiniMaxH3TextCondition | dict[str, torch.Tensor],
        conditions: list[MiniMaxH3PreparedCondition] | list[dict[str, Any]],
        num_inference_steps: int,
    ) -> dict[str, Any]:
        """Keep the video latent on device while returning parent-consumed outputs separately."""
        result = self.denoise(
            plan=plan,
            text=text,
            conditions=conditions,
            num_inference_steps=num_inference_steps,
            _transport_video=True,
        )
        return {
            "video_latent": result.video_latent,
            "remainder": MiniMaxH3DenoiseRemainder(
                audio_latent=result.audio_latent,
                packed=result.packed,
                runtime_metrics=result.runtime_metrics,
            ),
        }


__all__ = ["MiniMaxH3DenoiseRemainder", "MiniMaxH3DenoiseResult", "MiniMaxH3DenoisingStage"]
