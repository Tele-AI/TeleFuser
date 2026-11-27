# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL layer-50 conditioning stage for MiniMax H3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.distributed.fsdp import shard_model_fsdp2_inference
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger

from .presentation import (
    minimax_h3_multi_image_presentation,
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
    minimax_h3_text_only_ids,
)


@dataclass(frozen=True)
class MiniMaxH3TextCondition:
    hidden_states: torch.Tensor
    token_tags: torch.Tensor

    @property
    def text_len(self) -> int:
        return int(self.hidden_states.shape[0])


class MiniMaxH3TextEncodingStage(BaseStage):
    def __init__(
        self,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        *,
        processor: Any,
    ) -> None:
        super().__init__("minimax_h3_text_encoding", model_runtime_config)
        self.text_encoder = module_manager.fetch_module("minimax_h3_text_encoder")
        if self.text_encoder is None:
            raise ValueError("ModuleManager must contain 'minimax_h3_text_encoder'")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_names = ["text_encoder"]

    def parallel_models(self) -> None:
        parallel_config = self.model_runtime_config.parallel_config
        unsupported = {
            "cfg_degree": parallel_config.cfg_degree,
            "sp_ring_degree": parallel_config.sp_ring_degree,
            "pp_degree": parallel_config.pp_degree,
        }
        invalid = {name: degree for name, degree in unsupported.items() if degree != 1}
        if invalid:
            raise NotImplementedError(f"MiniMax H3 text encoder does not support these parallel degrees: {invalid}")
        if parallel_config.tp_degree <= 1 and not parallel_config.enable_fsdp:
            raise NotImplementedError("MiniMax H3 multi-GPU text encoding requires TP or FSDP inference")
        device_mesh = create_device_mesh_from_config(parallel_config)
        if parallel_config.tp_degree > 1:
            if parallel_config.sp_ulysses_degree > 1:
                raise ValueError("MiniMax H3 text encoder TP requires a dedicated one-dimensional TP mesh")
            if parallel_config.enable_fsdp:
                raise ValueError("MiniMax H3 text encoder TP cannot be combined with FSDP")
            if (
                self.device.type != "npu"
                and self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD
            ):
                raise ValueError("MiniMax H3 text encoder TP cannot be combined with model CPU offload")
            logger.info(f"Enabling tensor parallelism for {self.name}")
            self.text_encoder.enable_tp(device_mesh)
            if self.model_runtime_config.offload_config.offload_type == WeightOffloadType.NO_CPU_OFFLOAD:
                self.text_encoder.to(self.device)
                self.onload_models_flag = True
        else:
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                raise ValueError("MiniMax H3 text encoder FSDP inference cannot be combined with model CPU offload")
            logger.info(f"Enabling block FSDP2 for {self.name}")
            self.text_encoder = shard_model_fsdp2_inference(
                module=self.text_encoder,
                device_mesh=device_mesh,
                wrap_module_names=self.text_encoder.get_fsdp_module_names(),
            )
            self.onload_models_flag = True
        current_platform.empty_cache()

    @with_model_offload(["text_encoder"])
    @torch.inference_mode()
    def encode(
        self,
        *,
        task: str,
        prompt: str,
        images: list[Any],
        videos: list[torch.Tensor],
        condition_labels: list[tuple[str, int]],
    ) -> MiniMaxH3TextCondition:
        condition = self._encode_impl(
            task=task,
            prompt=prompt,
            images=images,
            videos=videos,
            condition_labels=condition_labels,
        )
        return MiniMaxH3TextCondition(condition.hidden_states.cpu(), condition.token_tags.cpu())

    @with_model_offload(["text_encoder"])
    @torch.inference_mode()
    def encode_for_denoising(
        self,
        *,
        task: str,
        prompt: str,
        images: list[Any],
        videos: list[torch.Tensor],
        condition_labels: list[tuple[str, int]],
    ) -> dict[str, torch.Tensor]:
        """Keep hidden states on the producer device while transporting control metadata on CPU."""
        condition = self._encode_impl(
            task=task,
            prompt=prompt,
            images=images,
            videos=videos,
            condition_labels=condition_labels,
        )
        return {
            "hidden_states": condition.hidden_states,
            "token_tags": condition.token_tags.cpu(),
        }

    def _encode_impl(
        self,
        *,
        task: str,
        prompt: str,
        images: list[Any],
        videos: list[torch.Tensor],
        condition_labels: list[tuple[str, int]],
    ) -> MiniMaxH3TextCondition:
        if task == "t2va":
            ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
            tags = torch.ones(ids.shape[0], dtype=torch.long)
            hidden = self.text_encoder.encode_ids(ids)
            return MiniMaxH3TextCondition(hidden, tags)
        if task == "fl2va":
            return self._encode_images(prompt, images)
        if task != "ref2va":
            raise ValueError(f"unsupported MiniMax H3 task {task!r}")
        return self._encode_references(prompt, images, videos, condition_labels)

    def _encode_images(self, prompt: str, images: list[Any]) -> MiniMaxH3TextCondition:
        if not images:
            raise ValueError("fl2va text conditioning requires keyframe images")
        vision = self.processor.image_processor(images=images, return_tensors="pt")
        grids = vision["image_grid_thw"]
        merge = int(self.processor.image_processor.merge_size) ** 2
        counts = [int(grids[index].prod().item()) // merge for index in range(len(images))]
        ids, tags = minimax_h3_multi_image_presentation(
            self.tokenizer,
            prompt=prompt,
            image_token_counts=counts,
        )
        hidden = self.text_encoder.encode_ids(
            ids,
            pixel_values=vision["pixel_values"],
            image_grid_thw=grids,
        )
        return MiniMaxH3TextCondition(hidden, tags)

    @staticmethod
    def _sample_video(frames: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
            raise ValueError("reference video frames must be non-empty [T,H,W,3]")
        sampled = frames[::12]
        timestamps = [index / 2.0 for index in range(int(sampled.shape[0]))]
        if len(timestamps) % 2:
            timestamps.append(timestamps[-1])
        block_timestamps = [(timestamps[index] + timestamps[index + 1]) / 2.0 for index in range(0, len(timestamps), 2)]
        return sampled.permute(0, 3, 1, 2), block_timestamps

    def _encode_references(
        self,
        prompt: str,
        images: list[Any],
        videos: list[torch.Tensor],
        condition_labels: list[tuple[str, int]],
    ) -> MiniMaxH3TextCondition:
        pixel_values = image_grids = None
        image_counts: list[int] = []
        if images:
            vision = self.processor.image_processor(images=images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grids = vision["image_grid_thw"]
            merge = int(self.processor.image_processor.merge_size) ** 2
            image_counts = [int(image_grids[index].prod().item()) // merge for index in range(len(images))]

        pixel_values_videos = video_grids = None
        video_counts: list[list[int]] = []
        video_timestamps: list[list[float]] = []
        if videos:
            sampled = [self._sample_video(frames) for frames in videos]
            video_output = self.processor.video_processor(
                videos=[item[0] for item in sampled],
                do_sample_frames=False,
                input_data_format="channels_first",
                return_tensors="pt",
            )
            pixel_values_videos = video_output["pixel_values_videos"]
            video_grids = video_output["video_grid_thw"]
            merge = int(self.processor.image_processor.merge_size) ** 2
            for index, (_, timestamps) in enumerate(sampled):
                blocks = int(video_grids[index, 0])
                per_block = int(video_grids[index, 1]) * int(video_grids[index, 2]) // merge
                if blocks != len(timestamps):
                    raise ValueError("Qwen video grid and timestamp block counts disagree")
                video_counts.append([per_block] * blocks)
                video_timestamps.append(timestamps)

        if videos:
            ids, tags = minimax_h3_ref2va_video_presentation(
                self.tokenizer,
                prompt=prompt,
                condition_labels=condition_labels,
                image_token_count=image_counts or None,
                video_block_token_counts=video_counts,
                video_block_timestamps=video_timestamps,
            )
        else:
            ids, tags = minimax_h3_ref2va_presentation(
                self.tokenizer,
                prompt=prompt,
                condition_labels=condition_labels,
                image_token_count=image_counts or None,
            )
        hidden = self.text_encoder.encode_ids(
            ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grids,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grids,
        )
        return MiniMaxH3TextCondition(hidden, tags)


__all__ = ["MiniMaxH3TextCondition", "MiniMaxH3TextEncodingStage"]
