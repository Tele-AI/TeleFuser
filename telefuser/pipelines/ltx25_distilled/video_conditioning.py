"""Still-image conditioning for LTX-2.5."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager

from .image import default_image_crf, preprocess_ltx25_image
from .latent import LatentState, VideoConditionByKeyframeIndex, VideoConditionByLatentIndex, VideoLatentTools


class LTX25ImageConditionProtocol(Protocol):
    image: Image.Image
    frame_idx: int
    strength: float
    crf: int | None


class LTX25VideoConditioningStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_video_conditioning", config)
        fetched = module_manager.fetch_module("ltx25_video_encoder", require_model_path=True)
        if fetched is None:
            raise ValueError("ModuleManager does not contain ltx25_video_encoder")
        self.video_encoder, video_encoder_path = fetched
        self.video_encoder_path = video_encoder_path
        self.model_names = ["video_encoder"]

    @with_model_offload(["video_encoder"])
    @torch.inference_mode()
    def apply(
        self,
        state: LatentState,
        tools: VideoLatentTools,
        conditions: Sequence[LTX25ImageConditionProtocol],
        height: int,
        width: int,
    ) -> LatentState:
        for condition in conditions:
            if condition.frame_idx < 0:
                raise ValueError(f"image frame_idx must be non-negative, got {condition.frame_idx}")
            if not 0.0 <= condition.strength <= 1.0:
                raise ValueError(f"image strength must be in [0, 1], got {condition.strength}")
            pixels = preprocess_ltx25_image(
                condition.image,
                height,
                width,
                default_image_crf(self.video_encoder_path) if condition.crf is None else condition.crf,
                device=self.device,
                dtype=self.torch_dtype,
            )
            encoded = self.video_encoder(pixels)
            conditioning = (
                VideoConditionByLatentIndex(encoded, condition.strength, 0)
                if condition.frame_idx == 0
                else VideoConditionByKeyframeIndex(encoded, condition.frame_idx, condition.strength)
            )
            state = conditioning.apply_to(state, tools)
        return state


__all__ = ["LTX25VideoConditioningStage"]
