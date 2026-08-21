"""Prompt encoding and duration prediction for LTX-2.5."""

from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx25 import LTX25DurationHead, LTX25EmbeddingsProcessor, LTX25Gemma4TextEncoder
from telefuser.models.ltx25.duration import seconds_to_num_frames


class LTX25TextEncodingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_text_encoding", config)
        self.text_encoder: LTX25Gemma4TextEncoder = module_manager.fetch_module("ltx25_gemma4")
        self.embeddings_processor: LTX25EmbeddingsProcessor = module_manager.fetch_module("ltx25_embeddings_processor")
        self.duration_head: LTX25DurationHead = module_manager.fetch_module("ltx25_duration_head")
        self.model_names = ["text_encoder", "embeddings_processor", "duration_head"]

    @with_model_offload(["text_encoder", "embeddings_processor", "duration_head"])
    @torch.inference_mode()
    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, _, attention_mask = self.text_encoder.encode([prompt])
        encoded = self.embeddings_processor(hidden_states, attention_mask)
        return encoded.video_encoding, encoded.audio_encoding

    @with_model_offload(["text_encoder", "embeddings_processor", "duration_head"])
    @torch.inference_mode()
    def predict_num_frames(self, video_context: torch.Tensor, audio_context: torch.Tensor, frame_rate: float) -> int:
        seconds = float(self.duration_head(video_context, audio_context).item())
        return seconds_to_num_frames(seconds, frame_rate=frame_rate)


__all__ = ["LTX25TextEncodingStage"]
