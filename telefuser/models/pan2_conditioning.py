# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible conditioning components for PAN2."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch

from telefuser.core.base_model import BaseModel


class PAN2TextEncoder(BaseModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        self.model = Qwen3_5TextModel(Qwen3_5TextConfig(**config))

    def to(self, *args: Any, **kwargs: Any) -> PAN2TextEncoder:
        # Match the Transformers loader: rotary frequencies remain FP32 rather than checkpoint dtype.
        preserved = {
            name: buffer.detach().clone()
            for name, buffer in self.named_buffers()
            if name.endswith("inv_freq") and not buffer.is_meta
        }
        result = super().to(*args, **kwargs)
        for name, buffer in result.named_buffers():
            if name.endswith("inv_freq"):
                buffer.data = preserved.get(name, buffer).to(device=buffer.device, dtype=torch.float32)
        return result

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states[-3]

    @staticmethod
    def state_dict_converter(config_path: str | Path) -> PAN2ConditioningConverter:
        return PAN2ConditioningConverter(config_path)


class PAN2VAE(BaseModel):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        from diffusers import AutoencoderKLWan

        allowed = inspect.signature(AutoencoderKLWan.__init__).parameters
        self.model = AutoencoderKLWan(**{k: v for k, v in config.items() if k in allowed})
        self.config = self.model.config

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.model.encode(pixels).latent_dist.mode()

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.model.decode(latents).sample

    @staticmethod
    def state_dict_converter(config_path: str | Path) -> PAN2ConditioningConverter:
        return PAN2ConditioningConverter(config_path)


class PAN2ConditioningConverter:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return {f"model.{name}": value for name, value in state_dict.items()}, {
            "config": json.loads(self.config_path.read_text())
        }

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return self.from_diffusers(state_dict)
