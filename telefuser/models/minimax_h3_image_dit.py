# SPDX-License-Identifier: Apache-2.0
"""Image-only H3 DiT using TeleFuser's maintained H3 SP/TP and attention paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.models.minimax_h3_dit import (
    MiniMaxH3DiT,
    MiniMaxH3DiTConfig,
    MiniMaxH3DiTStateDictConverter,
)


class _AbsentAudioInput(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0]:
            raise ValueError("MiniMax-H3-Image has no audio input branch")
        return value.new_empty((0, self.hidden_size))


class _AbsentAudioOutput(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # The inherited packed forward selects zero audio rows from this tensor.
        return value.new_empty((value.shape[0], 0))


class MiniMaxH3ImagePDDProjection(nn.Module):
    """Released fused FP32 output heads; head i belongs to sampling step i."""

    def __init__(self, hidden_size: int, patch_dim: int, num_heads: int) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        self.weight = nn.Parameter(torch.empty(num_heads, patch_dim, hidden_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_heads, patch_dim, dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.02)
        self.step_index = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value.float(), self.weight[self.step_index], self.bias[self.step_index])


class MiniMaxH3ImageDiT(MiniMaxH3DiT):
    """H3-Image preserves the three-modality AdaLN table, but has no audio rows.

    Only the input/output branches and the step-dependent PDD projection differ
    from the maintained H3 DiT. SP, TP, Sol-Attn and FP8 use the inherited paths.
    """

    def __init__(self, config: MiniMaxH3DiTConfig | None = None, num_pdd_heads: int = 6) -> None:
        super().__init__(config)
        self.audio_patch_proj = _AbsentAudioInput(self.config.hidden_size)
        self.final_layer.audio_out = _AbsentAudioOutput()
        self.final_layer.video_out = MiniMaxH3ImagePDDProjection(
            self.config.hidden_size, self.config.video_patch_dim, num_pdd_heads
        )

    def forward(self, *, pdd_step_index: int, **kwargs: Any) -> torch.Tensor:
        projection = self.final_layer.video_out
        if not 0 <= pdd_step_index < projection.weight.shape[0]:
            raise ValueError(f"PDD step index {pdd_step_index} is outside the released output-head table")
        audio = kwargs.get("audio_x")
        if audio is not None and audio.numel():
            raise ValueError("MiniMax-H3-Image does not accept audio")
        device = kwargs["x"].device
        kwargs["audio_x"] = torch.empty((1, 0, self.config.audio_latents_dim), device=device)
        kwargs["audio_pos_info"] = {"position_ids": torch.empty((0,), device=device, dtype=torch.long)}
        projection.step_index = int(pdd_step_index)
        video, _ = super().forward(**kwargs)
        return video

    def enable_quant(self, quant_type: QuantConfig | QuantType | str | torch.dtype) -> None:
        if not isinstance(quant_type, QuantConfig):
            if quant_type not in (QuantType.FP8, "fp8", torch.float8_e4m3fn):
                raise ValueError("Use QuantConfig to select a supported H3-Image quantization backend")
            quant_type = QuantConfig(
                enabled=True, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL
            )
        super().enable_quant(quant_type)

    @staticmethod
    def state_dict_converter(config_path: str | Path | None = None) -> MiniMaxH3ImageDiTStateDictConverter:
        return MiniMaxH3ImageDiTStateDictConverter(config_path)


class MiniMaxH3ImageDiTStateDictConverter:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = config_path

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if self.config_path is None:
            raise ValueError("H3-Image loading requires the released transformer/config.json")
        if "audio_patch_proj.weight" in state_dict:
            raise ValueError("An audio-enabled H3 checkpoint is not an H3-Image checkpoint")
        heads = state_dict.get("video_out.weight")
        bias = state_dict.get("video_out.bias")
        if heads is None or bias is None or heads.ndim != 3 or bias.shape != heads.shape[:2]:
            raise ValueError("Load transformer shards together with pdd_fused_heads.safetensors")
        converted, kwargs = MiniMaxH3DiTStateDictConverter(self.config_path).from_official(state_dict)
        config = kwargs["config"]
        if tuple(heads.shape[1:]) != (config.video_patch_dim, config.hidden_size):
            raise ValueError("PDD output heads do not match the transformer configuration")
        converted.pop("video_out.weight")
        converted.pop("video_out.bias")
        converted["final_layer.video_out.weight"] = heads
        converted["final_layer.video_out.bias"] = bias
        kwargs["num_pdd_heads"] = int(heads.shape[0])
        return converted, kwargs

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if self.config_path is None:
            raise ValueError("H3-Image loading requires the released native transformer/config.json")
        regular = {name: value for name, value in state_dict.items() if not name.startswith("pdd_proj_out.")}
        converted, kwargs = MiniMaxH3DiTStateDictConverter(self.config_path).from_diffusers(regular)
        config = kwargs["config"]
        indices = sorted({int(name.split(".")[1]) for name in state_dict if name.startswith("pdd_proj_out.")})
        if not indices or indices != list(range(len(indices))):
            raise ValueError("Diffusers PDD output heads must be contiguous and start at zero")
        for suffix in ("weight", "bias"):
            converted[f"final_layer.video_out.{suffix}"] = torch.stack(
                [state_dict[f"pdd_proj_out.{index}.{suffix}"] for index in indices]
            )
        # Native H3 stores [gate, value], while Diffusers SwiGLU stores [value, gate].
        for name, value in list(converted.items()):
            if name.endswith(".mlp.fc1.weight"):
                first, second = value.chunk(2, dim=0)
                converted[name] = torch.cat((second, first), dim=0)
        converted["rope.inv_freq"] = 10000.0 ** (
            -torch.arange(config.rope_inv_freq_len, dtype=torch.float32) / config.rope_inv_freq_len
        )
        kwargs["num_pdd_heads"] = len(indices)
        return converted, kwargs
