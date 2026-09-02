# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Qwen3-VL layer-50 encoder."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import Qwen3VLConfig, Qwen3VLModel

from telefuser.core.base_model import BaseModel
from telefuser.distributed.collectives import all_reduce_sum_
from telefuser.distributed.device_mesh import get_tp_group, get_tp_rank, get_tp_world_size

MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50
MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120
_LAYER_WEIGHT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def _is_unconsumed_checkpoint_weight(name: str) -> bool:
    if name == "lm_head.weight" or name.startswith("model.language_model.norm."):
        return True
    match = _LAYER_WEIGHT_RE.match(name)
    return bool(match and int(match.group(1)) >= MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER)


def load_minimax_h3_encoder_config(path: str | Path) -> Qwen3VLConfig:
    config_path = Path(path)
    source = config_path.parent if config_path.is_file() else config_path
    config = Qwen3VLConfig.from_pretrained(source, local_files_only=True)
    config.text_config.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    config.text_config.output_hidden_states = False
    config.text_config.use_cache = False
    return config


def _replace_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    parameter = module.get_parameter(name)
    setattr(module, name, nn.Parameter(value.contiguous(), requires_grad=parameter.requires_grad))


def _shard_linear_output(
    linear: nn.Linear,
    *,
    rank: int,
    world_size: int,
    sections: tuple[int, ...] | None = None,
) -> None:
    sections = sections or (linear.out_features,)
    if sum(sections) != linear.out_features or any(size % world_size for size in sections):
        raise ValueError(
            f"linear output sections {sections} must sum to {linear.out_features} and divide TP degree {world_size}"
        )
    weight_sections = linear.weight.split(sections, dim=0)
    local_weight = torch.cat(tuple(section.chunk(world_size, dim=0)[rank] for section in weight_sections), dim=0)
    _replace_parameter(linear, "weight", local_weight)
    if linear.bias is not None:
        bias_sections = linear.bias.split(sections, dim=0)
        local_bias = torch.cat(tuple(section.chunk(world_size, dim=0)[rank] for section in bias_sections), dim=0)
        _replace_parameter(linear, "bias", local_bias)
    linear.out_features //= world_size


def _shard_linear_input(linear: nn.Linear, *, rank: int, world_size: int) -> None:
    if linear.in_features % world_size:
        raise ValueError(f"linear input size {linear.in_features} must divide TP degree {world_size}")
    _replace_parameter(linear, "weight", linear.weight.chunk(world_size, dim=1)[rank])
    if linear.bias is not None and rank != 0:
        _replace_parameter(linear, "bias", torch.zeros_like(linear.bias))
    linear.in_features //= world_size


def _register_row_parallel_reduce(linear: nn.Linear, group: dist.ProcessGroup) -> None:
    def reduce_output(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        all_reduce_sum_((output,), group=group)
        return output

    linear.register_forward_hook(reduce_output)


class MiniMaxH3Encoder(BaseModel):
    """Qwen3-VL multimodal backbone ending at unnormalized layer 50."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        if int(config.text_config.num_hidden_layers) != MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER:
            raise ValueError("MiniMax H3 encoder config must be trimmed to 50 language layers")
        if int(config.text_config.hidden_size) != MINIMAX_H3_QWEN3VL_HIDDEN_DIM:
            raise ValueError("MiniMax H3 encoder hidden size must be 5120")
        self.model = Qwen3VLModel(config)
        self.model.language_model.norm = nn.Identity()
        self.config = config
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)
        self.selected_lm_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM
        self.layer_name_list = ["model"]
        self.tp_flag = False

    @property
    def fsdp_language_layers(self) -> nn.ModuleList:
        """Return language blocks as inference FSDP wrapping boundaries."""
        return self.model.language_model.layers

    @property
    def fsdp_visual_blocks(self) -> nn.ModuleList:
        """Return vision blocks as inference FSDP wrapping boundaries."""
        return self.model.visual.blocks

    @property
    def fsdp_visual_deepstack_mergers(self) -> nn.ModuleList:
        """Return vision deep-stack mergers as inference FSDP boundaries."""
        return self.model.visual.deepstack_merger_list

    def get_fsdp_module_names(self) -> list[str]:
        return [
            "fsdp_language_layers",
            "fsdp_visual_blocks",
            "fsdp_visual_deepstack_mergers",
        ]

    def enable_tp(self, device_mesh: Any) -> None:
        world_size = get_tp_world_size(device_mesh)
        if world_size <= 1:
            return
        if self.tp_flag:
            raise RuntimeError("MiniMax H3 encoder tensor parallelism is already enabled")
        group = get_tp_group(device_mesh)
        if group is None:
            raise RuntimeError("MiniMax H3 encoder TP requires a tensor-parallel process group")
        rank = get_tp_rank(device_mesh)
        text_config = self.config.text_config
        if int(text_config.num_attention_heads) % world_size:
            raise ValueError("Qwen3-VL query heads must divide the encoder TP degree")
        if int(text_config.num_key_value_heads) % world_size:
            raise ValueError("Qwen3-VL key/value heads must divide the encoder TP degree")
        if int(text_config.intermediate_size) % world_size:
            raise ValueError("Qwen3-VL language FFN size must divide the encoder TP degree")

        for layer in self.model.language_model.layers:
            attention = layer.self_attn
            for projection in (attention.q_proj, attention.k_proj, attention.v_proj):
                _shard_linear_output(projection, rank=rank, world_size=world_size)
            _shard_linear_input(attention.o_proj, rank=rank, world_size=world_size)
            _register_row_parallel_reduce(attention.o_proj, group)

            mlp = layer.mlp
            _shard_linear_output(mlp.gate_proj, rank=rank, world_size=world_size)
            _shard_linear_output(mlp.up_proj, rank=rank, world_size=world_size)
            _shard_linear_input(mlp.down_proj, rank=rank, world_size=world_size)
            _register_row_parallel_reduce(mlp.down_proj, group)
            mlp.intermediate_size //= world_size

        vision_config = self.config.vision_config
        if int(vision_config.num_heads) % world_size:
            raise ValueError("Qwen3-VL vision heads must divide the encoder TP degree")
        if int(vision_config.intermediate_size) % world_size:
            raise ValueError("Qwen3-VL vision FFN size must divide the encoder TP degree")
        for block in self.model.visual.blocks:
            attention = block.attn
            _shard_linear_output(
                attention.qkv,
                rank=rank,
                world_size=world_size,
                sections=(attention.dim, attention.dim, attention.dim),
            )
            _shard_linear_input(attention.proj, rank=rank, world_size=world_size)
            _register_row_parallel_reduce(attention.proj, group)
            attention.num_heads //= world_size

            mlp = block.mlp
            _shard_linear_output(mlp.linear_fc1, rank=rank, world_size=world_size)
            _shard_linear_input(mlp.linear_fc2, rank=rank, world_size=world_size)
            _register_row_parallel_reduce(mlp.linear_fc2, group)
            mlp.intermediate_size //= world_size
        self.tp_flag = True

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
            **kwargs,
        )
        return outputs.last_hidden_state

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError(f"input_ids must be one-dimensional, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be provided together")

        host_ids = input_ids.to(device="cpu", dtype=torch.long).unsqueeze(0)
        host_image_grid = None if image_grid_thw is None else image_grid_thw.to(device="cpu", dtype=torch.long)
        host_video_grid = None if video_grid_thw is None else video_grid_thw.to(device="cpu", dtype=torch.long)
        position_ids = None
        mm_token_type_ids = None
        if host_image_grid is not None or host_video_grid is not None:
            mm_token_type_ids = torch.zeros_like(host_ids)
            if host_image_grid is not None:
                mm_token_type_ids[host_ids == self.model.config.image_token_id] = 1
            if host_video_grid is not None:
                mm_token_type_ids[host_ids == self.model.config.video_token_id] = 2
            position_ids, _ = self.model.get_rope_index(
                input_ids=host_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=host_image_grid,
                video_grid_thw=host_video_grid,
                attention_mask=torch.ones_like(host_ids),
            )
        call_kwargs: dict[str, Any] = {
            "input_ids": host_ids.to(self.device),
            "attention_mask": torch.ones_like(host_ids).to(self.device),
        }
        if position_ids is not None:
            assert mm_token_type_ids is not None
            call_kwargs["position_ids"] = position_ids.to(self.device)
            call_kwargs["mm_token_type_ids"] = mm_token_type_ids.to(self.device)
        if pixel_values is not None:
            assert host_image_grid is not None
            call_kwargs["pixel_values"] = pixel_values.to(self.device, torch.bfloat16)
            call_kwargs["image_grid_thw"] = host_image_grid.to(self.device)
        if pixel_values_videos is not None:
            assert host_video_grid is not None
            call_kwargs["pixel_values_videos"] = pixel_values_videos.to(self.device, torch.bfloat16)
            call_kwargs["video_grid_thw"] = host_video_grid.to(self.device)
        hidden = self(**call_kwargs)[0].to(torch.bfloat16)
        expected = (input_ids.numel(), self.hidden_dim)
        if tuple(hidden.shape) != expected:
            raise ValueError(f"unexpected MiniMax H3 encoder shape {tuple(hidden.shape)}, expected {expected}")
        return hidden

    @staticmethod
    def state_dict_converter(config_path: str | Path) -> MiniMaxH3EncoderStateDictConverter:
        return MiniMaxH3EncoderStateDictConverter(config_path)


class MiniMaxH3EncoderStateDictConverter:
    def __init__(self, config_path: str | Path) -> None:
        self.config = load_minimax_h3_encoder_config(config_path)

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        converted = {
            name: value
            for name, value in state_dict.items()
            if "rotary_emb.inv_freq" not in name and not _is_unconsumed_checkpoint_weight(name)
        }
        return converted, {"config": self.config}

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return self.from_official(state_dict)


__all__ = [
    "MINIMAX_H3_QWEN3VL_HIDDEN_DIM",
    "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER",
    "MiniMaxH3Encoder",
    "MiniMaxH3EncoderStateDictConverter",
    "load_minimax_h3_encoder_config",
]
