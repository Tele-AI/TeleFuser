"""Checkpoint loading diagnostics for LingBot-Video components."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from telefuser.models.lingbot_video_dit import LingBotVideoTransformer3DModel
from telefuser.models.lingbot_video_moe import LingBotVideoMoeTransformer3DModel


def checkpoint_key_coverage(
    module: nn.Module, state_dict: Mapping[str, torch.Tensor] | Collection[str]
) -> dict[str, Any]:
    """Return exact checkpoint key coverage without mutating the module."""
    expected = set(module.state_dict())
    available = set(state_dict)
    matched = expected & available
    return {
        "expected_key_count": len(expected),
        "checkpoint_key_count": len(available),
        "matched_key_count": len(matched),
        "coverage": len(matched) / len(expected) if expected else 1.0,
        "missing_keys": sorted(expected - available),
        "unexpected_keys": sorted(available - expected),
        "matched_numel": sum(module.state_dict()[name].numel() for name in matched),
    }


def load_lingbot_video_dense_transformer(
    checkpoint_dir: str | Path, *, device: torch.device | str = "cuda", torch_dtype: torch.dtype = torch.bfloat16
) -> "LingBotVideoTransformer3DModel":
    """Strictly load the official Diffusers Dense transformer checkpoint."""

    from safetensors.torch import load_model

    directory = Path(checkpoint_dir)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    keys = (
        "patch_size",
        "in_channels",
        "out_channels",
        "hidden_size",
        "num_attention_heads",
        "depth",
        "intermediate_size",
        "text_dim",
        "freq_dim",
        "norm_eps",
        "rope_theta",
        "axes_dims",
        "qkv_bias",
        "out_bias",
        "patch_embed_bias",
        "timestep_mlp_bias",
    )
    transformer = LingBotVideoTransformer3DModel(**{key: config[key] for key in keys}).to(
        device=device, dtype=torch_dtype
    )
    fp32_names = (
        "time_embedder",
        "time_modulation",
        "scale_shift_table",
        "norm",
        "norm1",
        "norm2",
        "norm_q",
        "norm_k",
        "norm_post_attn",
        "norm_post_ffn",
        "norm_out",
        "norm_out_modulation",
        "router",
    )
    for name, module in transformer.named_modules():
        if any(part in fp32_names for part in name.split(".")):
            module.float()
    for name, parameter in transformer.named_parameters():
        if any(part in fp32_names for part in name.split(".")):
            parameter.data = parameter.data.float()

    missing, unexpected = load_model(
        transformer, directory / "diffusion_pytorch_model.safetensors", strict=True, device=str(device)
    )
    if missing or unexpected:
        raise RuntimeError(f"LingBot checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return transformer.eval()


def load_lingbot_video_moe_transformer(
    checkpoint_dir: str | Path, *, device: torch.device | str = "cuda", torch_dtype: torch.dtype = torch.bfloat16
) -> LingBotVideoMoeTransformer3DModel:
    """Strictly load the official sharded MoE/refiner transformer checkpoint."""
    from safetensors.torch import load_file

    directory = Path(checkpoint_dir)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    keys = (
        "patch_size",
        "in_channels",
        "out_channels",
        "hidden_size",
        "num_attention_heads",
        "depth",
        "intermediate_size",
        "text_dim",
        "freq_dim",
        "norm_eps",
        "rope_theta",
        "axes_dims",
        "qkv_bias",
        "out_bias",
        "patch_embed_bias",
        "timestep_mlp_bias",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "decoder_sparse_step",
        "mlp_only_layers",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "n_shared_experts",
    )
    transformer = LingBotVideoMoeTransformer3DModel(**{key: config[key] for key in keys}).to(
        device=device, dtype=torch_dtype
    )
    fp32_names = (
        "time_embedder",
        "time_modulation",
        "scale_shift_table",
        "norm",
        "norm1",
        "norm2",
        "norm_q",
        "norm_k",
        "norm_post_attn",
        "norm_post_ffn",
        "norm_out",
        "norm_out_modulation",
        "router",
    )
    for name, module in transformer.named_modules():
        if any(part in fp32_names for part in name.split(".")):
            module.float()
    for name, parameter in transformer.named_parameters():
        if any(part in fp32_names for part in name.split(".")):
            parameter.data = parameter.data.float()

    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8"))[
        "weight_map"
    ]
    expected = set(transformer.state_dict())
    found: set[str] = set()
    unexpected: set[str] = set()
    for shard in sorted(set(index.values())):
        shard_state = load_file(directory / shard, device=str(device))
        shard_unexpected = transformer.load_state_dict(shard_state, strict=False).unexpected_keys
        unexpected.update(shard_unexpected)
        found.update(shard_state)
        del shard_state
    missing = expected - found
    if missing or unexpected:
        raise RuntimeError(
            f"LingBot MoE checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return transformer.eval()
