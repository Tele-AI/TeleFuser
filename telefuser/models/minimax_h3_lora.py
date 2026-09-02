# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 LoRA and FastVideo adapter mapping rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from telefuser.core.config import LoraConfig
from telefuser.utils.logging import logger
from telefuser.utils.lora_loader import LoRALoader, LoRATarget

MINIMAX_H3_LORA_KEY_MAPPING_RULES = [
    (r"^(?:base_model\.model\.|model\.diffusion_model\.|diffusion_model\.|transformer\.|model\.)", ""),
    (r"^proj_in\.", "video_patch_proj."),
    (r"^audio_proj_in\.", "audio_patch_proj."),
    (r"^context_embedder\.", "condition_proj."),
    (r"^time_embedder\.linear_1\.", "time_embedder.proj_in."),
    (r"^time_embedder\.linear_2\.", "time_embedder.proj_out."),
    (r"^norm_out\.norm\.", "final_layer.norm."),
    (r"^norm_out\.linear\.", "final_layer.adaln_proj.linear."),
    (r"^proj_out\.", "final_layer.video_out."),
    (r"^audio_proj_out\.", "final_layer.audio_out."),
    (r"^transformer_blocks\.", "blocks."),
    (r"^token_refiner\.refiner_blocks\.", "token_refiner.blocks."),
    (r"\.attn\.norm_q\.", ".attn.q_norm."),
    (r"\.attn\.norm_k\.", ".attn.k_norm."),
    (r"\.attn\.to_out\.0\.", ".attn.out_proj."),
    (r"\.ff\.net\.0\.proj\.", ".mlp.fc1."),
    (r"\.ff\.net\.2\.", ".mlp.fc2."),
]

FASTVIDEO_LORA_FORMAT = "fastvideo-lora-v2"
FASTVIDEO_REPLACEMENT_SUFFIX = ".set_weight"


def _adapter_header(path: str | Path) -> tuple[Path, dict[str, str], tuple[str, ...]]:
    adapter_path = Path(path).expanduser()
    if adapter_path.is_dir():
        files = sorted(adapter_path.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(
                f"MiniMax H3 adapter directory must contain exactly one safetensors file, got {len(files)}: "
                f"{adapter_path}"
            )
        adapter_path = files[0]
    if not adapter_path.is_file():
        raise FileNotFoundError(f"MiniMax H3 adapter not found: {adapter_path}")
    with safe_open(str(adapter_path), framework="pt", device="cpu") as source:
        return adapter_path, source.metadata() or {}, tuple(source.keys())


def minimax_h3_lora_target(
    model_key: str,
    weights: Mapping[str, torch.Tensor],
) -> LoRATarget | None:
    """Resolve Diffusers H3 projections to native parameters or fused QKV slices."""
    for source, offset in ((".attn.to_q.", 0), (".attn.to_k.", 1), (".attn.to_v.", 2)):
        if source not in model_key:
            continue
        target_key = model_key.replace(source, ".attn.qkv_proj.")
        parameter = weights.get(target_key)
        if parameter is None or parameter.shape[0] % 3:
            return None
        width = parameter.shape[0] // 3
        return LoRATarget(target_key, parameter[offset * width : (offset + 1) * width])
    parameter = weights.get(model_key)
    return None if parameter is None else LoRATarget(model_key, parameter)


class MiniMaxH3LoraAdapter:
    """Merge released Turbo LoRAs and FastVideo hybrid MiniMax H3 adapters."""

    DEFAULT_ALPHA = 128.0

    @classmethod
    def apply(cls, model: torch.nn.Module, configs: Iterable[LoraConfig]) -> int:
        if getattr(model, "quant_type", None) is not None:
            raise ValueError("MiniMax H3 adapters require original DiT weights during merging")
        total = 0
        for config in configs:
            adapter_path, metadata, keys = _adapter_header(config.path)
            replacement_keys = tuple(key for key in keys if key.endswith(FASTVIDEO_REPLACEMENT_SUFFIX))
            if replacement_keys:
                raise ValueError(
                    "This MiniMax H3 adapter contains FastVideo VSA compression-gate .set_weight tensors, "
                    "but TeleFuser does not implement the VSA-H3 gate backend. Use the dense FastH3 adapter "
                    f"variant instead; first unsupported key: {replacement_keys[0]}"
                )
            fastvideo_format = metadata.get("format") == FASTVIDEO_LORA_FORMAT
            loader = LoRALoader(
                MINIMAX_H3_LORA_KEY_MAPPING_RULES,
                target_resolver=minimax_h3_lora_target,
                strict=True,
                # FastVideo adapters define W = W_base + B @ A. Turbo LoRAs
                # omit alpha metadata and use their released alpha=128 contract.
                default_alpha=None if fastvideo_format else cls.DEFAULT_ALPHA,
                stream_safetensors=True,
                merge_dtype=torch.float32,
            )
            applied = loader.apply_lora(model, adapter_path, strength=config.strength)
            total += applied
            logger.info(
                "Loaded MiniMax H3 adapter: {} (format={}, strength={}, tensors={})",
                config.path,
                metadata.get("format", "turbo-lora"),
                config.strength,
                applied,
            )
        return total


__all__ = [
    "FASTVIDEO_LORA_FORMAT",
    "FASTVIDEO_REPLACEMENT_SUFFIX",
    "MINIMAX_H3_LORA_KEY_MAPPING_RULES",
    "MiniMaxH3LoraAdapter",
    "minimax_h3_lora_target",
]
