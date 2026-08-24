"""Checkpoint resolution and loading for LingBot-VLA v2."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import torch
from transformers import AutoConfig

from telefuser.core.config import QuantConfig

if TYPE_CHECKING:
    from telefuser.core.module_manager import ModuleManager
    from telefuser.models.lingbot_vla_v2 import LingBotVlaV2Model, LingbotVLAV2Config

OFFICIAL_6B_MODEL_CONFIG: dict[str, Any] = {
    "post_training": False,
    "adanorm_time": True,
    "moe_implementation": "fused",
    "use_robby_moe_kernel": True,
    "attention_implementation": "eager",
    "precompute_grid_thw": True,
    "vlm_causal": True,
    "use_moe": True,
    "token_moe_layers": list(range(36)),
    "token_num_experts": 32,
    "token_top_k": 4,
    "token_moe_intermediate_size": 512,
    "token_shared_intermediate_size": 704,
    "bias_update_speed": 0.0,
    "sequence_wise_mode": "per_sequence",
    "sequence_wise_loss_coeff": 1e-3,
    "router_z_loss_coeff": 1e-4,
    "router_activation": "sigmoid",
    "routed_scaling_factor": 4.0,
    "use_shared_expert_gate": False,
    "freeze_vision_encoder": False,
    "tokenizer_max_length": 72,
    "loss_type": "L1_fm",
    "action_dim": 55,
    "max_action_dim": 55,
    "max_state_dim": 55,
    "align_params": {
        "mode": "query",
        "num_task_tokens": 8,
        "depth_loss_weight": 0.004,
        "future_depth_loss_weight": 0.004,
        "use_future_video": True,
        "llm": {
            "dim_out": 2560,
            "image_token_size": 8,
            "image_input_size": 224,
        },
        "depth": {
            "model_type": "MoRGBD",
            "num_layers": 1,
            "num_heads": 4,
            "dim_head": 32,
            "ff_mult": 1,
            "num_backbone_tokens": 256,
            "token_size": 16,
            "dim_out": 1024,
            "input_size": 224,
            "use_future_depth": True,
            "block_future_depth_to_action": True,
            "future_depth_head_type": "resampler",
            "detach_future_image_feats": True,
        },
        "video": {
            "attention_mode": "flex_block_causal",
            "input_size": 256,
            "block_suffix_to_future_video": True,
            "share_future_depth_query": True,
            "use_shared_future_task_proj": True,
            "use_current_shared_task_proj": True,
            "num_future_frames": 1,
            "use_warmup_frame": True,
            "effective_fps": 1.0,
            "n_blocks": 1,
            "cls_pool": "last",
            "detach_image_feats": True,
            "num_layers": 1,
            "num_heads": 4,
            "dim_head": 32,
            "ff_mult": 1,
            "num_backbone_tokens": 256,
            "dim_out": 1024,
            "future_video_loss_weight": 0.004,
            "use_smooth_l1_loss": False,
            "use_mse_loss": True,
            "mse_loss_weight": 1.0,
            "use_patch_loss": True,
            "use_current_patch_loss": True,
            "use_cosine_loss": False,
            "cosine_loss_weight": 0.2,
            "use_cls_loss": False,
            "cls_loss_type": "mse",
            "cls_loss_weight": 0.2,
        },
    },
}


def resolve_lingbot_vla_v2_checkpoint(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.name != "model.safetensors.index.json":
            raise ValueError(f"Expected model.safetensors.index.json, got: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"LingBot-VLA v2 model path does not exist: {path}")
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sharded checkpoint index: {index_path}")
    return index_path


def resolve_lingbot_vla_v2_shards(model_path: str | Path) -> list[str]:
    index_path = resolve_lingbot_vla_v2_checkpoint(model_path)
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index without weight_map: {index_path}")

    shard_paths = [index_path.parent / name for name in sorted(set(weight_map.values()))]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LingBot-VLA v2 checkpoint shards: {missing}")
    return [str(path) for path in shard_paths]


def build_official_6b_config(
    qwen3vl_path: str | Path,
    *,
    checkpoint_variant: str = "base",
    checkpoint_path: str | Path | None = None,
) -> LingbotVLAV2Config:
    from telefuser.models.lingbot_vla_v2 import LingbotVLAV2Config

    if checkpoint_variant != "base":
        raise ValueError(f"Unsupported LingBot-VLA v2 checkpoint variant: {checkpoint_variant!r}")

    qwen_path = Path(qwen3vl_path).expanduser().resolve()
    qwen_config = AutoConfig.from_pretrained(str(qwen_path), local_files_only=True)
    if not hasattr(qwen_config, "text_config") or not hasattr(qwen_config, "vision_config"):
        raise ValueError(
            "LingBot-VLA v2 requires the local Qwen3-VL-4B-Instruct architecture/tokenizer "
            f"directory; this is not a complete Qwen3-VL directory: {qwen_path}"
        )

    text_config = qwen_config.text_config
    expected_architecture = {"hidden_size": 2560, "num_hidden_layers": 36}
    mismatches = {
        key: (expected, getattr(text_config, key, None))
        for key, expected in expected_architecture.items()
        if getattr(text_config, key, None) != expected
    }
    if mismatches:
        raise ValueError(
            "LingBot-VLA v2 6B was trained with Qwen3-VL-4B-Instruct; "
            f"the supplied architecture is incompatible: {mismatches}"
        )

    values = deepcopy(OFFICIAL_6B_MODEL_CONFIG)
    values["tokenizer_path"] = str(qwen_path)
    config = LingbotVLAV2Config(**values)
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
    ):
        if hasattr(text_config, key):
            setattr(config, key, getattr(text_config, key))
    config.vision_config = qwen_config.vision_config
    config.tokenizer_path = str(qwen_path)
    config.use_cache = True
    config.attention_implementation = "eager"
    config.checkpoint_variant = checkpoint_variant
    config.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve())
    config.policy_verified = False
    config.verification_status = "unverified_official_6b_base"
    return config


def validate_official_6b_checkpoint(state_dict: dict[str, torch.Tensor]) -> None:
    gate = "model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.gate_proj"
    last_gate = "model.qwenvl_with_expert.qwen_expert.model.layers.35.mlp.experts.gate_proj"
    expected = (32, 512, 768)
    for key in (gate, last_gate):
        if key not in state_dict:
            raise ValueError(f"Missing official LingBot-VLA v2 weight: {key}")
        if tuple(state_dict[key].shape) != expected:
            raise ValueError(f"Unexpected shape for {key}: expected {expected}, got {tuple(state_dict[key].shape)}")


class LingBotVlaV2StateDictConverter:
    def __init__(
        self,
        qwen3vl_path: str | Path,
        checkpoint_variant: str = "base",
        checkpoint_path: str | Path | None = None,
    ):
        self.qwen3vl_path = Path(qwen3vl_path)
        self.checkpoint_variant = checkpoint_variant
        self.checkpoint_path = checkpoint_path

    def from_official(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        validate_official_6b_checkpoint(state_dict)
        config = build_official_6b_config(
            self.qwen3vl_path,
            checkpoint_variant=self.checkpoint_variant,
            checkpoint_path=self.checkpoint_path,
        )
        return state_dict, {"config": config, "eval": True}

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> NoReturn:
        del state_dict
        raise ValueError("LingBot-VLA v2 does not provide a Diffusers checkpoint")


def load_lingbot_vla_v2(
    module_manager: ModuleManager,
    model_path: str | Path,
    qwen3vl_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    checkpoint_variant: str = "base",
    quant_config: QuantConfig | None = None,
) -> LingBotVlaV2Model:
    from telefuser.models.lingbot_vla_v2 import LingBotVlaV2Model

    checkpoint_path = resolve_lingbot_vla_v2_checkpoint(model_path).parent
    shard_paths = resolve_lingbot_vla_v2_shards(checkpoint_path)
    module_manager.load_model(
        shard_paths,
        device=device,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        name="lingbot_vla_v2",
        model_class=LingBotVlaV2Model,
        model_resource="official",
        converter_kwargs={
            "qwen3vl_path": str(qwen3vl_path),
            "checkpoint_variant": checkpoint_variant,
            "checkpoint_path": str(checkpoint_path),
        },
        quant_config=quant_config,
    )
    return module_manager.fetch_module("lingbot_vla_v2")
