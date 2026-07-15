"""LingBot-World v2 causal-fast facade and checkpoint validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from telefuser.pipelines.lingbot_world_fast.pipeline import (
    LingBotWorldFastPipeline,
    LingBotWorldFastPipelineConfig,
)
from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig


def resolve_lingbot_world_v2_transformers(checkpoint_path: str | Path) -> Path:
    """Validate and return the v2 transformer directory for either accepted layout."""
    candidate = Path(checkpoint_path).expanduser().resolve()
    transformers_dir = candidate if candidate.name == "transformers" else candidate / "transformers"
    checkpoint_root = transformers_dir.parent
    config_path = checkpoint_root / "config.json"
    index_path = transformers_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LingBot v2 config.json not found: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"LingBot v2 weight index not found: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"LingBot v2 weight index has no weight_map: {index_path}")
    shard_names = set(weight_map.values())
    if not all(isinstance(name, str) for name in shard_names):
        raise ValueError(f"LingBot v2 weight index has invalid shard names: {index_path}")
    missing = sorted(name for name in shard_names if not (transformers_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"LingBot v2 is missing checkpoint shards: {missing}")

    control_key = "patch_embedding_wancamctrl.weight"
    if control_key not in weight_map:
        raise ValueError(f"LingBot v2 checkpoint is missing {control_key}")
    with safe_open(transformers_dir / weight_map[control_key], framework="pt", device="cpu") as tensors:
        control_shape = tuple(tensors.get_tensor(control_key).shape)
    if len(control_shape) != 2 or control_shape[1] % (64 * 2 * 2):
        raise ValueError(f"LingBot v2 control projection has invalid shape: {control_shape}")
    control_channels = control_shape[1] // (64 * 2 * 2)
    if control_channels != 6:
        raise ValueError(f"LingBot v2 requires a six-channel camera checkpoint, got {control_channels} channels")
    return transformers_dir


@dataclass
class LingBotWorldV2PipelineConfig(LingBotWorldFastPipelineConfig):
    """Configuration for the public LingBot-World v2 causal-fast checkpoint."""

    fast_checkpoint_path: str = "transformers"
    control_type: str = "cam"
    local_attn_size: int = 18
    sink_size: int = 6
    timestep_indices: tuple[int, ...] = (0, 250, 500, 750)


class LingBotWorldV2Pipeline(LingBotWorldFastPipeline):
    """Thin v2 facade over the shared LingBot causal-fast engine."""

    def init(self, config: LingBotWorldV2PipelineConfig) -> None:
        if config.control_type != "cam":
            raise ValueError("LingBot-World v2 causal-fast supports camera control only")
        transformers_dir = resolve_lingbot_world_v2_transformers(
            Path(config.checkpoint_dir) / config.fast_checkpoint_path
            if not Path(config.fast_checkpoint_path).is_absolute()
            else config.fast_checkpoint_path
        )
        config.fast_checkpoint_path = str(transformers_dir)
        super().init(config)
