"""Report LingBot-Video transformer checkpoint loading acceptance evidence."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from telefuser.core.module_manager import ModuleManager

_DENSE_CONFIG_KEYS = (
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
_MOE_CONFIG_KEYS = _DENSE_CONFIG_KEYS + (
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


def checkpoint_keys(checkpoint_dir: Path) -> set[str]:
    """Read checkpoint keys without materializing all checkpoint tensors."""
    index_path = checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"invalid safetensors index: {index_path}")
        return set(weight_map)
    checkpoint_path = checkpoint_dir / "diffusion_pytorch_model.safetensors"
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        return set(handle.keys())


def checkpoint_key_coverage(model: torch.nn.Module, available_keys: set[str]) -> dict[str, Any]:
    """Report exact checkpoint key coverage without retaining checkpoint tensors."""
    expected = set(model.state_dict())
    matched = expected & available_keys
    return {
        "expected_key_count": len(expected),
        "checkpoint_key_count": len(available_keys),
        "matched_key_count": len(matched),
        "coverage": len(matched) / len(expected) if expected else 1.0,
        "missing_keys": sorted(expected - available_keys),
        "unexpected_keys": sorted(available_keys - expected),
        "matched_numel": sum(model.state_dict()[name].numel() for name in matched),
    }


def _parameter_report(model: torch.nn.Module) -> dict[str, Any]:
    """Summarize parameter count, placement, precision, and model structure."""
    by_dtype: dict[str, int] = defaultdict(int)
    by_device: dict[str, int] = defaultdict(int)
    by_component: dict[str, int] = defaultdict(int)
    by_block: dict[str, int] = defaultdict(int)
    total = 0
    fp32 = 0
    byte_count = 0
    for name, parameter in model.named_parameters():
        numel = parameter.numel()
        total += numel
        byte_count += numel * parameter.element_size()
        by_dtype[str(parameter.dtype).removeprefix("torch.")] += numel
        by_device[str(parameter.device)] += numel
        by_component[name.split(".", 1)[0]] += numel
        if name.startswith("blocks."):
            by_block[name.split(".", 2)[1]] += numel
        if parameter.dtype == torch.float32:
            fp32 += numel
    return {
        "total_parameter_count": total,
        "estimated_parameter_bytes": byte_count,
        "retained_fp32_parameter_count": fp32,
        "parameter_count_by_dtype": dict(sorted(by_dtype.items())),
        "parameter_count_by_device": dict(sorted(by_device.items())),
        "parameter_count_by_component": dict(sorted(by_component.items())),
        "parameter_count_by_block": dict(sorted(by_block.items(), key=lambda item: int(item[0]))),
    }


def build_load_report(
    model: torch.nn.Module,
    *,
    checkpoint_dir: Path,
    variant: str,
    config: dict[str, Any],
    available_keys: set[str],
) -> dict[str, Any]:
    """Build the plan-required acceptance report after a strict model load."""
    consumed_keys = _DENSE_CONFIG_KEYS if variant == "dense" else _MOE_CONFIG_KEYS
    return {
        "variant": variant,
        "checkpoint_dir": str(checkpoint_dir),
        "consumed_config": {key: config[key] for key in consumed_keys},
        "checkpoint_key_coverage": checkpoint_key_coverage(model, available_keys),
        "parameters": _parameter_report(model),
    }


def _dtype(value: str) -> torch.dtype:
    values = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    try:
        return values[value]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {value}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Checkpoint root containing transformer/ and refiner/."
    )
    parser.add_argument("--variant", choices=("dense", "moe", "refiner"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    checkpoint_dir = args.model_dir / ("transformer" if args.variant in {"dense", "moe"} else "refiner")
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    available_keys = checkpoint_keys(checkpoint_dir)
    cuda_before = (
        torch.cuda.memory_allocated(args.device) if torch.cuda.is_available() and "cuda" in args.device else None
    )
    started = time.perf_counter()
    module_manager = ModuleManager(device=args.device, torch_dtype=_dtype(args.dtype))
    module_manager.load_model(str(checkpoint_dir), name="transformer")
    model = module_manager.fetch_module("transformer")
    if model is None:
        raise RuntimeError(f"Unable to load LingBot-Video transformer from {checkpoint_dir}")
    model.promote_stability_layers_to_fp32()
    report = build_load_report(
        model,
        checkpoint_dir=checkpoint_dir,
        variant=args.variant,
        config=config,
        available_keys=available_keys,
    )
    report["load_seconds"] = time.perf_counter() - started
    if cuda_before is not None:
        report["measured_cuda_allocated_bytes"] = torch.cuda.memory_allocated(args.device) - cuda_before
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
