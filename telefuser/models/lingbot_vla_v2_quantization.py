"""LingBot-VLA v2 helpers for quantized Linear compatibility and identity."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from importlib import metadata
from typing import Iterable

import torch
from torch import nn

LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZED_LINEAR_COUNT = 492
LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZATION_MANIFEST_SHA256 = (
    "f9efe28620796060ccc46bd18ac153a580b28d01c7719fa55a8e80631f2ce833"
)


def _matches_tokens(name: str, tokens: Iterable[str]) -> bool:
    return any(token and token in name for token in tokens)


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _linear_group(name: str) -> str:
    if "qwenvl.model.language_model.layers." in name:
        return "qwen_language"
    if "qwenvl.model.visual.blocks." in name:
        return "qwen_visual"
    if "qwen_expert" in name and ".self_attn." in name:
        return "action_expert_attention"
    return "other"


def build_lingbot_vla_v2_linear_manifest(
    module: nn.Module,
    *,
    include_names: Iterable[str],
    exclude_names: Iterable[str],
) -> dict[str, object]:
    """Describe the exact Linear modules selected before online quantization."""
    include_tokens = tuple(include_names)
    exclude_tokens = tuple(exclude_names)
    selected_names: list[str] = []
    excluded_names: list[str] = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if _matches_tokens(name, exclude_tokens):
            excluded_names.append(name)
        elif _matches_tokens(name, include_tokens):
            selected_names.append(name)

    selected_names.sort()
    excluded_names.sort()
    group_counts = Counter(_linear_group(name) for name in selected_names)
    canonical = json.dumps(
        {
            "include_names": include_tokens,
            "exclude_names": exclude_tokens,
            "selected_names": selected_names,
            "excluded_names": excluded_names,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "selected_count": len(selected_names),
        "selected_names": selected_names,
        "excluded_count": len(excluded_names),
        "excluded_names": excluded_names,
        "group_counts": dict(sorted(group_counts.items())),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def finalize_lingbot_vla_v2_quantization_identity(
    module: nn.Module,
    *,
    profile: str,
    quant_type: str,
    kernel_backend: str,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Attach JSON-safe backend and wrapper facts after conversion."""
    modules = dict(module.named_modules())
    selected_names = manifest.get("selected_names", [])
    if not isinstance(selected_names, list):
        raise TypeError("LingBot-VLA v2 quantization manifest selected_names must be a list")

    module_types: Counter[str] = Counter()
    weight_types: Counter[str] = Counter()
    weight_dtypes: Counter[str] = Counter()
    missing_names: list[str] = []
    for name in selected_names:
        selected = modules.get(str(name))
        if selected is None:
            missing_names.append(str(name))
            continue
        module_types[_qualified_type(selected)] += 1
        weight = getattr(selected, "weight", None)
        if weight is None:
            weight_types["none"] += 1
            weight_dtypes["none"] += 1
        else:
            weight_types[_qualified_type(weight)] += 1
            weight_dtypes[str(getattr(weight, "dtype", "unknown")).removeprefix("torch.")] += 1
    if missing_names:
        raise RuntimeError(f"quantized Linear modules disappeared from the model: {missing_names[:5]}")

    identity = {
        "enabled": True,
        "profile": profile,
        "quant_type": quant_type,
        "kernel_backend": kernel_backend,
        "packages": {
            "torchao": _package_version("torchao"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "tf-kernel": _package_version("tf-kernel"),
        },
        "implementation": {
            "module_types": dict(sorted(module_types.items())),
            "weight_types": dict(sorted(weight_types.items())),
            "weight_dtypes": dict(sorted(weight_dtypes.items())),
        },
        "manifest": copy.deepcopy(manifest),
    }
    module._lingbot_vla_v2_quantization_identity = identity
    return copy.deepcopy(identity)


def lingbot_vla_v2_quantization_identity(module: nn.Module) -> dict[str, object]:
    """Return bounded runtime identity for BF16 or an applied quantization profile."""
    identity = getattr(module, "_lingbot_vla_v2_quantization_identity", None)
    if isinstance(identity, dict):
        return copy.deepcopy(identity)
    return {
        "enabled": False,
        "profile": "bf16",
        "quant_type": None,
        "kernel_backend": None,
        "packages": {
            "torchao": _package_version("torchao"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "tf-kernel": _package_version("tf-kernel"),
        },
        "implementation": {},
        "manifest": None,
    }


def linear_compute_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Return a Linear wrapper's activation dtype rather than its packed weight dtype."""
    compute_dtype = getattr(module, "compute_dtype", None)
    if isinstance(compute_dtype, torch.dtype):
        return compute_dtype
    weight = getattr(module, "weight", None)
    weight_dtype = getattr(weight, "dtype", None)
    return weight_dtype if isinstance(weight_dtype, torch.dtype) else fallback
