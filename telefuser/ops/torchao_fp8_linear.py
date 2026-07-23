"""TorchAO FP8 helpers for TeleFuser DiT linear layers.

This backend applies TorchAO dynamic-activation FP8 + FP8 weight quantization
to selected ``nn.Linear`` modules. It targets W8A8 inference on Hopper/H100
and keeps the integration close to TorchAO's native ``quantize_`` API.
"""

from __future__ import annotations

import inspect
from importlib import metadata
from typing import Iterable

import torch.nn as nn

from telefuser.utils.logging import logger


def _import_first_attr(module_names: tuple[str, ...], attr_names: tuple[str, ...]):
    errors: list[str] = []
    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=list(attr_names))
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        for attr_name in attr_names:
            if hasattr(module, attr_name):
                return getattr(module, attr_name)
    raise ImportError("; ".join(errors) if errors else f"none of {attr_names} found")


def _instantiate_config(config_cls, **kwargs):
    try:
        signature = inspect.signature(config_cls)
        accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
    except (TypeError, ValueError):
        accepted = kwargs
    try:
        return config_cls(**accepted)
    except TypeError:
        return config_cls()


def _check_torchao_fp8_available() -> None:
    try:
        metadata.version("torchao")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("TorchAO FP8 requires torchao to be installed") from exc


def _matches_filter(name: str, include_names: Iterable[str] | None, exclude_names: Iterable[str]) -> bool:
    if include_names is not None and not any(token in name for token in include_names):
        return False
    return not any(token and token in name for token in exclude_names)


def _count_matching_linear_layers(
    module: nn.Module,
    *,
    include_names: Iterable[str] | None,
    exclude_names: Iterable[str],
    _prefix: str = "",
) -> int:
    count = 0
    for child_name, child in module.named_children():
        full_name = f"{_prefix}.{child_name}" if _prefix else child_name
        if isinstance(child, nn.Linear):
            if _matches_filter(full_name, include_names, exclude_names):
                count += 1
            continue
        count += _count_matching_linear_layers(
            child,
            include_names=include_names,
            exclude_names=exclude_names,
            _prefix=full_name,
        )
    return count


def replace_linear_layers_with_torchao_fp8(
    module: nn.Module,
    *,
    include_names: Iterable[str] | None = None,
    exclude_names: Iterable[str] = ("head", "time_embedding", "time_projection", "patch_embedding"),
) -> int:
    """Quantize selected ``nn.Linear`` modules with TorchAO FP8.

    Returns the number of selected Linear layers. TorchAO performs in-place
    conversion through ``quantize_``.
    """
    _check_torchao_fp8_available()

    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError("TorchAO FP8 requires torchao.quantization.quantize_") from exc

    fp8_api = _import_first_attr(
        ("torchao.quantization", "torchao.quantization.quant_api", "torchao.float8"),
        (
            "float8_dynamic_activation_float8_weight",
            "Float8DynamicActivationFloat8WeightConfig",
            "Float8WeightOnlyConfig",
        ),
    )
    quant_config = fp8_api() if not inspect.isclass(fp8_api) else _instantiate_config(fp8_api)
    selected = _count_matching_linear_layers(module, include_names=include_names, exclude_names=exclude_names)

    def filter_fn(target: nn.Module, *args) -> bool:
        name = next((str(arg) for arg in args if isinstance(arg, str)), "")
        return isinstance(target, nn.Linear) and _matches_filter(name, include_names, exclude_names)

    quantize_(module, quant_config, filter_fn=filter_fn)
    logger.info(f"TorchAO FP8 quantized {selected} Linear layers")
    return selected
