"""TorchAO NVFP4 helpers for Blackwell linear layers."""

from __future__ import annotations

from importlib import metadata
from typing import Iterable

import torch.nn as nn

from telefuser.utils.logging import logger


def _matches_filter(name: str, include_names: Iterable[str] | None, exclude_names: Iterable[str]) -> bool:
    if include_names is not None and not any(token in name for token in include_names):
        return False
    return not any(token and token in name for token in exclude_names)


def replace_linear_layers_with_torchao_nvfp4(
    module: nn.Module,
    *,
    include_names: Iterable[str] | None = None,
    exclude_names: Iterable[str] = (),
) -> int:
    """Replace selected Linear layers with dynamic-activation W4A4 NVFP4.

    The resulting tensors dispatch NVFP4 GEMMs on supported Blackwell GPUs.
    This is deliberately not a weight-only path, whose BF16 activation GEMM
    would not exercise native FP4 compute.
    """
    try:
        metadata.version("torchao")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("NVFP4 requires the TeleFuser CUDA extra (torchao)") from exc

    try:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError("Installed torchao does not provide dynamic W4A4 NVFP4") from exc

    selected = 0
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and _matches_filter(name, include_names, exclude_names):
            selected += 1

    def filter_fn(target: nn.Module, *args: object) -> bool:
        name = next((str(arg) for arg in args if isinstance(arg, str)), "")
        return isinstance(target, nn.Linear) and _matches_filter(name, include_names, exclude_names)

    config = NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False)
    quantize_(module, config, filter_fn=filter_fn)
    logger.info(f"TorchAO NVFP4 quantized {selected} Linear layers with dynamic W4A4")
    return selected


def enable_torchao_nvfp4_triton(module: nn.Module) -> int:
    """Enable the CUDA Triton W4A4 dispatch after a quantized model is on GPU."""
    enabled = 0
    for parameter in module.parameters():
        tensor = parameter.data
        kwargs = getattr(tensor, "act_quant_kwargs", None)
        if kwargs is not None:
            kwargs.use_triton_kernel = True
            enabled += 1
    if not enabled:
        raise RuntimeError("No dynamic NVFP4 weights were found after quantization")
    return enabled
