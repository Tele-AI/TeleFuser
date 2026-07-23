"""bitsandbytes NF4 helpers for TeleFuser DiT linear layers."""

from __future__ import annotations

from importlib import metadata
from typing import Iterable

import torch
import torch.nn as nn

from telefuser.utils.logging import logger


def _check_bnb_available() -> None:
    try:
        metadata.version("bitsandbytes")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("BNB NF4 requires bitsandbytes to be installed") from exc
    try:
        import bitsandbytes as bnb  # noqa: F401
    except OSError as exc:
        raise RuntimeError(
            "bitsandbytes failed to load its CUDA library. Check LD_LIBRARY_PATH/CUDA_HOME; "
            "for CUDA 13 PyTorch wheels, libnvJitLink.so.13 must be discoverable."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"bitsandbytes import failed: {exc}") from exc


def _matches_filter(name: str, include_names: Iterable[str] | None, exclude_names: Iterable[str]) -> bool:
    if include_names is not None and not any(token in name for token in include_names):
        return False
    return not any(token and token in name for token in exclude_names)


def _make_bnb_nf4_linear(linear: nn.Linear, *, compute_dtype: torch.dtype, compress_statistics: bool) -> nn.Module:
    import bitsandbytes as bnb

    device = linear.weight.device
    new_linear = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type="nf4",
        device=device,
    )
    new_linear.weight = bnb.nn.Params4bit(
        linear.weight.detach().contiguous(),
        requires_grad=False,
        compress_statistics=compress_statistics,
        quant_type="nf4",
    )
    if linear.bias is not None:
        new_linear.bias = nn.Parameter(linear.bias.detach().to(device=device, dtype=compute_dtype), requires_grad=False)
    new_linear = new_linear.to(device=device)
    new_linear.requires_grad_(False)
    return new_linear


def replace_linear_layers_with_bnb_nf4(
    module: nn.Module,
    *,
    compute_dtype: torch.dtype = torch.bfloat16,
    include_names: Iterable[str] | None = None,
    exclude_names: Iterable[str] = ("head", "time_embedding", "time_projection", "patch_embedding"),
    compress_statistics: bool = True,
    _prefix: str = "",
) -> int:
    """Replace selected ``nn.Linear`` modules with bitsandbytes NF4 Linear4bit."""
    _check_bnb_available()
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{_prefix}.{child_name}" if _prefix else child_name
        if isinstance(child, nn.Linear):
            if _matches_filter(full_name, include_names, exclude_names):
                setattr(
                    module,
                    child_name,
                    _make_bnb_nf4_linear(
                        child,
                        compute_dtype=compute_dtype,
                        compress_statistics=compress_statistics,
                    ),
                )
                replaced += 1
            continue
        replaced += replace_linear_layers_with_bnb_nf4(
            child,
            compute_dtype=compute_dtype,
            include_names=include_names,
            exclude_names=exclude_names,
            compress_statistics=compress_statistics,
            _prefix=full_name,
        )
    if not _prefix:
        logger.info(f"BNB NF4 replaced {replaced} Linear layers")
    return replaced
