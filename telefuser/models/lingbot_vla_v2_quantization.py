"""LingBot-VLA v2 helpers for quantized Linear compatibility."""

from __future__ import annotations

import torch
from torch import nn


def linear_compute_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Return a Linear wrapper's activation dtype rather than its packed weight dtype."""
    compute_dtype = getattr(module, "compute_dtype", None)
    if isinstance(compute_dtype, torch.dtype):
        return compute_dtype
    weight = getattr(module, "weight", None)
    weight_dtype = getattr(weight, "dtype", None)
    return weight_dtype if isinstance(weight_dtype, torch.dtype) else fallback
