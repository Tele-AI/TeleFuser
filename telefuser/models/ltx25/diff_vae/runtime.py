"""Runtime primitives required by the isolated LTX-2.5 DiffVAE path."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import torch
from torch import nn


class AttentionCallable(Protocol):
    """Callable interface used by the ConvVAE spatial-attention blocks."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor: ...


def _sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor:
    batch, _, channels = q.shape
    head_dim = channels // heads
    q, k, v = (tensor.view(batch, -1, heads, head_dim).transpose(1, 2) for tensor in (q, k, v))
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    return output.transpose(1, 2).reshape(batch, -1, channels)


class AttentionFunction(Enum):
    """Supported eager attention backend for the isolated ConvVAE."""

    PYTORCH = "pytorch"

    def to_callable(self) -> AttentionCallable:
        """Resolve the configured backend without importing upstream LTX-Core."""
        return _sdpa_attention


class PixelNorm(nn.Module):
    """Per-location RMS normalization used by the LTX video VAE."""

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize values over the configured channel dimension."""
        return value / torch.sqrt(torch.mean(value.square(), dim=self.dim, keepdim=True) + self.eps)


class Disposable:
    """Mixin that releases parameters and persistent buffers to the meta device."""

    def dispose(self) -> None:
        """Release tensor storage while preserving the module structure."""
        if not isinstance(self, nn.Module):
            raise TypeError(f"{type(self).__name__} must be an nn.Module to dispose")
        persistent = set(self.state_dict())
        for name, parameter in list(self.named_parameters()):
            parent_name, _, attribute = name.rpartition(".")
            parent = self.get_submodule(parent_name) if parent_name else self
            setattr(
                parent,
                attribute,
                nn.Parameter(torch.empty_like(parameter, device="meta"), requires_grad=parameter.requires_grad),
            )
        for name, buffer in list(self.named_buffers()):
            if name not in persistent:
                continue
            parent_name, _, attribute = name.rpartition(".")
            parent = self.get_submodule(parent_name) if parent_name else self
            parent.register_buffer(attribute, torch.empty_like(buffer, device="meta"), persistent=True)


@dataclass(frozen=True)
class CompilationConfig:
    """Minimal compile settings accepted by the isolated DiffVAE implementation."""

    mode: str | None = None
    backend: str | None = None
    fullgraph: bool = False
    dynamic: bool = False
    inductor_config: dict[str, Any] = field(default_factory=dict)
    dynamo_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModuleOps:
    """Local model mutator used by DiffVAE configuration helpers."""

    name: str
    matcher: Callable[[nn.Module], bool]
    mutator: Callable[[nn.Module], nn.Module]
