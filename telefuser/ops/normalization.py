"""Normalization layers including RMSNorm, LayerNorm, and adaptive layer norm.

Uses CustomOp base class for automatic dispatch between optimized kernels
(Triton on CUDA) and PyTorch-native implementations based on compile state.
"""

from __future__ import annotations

import functools
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CustomOp

KernelName = Literal["norm_infer", "layer_norm_fn", "fused_scale_shift"]


@functools.lru_cache(maxsize=None)
def _get_triton_kernel(name: KernelName) -> Callable:
    """Lazily import Triton kernel to avoid import errors on non-CUDA platforms."""
    if name == "norm_infer":
        from telefuser.kernel.triton import norm_infer

        return norm_infer
    elif name == "layer_norm_fn":
        from telefuser.kernel.triton import layer_norm_fn

        return layer_norm_fn
    elif name == "fused_scale_shift":
        from telefuser.kernel.triton import fused_scale_shift

        return fused_scale_shift
    raise ValueError(f"Unknown kernel: {name}")


class RMSNorm(CustomOp):
    """Root Mean Square Layer Normalization.

    Reference: https://huggingface.co/papers/1910.07467

    Automatically uses Triton kernel on CUDA in eager mode for better performance.
    Falls back to PyTorch implementation in compile mode or on non-CUDA devices.

    Args:
        dim: Dimension for learnable weights.
        eps: Epsilon for numerical stability.
        elementwise_affine: Whether to use learnable weights.
        bias: Whether to use learnable bias.
        device: Device for parameters.
        dtype: Data type for parameters.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward_cuda(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """CUDA-optimized implementation using Triton kernel.

        Note: Device check is needed because CustomOp dispatches based on platform
        availability, not tensor device. CPU tensors on CUDA platform should fall back.
        """
        if hidden_states.device.type != "cuda":
            return self.forward_native(hidden_states)
        if self.elementwise_affine and self.bias is None:
            assert self.weight is not None
            # Ensure input is contiguous for Triton kernel
            hidden_states = hidden_states.contiguous()
            norm_infer = _get_triton_kernel("norm_infer")
            # weight is nn.Parameter - always contiguous by PyTorch convention
            return norm_infer(hidden_states, self.weight, None, self.eps, is_rms_norm=True)
        return self.forward_native(hidden_states)

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation for compile compatibility."""
        input_dtype = hidden_states.dtype
        # FP32 computation for numerical stability
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # Cast to weight dtype for half-precision support
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class LayerNorm(CustomOp):
    """Layer Normalization with optimized Triton kernel support on CUDA.

    Uses Triton kernel when on CUDA in eager mode for better performance.
    Falls back to PyTorch for non-CUDA tensors, non-affine case, or when compiling.

    Args:
        dim: Dimension for learnable weights.
        eps: Epsilon for numerical stability.
        elementwise_affine: Whether to use learnable weights and bias.
        bias: Whether to use bias.
        device: Device for parameters.
        dtype: Data type for parameters.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """CUDA-optimized implementation using Triton kernel.

        Note: Device check is needed because CustomOp dispatches based on platform
        availability, not tensor device. CPU tensors on CUDA platform should fall back.
        """
        if x.device.type != "cuda":
            return self.forward_native(x)
        if not self.elementwise_affine:
            return self.forward_native(x)
        # Triton kernel in eager mode currently bring performance degradation
        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation for compile compatibility."""
        if self.elementwise_affine:
            return F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return F.layer_norm(x, (x.shape[-1],), eps=self.eps)


class AdaLayerNormContinuous(nn.Module):
    """Adaptive layer normalization with continuous conditioning.

    Applies scale and shift from conditioning embedding after normalization.

    Args:
        embedding_dim: Dimension to normalize.
        conditioning_embedding_dim: Conditioning input dimension.
        elementwise_affine: Whether norm has learnable affine parameters.
        eps: Numerical stability epsilon.
        bias: Whether to use bias in conditioning projection.
        norm_type: "layer_norm" or "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        # Output scale + shift (2x embedding_dim)
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # Cast conditioning to input dtype for mixed-precision compatibility
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        # Apply scale-shift: (1 + scale) for multiplicative, + shift for additive
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


def fused_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float = 1.0,
) -> torch.Tensor:
    """Fused scale and shift operation.

    Computes: output = x * (scale_constant + scale) + shift

    Uses Triton kernel on CUDA in eager mode for better performance.
    Falls back to PyTorch implementation in compile mode or on non-CUDA devices.

    Args:
        x: Input tensor of shape [B, L, C]
        scale: Scale tensor
        shift: Shift tensor
        scale_constant: Constant to add to scale (default 1.0)

    Returns:
        Output tensor of same shape as input
    """
    if torch.compiler.is_compiling():
        return x * (scale_constant + scale) + shift

    if x.device.type == "cuda":
        # Ensure all tensors are contiguous for Triton kernel
        x = x.contiguous()
        scale = scale.contiguous()
        shift = shift.contiguous()
        fused_scale_shift_kernel = _get_triton_kernel("fused_scale_shift")
        return fused_scale_shift_kernel(x, scale, shift, scale_constant)

    return x * (scale_constant + scale) + shift


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply modulation: x * (1 + scale) + shift.

    Uses fused_scale_shift on CUDA in eager mode for better performance.

    Args:
        x: Input tensor to modulate.
        shift: Shift tensor for additive modulation.
        scale: Scale tensor for multiplicative modulation.

    Returns:
        Modulated tensor: x * (1 + scale) + shift
    """
    return fused_scale_shift(x, scale, shift, scale_constant=1.0)


__all__ = [
    "RMSNorm",
    "LayerNorm",
    "AdaLayerNormContinuous",
    "fused_scale_shift",
    "modulate",
]
