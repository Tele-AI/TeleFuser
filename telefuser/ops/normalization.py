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

_compiled_rmsnorm: Callable | None = None
try:
    from tf_kernel import rmsnorm as _compiled_rmsnorm
except ImportError:
    pass

KernelName = Literal[
    "norm_infer",
    "layer_norm_fn",
    "fused_scale_shift",
    "fused_layernorm_scale_shift",
    "fused_add_layernorm_scale_shift",
    "indexed_gate_bf16_",
    "indexed_scale_shift_bf16_",
]


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
    elif name == "fused_layernorm_scale_shift":
        from telefuser.kernel.triton import fused_layernorm_scale_shift

        return fused_layernorm_scale_shift
    elif name == "fused_add_layernorm_scale_shift":
        from telefuser.kernel.triton import fused_add_layernorm_scale_shift

        return fused_add_layernorm_scale_shift
    elif name == "indexed_gate_bf16_":
        from telefuser.kernel.triton import indexed_gate_bf16_

        return indexed_gate_bf16_
    elif name == "indexed_scale_shift_bf16_":
        from telefuser.kernel.triton import indexed_scale_shift_bf16_

        return indexed_scale_shift_bf16_
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
            if (
                _compiled_rmsnorm is not None
                and hidden_states.dtype in (torch.float16, torch.bfloat16)
                and self.weight.dtype == hidden_states.dtype
            ):
                shape = hidden_states.shape
                normalized = _compiled_rmsnorm(
                    hidden_states.view(-1, shape[-1]),
                    self.weight,
                    self.eps,
                )
                return normalized.view(shape)
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


def _validate_indexed_modulation_inputs(
    x: torch.Tensor,
    indices: torch.Tensor,
    *,
    row_parameters: tuple[torch.Tensor, ...],
    per_row_tensors: tuple[torch.Tensor, ...] = (),
) -> None:
    if x.ndim != 2:
        raise ValueError("indexed modulation input must be a two-dimensional tensor")
    if indices.ndim != 1 or indices.numel() != x.shape[0]:
        raise ValueError("indexed modulation indices must contain one entry per input row")
    if indices.dtype not in {torch.int32, torch.int64}:
        raise TypeError("indexed modulation indices must use int32 or int64")
    tensors = (*row_parameters, *per_row_tensors, indices)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("indexed modulation tensors must be on the same device")
    if any(parameter.ndim != 2 or parameter.shape[1] != x.shape[1] for parameter in row_parameters):
        raise ValueError("indexed modulation parameter tensors must match the input hidden dimension")
    if any(parameter.shape[0] == 0 for parameter in row_parameters):
        raise ValueError("indexed modulation parameter tensors must contain at least one row")
    parameter_rows = row_parameters[0].shape[0]
    if any(parameter.shape[0] != parameter_rows for parameter in row_parameters[1:]):
        raise ValueError("indexed modulation parameter tensors must contain the same number of rows")
    if any(tensor.shape != x.shape for tensor in per_row_tensors):
        raise ValueError("indexed modulation per-row tensors must match the input shape")


def indexed_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Apply row-indexed scale and shift, reusing disposable CUDA BF16 input."""
    _validate_indexed_modulation_inputs(x, indices, row_parameters=(shift, scale))
    if (
        not torch.compiler.is_compiling()
        and x.device.type == "cuda"
        and x.dtype == shift.dtype == scale.dtype == torch.bfloat16
        and x.ndim == shift.ndim == scale.ndim == 2
        and indices.ndim == 1
        and x.is_contiguous()
        and shift.stride(-1) == 1
        and scale.stride(-1) == 1
    ):
        kernel = _get_triton_kernel("indexed_scale_shift_bf16_")
        return kernel(x, shift, scale, indices.contiguous())
    return x * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def indexed_rmsnorm_scale_shift(
    norm: RMSNorm,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Apply RMSNorm and indexed modulation through the public ops boundary.

    The fused kernel is selected here so model implementations do not import
    internal Triton modules or make compile-mode decisions themselves.
    """
    _validate_indexed_modulation_inputs(x, indices, row_parameters=(shift, scale))
    if (
        not torch.compiler.is_compiling()
        and x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and norm.weight is not None
        and norm.weight.dtype == shift.dtype == scale.dtype == x.dtype
        and norm.weight.is_contiguous()
        and shift.stride(-1) == scale.stride(-1) == 1
    ):
        from telefuser.kernel.triton.indexed_rmsnorm_modulation import indexed_rmsnorm_scale_shift_bf16

        return indexed_rmsnorm_scale_shift_bf16(
            x,
            norm.weight,
            shift,
            scale,
            indices.contiguous(),
            norm.eps,
        )
    return indexed_scale_shift(norm(x), shift, scale, indices)


def indexed_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Apply a row-indexed gated residual, reusing disposable CUDA BF16 input."""
    _validate_indexed_modulation_inputs(x, indices, row_parameters=(gate,), per_row_tensors=(other,))
    if (
        not torch.compiler.is_compiling()
        and x.device.type == "cuda"
        and x.dtype == gate.dtype == other.dtype == torch.bfloat16
        and x.ndim == gate.ndim == other.ndim == 2
        and indices.ndim == 1
        and x.is_contiguous()
        and gate.stride(-1) == 1
        and other.is_contiguous()
    ):
        kernel = _get_triton_kernel("indexed_gate_bf16_")
        return kernel(x, gate, other, indices.contiguous())
    return x + gate.index_select(0, indices) * other


def _fused_layer_norm_scale_shift(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fuse LayerNorm and adaptive modulation while preserving the input dtype."""
    if torch.compiler.is_compiling() or x.device.type != "cuda":
        normalized = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        return (normalized * (1 + scale) + shift).to(dtype=x.dtype)
    kernel = _get_triton_kernel("fused_layernorm_scale_shift")
    return kernel(x.contiguous(), weight, bias, scale, shift, eps)


def _fused_add_layer_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse residual add, LayerNorm, and adaptive modulation."""
    if torch.compiler.is_compiling() or x.device.type != "cuda":
        residual_out = (residual + x).to(dtype=residual.dtype)
        normalized = F.layer_norm(residual_out, (residual_out.shape[-1],), weight, bias, eps)
        return (normalized * (1 + scale) + shift).to(dtype=residual.dtype), residual_out
    kernel = _get_triton_kernel("fused_add_layernorm_scale_shift")
    return kernel(residual.contiguous(), x.contiguous(), weight, bias, scale, shift, eps)


__all__ = [
    "RMSNorm",
    "LayerNorm",
    "AdaLayerNormContinuous",
    "fused_scale_shift",
    "modulate",
    "indexed_gate",
    "indexed_scale_shift",
]
