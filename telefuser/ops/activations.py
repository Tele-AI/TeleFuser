"""Activation functions for neural networks.

Provides standard and specialized activation functions including:
- GELU variants (exact, approximate, gating)
- SiLU/Swish variants
- Gated linear units (GEGLU, SwiGLU)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from telefuser.platforms import CudaPlatform, current_platform

# Check if tf_kernel is available (only supported on NVIDIA CUDA)
_has_tf_kernel = False
if isinstance(current_platform, CudaPlatform):
    try:
        from tf_kernel import gelu_and_mul as _tf_gelu_and_mul
        from tf_kernel import silu_and_mul as _tf_silu_and_mul

        _has_tf_kernel = True
    except ImportError:
        pass


ACT2CLS: dict[str, type[nn.Module]] = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Get activation function by name."""
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    raise ValueError(f"activation function {act_fn} not found")


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Apply SiLU gated multiplication: silu(x1) * x2.

    Uses tf_kernel fused kernel when available on NVIDIA CUDA for better performance,
    falls back to PyTorch for ROCm and other platforms.

    Args:
        x: Input tensor where last dimension is split into [x1, x2].

    Returns:
        Gated output: silu(x1) * x2
    """
    if _has_tf_kernel and x.device.type == "cuda":
        return _tf_silu_and_mul(x)
    if not torch.compiler.is_compiling() and x.device.type == "npu":
        import torch_npu

        return torch_npu.npu_swiglu(x, dim=-1)
    # Fallback to PyTorch
    gate, val = x.chunk(2, dim=-1)
    return F.silu(gate) * val


def silu_and_mul_reuse_input(x: torch.Tensor) -> torch.Tensor:
    """Apply H3-compatible rounded SiLU gating while reusing disposable BF16 input."""
    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[-1] % 16 == 0
        and not torch.compiler.is_compiling()
    ):
        from telefuser.kernel.triton import silu_and_mul_bf16_input_inplace_

        return silu_and_mul_bf16_input_inplace_(x)
    return silu_and_mul(x)


def gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU gated multiplication: gelu(x1) * x2.

    Uses tf_kernel fused kernel when available on NVIDIA CUDA for better performance,
    falls back to PyTorch for ROCm and other platforms.

    Args:
        x: Input tensor where last dimension is split into [x1, x2].

    Returns:
        Gated output: gelu(x1) * x2
    """
    if _has_tf_kernel and x.device.type == "cuda":
        return _tf_gelu_and_mul(x)
    # Fallback to PyTorch
    gate, val = x.chunk(2, dim=-1)
    return F.gelu(gate) * val


class FP32SiLU(nn.Module):
    """SiLU activation with FP32 upcast for numerical stability."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)


class GELU(nn.Module):
    """GELU activation with optional tanh approximation.

    Reference: https://huggingface.co/papers/1606.08415 (Section 2)

    Args:
        dim_in: Input dimension.
        dim_out: Output dimension.
        approximate: Use "tanh" approximation or "none" for exact.
        bias: Whether to use bias in linear projection.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return self.gelu(hidden_states)


class GEGLU(nn.Module):
    """Gated GELU unit: splits input, applies GELU to gate, multiplies.

    Reference: https://huggingface.co/papers/2002.05202

    Args:
        dim_in: Input dimension.
        dim_out: Output dimension.
        bias: Whether to use bias in linear projection.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        # Output 2x dim for gate splitting
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return gelu_and_mul(hidden_states)


class SwiGLU(nn.Module):
    """Swish-gated linear unit: like GEGLU but with SiLU activation.

    Reference: https://huggingface.co/papers/2002.05202

    Args:
        dim_in: Input dimension.
        dim_out: Output dimension.
        bias: Whether to use bias in linear projection.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return silu_and_mul(hidden_states)


class ApproximateGELU(nn.Module):
    """Fast GELU approximation using sigmoid: x * sigmoid(1.702 * x).

    Reference: https://huggingface.co/papers/1606.08415 (Section 2)

    Args:
        dim_in: Input dimension.
        dim_out: Output dimension.
        bias: Whether to use bias in linear projection.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class LinearActivation(nn.Module):
    """Linear projection followed by activation."""

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, activation: str = "silu"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = get_activation(activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return self.activation(hidden_states)
