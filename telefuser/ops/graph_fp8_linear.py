"""CUDA-Graph-compatible FP8 Linear operations for fixed-shape H100 inference."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


def _quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size, torch.dtype]:
    input_dtype = x.dtype
    x_bf16 = x if input_dtype == torch.bfloat16 else x.to(torch.bfloat16)
    shape = x_bf16.shape
    x_2d = x_bf16.reshape(-1, shape[-1]).contiguous()
    scale = x_2d.abs().amax(dim=1, keepdim=True).float().clamp_min_(1e-12).div_(_FP8_MAX)
    quantized = (x_2d / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return quantized, scale, shape, input_dtype


class GraphFP8Linear(nn.Module):
    """Inference-only W8A8 Linear backed by PyTorch's fused scaled GEMM."""

    compute_dtype = torch.bfloat16

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if not linear.weight.is_cuda:
            raise ValueError("GraphFP8Linear weights must be materialized on CUDA")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError(
                "GraphFP8Linear requires in_features and out_features divisible by 16, got "
                f"{linear.in_features}x{linear.out_features}"
            )
        weight = linear.weight.detach().to(torch.bfloat16)
        weight_scale = weight.abs().amax(dim=1, keepdim=True).float().clamp_min_(1e-12).div_(_FP8_MAX)
        weight_fp8 = (weight / weight_scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE).contiguous()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer("weight", weight_fp8, persistent=True)
        self.register_buffer("weight_scale", weight_scale.t().contiguous(), persistent=True)
        if linear.bias is None:
            self.register_buffer("bias", None, persistent=True)
        else:
            self.register_buffer("bias", linear.bias.detach().to(torch.bfloat16), persistent=True)

    def _forward_quantized(
        self,
        quantized: torch.Tensor,
        input_scale: torch.Tensor,
        shape: torch.Size,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch._scaled_mm(
            quantized,
            self.weight.t(),
            scale_a=input_scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
        ).reshape(*shape[:-1], self.out_features)
        return output if input_dtype == torch.bfloat16 else output.to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("GraphFP8Linear requires CUDA inputs")
        return self._forward_quantized(*_quantize_activation(x))


def graph_fp8_linear_forward_many(
    linears: tuple[GraphFP8Linear, ...],
    x: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Reuse one per-token activation quantization across compatible FP8 Linears."""
    if not linears:
        return ()
    quantized, input_scale, shape, input_dtype = _quantize_activation(x)
    return tuple(linear._forward_quantized(quantized, input_scale, shape, input_dtype) for linear in linears)


def replace_linear_layers_with_graph_fp8(
    module: nn.Module,
    *,
    module_filter: Callable[[str, nn.Linear], bool],
) -> int:
    """Replace selected CUDA Linear modules and release their BF16 weights."""
    replaced = 0

    def recurse(prefix: str, parent: nn.Module) -> None:
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and module_filter(full_name, child):
                setattr(parent, child_name, GraphFP8Linear(child))
                replaced += 1
            else:
                recurse(full_name, child)

    recurse("", module)
    return replaced


__all__ = ["GraphFP8Linear", "graph_fp8_linear_forward_many", "replace_linear_layers_with_graph_fp8"]
