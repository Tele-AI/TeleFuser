"""Custom operations module for TeleFuser.

Provides activation functions, feed-forward networks, normalization layers,
rotary position embeddings, and attention implementations optimized for video generation.

All operations support torch.compile via automatic dispatch:
- In compile mode: PyTorch native implementations
- In eager mode: Optimized kernels (Triton on CUDA)
"""

from __future__ import annotations

from .base import CustomOp, CustomOpFunction
from .custom_op import TritonKernelWrapper, register_custom_op
from .moe import grouped_expert_forward, route_topk
from .normalization import AdaLayerNormContinuous, LayerNorm, RMSNorm, fused_scale_shift, modulate
from .rotary import apply_rotary_emb

__all__ = [
    # Base classes
    "CustomOp",
    "CustomOpFunction",
    # Custom op registration
    "register_custom_op",
    "TritonKernelWrapper",
    # Normalization
    "RMSNorm",
    "LayerNorm",
    "AdaLayerNormContinuous",
    "fused_scale_shift",
    "modulate",
    "route_topk",
    "grouped_expert_forward",
    # Rotary
    "apply_rotary_emb",
]
