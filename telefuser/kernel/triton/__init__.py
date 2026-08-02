"""Triton kernels for TeleFuser.

This module provides optimized Triton kernels for:
- Normalization: LayerNorm, RMSNorm, fused add + RMSNorm, tiled RMSNorm
- Position Encoding: Rotary Position Embedding (RoPE)
- Element-wise Operations: Fused scale and shift
- Quantization: FP8 per-token quantization
- Attention: Merge attention states for Ring Attention

Note: All functions in this module require triton to be installed.
"""

from .merge_attn_states import fused_merge_attn_states
from .norm import (
    fused_add_rms_norm,
    layer_norm_fn,
    norm_infer,
    triton_one_pass_rms_norm,
)
from .quant import per_token_dequant_fp8, per_token_quant_fp8
from .rotary import apply_rotary_embedding
from .scale_shift import (
    fused_add_layernorm_scale_shift,
    fused_layernorm_scale_shift,
    fused_layernorm_scale_shift_gate_select01,
    fused_residual_layernorm_scale_shift_gate_select01,
    fused_scale_shift,
    fused_scale_shift_gate_select,
)

__all__ = [
    "layer_norm_fn",
    "norm_infer",
    "triton_one_pass_rms_norm",
    "fused_add_rms_norm",
    "apply_rotary_embedding",
    "apply_rotary_embedding_inplace",
    "fused_scale_shift",
    "fused_layernorm_scale_shift",
    "fused_add_layernorm_scale_shift",
    "fused_scale_shift_gate_select",
    "fused_layernorm_scale_shift_gate_select01",
    "fused_residual_layernorm_scale_shift_gate_select01",
    "fused_merge_attn_states",
    "per_token_quant_fp8",
    "per_token_dequant_fp8",
]
