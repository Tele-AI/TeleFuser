"""Unified attention implementation with dense and sparse support.

Supports multiple attention implementations:
- Dense: TORCH_SDPA, TORCH_CUDNN, FLASH_ATTN_2/3/4, SAGE_ATTN variants, SPARGE_ATTN
- Sparse: RADIAL_ATTN, LOCAL_SPARSE_ATTN

Note: Attention functions are decorated with @torch.compiler.disable because:
1. SageAttention requires static tensor shapes for quantization scales
2. Custom CUDA kernels (FlashAttention, etc.) are not compile-friendly
3. Keeping attention in eager mode preserves optimal performance

Example:
    >>> from telefuser.core.config import AttentionConfig, AttnImplType
    >>> config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
    >>> output = attention(q, k, v, attention_config=config)
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from telefuser.core.config import AttentionConfig, AttnImplType, SparseAttentionConfig
from telefuser.distributed.device_mesh import get_attention_strategy, get_ring_group, get_ulysses_group
from telefuser.distributed.ring import ring_attention_forward
from telefuser.distributed.ulysses_comm import ulysses_gather_heads, ulysses_scatter_heads
from telefuser.ops.attention.backends import (
    FLASH_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_4_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    SDPA_AVAILABLE,
    flash_attn2,
    flash_attn3,
    flash_attn4,
    get_lse_fallback_impl,
    sageattention,
    sdpa_attn_cudnn,
    sparge_attn,
    supports_return_lse,
)
from telefuser.ops.attention.sparse_patterns import MaskMap, radial_attention
from telefuser.utils.logging import logger

_warned_attn_fallback: set[str] = set()


def _packed_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sequence_lengths: list[int],
    *,
    scale: float | None,
    is_causal: bool,
) -> Tensor:
    """Run SDPA independently for sequences packed along the BNSD sequence axis."""
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("packed SDPA requires matching Q/K/V tensors with shape [1, heads, sequence, dim]")
    if any(length <= 0 for length in sequence_lengths) or sum(sequence_lengths) != q.shape[2]:
        raise ValueError("packed SDPA sequence lengths must be positive and sum to the packed sequence dimension")
    outputs = [
        F.scaled_dot_product_attention(q_part, k_part, v_part, scale=scale, is_causal=is_causal)
        for q_part, k_part, v_part in zip(
            torch.split(q, sequence_lengths, dim=2),
            torch.split(k, sequence_lengths, dim=2),
            torch.split(v, sequence_lengths, dim=2),
        )
    ]
    return torch.cat(outputs, dim=2)


class SparseAttentionState:
    """Runtime state for sparse attention computation.

    Tracks current timestep and layer to determine whether to use
    dense or sparse attention patterns.
    """

    def __init__(
        self,
        config: SparseAttentionConfig,
        mask_map: MaskMap,
        model_type: str = "wan",
    ) -> None:
        self.config = config
        self.mask_map = mask_map
        self.model_type = model_type
        self.numeral_timestep = 0
        self.layer_idx = 0

    def update(self, numeral_timestep: int | None = None, layer_idx: int | None = None) -> None:
        """Update runtime state."""
        if numeral_timestep is not None:
            self.numeral_timestep = numeral_timestep
        if layer_idx is not None:
            self.layer_idx = layer_idx

    def should_use_dense(self) -> bool:
        """Check if dense attention should be used for current state."""
        return self.config.should_use_dense(self.numeral_timestep, self.layer_idx)

    def get_sparsity_type(self) -> str:
        """Get sparsity type for current state."""
        return "dense" if self.should_use_dense() else self.config.sparse_impl


@torch.compiler.disable
def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_config: AttentionConfig | None = None,
    sparse_state: SparseAttentionState | None = None,
    attn_impl: AttnImplType | None = None,
    attn_mask: Tensor | None = None,
    scale: float | None = None,
    input_layout: Literal["BSND", "BNSD"] = "BSND",
    output_layout: Literal["BSND", "BNSD"] = "BSND",
    is_causal: bool = False,
    return_lse: bool = False,
    sequence_lengths: list[int] | None = None,
    **kwargs: Any,
) -> Tensor | tuple[Tensor, Tensor]:
    """Unified attention function.

    Args:
        q: Query tensor [B, S, N, D] or [B, N, S, D].
        k: Key tensor [B, S, N, D] or [B, N, S, D].
        v: Value tensor [B, S, N, D] or [B, N, S, D].
        attention_config: Unified attention configuration.
        sparse_state: Runtime state for sparse attention.
        attn_impl: Attention implementation (if config not provided).
        attn_mask: Attention mask.
        scale: Attention scale factor.
        input_layout: Input tensor layout.
        output_layout: Output tensor layout.
        is_causal: Use causal masking.
        return_lse: Return log-sum-exp values.
        **kwargs: Implementation-specific arguments.

    Returns:
        Attention output, or (output, lse) if return_lse=True.
    """
    # Build config from parameters if not provided
    if attention_config is None:
        attn_impl = attn_impl or AttnImplType.TORCH_SDPA
        attention_config = AttentionConfig(
            attn_impl=attn_impl,
            scale=scale,
            is_causal=is_causal,
        )

    attn_impl = attention_config.attn_impl
    scale = scale if scale is not None else attention_config.scale
    is_causal = is_causal or attention_config.is_causal

    # Handle sparse attention
    if attn_impl in (AttnImplType.RADIAL_ATTN, AttnImplType.LOCAL_SPARSE_ATTN):
        if sparse_state is None:
            msg = "Sparse attention requires sparse_state, falling back to FLASH_ATTN_2"
            if msg not in _warned_attn_fallback:
                _warned_attn_fallback.add(msg)
                logger.warning(msg)
            attn_impl = AttnImplType.FLASH_ATTN_2
        else:
            return radial_attention(
                query=q,
                key=k,
                value=v,
                mask_map=sparse_state.mask_map,
                sparsity_type=sparse_state.get_sparsity_type(),
                block_size=sparse_state.config.block_size,
                decay_factor=sparse_state.config.decay_factor,
                model_type=sparse_state.model_type,
                pre_defined_mask=None,
                use_sage_attention=sparse_state.config.use_sage_attention,
            )

    # Dense attention: handle layout conversion
    # - Flash Attention expects BSND (NHD) layout
    # - PyTorch SDPA variants expect BNSD (HND) layout
    # - SageAttention can accept both via tensor_layout parameter
    BNSD_IMPLS = {AttnImplType.TORCH_CUDNN, AttnImplType.TORCH_SDPA, AttnImplType.SPARGE_ATTN}

    # Track current layout after potential conversions
    current_layout = input_layout

    if input_layout == "BSND" and attn_impl in BNSD_IMPLS:
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        current_layout = "BNSD"
    elif (
        input_layout == "BNSD"
        and attn_impl not in BNSD_IMPLS
        and attn_impl not in (AttnImplType.RADIAL_ATTN, AttnImplType.LOCAL_SPARSE_ATTN)
    ):
        # Flash Attention and SageAttention need BSND layout, so convert if input is BNSD
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        current_layout = "BSND"

    output: Tensor | None = None
    lse: Tensor | None = None

    # Flash Attention implementations
    if attn_impl == AttnImplType.FLASH_ATTN_4 and FLASH_ATTN_4_AVAILABLE and flash_attn4 is not None:
        result = flash_attn4(q, k, v, softmax_scale=scale, return_softmax_lse=return_lse, **kwargs)
        if return_lse:
            output, lse = result
        else:
            output = result
    elif attn_impl == AttnImplType.FLASH_ATTN_3 and FLASH_ATTN_3_AVAILABLE and flash_attn3 is not None:
        result = flash_attn3(q, k, v, softmax_scale=scale, return_softmax_lse=return_lse, **kwargs)
        if return_lse:
            output, lse = result
        else:
            output = result
    elif attn_impl == AttnImplType.FLASH_ATTN_2 and FLASH_ATTN_2_AVAILABLE and flash_attn2 is not None:
        result = flash_attn2(q, k, v, softmax_scale=scale, return_attn_probs=return_lse, **kwargs)
        if return_lse:
            output, lse, _ = result
        else:
            output = result

    # PyTorch SDPA
    elif attn_impl == AttnImplType.TORCH_CUDNN and SDPA_AVAILABLE:
        output = sdpa_attn_cudnn(q, k, v, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
    elif attn_impl == AttnImplType.TORCH_SDPA and SDPA_AVAILABLE:
        if sequence_lengths is None:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
        else:
            if attn_mask is not None or current_layout != "BNSD":
                raise ValueError("packed SDPA requires BNSD layout without an attention mask")
            output = _packed_sdpa(q, k, v, sequence_lengths, scale=scale, is_causal=is_causal)

    # Sage Attention variants
    elif SAGE_ATTN_AVAILABLE and sageattention is not None:
        # SageAttention tensor_layout: "NHD" for BSND, "HND" for BNSD
        sage_tensor_layout = "NHD" if current_layout == "BSND" else "HND"
        if attn_impl == AttnImplType.SAGE_ATTN_2_8_8:
            result = sageattention.sageattn_qk_int8_pv_fp8_cuda(
                q,
                k,
                v,
                attn_mask=attn_mask,
                sm_scale=scale,
                tensor_layout=sage_tensor_layout,
                pv_accum_dtype="fp32+fp32",
                return_lse=return_lse,
            )
            output, lse = result if return_lse else (result, None)
        elif attn_impl == AttnImplType.SAGE_ATTN_2_8_16:
            result = sageattention.sageattn_qk_int8_pv_fp16_cuda(
                q,
                k,
                v,
                attn_mask=attn_mask,
                sm_scale=scale,
                tensor_layout=sage_tensor_layout,
                pv_accum_dtype="fp32",
                return_lse=return_lse,
            )
            output, lse = result if return_lse else (result, None)
        elif attn_impl == AttnImplType.SAGE_ATTN_2_8_8_SM90:
            result = sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90(
                q,
                k,
                v,
                attn_mask=attn_mask,
                sm_scale=scale,
                tensor_layout=sage_tensor_layout,
                pv_accum_dtype="fp32+fp32",
                return_lse=return_lse,
            )
            output, lse = result if return_lse else (result, None)

    # Sparge Attention
    elif attn_impl == AttnImplType.SPARGE_ATTN:
        output = sparge_attn(q, k, v, attn_mask=attn_mask, scale=scale)

    # Fallback to SDPA
    if output is None:
        msg = f"Attention {attn_impl} not available, falling back to TORCH_SDPA"
        if msg not in _warned_attn_fallback:
            _warned_attn_fallback.add(msg)
            logger.warning(msg)
        if current_layout == "BSND":
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            current_layout = "BNSD"

        if sequence_lengths is None:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale, is_causal=is_causal)
        else:
            if attn_mask is not None or current_layout != "BNSD":
                raise ValueError("packed SDPA requires BNSD layout without an attention mask")
            output = _packed_sdpa(q, k, v, sequence_lengths, scale=scale, is_causal=is_causal)
        attn_impl = AttnImplType.TORCH_SDPA

    # Handle output layout conversion - output matches current_layout, may need to convert to output_layout
    if current_layout != output_layout:
        output = output.transpose(1, 2).contiguous()

    if return_lse:
        return output, lse
    return output


@torch.compiler.disable
def long_context_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_config: AttentionConfig | None = None,
    sparse_state: SparseAttentionState | None = None,
    attn_mask: Tensor | None = None,
    scale: float | None = None,
    input_layout: Literal["BSND", "BNSD"] = "BSND",
    output_layout: Literal["BSND", "BNSD"] = "BSND",
    device_mesh: torch.distributed.device_mesh.DeviceMesh | None = None,
    is_causal: bool = False,
) -> Tensor:
    """Distributed long-context attention with sequence parallelism.

    Supports three strategies:
    - Ulysses: All-to-All communication across heads.
    - Ring: P2P KV rotation with online softmax merging.
    - USP: Combined Ulysses + Ring for large-scale training.
    """
    if device_mesh is None:
        raise RuntimeError("Device mesh required for long context attention")

    strategy = get_attention_strategy(device_mesh)

    # Convert to BSND layout for internal processing
    if input_layout == "BNSD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

    if strategy == "local":
        output = attention(
            q,
            k,
            v,
            attention_config=attention_config,
            sparse_state=sparse_state,
            input_layout="BSND",
            output_layout="BSND",
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
        )
    elif strategy == "ulysses":
        output = _ulysses_attention(q, k, v, attention_config, sparse_state, attn_mask, scale, device_mesh, is_causal)
    elif strategy == "ring":
        output = _ring_attention(q, k, v, attention_config, sparse_state, attn_mask, scale, device_mesh, is_causal)
    elif strategy == "usp":
        output = _usp_attention(q, k, v, attention_config, sparse_state, attn_mask, scale, device_mesh, is_causal)
    else:
        raise RuntimeError(f"Unknown attention strategy: {strategy}")

    if output_layout == "BNSD":
        output = output.transpose(1, 2).contiguous()

    return output


def _ulysses_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_config: AttentionConfig | None,
    sparse_state: SparseAttentionState | None,
    attn_mask: Tensor | None,
    scale: float | None,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
    is_causal: bool,
) -> Tensor:
    """Ulysses attention: All-to-All on sequence dimension."""
    group = get_ulysses_group(device_mesh)
    if group is None:
        raise RuntimeError("Ulysses group not found in device mesh")

    num_heads = q.shape[2]

    # All-to-All QKV
    v_wait = ulysses_scatter_heads(v, group)
    k_wait = ulysses_scatter_heads(k, group)
    q_wait = ulysses_scatter_heads(q, group)

    q, k, v = q_wait(), k_wait(), v_wait()

    # Local attention
    output = attention(
        q,
        k,
        v,
        attention_config=attention_config,
        sparse_state=sparse_state,
        input_layout="BSND",
        output_layout="BSND",
        attn_mask=attn_mask,
        scale=scale,
        is_causal=is_causal,
    )

    # All-to-All O
    out_wait = ulysses_gather_heads(output, group, num_heads=num_heads)
    return out_wait()


def _get_ring_attn_config(
    attention_config: AttentionConfig | None, scale: float | None, is_causal: bool
) -> AttentionConfig:
    """Get attention config with LSE support for ring attention."""
    attn_impl = attention_config.attn_impl if attention_config else AttnImplType.TORCH_SDPA

    if not supports_return_lse(attn_impl.value):
        fallback = get_lse_fallback_impl()
        if fallback is None:
            raise RuntimeError("Ring attention requires LSE support. Install flash-attn or sageattention.")
        logger.info(f"Ring attention: falling back to {fallback}")
        return AttentionConfig(attn_impl=AttnImplType(fallback), scale=scale, is_causal=is_causal)

    return attention_config


def _ring_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_config: AttentionConfig | None,
    sparse_state: SparseAttentionState | None,
    attn_mask: Tensor | None,
    scale: float | None,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
    is_causal: bool,
) -> Tensor:
    """Ring attention: P2P KV rotation with online softmax."""
    group = get_ring_group(device_mesh)
    if group is None:
        raise RuntimeError("Ring group not found in device mesh")

    ring_attn_config = _get_ring_attn_config(attention_config, scale, is_causal)

    def attention_fn(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float | None = None,
        is_causal: bool = False,
        return_lse: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return attention(
            query,
            key,
            value,
            attention_config=ring_attn_config,
            sparse_state=sparse_state,
            input_layout="BSND",
            output_layout="BSND",
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
            return_lse=return_lse,
        )

    output, _ = ring_attention_forward(
        query=q,
        key=k,
        value=v,
        attention_fn=attention_fn,
        process_group=group,
        scale=scale,
        is_causal=is_causal,
        return_lse=True,
    )
    return output


def _usp_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention_config: AttentionConfig | None,
    sparse_state: SparseAttentionState | None,
    attn_mask: Tensor | None,
    scale: float | None,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
    is_causal: bool,
) -> Tensor:
    """USP (Ulysses + Ring) attention for large-scale distributed training."""
    ulysses_group = get_ulysses_group(device_mesh)
    ring_group = get_ring_group(device_mesh)

    if ulysses_group is None:
        raise RuntimeError("Ulysses group not found for USP")
    if ring_group is None:
        raise RuntimeError("Ring group not found for USP")

    num_heads = q.shape[2]

    # Step 1: Ulysses All-to-All QKV
    v_wait = ulysses_scatter_heads(v, ulysses_group)
    k_wait = ulysses_scatter_heads(k, ulysses_group)
    q_wait = ulysses_scatter_heads(q, ulysses_group)

    q, k, v = q_wait(), k_wait(), v_wait()

    # Step 2: Ring attention
    ring_attn_config = _get_ring_attn_config(attention_config, scale, is_causal)

    def attention_fn(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float | None = None,
        is_causal: bool = False,
        return_lse: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return attention(
            query,
            key,
            value,
            attention_config=ring_attn_config,
            sparse_state=sparse_state,
            input_layout="BSND",
            output_layout="BSND",
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
            return_lse=return_lse,
        )

    output, _ = ring_attention_forward(
        query=q,
        key=k,
        value=v,
        attention_fn=attention_fn,
        process_group=ring_group,
        scale=scale,
        is_causal=is_causal,
        return_lse=True,
    )

    # Step 3: Ulysses All-to-All O
    out_wait = ulysses_gather_heads(output, ulysses_group, num_heads=num_heads)
    return out_wait()


__all__ = [
    "attention",
    "long_context_attention",
    "SparseAttentionState",
    "AttentionConfig",
    "AttnImplType",
]
