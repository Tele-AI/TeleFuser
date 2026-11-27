"""Rotary Position Embedding (RoPE) operations.

Provides optimized RoPE implementations with automatic kernel selection
based on platform.

Note: apply_rotary_emb is decorated with @torch.compiler.disable because:
1. The Triton kernel provides better performance than native PyTorch implementation
2. Attention (which follows RoPE in typical execution flow) already uses @torch.compiler.disable
3. Fusion opportunities with Attention are blocked anyway, so preserving Triton performance is optimal

Execution flow: Linear → RMSNorm → RoPE → Attention (disabled)
"""

from __future__ import annotations

import torch

_tf_apply_rope_neox = None
try:
    from tf_kernel import apply_rope_with_cos_sin_cache_inplace as _tf_apply_rope_neox
except ImportError:
    pass


def _apply_rotary_emb_cuda(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """CUDA-optimized implementation using Triton kernel."""
    from telefuser.kernel.triton import apply_rotary_embedding

    # Ensure all tensors are contiguous for Triton kernel
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    return apply_rotary_embedding(x, cos, sin, interleaved=True)


def _apply_rotary_emb_npu(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NPU-optimized implementation using torch_npu.npu_rotary_mul."""
    import torch_npu

    head_size = x.shape[-1]
    batch_size = x.shape[0]

    # Zero-overhead fast path when caller supplies pre-expanded 4D [1 or B, S, 1, D] on target device/dtype.
    if (
        cos.dim() == 4
        and cos.shape[-1] == head_size
        and (cos.shape[0] == 1 or cos.shape[0] == batch_size)
        and cos.device == x.device
        and sin.device == x.device
        and cos.dtype == x.dtype
        and sin.dtype == x.dtype
    ):
        return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")

    # Normalize cos/sin to [B or 1, S, 1, D_cos] for BNSD/BSND broadcasting with x [B, S, H, D].
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(-2)
        sin = sin.unsqueeze(0).unsqueeze(-2)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    elif cos.dim() == 4:
        if cos.shape[0] != batch_size and cos.shape[1] == batch_size:
            cos = cos.swapaxes(0, 1)
            sin = sin.swapaxes(0, 1)
        elif cos.shape[0] != batch_size and cos.shape[0] != 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

    if cos.device != x.device:
        cos = cos.to(x.device)
    if sin.device != x.device:
        sin = sin.to(x.device)
    if cos.dtype != x.dtype:
        cos = cos.to(x.dtype)
    if sin.dtype != x.dtype:
        sin = sin.to(x.dtype)

    if cos.shape[-1] == head_size // 2:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

    return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")


def _apply_rotary_emb_native(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """PyTorch-native implementation for compile compatibility.

    Handles both full head_size and half head_size (interleaved) formats for cos/sin.
    When cos/sin have head_size//2, they are in interleaved format.
    """
    head_size = x.shape[-1]
    batch_size = x.shape[0]

    # Handle interleaved format: cos/sin have head_size//2
    if cos.shape[-1] == head_size // 2:
        # Interleaved format: cos[i] applies to x[2i] and x[2i+1]
        # x: [B, S, H, D], cos: [..., D//2]

        # Normalize cos/sin shape for broadcasting with x [B, S, H, D//2]
        # Target shape: [B, S, 1, D//2] or [1, S, 1, D//2]
        if cos.dim() == 2:
            # [S, D//2] -> [1, S, 1, D//2]
            cos = cos.unsqueeze(0).unsqueeze(-2)
            sin = sin.unsqueeze(0).unsqueeze(-2)
        elif cos.dim() == 3:
            # [S, 1, D//2] -> [1, S, 1, D//2]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        elif cos.dim() == 4:
            # Check if first dim is batch or sequence
            if cos.shape[0] != batch_size and cos.shape[1] == batch_size:
                # Shape is [1, B, 1, D//2] or [S, B, 1, D//2] - unusual, treat as [S, ...]
                cos = cos.swapaxes(0, 1)  # [B, S, 1, D//2]
                sin = sin.swapaxes(0, 1)
            elif cos.shape[0] != batch_size:
                # Assume [S, 1, 1, D//2] -> [1, S, 1, D//2]
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            # else: [B, S, 1, D//2] - correct

        if cos.device != x.device:
            cos = cos.to(x.device)
        if sin.device != x.device:
            sin = sin.to(x.device)

        # Apply interleaved RoPE: split x into pairs
        # x[..., 0::2] and x[..., 1::2] are the pairs
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)  # [..., D//2, 2]
        x0, x1 = x_reshaped.unbind(-1)  # each [..., D//2]

        # Rotate: out0 = x0 * cos - x1 * sin, out1 = x0 * sin + x1 * cos
        out0 = x0.float() * cos.float() - x1.float() * sin.float()
        out1 = x0.float() * sin.float() + x1.float() * cos.float()

        # Interleave back
        out = torch.stack([out0, out1], dim=-1).flatten(-2)  # [..., D]
        return out.to(x.dtype)
    else:
        # Full head_size format (non-interleaved)
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(-2)
            sin = sin.unsqueeze(0).unsqueeze(-2)
        elif cos.dim() == 3:
            cos = cos.unsqueeze(-2)
            sin = sin.unsqueeze(-2)

        if cos.device != x.device:
            cos = cos.to(x.device)
        if sin.device != x.device:
            sin = sin.to(x.device)

        # Rotate: x_real = x[..., 0::2], x_imag = x[..., 1::2]
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)

        return (x.float() * cos.float() + x_rotated.float() * sin.float()).to(x.dtype)


def _apply_rotary_emb_neox_native(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos_sin_cache.shape[-1]
    half = rotary_dim // 2
    cosine = cos_sin_cache[..., :half].unsqueeze(-2).to(query.dtype)
    sine = cos_sin_cache[..., half:].unsqueeze(-2).to(query.dtype)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        rotary, passthrough = value[..., :rotary_dim], value[..., rotary_dim:]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)
        return torch.cat((rotated, passthrough), dim=-1)

    return rotate(query), rotate(key)


def _apply_rotary_emb_neox_npu(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    inplace_normalized: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial NeoX half-split RoPE on Ascend NPU using torch_npu.npu_rotary_mul."""
    import torch_npu

    rotary_dim = cos_sin_cache.shape[-1]
    half = rotary_dim // 2
    cos_sin = cos_sin_cache.to(device=query.device, dtype=query.dtype)
    cos = cos_sin[..., :half].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
    sin = cos_sin[..., half:].unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        rotary = value[..., :rotary_dim].unsqueeze(0).contiguous()
        rotated = torch_npu.npu_rotary_mul(rotary, cos, sin).squeeze(0)
        if rotary_dim == value.shape[-1]:
            return rotated
        if inplace_normalized:
            value[..., :rotary_dim] = rotated
            return value
        return torch.cat((rotated, value[..., rotary_dim:]), dim=-1)

    return rotate(query), rotate(key)


def apply_rotary_emb_neox(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply partial NeoX RoPE to Q and K with optional tf-kernel or Ascend NPU dispatch."""
    if query.shape != key.shape or query.ndim != 3:
        raise ValueError("NeoX RoPE requires matching query and key tensors with shape [sequence, heads, dim]")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[0] != query.shape[0]:
        raise ValueError("NeoX RoPE cache must have shape [sequence, rotary_dim]")
    rotary_dim = cos_sin_cache.shape[-1]
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > query.shape[-1]:
        raise ValueError("NeoX RoPE rotary_dim must be positive, even, and no larger than the head dimension")

    if not torch.compiler.is_compiling() and query.device.type == "cuda" and _tf_apply_rope_neox is not None:
        query = query.contiguous()
        key = key.contiguous()
        positions = torch.arange(query.shape[0], dtype=torch.long, device=query.device)
        _tf_apply_rope_neox(
            positions,
            query,
            key,
            head_size=query.shape[-1],
            cos_sin_cache=cos_sin_cache.float().contiguous(),
            is_neox=True,
        )
        return query, key

    if (
        not torch.compiler.is_compiling()
        and query.device.type == "npu"
        and query.dtype == key.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
    ):
        return _apply_rotary_emb_neox_npu(query, key, cos_sin_cache)

    return _apply_rotary_emb_neox_native(query, key, cos_sin_cache)


def apply_qk_norm_rope_neox(
    query: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K RMSNorm and partial NeoX RoPE with fused eager CUDA or Ascend NPU dispatch."""
    use_fused_cuda = (
        not torch.compiler.is_compiling()
        and query.device.type == "cuda"
        and query.dtype == key.dtype == q_weight.dtype == k_weight.dtype == torch.bfloat16
        and query.shape == key.shape
        and query.ndim == 3
        and q_weight.shape == k_weight.shape == (query.shape[-1],)
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[0] == query.shape[0]
        and 0 < cos_sin_cache.shape[-1] <= query.shape[-1]
        and cos_sin_cache.shape[-1] % 2 == 0
        and query.stride(-1) == key.stride(-1) == 1
        and query.stride(-2) == key.stride(-2) == query.shape[-1]
    )
    if use_fused_cuda:
        from telefuser.kernel.triton import qknorm_rope_neox_bf16_

        return qknorm_rope_neox_bf16_(
            query,
            key,
            q_weight.contiguous(),
            k_weight.contiguous(),
            cos_sin_cache.contiguous(),
            eps,
        )

    use_fused_npu = (
        not torch.compiler.is_compiling()
        and query.device.type == "npu"
        and query.dtype == key.dtype == q_weight.dtype == k_weight.dtype
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape == key.shape
        and query.ndim == 3
        and q_weight.shape == k_weight.shape == (query.shape[-1],)
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[0] == query.shape[0]
        and 0 < cos_sin_cache.shape[-1] <= query.shape[-1]
        and cos_sin_cache.shape[-1] % 2 == 0
    )
    if use_fused_npu:
        import torch_npu

        q_norm = torch_npu.npu_rms_norm(query.contiguous(), q_weight.contiguous(), eps)[0]
        k_norm = torch_npu.npu_rms_norm(key.contiguous(), k_weight.contiguous(), eps)[0]
        return _apply_rotary_emb_neox_npu(q_norm, k_norm, cos_sin_cache, inplace_normalized=True)

    def normalize(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = value.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (value * torch.rsqrt(variance + eps)).to(weight.dtype) * weight

    return _apply_rotary_emb_neox_native(
        normalize(query, q_weight),
        normalize(key, k_weight),
        cos_sin_cache,
    )


@torch.compiler.disable
def apply_rotary_emb(
    x: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
    sequence_dim: int = 1,
) -> torch.Tensor:
    """Apply rotary positional embeddings to input tensor.

    Uses Triton kernel on CUDA for optimal performance. Falls back to
    PyTorch native implementation on other platforms.

    Note: This function is disabled for torch.compile to preserve Triton
    kernel performance, since the subsequent Attention operation also
    uses @torch.compiler.disable and blocks fusion opportunities anyway.

    Args:
        x: Input tensor of shape (B, S, H, D).
        freqs: Tuple of (cos, sin) tensors.
            For interleaved format: (S, D//2) or (S, 1, 1, D//2)
            For full format: (S, D) or (S, 1, D)
        sequence_dim: Dimension for sequence (default: 1).

    Returns:
        Tensor with rotary embeddings applied.
    """
    cos, sin = freqs

    # Use Triton kernel on CUDA and fused rotary op on Ascend NPU
    if x.device.type == "cuda":
        return _apply_rotary_emb_cuda(x, cos, sin)
    if x.device.type == "npu":
        return _apply_rotary_emb_npu(x, cos, sin)

    return _apply_rotary_emb_native(x, cos, sin)


__all__ = ["apply_qk_norm_rope_neox", "apply_rotary_emb", "apply_rotary_emb_neox"]
