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


def _apply_rotary_emb_cuda(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """CUDA-optimized implementation using Triton kernel."""
    from telefuser.kernel.triton import apply_rotary_embedding

    # Ensure all tensors are contiguous for Triton kernel
    x = x.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    return apply_rotary_embedding(x, cos, sin, interleaved=True)


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

    # Use Triton kernel on CUDA for optimal performance
    if x.device.type == "cuda":
        return _apply_rotary_emb_cuda(x, cos, sin)

    return _apply_rotary_emb_native(x, cos, sin)


__all__ = ["apply_rotary_emb"]
