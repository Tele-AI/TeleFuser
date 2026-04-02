"""Block Sparse Attention (BSA) for 3D video latents.

Provides flash_attn_bsa_3d: reorder QKV from spatial (T,H,W) layout to
3D-block layout, run block-sparse attention with mean-pooling gating,
then reorder output back to spatial layout.

Backend priority:
  1. tf_kernel / block_sparse_attn library (optimized CUDA kernels)
  2. Pure-PyTorch gating + sparse_sageattn fallback
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange

from telefuser.ops.attention.sparse_sage import sparse_sageattn

# ── Backend detection ────────────────────────────────────────────────────

# Priority 1: tf_kernel or block_sparse_attn library (optional)
_block_sparse_attn_func = None
try:
    from tf_kernel.block_sparse_attn import block_sparse_attn_func as _block_sparse_attn_func
except ImportError:
    try:
        from block_sparse_attn import block_sparse_attn_func as _block_sparse_attn_func
    except ImportError:
        pass


# ── Block reorder (from official LongCat) ────────────────────────────────


def _rearrange_THW_to_3d_block(
    x: torch.Tensor,
    Nt: int,
    Nh: int,
    Nw: int,
    t: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """Reorder from spatial (T,H,W) layout to 3D-block layout.

    Input:  [B, H_heads, Nt*t * Nh*h * Nw*w, D]
    Output: [B, H_heads, (Nt*Nh*Nw) * (t*h*w), D]
    """
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, t, Nh, h, Nw, w, D)
    x = x.permute(0, 1, 2, 4, 6, 3, 5, 7, 8)  # B H Nt Nh Nw t h w D
    return x.contiguous().view(B, H, Nt * Nh * Nw * t * h * w, D)


def _rearrange_3d_block_to_THW(
    x: torch.Tensor,
    Nt: int,
    Nh: int,
    Nw: int,
    t: int,
    h: int,
    w: int,
) -> torch.Tensor:
    """Reorder from 3D-block layout back to spatial (T,H,W) layout.

    Input:  [B, H_heads, (Nt*Nh*Nw) * (t*h*w), D]
    Output: [B, H_heads, Nt*t * Nh*h * Nw*w, D]
    """
    B, H, _, D = x.shape
    x = x.view(B, H, Nt, Nh, Nw, t, h, w, D)
    x = x.permute(0, 1, 2, 5, 3, 6, 4, 7, 8)  # B H Nt t Nh h Nw w D
    return x.contiguous().view(B, H, Nt * t * Nh * h * Nw * w, D)


# ── Gating + attention core ───────────────────────────────────────────

# Cache for small helper tensors to avoid repeated GPU allocations in the hot path.
_tensor_cache: dict[tuple, torch.Tensor] = {}


def _get_cu_seqlens(seqlen: int, device: torch.device) -> torch.Tensor:
    """Return cached cu_seqlens [0, seqlen] tensor for B=1."""
    key = ("cu_seqlens", seqlen, device)
    if key not in _tensor_cache:
        _tensor_cache[key] = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
    return _tensor_cache[key]


def _get_head_mask_type(num_heads: int, device: torch.device) -> torch.Tensor:
    """Return cached head_mask_type [1, 1, ...] tensor."""
    key = ("head_mask_type", num_heads, device)
    if key not in _tensor_cache:
        _tensor_cache[key] = torch.ones(num_heads, device=device, dtype=torch.int32)
    return _tensor_cache[key]


@torch.compile
def _mean_pooling_compression(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Mean-pool along sequence dim to get block-level representations."""
    B, H, S = x.shape[:3]
    num_block = math.ceil(S / block_size)
    if S % block_size != 0:
        x = F.pad(x, (0, 0, 0, num_block * block_size - S))
    return x.view(B, H, num_block, block_size, -1).mean(dim=3)


@torch.compile
def _get_topk_indices(q_cmp: torch.Tensor, k_cmp: torch.Tensor, sparsity: float):
    """Compute block-level gating scores and select top-k block pairs."""
    score = torch.matmul(q_cmp, k_cmp.transpose(-1, -2))
    num_selected = max(1, int((1 - sparsity) * score.shape[-1]))
    block_indices = torch.topk(score, num_selected, dim=-1).indices
    block_indices, _ = torch.sort(block_indices, dim=-1)
    return block_indices, num_selected


def _topk_to_bool_mask(block_indices: torch.Tensor, num_k_blocks: int) -> torch.Tensor:
    """Convert top-k indices to dense boolean mask [B, H, Q_blocks, K_blocks]."""
    B, H, Q, K_sel = block_indices.shape
    mask = torch.zeros(B, H, Q, num_k_blocks, dtype=torch.bool, device=block_indices.device)
    mask.scatter_(3, block_indices, True)
    return mask


def _bsa_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size_q: int,
    chunk_size_k: int,
    sparsity: float,
    num_heads: int,
) -> torch.Tensor:
    """BSA core: mean-pooling gating -> bool mask -> block_sparse_attn or sparse_sageattn.

    Input/output: [B, H, S, D] in block order.
    """
    # Gating: mean-pool Q and K to block-level, then top-k selection
    q_cmp = _mean_pooling_compression(q, chunk_size_q)
    k_cmp = _mean_pooling_compression(k, chunk_size_k)
    block_indices, _ = _get_topk_indices(q_cmp, k_cmp, sparsity)

    S = q.shape[2]
    num_k_blocks = math.ceil(S / chunk_size_k)
    bool_mask = _topk_to_bool_mask(block_indices, num_k_blocks)

    if _block_sparse_attn_func is not None:
        # tf_kernel / block_sparse_attn library
        seqlen = S
        q_flat = rearrange(q, "b h s d -> (b s) h d")
        k_flat = rearrange(k, "b h s d -> (b s) h d")
        v_flat = rearrange(v, "b h s d -> (b s) h d")
        cu_seqlens_q = _get_cu_seqlens(seqlen, q.device)
        cu_seqlens_k = _get_cu_seqlens(seqlen, q.device)
        head_mask = _get_head_mask_type(num_heads, q.device)
        x = _block_sparse_attn_func(
            q_flat,
            k_flat,
            v_flat,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask,
            None,
            bool_mask,
            seqlen,
            seqlen,
            0.0,
            deterministic=False,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        ).unsqueeze(0)
        x = rearrange(x, "b s h d -> b h s d")
        return x

    # Priority 2: sparse_sageattn
    # Kernel expects mask [B, H, ceil(S/128), ceil(S/64)].
    # Our blocks are chunk_size_q tokens; merge adjacent Q-block pairs with OR
    # so each mask row covers chunk_size_q*2 tokens (matching BLOCK_M=128 when chunk_size_q=64).
    Q_blocks = bool_mask.shape[2]
    if Q_blocks % 2 == 0:
        merged = bool_mask.view(bool_mask.shape[0], bool_mask.shape[1], Q_blocks // 2, 2, num_k_blocks)
        bool_mask = merged.any(dim=3)

    x = sparse_sageattn(q, k, v, mask_id=bool_mask.to(torch.int8), is_causal=False, tensor_layout="HND")
    return x


# ── Public API ───────────────────────────────────────────────────────────


def flash_attn_bsa_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    latent_shape_q: tuple[int, int, int],
    latent_shape_k: tuple[int, int, int],
    sparsity: float = 0.9375,
    chunk_3d_shape_q: list[int] | None = None,
    chunk_3d_shape_k: list[int] | None = None,
) -> torch.Tensor:
    """3D block sparse attention for video latents.

    Reorders QKV from spatial (T,H,W) layout to 3D-block layout, runs
    block-sparse attention, then reorders back.

    Args:
        q: Query [B, H, S, D] in spatial order.
        k: Key   [B, H, S, D] in spatial order.
        v: Value [B, H, S, D] in spatial order.
        latent_shape_q: (T, H, W) for queries.
        latent_shape_k: (T, H, W) for keys.
        sparsity: Fraction of blocks to prune (0.9375 = keep 6.25%).
        chunk_3d_shape_q: Block size [t, h, w] for queries.
        chunk_3d_shape_k: Block size [t, h, w] for keys.

    Returns:
        Output [B, H, S, D] in spatial order.
    """
    if chunk_3d_shape_q is None:
        chunk_3d_shape_q = [4, 4, 4]
    if chunk_3d_shape_k is None:
        chunk_3d_shape_k = [4, 4, 4]

    Tq, Hq, Wq = latent_shape_q
    Tk, Hk, Wk = latent_shape_k
    tq, hq, wq = chunk_3d_shape_q
    tk, hk, wk = chunk_3d_shape_k

    Ntq, Nhq, Nwq = Tq // tq, Hq // hq, Wq // wq
    Ntk, Nhk, Nwk = Tk // tk, Hk // hk, Wk // wk

    # Reorder to block layout
    q = _rearrange_THW_to_3d_block(q, Ntq, Nhq, Nwq, tq, hq, wq)
    k = _rearrange_THW_to_3d_block(k, Ntk, Nhk, Nwk, tk, hk, wk)
    v = _rearrange_THW_to_3d_block(v, Ntk, Nhk, Nwk, tk, hk, wk)

    chunk_size_q = tq * hq * wq
    chunk_size_k = tk * hk * wk
    num_heads = q.shape[1]

    x = _bsa_core(q, k, v, chunk_size_q, chunk_size_k, sparsity, num_heads)

    # Reorder back to spatial layout
    x = _rearrange_3d_block_to_THW(x, Ntq, Nhq, Nwq, tq, hq, wq)
    return x
