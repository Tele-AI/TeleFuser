"""Attention and mask helpers for LingBot-VLA v2.

Adapted from the Apache-2.0 licensed LingBot-VLA v2 implementation.
"""

from __future__ import annotations

import math
from typing import Any

import einops
import torch
import torch.nn.functional as F
from packaging.version import Version
from torch import Tensor


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def prefix_query_segments(
    use_depth_align: bool,
    use_future_depth: bool,
    use_future_video: bool = False,
    use_future_video_cls: bool = False,
    use_future_video_patch: bool = True,
    future_video_share_future_depth_query: bool = False,
) -> tuple[str, ...]:
    """Return prefix segment order after the image block.

    Task-specific query tokens are always placed after language tokens. Current
    task queries precede future task queries; future-depth remains the last
    query segment so the existing suffix-to-future-depth blocking can keep using
    the tail span.
    """
    segments = ["language"]
    if not use_depth_align:
        return tuple(segments)

    segments.append("current_depth")
    if use_future_video:
        if use_future_video_cls:
            segments.append("future_video_cls")
        if use_future_video_patch and not future_video_share_future_depth_query:
            segments.append("future_video")
    if use_future_depth:
        segments.append("future_depth")
    return tuple(segments)


def prefix_query_token_spans(
    prefix_len: int,
    num_task_tokens: int,
    use_depth_align: bool,
    use_future_depth: bool,
    use_future_video: bool = False,
    use_future_video_cls: bool = False,
    use_future_video_patch: bool = True,
    future_video_share_future_depth_query: bool = False,
) -> dict[str, tuple[int, int]]:
    """Return [start, end) spans for non-language task query segments."""
    counts = {
        "current_depth": num_task_tokens,
        "future_video_cls": 1,
        "future_video": num_task_tokens,
        "future_depth": num_task_tokens,
    }
    ordered = prefix_query_segments(
        use_depth_align=use_depth_align,
        use_future_depth=use_future_depth,
        use_future_video=use_future_video,
        use_future_video_cls=use_future_video_cls,
        use_future_video_patch=use_future_video_patch,
        future_video_share_future_depth_query=future_video_share_future_depth_query,
    )
    query_segments = [name for name in ordered if name != "language"]
    cursor = prefix_len - sum(counts[name] for name in query_segments)
    spans = {}
    for name in query_segments:
        count = counts[name]
        spans[name] = (cursor, cursor + count)
        cursor += count
    return spans


def fv_col_span(prefix_len: int, num_task_tokens: int, use_cls: bool, use_patch: bool) -> tuple[int, int]:
    """Return [start, end) of a tail query block inside the prefix.

    This legacy helper is still used for future-depth tail blocking in V2.
    New prefix layout code should prefer prefix_query_token_spans(), which also
    handles current-depth and separate future-video spans.
    """
    fv_len = (1 if use_cls else 0) + (num_task_tokens if use_patch else 0)
    return prefix_len - fv_len, prefix_len


def block_suffix_to_fv_(
    att_2d_masks: torch.Tensor,
    suffix_row_start: int,
    prefix_len: int,
    num_task_tokens: int,
    use_cls: bool = False,
    use_patch: bool = True,
    drop_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-place mask out the suffix-to-future-video attention edge.

    `make_att_2d_masks`' cumsum scheme cannot express "a query cannot see a
    segment that precedes it", so we zero the rectangular [suffix rows, FV cols]
    block on the already-built 2D mask instead of touching mask_ar.

    att_2d_masks: bool[B, Q, K], True == visible. `suffix_row_start` is the first
    query row belonging to the suffix: prefix_len in the square training mask,
    0 in the suffix-only inference mask. Leaves FV -> img/lang rows untouched so
    the distillation query still reads the current observation.

    `drop_mask`: optional bool[B], True where this sample's suffix must NOT see
    FV. None == block every sample (hard mask). Used for per-sample stochastic
    masking (FV-attention dropout): keep = visible iff not dropped, applied via
    broadcast multiply so it stays a static graph under torch.compile.
    """
    fv_start, fv_end = fv_col_span(prefix_len, num_task_tokens, use_cls, use_patch)
    if fv_end <= fv_start:
        return att_2d_masks
    if drop_mask is None:
        att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = False
    else:
        # keep[b] = True where the sample is NOT dropped -> AND keeps those rows
        # visible and zeros the dropped ones, with no data-dependent indexing.
        keep = (~drop_mask).view(-1, 1, 1)
        block = att_2d_masks[:, suffix_row_start:, fv_start:fv_end]
        att_2d_masks[:, suffix_row_start:, fv_start:fv_end] = block & keep
    return att_2d_masks


def our_eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Performs eager attention, optimized with torch.einsum.

    Args:
        query_states: Query tensor of shape [batch_size, seq_len, num_attention_heads, head_dim].
        key_states: Key tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        value_states: Value tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        attention_mask: Attention mask tensor, typically
            [batch_size, 1, seq_len, seq_len] or [batch_size, seq_len, seq_len].

    Returns:
        Output tensor of shape [batch_size, seq_len, num_attention_heads * head_dim].
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[2]
    num_key_value_groups = num_att_heads // num_key_value_heads

    key_states = einops.repeat(key_states, "b l h d -> b l (h g) d", g=num_key_value_groups)
    value_states = einops.repeat(value_states, "b l h d -> b l (h g) d", g=num_key_value_groups)

    query_states_permuted = torch.einsum("blhd->bhld", query_states)
    key_states_permuted = torch.einsum("blhd->bhld", key_states)

    att_weights = torch.einsum("bhqd,bhkd->bhqk", query_states_permuted, key_states_permuted)
    att_weights *= head_dim**-0.5

    big_neg = -2.3819763e38
    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

    probs = F.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    value_states_permuted = torch.einsum("blhd->bhld", value_states)  # [B, H, L_v, D]
    att_output = torch.einsum("bhqk,bhkv->bhqv", probs, value_states_permuted)  # [B, H, L_q, D]
    att_output = torch.einsum("bhld->blhd", att_output)  # [B, L, H, D]
    att_output = att_output.reshape(bsize, seq_len, num_att_heads * head_dim)

    return att_output


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: float = 10_000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    original_dtype = x.dtype  # bf16
    d = x.shape[-1]
    d_half = d // 2
    device = x.device

    # Cast input to compute_dtype for all internal operations
    x_casted = x.to(dtype)
    positions_casted = positions.to(dtype)

    freq_exponents = (2.0 / d) * torch.arange(d_half, dtype=dtype, device=device)
    timescale = max_wavelength**freq_exponents
    radians = torch.einsum("bl,h->blh", positions_casted, 1.0 / timescale)  # fp32 -> bf16

    radians = radians[..., None, :]  # [B, L, 1, D_half]

    sin = torch.sin(radians)  # bf16
    cos = torch.cos(radians)  # bf16

    x1, x2 = x_casted.split(d_half, dim=-1)  # fp32

    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)  # fp32

    return res.to(original_dtype)  # bf16


# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FLEX_SPARSE_BLOCK_SIZE = 128
FLEX_KERNEL_OPTIONS = {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}

if Version(torch.__version__) >= Version("2.5.0"):
    # Flex attention is available from torch 2.5 onwards.
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
        flex_attention,
    )


def flex_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float | None = None,
) -> torch.Tensor:
    """
    This is defined out of classes to make compile happy.
    """
    batch_size, seq_len, num_att_heads, head_dim = query_states.shape
    original_dtype = query_states.dtype
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)

    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, None, :, : key_states.shape[2]]

        if causal_mask.shape[1] == 1 and query_states.shape[1] > 1:
            causal_mask = causal_mask.expand(-1, query_states.shape[1], -1, -1)

    def precomputed_mask_factory(precomputed_mask: torch.Tensor) -> _mask_mod_signature:
        def mask_mod(b, h, q_idx, kv_idx):
            # Danger zone: if b,h,q_idx,kv_idx exceed the shape, device-side assert occurs.
            return precomputed_mask[b][h][q_idx][kv_idx]

        return mask_mod

    b_mask, h_mask, q_len, kv_len = causal_mask.shape  # The shape of your mask
    block_size = FLEX_SPARSE_BLOCK_SIZE
    q_len_rounded = _round_up_to_multiple(q_len, block_size)
    kv_len_rounded = _round_up_to_multiple(kv_len, block_size)

    # *CRITICAL* we do need to expand here, else we get a CUDA index error

    pad_q = q_len_rounded - q_len
    pad_k = kv_len_rounded - kv_len

    if pad_q > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_q), value=0.0)  # [B, H, q_len_rounded, D]
    if pad_k > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_k), value=0.0)
        value_states = F.pad(value_states, (0, 0, 0, pad_k), value=0.0)
    padded_causal_mask = F.pad(causal_mask, (0, pad_k, 0, pad_q), value=0.0)
    mask_mod_fn_orig = precomputed_mask_factory(padded_causal_mask)

    mask_4d = create_mask(
        mod_fn=mask_mod_fn_orig,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        device=causal_mask.device,
    )

    mask_mod_fn_padded = precomputed_mask_factory(mask_4d)
    block_mask = create_block_mask(
        mask_mod=mask_mod_fn_padded,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=block_size,
        device=causal_mask.device,
        _compile=False,
    )

    #  mask is applied inside the kernel, ideally more efficiently than score_mod.
    attn_output, attention_weights = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=True,  # because we shaped query/key states for GQA
        scale=head_dim**-0.5 if scaling is None else scaling,
        return_lse=True,
        kernel_options=FLEX_KERNEL_OPTIONS,
    )
    attn_output = attn_output[:, :, :seq_len, :].to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q_LEN, H, head_dim]
    attn_output = attn_output.reshape(
        batch_size,
        -1,
        attn_output.shape[2] * attn_output.shape[3],  # merges [H, head_dim]
    )
    return attn_output


@torch.compiler.disable
def build_block_mask(
    attention_mask_3d: torch.Tensor,
    num_heads: int,
    q_len: int,
    kv_len: int,
    block_size: int = FLEX_SPARSE_BLOCK_SIZE,
) -> Any:
    """
    Build a reusable BlockMask from a 3D attention mask [B, Q, KV].
    This allocates the dense 4D mask once; the returned BlockMask can be reused across layers.
    """
    causal_mask = attention_mask_3d[:, None, :, :].expand(-1, num_heads, -1, -1).contiguous()
    b_mask, h_mask = causal_mask.shape[0], causal_mask.shape[1]

    q_len_rounded = _round_up_to_multiple(q_len, block_size)
    kv_len_rounded = _round_up_to_multiple(kv_len, block_size)

    pad_q = q_len_rounded - q_len
    pad_k = kv_len_rounded - kv_len
    padded_mask = F.pad(causal_mask, (0, pad_k, 0, pad_q), value=0.0)

    def precomputed_mask_factory(precomputed_mask: torch.Tensor):
        def mask_mod(b, h, q_idx, kv_idx):
            return precomputed_mask[b][h][q_idx][kv_idx]

        return mask_mod

    mask_4d = create_mask(
        mod_fn=precomputed_mask_factory(padded_mask),
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        device=causal_mask.device,
    )

    block_mask = create_block_mask(
        mask_mod=precomputed_mask_factory(mask_4d),
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=block_size,
        device=causal_mask.device,
        _compile=False,
    )
    return block_mask


def flex_attention_with_block_mask(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_mask: Any,
    seq_len: int,
    scaling: float | None = None,
) -> torch.Tensor:
    """
    Run flex_attention with a pre-built BlockMask (no create_mask allocation per call).
    """
    batch_size = query_states.shape[0]
    head_dim = query_states.shape[3]
    original_dtype = query_states.dtype

    query_states = query_states.transpose(1, 2).to(torch.float32)
    key_states = key_states.transpose(1, 2).to(torch.float32)
    value_states = value_states.transpose(1, 2).to(torch.float32)

    q_len_rounded = block_mask.shape[-2] if hasattr(block_mask, "shape") else query_states.shape[2]
    kv_len_rounded = block_mask.shape[-1] if hasattr(block_mask, "shape") else key_states.shape[2]

    pad_q = q_len_rounded - query_states.shape[2]
    pad_k = kv_len_rounded - key_states.shape[2]

    if pad_q > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_q), value=0.0)
    if pad_k > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_k), value=0.0)
        value_states = F.pad(value_states, (0, 0, 0, pad_k), value=0.0)

    attn_output, _ = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=True,
        scale=head_dim**-0.5 if scaling is None else scaling,
        return_lse=True,
        kernel_options=FLEX_KERNEL_OPTIONS,
    )
    attn_output = attn_output[:, :, :seq_len, :].to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, -1, attn_output.shape[2] * attn_output.shape[3])
    return attn_output
