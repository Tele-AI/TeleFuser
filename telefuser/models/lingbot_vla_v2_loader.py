"""Native utility, alignment, and checkpoint support for LingBot-VLA v2.

Adapted from the Apache-2.0 licensed LingBot-VLA v2 implementation.
"""

import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version
from torch import Tensor

# from xformers.ops import memory_efficient_attention


def find_next_divisible_by_8_numpy(n: np.ndarray) -> np.ndarray:
    """
    Finds the smallest integers greater than each element in a NumPy array 'n'
    that are divisible by 8. Assumes non-negative integers.

    Args:
        n: A NumPy array of integers.

    Returns:
        A NumPy array containing the smallest integers greater than each input element
        that are divisible by 8.
    """
    remainder = n % 8
    # Calculate the amount to add: 0 if already divisible, otherwise 8 - remainder
    # np.where is efficient for conditional operations on arrays
    amount_to_add = np.where(remainder == 0, 8, 8 - remainder)
    return n + amount_to_add


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
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


def make_att_2d_masks(pad_masks, att_masks):
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
    use_depth_align,
    use_future_depth,
    use_future_video=False,
    use_future_video_cls=False,
    use_future_video_patch=True,
    future_video_share_future_depth_query=False,
):
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
    prefix_len,
    num_task_tokens,
    use_depth_align,
    use_future_depth,
    use_future_video=False,
    use_future_video_cls=False,
    use_future_video_patch=True,
    future_video_share_future_depth_query=False,
):
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


def fv_col_span(prefix_len, num_task_tokens, use_cls, use_patch):
    """Return [start, end) of a tail query block inside the prefix.

    This legacy helper is still used for future-depth tail blocking in V2.
    New prefix layout code should prefer prefix_query_token_spans(), which also
    handles current-depth and separate future-video spans.
    """
    fv_len = (1 if use_cls else 0) + (num_task_tokens if use_patch else 0)
    return prefix_len - fv_len, prefix_len


def block_suffix_to_fv_(
    att_2d_masks, suffix_row_start, prefix_len, num_task_tokens, use_cls=False, use_patch=True, drop_mask=None
):
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


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


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

    probs = nn.functional.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    value_states_permuted = torch.einsum("blhd->bhld", value_states)  # [B, H, L_v, D]
    att_output = torch.einsum("bhqk,bhkv->bhqv", probs, value_states_permuted)  # [B, H, L_q, D]
    att_output = torch.einsum("bhld->blhd", att_output)  # [B, L, H, D]
    att_output = att_output.reshape(bsize, seq_len, num_att_heads * head_dim)

    return att_output


# @torch.jit.script
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

import torch

FLEX_SPARSE_BLOCK_SIZE = 128
FLEX_KERNEL_OPTIONS = {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}

if Version(torch.__version__) > Version("2.5.0"):
    # Ffex attention is only available from torch 2.5 onwards
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
        flex_attention,
    )


# @torch.compile(dynamic=False)
def flex_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling=None,
):
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
    # ipdb.set_trace()
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
):
    """
    Build a reusable BlockMask from a 3D attention mask [B, Q, KV].
    This allocates the dense 4D mask once; the returned BlockMask can be reused across layers.
    """
    from torch.nn.attention.flex_attention import (
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
    )

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
    block_mask,
    seq_len: int,
    scaling=None,
):
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


# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
import torch


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        batch_size, latent_length, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(batch_size, latent_length, -1)

        return self.to_out(out)


class TaskTokenResampler(nn.Module):
    def __init__(
        self,
        dim_in=768,
        dim_mid=1024,
        dim_head=64,
        dim_out=1024,
        num_layers=8,
        num_queries=8,
        num_heads=16,
        ff_mult=4,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                        FeedForward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, queries):
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries


class TaskTokenDepthHead(nn.Module):
    def __init__(
        self,
        proj_config=None,
        llm_hidden_size=4096,
        use_intermediate_depth=False,
    ):
        super(TaskTokenDepthHead, self).__init__()

        self.projector = TaskTokenResampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            num_queries=proj_config["num_backbone_tokens"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats, queries):
        queries = self.projector(llm_feats, queries)
        return queries


import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from telefuser.core.config import QuantConfig


class LingBotVLAWeightLoader:
    """Minimal native weight-name mapper retained for model compatibility."""

    def get_vlm_submodule(self, model):
        return model.model.qwenvl_with_expert.qwenvl

    def get_expert_vision_submodule(self, model):
        return getattr(model.model.qwenvl_with_expert, "expert_visual", None)

    def map_ckpt_key(self, key, load_vlm_only=False, post_training=False):
        if key.startswith("expert_visual.") and not post_training:
            return "model.qwenvl_with_expert." + key
        if load_vlm_only:
            return "model.qwenvl_with_expert.qwenvl." + key
        return key


OFFICIAL_6B_MODEL_CONFIG: dict[str, Any] = {
    "post_training": False,
    "adanorm_time": True,
    "moe_implementation": "fused",
    "use_robby_moe_kernel": True,
    "attention_implementation": "eager",
    "precompute_grid_thw": True,
    "vlm_causal": True,
    "use_moe": True,
    "token_moe_layers": list(range(36)),
    "token_num_experts": 32,
    "token_top_k": 4,
    "token_moe_intermediate_size": 512,
    "token_shared_intermediate_size": 704,
    "bias_update_speed": 0.0,
    "sequence_wise_mode": "per_sequence",
    "sequence_wise_loss_coeff": 1e-3,
    "router_z_loss_coeff": 1e-4,
    "router_activation": "sigmoid",
    "routed_scaling_factor": 4.0,
    "use_shared_expert_gate": False,
    "freeze_vision_encoder": False,
    "tokenizer_max_length": 72,
    "loss_type": "L1_fm",
    "action_dim": 55,
    "max_action_dim": 55,
    "max_state_dim": 55,
    "align_params": {
        "mode": "query",
        "num_task_tokens": 8,
        "depth_loss_weight": 0.004,
        "future_depth_loss_weight": 0.004,
        "use_future_video": True,
        "llm": {
            "dim_out": 2560,
            "image_token_size": 8,
            "image_input_size": 224,
        },
        "depth": {
            "model_type": "MoRGBD",
            "num_layers": 1,
            "num_heads": 4,
            "dim_head": 32,
            "ff_mult": 1,
            "num_backbone_tokens": 256,
            "token_size": 16,
            "dim_out": 1024,
            "input_size": 224,
            "use_future_depth": True,
            "block_future_depth_to_action": True,
            "future_depth_head_type": "resampler",
            "detach_future_image_feats": True,
        },
        "video": {
            "attention_mode": "flex_block_causal",
            "input_size": 256,
            "block_suffix_to_future_video": True,
            "share_future_depth_query": True,
            "use_shared_future_task_proj": True,
            "use_current_shared_task_proj": True,
            "num_future_frames": 1,
            "use_warmup_frame": True,
            "effective_fps": 1.0,
            "n_blocks": 1,
            "cls_pool": "last",
            "detach_image_feats": True,
            "num_layers": 1,
            "num_heads": 4,
            "dim_head": 32,
            "ff_mult": 1,
            "num_backbone_tokens": 256,
            "dim_out": 1024,
            "future_video_loss_weight": 0.004,
            "use_smooth_l1_loss": False,
            "use_mse_loss": True,
            "mse_loss_weight": 1.0,
            "use_patch_loss": True,
            "use_current_patch_loss": True,
            "use_cosine_loss": False,
            "cosine_loss_weight": 0.2,
            "use_cls_loss": False,
            "cls_loss_type": "mse",
            "cls_loss_weight": 0.2,
        },
    },
}


def resolve_lingbot_vla_v2_checkpoint(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.name != "model.safetensors.index.json":
            raise ValueError(f"Expected model.safetensors.index.json, got: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"LingBot-VLA v2 model path does not exist: {path}")
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing sharded checkpoint index: {index_path}")
    return index_path


def resolve_lingbot_vla_v2_shards(model_path: str | Path) -> list[str]:
    index_path = resolve_lingbot_vla_v2_checkpoint(model_path)
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid safetensors index without weight_map: {index_path}")

    shard_paths = [index_path.parent / name for name in sorted(set(weight_map.values()))]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing LingBot-VLA v2 checkpoint shards: {missing}")
    return [str(path) for path in shard_paths]


def build_official_6b_config(
    qwen3vl_path: str | Path,
    *,
    checkpoint_variant: str = "base",
    checkpoint_path: str | Path | None = None,
):
    from telefuser.models.lingbot_vla_v2 import LingbotVLAV2Config

    if checkpoint_variant != "base":
        raise ValueError(f"Unsupported LingBot-VLA v2 checkpoint variant: {checkpoint_variant!r}")

    qwen_path = Path(qwen3vl_path).expanduser().resolve()
    qwen_config = AutoConfig.from_pretrained(str(qwen_path), local_files_only=True)
    if not hasattr(qwen_config, "text_config") or not hasattr(qwen_config, "vision_config"):
        raise ValueError(
            "LingBot-VLA v2 requires the local Qwen3-VL-4B-Instruct architecture/tokenizer "
            f"directory; this is not a complete Qwen3-VL directory: {qwen_path}"
        )

    text_config = qwen_config.text_config
    expected_architecture = {"hidden_size": 2560, "num_hidden_layers": 36}
    mismatches = {
        key: (expected, getattr(text_config, key, None))
        for key, expected in expected_architecture.items()
        if getattr(text_config, key, None) != expected
    }
    if mismatches:
        raise ValueError(
            "LingBot-VLA v2 6B was trained with Qwen3-VL-4B-Instruct; "
            f"the supplied architecture is incompatible: {mismatches}"
        )

    values = deepcopy(OFFICIAL_6B_MODEL_CONFIG)
    values["tokenizer_path"] = str(qwen_path)
    config = LingbotVLAV2Config(**values)
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
    ):
        if hasattr(text_config, key):
            setattr(config, key, getattr(text_config, key))
    config.vision_config = qwen_config.vision_config
    config.tokenizer_path = str(qwen_path)
    config.use_cache = True
    config.attention_implementation = "eager"
    config.checkpoint_variant = checkpoint_variant
    config.checkpoint_path = None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve())
    config.policy_verified = False
    config.verification_status = "unverified_official_6b_base"
    return config


def validate_official_6b_checkpoint(state_dict):
    gate = "model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.gate_proj"
    last_gate = "model.qwenvl_with_expert.qwen_expert.model.layers.35.mlp.experts.gate_proj"
    expected = (32, 512, 768)
    for key in (gate, last_gate):
        if key not in state_dict:
            raise ValueError(f"Missing official LingBot-VLA v2 weight: {key}")
        if tuple(state_dict[key].shape) != expected:
            raise ValueError(f"Unexpected shape for {key}: expected {expected}, got {tuple(state_dict[key].shape)}")


class LingBotVlaV2StateDictConverter:
    def __init__(
        self,
        qwen3vl_path: str | Path,
        checkpoint_variant: str = "base",
        checkpoint_path: str | Path | None = None,
    ):
        self.qwen3vl_path = Path(qwen3vl_path)
        self.checkpoint_variant = checkpoint_variant
        self.checkpoint_path = checkpoint_path

    def from_official(self, state_dict):
        validate_official_6b_checkpoint(state_dict)
        config = build_official_6b_config(
            self.qwen3vl_path,
            checkpoint_variant=self.checkpoint_variant,
            checkpoint_path=self.checkpoint_path,
        )
        return state_dict, {"config": config, "eval": True}

    def from_diffusers(self, state_dict):
        del state_dict
        raise ValueError("LingBot-VLA v2 does not provide a Diffusers checkpoint")


def load_lingbot_vla_v2(
    module_manager,
    model_path: str | Path,
    qwen3vl_path: str | Path,
    *,
    torch_dtype=torch.bfloat16,
    device=None,
    checkpoint_variant: str = "base",
    quant_config: QuantConfig | None = None,
):
    from telefuser.models.lingbot_vla_v2 import LingBotVlaV2Model

    checkpoint_path = resolve_lingbot_vla_v2_checkpoint(model_path).parent
    shard_paths = resolve_lingbot_vla_v2_shards(checkpoint_path)
    module_manager.load_model(
        shard_paths,
        device=device,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        name="lingbot_vla_v2",
        model_class=LingBotVlaV2Model,
        model_resource="official",
        converter_kwargs={
            "qwen3vl_path": str(qwen3vl_path),
            "checkpoint_variant": checkpoint_variant,
            "checkpoint_path": str(checkpoint_path),
        },
        quant_config=quant_config,
    )
    return module_manager.fetch_module("lingbot_vla_v2")
