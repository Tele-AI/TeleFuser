# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""LiveAct DiT Model - Faithful copy from SoulX-LiveAct.

This is a direct copy of model_memory.py with minimal modifications for TeleFuser integration.
Only modifications:
- Inherit from BaseModel instead of ModelMixin
- Add type annotations
- Remove diffusers dependency
- Add StateDictConverter
- Add Ulysses Sequence Parallel support
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh

from telefuser.cache import KVCache
from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.distributed import get_ulysses_group, get_ulysses_rank, get_ulysses_world_size
from telefuser.distributed.parallel_shard import sequence_parallel_shard, sequence_parallel_unshard
from telefuser.distributed.ulysses_comm import ulysses_gather_heads, ulysses_scatter_heads
from telefuser.ops.attention import attention
from telefuser.utils.logging import logger


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Create sinusoidal position embeddings."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """Precompute RoPE frequencies."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def build_freqs_combined(freqs: torch.Tensor, f: int, h: int, w: int, start_frame: int = 0) -> torch.Tensor:
    """Build combined RoPE frequencies for 3D positions (temporal, height, width).

    Args:
        freqs: Base frequencies tensor [max_len, head_dim_half]
        f: Number of frames
        h: Height dimension
        w: Width dimension
        start_frame: Starting frame index (for incremental generation)

    Returns:
        Combined frequencies tensor [f*h*w, 1, head_dim_half] in complex128
    """
    seq_len = f * h * w
    head_dim_half = freqs.size(1)

    # Split freqs for temporal, height, width dimensions
    c = head_dim_half
    freqs_t, freqs_h, freqs_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # Build combined freqs tensor
    freqs_combined = torch.cat(
        [
            freqs_t[start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(seq_len, 1, -1)

    return freqs_combined


def apply_rope(x: torch.Tensor, freqs_combined: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Apply RoPE with pre-computed frequencies.

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim]
        freqs_combined: Pre-computed frequencies [seq_len, 1, head_dim_half]
        seq_len: Sequence length to apply RoPE to

    Returns:
        RoPE-applied tensor [batch, seq_len, num_heads, head_dim]
    """
    batch_size = x.size(0)
    num_heads = x.size(2)

    # Apply RoPE to entire batch at once
    x_complex = torch.view_as_complex(x[:, :seq_len].to(torch.float64).reshape(batch_size, seq_len, num_heads, -1, 2))
    x_rope = torch.view_as_real(x_complex * freqs_combined).flatten(2)
    # Reshape back to NHD layout for SageAttention: [batch, seq_len, num_heads, head_dim]
    x_rope = x_rope.view(batch_size, seq_len, num_heads, -1)

    # Handle remaining tokens if any
    if x.size(1) > seq_len:
        x_rope = torch.cat([x_rope, x[:, seq_len:]], dim=1)

    return x_rope


class WanRMSNorm(nn.Module):
    """RMS normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm with FP32 computation."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(origin_dtype)
        return out


class SingleStreamAttention(nn.Module):
    """Cross-attention for audio conditioning - direct copy from SoulX-LiveAct."""

    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # SP support
        self.sp_flag = False
        self.sp_rank = 0
        self.sp_size = 1

        # Attention config - defaults to SAGE_ATTN for best performance
        self._attn_config = AttentionConfig(attn_impl=AttnImplType.SAGE_ATTN_2_8_8)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        shape: tuple | None = None,
        start_f: int = 0,
    ) -> torch.Tensor:
        """Audio cross-attention forward.

        In SP mode: each rank processes its corresponding frame chunk of encoder_kv.
        encoder_kv is sliced by sp_rank: [start_f + sp_rank*B : start_f + (sp_rank+1)*B]
        """
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        _, N_h, N_w = shape

        # In SP mode, each rank only holds partial frames
        N_t = x.size(1) // (N_h * N_w)
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

        B, N, C = x.shape
        frames_per_rank = B

        q = self.q_linear(x)
        q = q.view(B, N, self.num_heads, self.head_dim).permute((0, 2, 1, 3))

        B_e, N_a, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv = encoder_kv.view(B_e, N_a, 2, self.num_heads, self.head_dim)

        # SP mode: slice encoder_kv by sp_rank
        if self.sp_flag:
            kv_start = start_f + self.sp_rank * frames_per_rank
            kv_end = start_f + (self.sp_rank + 1) * frames_per_rank
            encoder_kv = encoder_kv[kv_start:kv_end]
        else:
            encoder_kv = encoder_kv[start_f : start_f + frames_per_rank]

        encoder_kv = encoder_kv.permute((2, 0, 3, 1, 4))
        encoder_k, encoder_v = encoder_kv.unbind(0)

        x = attention(
            q,
            encoder_k,
            encoder_v,
            attention_config=self._attn_config,
            input_layout="BNSD",
            output_layout="BNSD",
            is_causal=False,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        x = rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)

        return x


class WanSelfAttention(nn.Module):
    """Self-attention with KV cache and memory compression - direct copy from SoulX-LiveAct."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
        mean_memory: bool = False,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.mean_memory = mean_memory

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn_mask = None
        self.memory_proj_k = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False)
        self.memory_proj_v = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False)

        # SP (Ulysses Sequence Parallel) support - added for TeleFuser
        self.sp_flag = False
        self.device_mesh = None
        self.ulysses_group = None

        # Attention config - defaults to SAGE_ATTN for best performance
        self._attn_config = AttentionConfig(attn_impl=AttnImplType.SAGE_ATTN_2_8_8)

    def post_init(self, device):
        self.memory_proj_k = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False).to(
            device, dtype=torch.bfloat16
        )
        self.memory_proj_v = nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=5, groups=self.dim, bias=False).to(
            device, dtype=torch.bfloat16
        )
        nn.init.constant_(self.memory_proj_k.weight, 1.0 / 5.0)
        nn.init.constant_(self.memory_proj_v.weight, 1.0 / 5.0)

    def k_compress(self, k: torch.Tensor, n_frame: int = 5) -> torch.Tensor:
        B, N, H, C = k.shape
        assert N % n_frame == 0
        T = N // n_frame
        k = k.view(B, N, H * C).transpose(1, 2)
        k = self.memory_proj_k(k)
        k = k.view(B, H, C, T).permute(0, 3, 1, 2)
        return k

    def v_compress(self, v: torch.Tensor, n_frame: int = 5) -> torch.Tensor:
        B, N, H, C = v.shape
        assert N % n_frame == 0
        T = N // n_frame
        v = v.view(B, N, H * C).transpose(1, 2)
        v = self.memory_proj_v(v)
        v = v.view(B, H, C, T).permute(0, 3, 1, 2)
        return v

    def kv_mean(self, kv: torch.Tensor, n_frame: int = 5) -> torch.Tensor:
        B, N, H, C = kv.shape
        assert N % n_frame == 0
        T = N // n_frame
        kv = kv.view(B, T, n_frame, H, C).mean(dim=2)
        return kv

    def _compress_kv_cache_simple(
        self,
        k_full: torch.Tensor,
        v_full: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        tokens_per_frame: int,
    ) -> None:
        """Compress KV cache from 14 frames to 6 frames.

        Args:
            k_full: Full key tensor
            v_full: Full value tensor
            k_cache: Cache key tensor to update
            v_cache: Cache value tensor to update
            tokens_per_frame: Tokens per frame
        """
        if self.mean_memory:
            k_compress, v_compress = self.kv_mean, self.kv_mean
        else:
            k_compress, v_compress = self.k_compress, self.v_compress

        k_cache[:, 2 * tokens_per_frame : 3 * tokens_per_frame] = k_compress(
            k_full[:, 2 * tokens_per_frame : 7 * tokens_per_frame]
        )
        v_cache[:, 2 * tokens_per_frame : 3 * tokens_per_frame] = v_compress(
            v_full[:, 2 * tokens_per_frame : 7 * tokens_per_frame]
        )
        k_cache[:, 3 * tokens_per_frame : 4 * tokens_per_frame] = k_compress(
            k_full[:, 7 * tokens_per_frame : 12 * tokens_per_frame]
        )
        v_cache[:, 3 * tokens_per_frame : 4 * tokens_per_frame] = v_compress(
            v_full[:, 7 * tokens_per_frame : 12 * tokens_per_frame]
        )
        k_cache[:, 4 * tokens_per_frame : 6 * tokens_per_frame] = k_full[
            :, 12 * tokens_per_frame : 14 * tokens_per_frame
        ]
        v_cache[:, 4 * tokens_per_frame : 6 * tokens_per_frame] = v_full[
            :, 12 * tokens_per_frame : 14 * tokens_per_frame
        ]

    def forward_sp(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        f: int,
        h: int,
        w: int,
        kv_cache: KVCache,
        start_idx: int | None = None,
        end_idx: int | None = None,
        tokens_per_frame: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Self-attention with KVCache in Ulysses SP mode.

        In Ulysses SP, we scatter heads across ranks:
        - QKV: [B, S/N, H, D] -> [B, S, H/N, D] after scatter
        - Each rank holds H/N heads for full sequence
        - KV cache stores [1, seq, H/N, D] (full sequence, local heads)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # QKV projection [B, S/N, H, D]
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Ulysses scatter heads: [B, S/N, H, D] -> [B, S, H/N, D]
        q = ulysses_scatter_heads(q, self.ulysses_group)()
        k = ulysses_scatter_heads(k, self.ulysses_group)()
        v = ulysses_scatter_heads(v, self.ulysses_group)()

        k_cache, v_cache = kv_cache.load(x.device, torch.bfloat16)

        if tokens_per_frame is None:
            tokens_per_frame = h * w
        current_start_frame = start_idx // tokens_per_frame

        # KV cache stores 6 frames, for attention from full KV
        if start_idx != 0:
            k_full = torch.cat([k_cache, k], dim=1)
            v_full = torch.cat([v_cache, v], dim=1)
        else:
            k_cache[:, : 6 * tokens_per_frame] = k
            v_cache[:, : 6 * tokens_per_frame] = v
            k_full = k_cache
            v_full = v_cache

        q_seq_len = q.size(1)
        k_seq_len = k_full.size(1)
        q_freqs = build_freqs_combined(freqs, q_seq_len // tokens_per_frame, h, w, start_frame=current_start_frame)
        k_freqs = build_freqs_combined(freqs, k_seq_len // tokens_per_frame, h, w, start_frame=0)

        roped_query = apply_rope(q, q_freqs, q_seq_len).type_as(v)
        roped_key = apply_rope(k_full, k_freqs, k_seq_len).type_as(v)

        x = attention(
            roped_query,
            roped_key[:, :end_idx, ...],
            v_full[:, :end_idx, ...],
            attention_config=self._attn_config,
            input_layout="BSND",
            output_layout="BSND",
            is_causal=False,
        ).type_as(x)

        if start_idx != 0:
            self._compress_kv_cache_simple(k_full, v_full, k_cache, v_cache, tokens_per_frame)

        kv_cache.store(k_cache, v_cache)

        # Ulysses gather heads: [B, S, H/N, D] -> [B, S/N, H, D]
        x = ulysses_gather_heads(x, self.ulysses_group, num_heads=self.num_heads)()

        x = x.flatten(2)
        x = self.o(x)
        return x, None

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        f: int,
        h: int,
        w: int,
        kv_cache: KVCache,
        start_idx: int | None = None,
        end_idx: int | None = None,
        tokens_per_frame: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Self-attention with KVCache.

        Args:
            x: Input tensor [batch, seq_len, dim]
            freqs: Base frequencies tensor [max_len, head_dim_half]
            f, h, w: Grid dimensions (frames, height, width)
            kv_cache: KVCache instance
            start_idx: Starting token index for incremental generation
            end_idx: Ending token index for attention window
            tokens_per_frame: Pre-computed h*w (optional, for compatibility)
        """
        if self.sp_flag:
            return self.forward_sp(x, freqs, f, h, w, kv_cache, start_idx, end_idx, tokens_per_frame)

        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # QKV projection
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        k_cache, v_cache = kv_cache.load(x.device, torch.bfloat16)

        if tokens_per_frame is None:
            tokens_per_frame = h * w
        current_start_frame = start_idx // tokens_per_frame

        if start_idx != 0:
            k_full = torch.cat([k_cache, k], dim=1)
            v_full = torch.cat([v_cache, v], dim=1)
        else:
            k_cache[:, : 6 * tokens_per_frame] = k
            v_cache[:, : 6 * tokens_per_frame] = v
            k_full = k_cache
            v_full = v_cache

        q_seq_len = q.size(1)
        k_seq_len = k_full.size(1)
        q_freqs = build_freqs_combined(freqs, q_seq_len // tokens_per_frame, h, w, start_frame=current_start_frame)
        k_freqs = build_freqs_combined(freqs, k_seq_len // tokens_per_frame, h, w, start_frame=0)

        roped_query = apply_rope(q, q_freqs, q_seq_len).type_as(v)
        roped_key = apply_rope(k_full, k_freqs, k_seq_len).type_as(v)

        x = attention(
            roped_query,
            roped_key[:, :end_idx, ...],
            v_full[:, :end_idx, ...],
            attention_config=self._attn_config,
            input_layout="BSND",
            output_layout="BSND",
            is_causal=False,
        ).type_as(x)

        if start_idx != 0:
            self._compress_kv_cache_simple(k_full, v_full, k_cache, v_cache, tokens_per_frame)

        kv_cache.store(k_cache, v_cache)

        x = x.flatten(2)
        x = self.o(x)
        return x, None


class WanI2VCrossAttention(nn.Module):
    """Cross-attention for text and image conditioning - direct copy from SoulX-LiveAct."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Attention config - defaults to SAGE_ATTN for best performance
        self._attn_config = AttentionConfig(attn_impl=AttnImplType.SAGE_ATTN_2_8_8)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention forward - same as original SoulX-LiveAct."""
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)

        # Cross-attention using unified ops - BSND layout (NHD)
        img_x = attention(
            q,
            k_img,
            v_img,
            attention_config=self._attn_config,
            input_layout="BSND",
            output_layout="BSND",
            is_causal=False,
        )
        x = attention(
            q,
            k,
            v,
            attention_config=self._attn_config,
            input_layout="BSND",
            output_layout="BSND",
            is_causal=False,
        )

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and audio cross-attention."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        output_dim: int = 768,
        norm_input_visual: bool = True,
        mean_memory: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, mean_memory)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # init audio module
        self.audio_cross_attn = SingleStreamAttention(
            dim=dim,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qkv_bias=True,
        )
        self.norm_x = WanLayerNorm(dim, eps, elementwise_affine=True) if norm_input_visual else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        f: int,
        h: int,
        w: int,
        context: torch.Tensor,
        kv_cache: KVCache,
        start_idx: int | None = None,
        end_idx: int | None = None,
        audio_embedding: torch.Tensor | None = None,
        skip_audio: bool = False,
        tokens_per_frame: int | None = None,
    ) -> torch.Tensor:
        """Transformer block forward.

        Args:
            x: Input tensor [batch, seq_len, dim]
            e: Time embedding modulation
            freqs: Base frequencies tensor [max_len, head_dim_half]
            f, h, w: Grid dimensions (frames, height, width)
            context: Text conditioning
            kv_cache: KVCache instance for this layer
            start_idx: Starting token index for incremental generation
            end_idx: Ending token index for attention window
            audio_embedding: Audio conditioning tensor
            skip_audio: Whether to skip audio cross-attention
            tokens_per_frame: Pre-computed h*w (optional, defaults to h*w)
        """
        dtype = x.dtype

        if len(e.shape) == 3:
            e = (self.modulation + e).chunk(6, dim=1)
        else:
            e = (self.modulation.unsqueeze(-2) + e)[0].chunk(6, dim=0)

        # self-attention (same as original, no ProfilingContext wrapper)
        y, _ = self.self_attn(
            (self.norm1(x).float() * (1 + e[1]) + e[0]).type_as(x),
            freqs,
            f,
            h,
            w,
            kv_cache=kv_cache,
            start_idx=start_idx,
            end_idx=end_idx,
            tokens_per_frame=tokens_per_frame,
        )
        x = x + y * e[2]

        x = x.to(dtype)

        # cross-attention of text
        x = x + self.cross_attn(self.norm3(x), context)

        # cross attn of audio
        if not skip_audio:
            if tokens_per_frame is None:
                tokens_per_frame = h * w
            start_f = start_idx // tokens_per_frame if tokens_per_frame else 0
            x_a = self.audio_cross_attn(
                self.norm_x(x),
                encoder_hidden_states=audio_embedding,
                shape=(f, h, w),
                start_f=start_f,
            )
            # Only zero first frame on sp_rank=0 to match original model behavior
            if start_f == 0 and (not self.audio_cross_attn.sp_flag or self.audio_cross_attn.sp_rank == 0):
                x_a[:, :tokens_per_frame] = 0
            x = x + x_a

        # FFN
        y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(dtype))
        x = x + y * e[5]

        x = x.to(dtype)

        return x


class Head(nn.Module):
    """Output head."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(nn.Module):
    """MLP projection for CLIP image features."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class AudioProjModel(nn.Module):
    """Audio projection model for wav2vec2 embeddings."""

    def __init__(
        self,
        seq_len: int = 5,
        seq_len_vf: int = 12,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        norm_output_audio: bool = False,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def forward(self, audio_embeds: torch.Tensor, audio_embeds_vf: torch.Tensor) -> torch.Tensor:
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B, _, _, S, C = audio_embeds.shape

        # process audio of first frame
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        # process audio of latter frame
        audio_embeds_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) w b c")
        batch_size_vf, window_size_vf, blocks_vf, channels_vf = audio_embeds_vf.shape
        audio_embeds_vf = audio_embeds_vf.view(batch_size_vf, window_size_vf * blocks_vf * channels_vf)

        # first projection
        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds_vf = torch.relu(self.proj1_vf(audio_embeds_vf))
        audio_embeds = rearrange(audio_embeds, "(bz f) c -> bz f c", bz=B)
        audio_embeds_vf = rearrange(audio_embeds_vf, "(bz f) c -> bz f c", bz=B)
        audio_embeds_c = torch.concat([audio_embeds, audio_embeds_vf], dim=1)
        batch_size_c, N_t, C_a = audio_embeds_c.shape
        audio_embeds_c = audio_embeds_c.view(batch_size_c * N_t, C_a)

        # second projection
        audio_embeds_c = torch.relu(self.proj2(audio_embeds_c))

        context_tokens = self.proj3(audio_embeds_c).reshape(batch_size_c * N_t, self.context_tokens, self.output_dim)

        # normalization and reshape
        context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(context_tokens, "(bz f) m c -> bz f m c", f=video_length)

        return context_tokens


class LiveActDiT(BaseModel):
    """LiveAct Diffusion Transformer for audio-conditioned video generation.

    Faithful copy of SoulX-LiveAct WanModel with minimal TeleFuser adaptations.
    """

    def __init__(
        self,
        model_type: str = "i2v",
        patch_size: tuple = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 36,  # LiveAct: 36 (includes mask channel)
        dim: int = 5120,  # LiveAct: 5120
        ffn_dim: int = 13824,  # LiveAct: 13824
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,  # LiveAct: 40
        num_layers: int = 40,  # LiveAct: 40
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        # audio params
        audio_window: int = 5,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4,
        norm_input_visual: bool = True,
        norm_output_audio: bool = True,
        weight_init: bool = True,
        # KV cache compression mode
        mean_memory: bool = False,
    ):
        super().__init__()

        assert model_type == "i2v", "LiveAct model requires model_type is i2v."
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.gradient_checkpointing = False
        self.mean_memory = mean_memory

        self.norm_output_audio = norm_output_audio
        self.audio_window = audio_window
        self.intermediate_dim = intermediate_dim
        self.vae_scale = vae_scale

        self.layer_name_list = ["blocks"]

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    output_dim=output_dim,
                    norm_input_visual=norm_input_visual,
                    mean_memory=mean_memory,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE frequencies - keep as regular attribute to preserve complex128 dtype
        # (buffers get converted when model.to(dtype=...) is called)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self._freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

        if model_type == "i2v":
            self.img_emb = MLPProj(1280, dim)
        else:
            raise NotImplementedError("Not supported model type.")

        # init audio adapter
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
            norm_output_audio=norm_output_audio,
        )

        # USP (Ulysses Sequence Parallel) support
        self.usp_flag = False
        self.device_mesh = None

        # initialize weights
        if weight_init:
            self.init_weights()

    @property
    def freqs(self) -> torch.Tensor:
        """Return RoPE frequencies (complex128)."""
        return self._freqs

    @property
    def time_embedding_dtype(self) -> torch.dtype:
        """Return the dtype expected by the time embedding MLP."""
        return next(self.time_embedding.parameters()).dtype

    def forward(
        self,
        x: list,
        t: torch.Tensor,
        context: list,
        clip_fea: torch.Tensor | None = None,
        y: list | None = None,
        audio: torch.Tensor | None = None,
        e0: torch.Tensor | None = None,
        kv_cache: list[KVCache] = [],
        start_idx: int | None = None,
        end_idx: int | None = None,
        skip_audio: bool = False,
    ) -> torch.Tensor:
        """DiT forward with KVCache."""
        assert clip_fea is not None and y is not None

        # Move freqs to GPU if needed
        self._freqs = self._freqs.to(x[0].device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        x[0] = x[0].to(context[0].dtype)

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # Extract grid dimensions (f, h, w) directly - assuming batch=1
        f, h, w = x[0].shape[2:]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        x = torch.cat(x)

        # time embeddings
        time_emb = sinusoidal_embedding_1d(self.freq_dim, t).to(device=t.device, dtype=self.time_embedding_dtype)
        if e0 is None:
            e = self.time_embedding(time_emb)
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        else:
            e = self.time_embedding(time_emb)

        # text embedding
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        # clip embedding
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1).to(x.dtype)

        # audio processing
        audio_cond = audio.to(device=x.device, dtype=x.dtype)
        first_frame_audio_emb_s = audio_cond[:, :1, ...]
        latter_frame_audio_emb = audio_cond[:, 1:, ...]
        latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale)
        middle_index = self.audio_window // 2
        latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, : middle_index + 1, ...]
        latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
        latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index : middle_index + 1, ...]
        latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
        latter_frame_audio_emb_s = torch.concat(
            [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2
        )
        audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
        audio_embedding = torch.concat(audio_embedding.split(1), dim=2).to(x.dtype)

        # Sequence parallel: shard x across ranks [B, S, D] -> [B, S/N, D]
        seq_len = x.size(1)
        if self.usp_flag:
            sequence_parallel_shard(self.device_mesh, [x], seq_dims=[1])

        # Pre-compute tokens_per_frame (h * w)
        tokens_per_frame = h * w
        for block_index, block in enumerate(self.blocks):
            x = block(
                x,
                kv_cache=kv_cache[block_index],
                e=e0,
                freqs=self.freqs,
                f=f,
                h=h,
                w=w,
                context=context,
                audio_embedding=audio_embedding,
                start_idx=start_idx,
                end_idx=end_idx,
                skip_audio=skip_audio,
                tokens_per_frame=tokens_per_frame,
            )

        # head
        x = self.head(x, e)

        # Sequence parallel: unshard x back to full sequence [B, S/N, D] -> [B, S, D]
        if self.usp_flag:
            (x,) = sequence_parallel_unshard(self.device_mesh, (x,), seq_dims=(1,), seq_lens=(seq_len,))

        # unpatchify
        x = self.unpatchify(x, f, h, w)

        return torch.stack(x)

    def unpatchify(self, x: torch.Tensor, f: int, h: int, w: int) -> list:
        """Reconstruct video tensors from patch embeddings.

        Args:
            x: Input tensor [batch, seq_len, dim]
            f: Number of frames
            h: Height dimension
            w: Width dimension

        Returns:
            List of reconstructed video tensors [c, f*p0, h*p1, w*p2]
        """
        c = self.out_dim
        batch_size = x.size(0)
        tokens_per_video = f * h * w

        # Process all batch items at once using tensor operations
        x_trimmed = x[:, :tokens_per_video]
        x_reshaped = x_trimmed.view(batch_size, f, h, w, *self.patch_size, c)
        # Batch einsum: bfhwpqrc -> cbfphqwr
        x_permuted = torch.einsum("bfhwpqrc->cbfphqwr", x_reshaped)
        # Reshape to final output shape
        x_out = x_permuted.reshape(
            batch_size,
            c,
            f * self.patch_size[0],
            h * self.patch_size[1],
            w * self.patch_size[2],
        )
        # Return as list of tensors (each batch item)
        return list(x_out.unbind(0))

    def init_weights(self):
        """Initialize model parameters using Xavier initialization."""
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

    def set_attention_config(self, attention_config: AttentionConfig) -> None:
        """Set attention implementation configuration.

        Args:
            attention_config: Unified attention configuration from TeleFuser ops.
        """
        for block in self.blocks:
            block.self_attn._attn_config = attention_config
            block.cross_attn._attn_config = attention_config
            block.audio_cross_attn._attn_config = attention_config
        logger.info(f"LiveAct DiT attention implementation: {attention_config.attn_impl}")

    def enable_usp(self, device_mesh: DeviceMesh | None = None) -> None:
        """Enable Ulysses sequence parallelism.

        Args:
            device_mesh: Device mesh for distributed communication.
        """
        logger.info("LiveAct DiT enable USP (Ulysses Sequence Parallel)")
        self.device_mesh = device_mesh

        if device_mesh is None:
            return

        sp_size = get_ulysses_world_size(device_mesh)
        if self.num_heads % sp_size != 0:
            raise ValueError(f"num_heads {self.num_heads} must be divisible by sp_size {sp_size}")

        self.usp_flag = True
        ulysses_group = get_ulysses_group(device_mesh)
        sp_rank = get_ulysses_rank(device_mesh)

        for block in self.blocks:
            block.self_attn.sp_flag = True
            block.self_attn.device_mesh = device_mesh
            block.self_attn.ulysses_group = ulysses_group

            block.audio_cross_attn.sp_flag = True
            block.audio_cross_attn.sp_rank = sp_rank
            block.audio_cross_attn.sp_size = sp_size

            # Replace memory_proj with sliced versions for SP mode
            local_dim = block.self_attn.dim // sp_size
            new_proj_k = nn.Conv1d(local_dim, local_dim, kernel_size=5, stride=5, groups=local_dim, bias=False)
            new_proj_v = nn.Conv1d(local_dim, local_dim, kernel_size=5, stride=5, groups=local_dim, bias=False)

            start = sp_rank * local_dim
            end = (sp_rank + 1) * local_dim
            with torch.no_grad():
                new_proj_k.weight.copy_(block.self_attn.memory_proj_k.weight[start:end, :, :])
                new_proj_v.weight.copy_(block.self_attn.memory_proj_v.weight[start:end, :, :])

            block.self_attn.memory_proj_k = new_proj_k
            block.self_attn.memory_proj_v = new_proj_v

    def get_fsdp_module_names(self) -> list[str]:
        """Get module names for FSDP sharding."""
        return ["blocks"]

    def set_mean_memory(self, mean_memory: bool) -> None:
        """Set mean_memory mode for KV cache compression.

        Args:
            mean_memory: Use mean pooling instead of Conv1d for compression
        """
        self.mean_memory = mean_memory
        for block in self.blocks:
            block.self_attn.mean_memory = mean_memory

    @staticmethod
    def state_dict_converter():
        return LiveActDiTStateDictConverter()


class LiveActDiTStateDictConverter:
    """State dict converter for LiveAct DiT."""

    def from_official(self, state_dict: dict) -> tuple[dict, dict]:
        """Convert from official SoulX-LiveAct format."""
        return state_dict, {}

    def from_diffusers(self, state_dict: dict) -> tuple[dict, dict]:
        """Convert from diffusers format (not supported)."""
        raise NotImplementedError("Diffusers format not supported for LiveAct DiT")


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "99c7bbedc6dd6c4824799e4f999fee0a", ["liveact_dit"], [LiveActDiT], "official")
