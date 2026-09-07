"""ABot-World causal Wan 2.2 DiT.

This is the inference-only, single-card implementation for the public
``ABot-World-0-5B-LF`` checkpoint.  Its parameter names intentionally match
the official checkpoint.  The causal cache stores unrotated keys and applies
RoPE at read time. A sink configuration uses fixed logical positions for the
sink prefix and rolling tail, so long sessions do not grow RoPE indices.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig
from telefuser.ops.attention import attention as attention_fn
from telefuser.ops.normalization import LayerNorm, RMSNorm

from .wan_video_dit import precompute_freqs_cis_3d, sinusoidal_embedding_1d


def _rope_apply(
    x: torch.Tensor,
    grid_size: tuple[int, int, int],
    freqs: torch.Tensor,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Apply Wan's 3D complex RoPE to ``[B, S, heads, head_dim]`` tensors."""
    frames, height, width = grid_size
    sequence_length = frames * height * width
    if x.shape[1] != sequence_length:
        raise ValueError(f"RoPE expected {sequence_length} tokens, got {x.shape[1]}")
    half_dim = x.shape[-1] // 2
    time_dim = half_dim - 2 * (half_dim // 3)
    height_dim = half_dim // 3
    width_dim = half_dim // 3
    freq_t, freq_h, freq_w = freqs.split([time_dim, height_dim, width_dim], dim=1)
    indices = frame_indices.to(device=x.device, dtype=torch.long)
    if indices.numel() != frames:
        raise ValueError(f"RoPE frame index count {indices.numel()} does not match {frames}")
    if indices.min().item() < 0 or indices.max().item() >= freq_t.shape[0]:
        raise ValueError(f"RoPE frame indices must be in [0, {freq_t.shape[0]}), got {indices.tolist()}")
    expanded = torch.cat(
        [
            freq_t[indices].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            freq_h[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            freq_w[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(sequence_length, 1, -1)
    rotated = torch.view_as_real(
        torch.view_as_complex(x.float().reshape(x.shape[0], sequence_length, x.shape[2], -1, 2)) * expanded
    ).flatten(3)
    return rotated.to(dtype=x.dtype)


def _rope_apply_static(
    x: torch.Tensor,
    grid_size: tuple[int, int, int],
    freqs: torch.Tensor,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Apply fixed, prevalidated RoPE indices without host scalar reads."""
    frames, height, width = grid_size
    sequence_length = frames * height * width
    if x.shape[1] != sequence_length:
        raise ValueError(f"RoPE expected {sequence_length} tokens, got {x.shape[1]}")
    half_dim = x.shape[-1] // 2
    time_dim = half_dim - 2 * (half_dim // 3)
    height_dim = half_dim // 3
    width_dim = half_dim // 3
    freq_t, freq_h, freq_w = freqs.split([time_dim, height_dim, width_dim], dim=1)
    indices = frame_indices.to(device=x.device, dtype=torch.long)
    expanded = torch.cat(
        [
            freq_t[indices].view(frames, 1, 1, -1).expand(frames, height, width, -1),
            freq_h[:height].view(1, height, 1, -1).expand(frames, height, width, -1),
            freq_w[:width].view(1, 1, width, -1).expand(frames, height, width, -1),
        ],
        dim=-1,
    ).reshape(sequence_length, 1, -1)
    rotated = torch.view_as_real(
        torch.view_as_complex(x.float().reshape(x.shape[0], sequence_length, x.shape[2], -1, 2)) * expanded
    ).flatten(3)
    return rotated.to(dtype=x.dtype)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class SimpleAdapter(nn.Module):
    """Official ABot action adapter from pixel-space key presses to DiT tokens."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        downscale_factor: int = 16,
    ) -> None:
        super().__init__()
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)
        self.conv = nn.Conv2d(
            in_dim * downscale_factor * downscale_factor,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.residual_blocks = nn.Sequential(_ResidualBlock(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        pixels = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        features = self.residual_blocks(self.conv(self.pixel_unshuffle(pixels)))
        return features.reshape(batch, frames, *features.shape[1:]).permute(0, 2, 1, 3, 4)


class CausalWanSelfAttention(nn.Module):
    """Single-GPU causal self-attention with ABot's rolling cache semantics."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        eps: float = 1e-6,
        use_relative_rope: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.use_relative_rope = use_relative_rope
        self.attention_config = AttentionConfig()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    @staticmethod
    def _cursor(cache: dict[str, Any], name: str) -> int:
        value = cache[name]
        return int(value.item()) if isinstance(value, torch.Tensor) else int(value)

    @staticmethod
    def _set_cursor(cache: dict[str, Any], name: str, value: int) -> None:
        cursor = cache[name]
        if isinstance(cursor, torch.Tensor):
            cursor.fill_(value)
        else:
            cache[name] = value

    def _update_cache(
        self,
        cache: dict[str, Any],
        key: torch.Tensor,
        value: torch.Tensor,
        current_start: int,
        frame_tokens: int,
    ) -> tuple[int, int]:
        """Store a block and return its local start/end positions."""
        token_count = key.shape[1]
        current_end = current_start + token_count
        global_end = self._cursor(cache, "global_end_index")
        local_end = self._cursor(cache, "local_end_index")
        capacity = cache["k"].shape[1]
        sink_tokens = self.sink_size * frame_tokens
        if self.local_attn_size != -1 and current_end > global_end and token_count + local_end > capacity:
            evicted = token_count + local_end - capacity
            rolled = local_end - evicted - sink_tokens
            if rolled < 0:
                raise RuntimeError("ABot KV cache is smaller than its causal attention window")
            cache["k"][:, sink_tokens : sink_tokens + rolled].copy_(
                cache["k"][:, sink_tokens + evicted : sink_tokens + evicted + rolled].clone()
            )
            cache["v"][:, sink_tokens : sink_tokens + rolled].copy_(
                cache["v"][:, sink_tokens + evicted : sink_tokens + evicted + rolled].clone()
            )
            local_end = local_end + current_end - global_end - evicted
        else:
            local_end = local_end + current_end - global_end
        local_start = local_end - token_count
        if local_start < 0 or local_end > capacity:
            raise RuntimeError(f"ABot KV cache write [{local_start}:{local_end}] exceeds capacity {capacity}")
        cache["k"][:, local_start:local_end].copy_(key.detach())
        cache["v"][:, local_start:local_end].copy_(value)
        self._set_cursor(cache, "global_end_index", current_end)
        self._set_cursor(cache, "local_end_index", local_end)
        return local_start, local_end

    def forward_steady_state(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int],
        freqs: torch.Tensor,
        kv_cache: dict[str, Any],
        current_end: torch.Tensor,
        roll_scratch_k: torch.Tensor,
        roll_scratch_v: torch.Tensor,
        *,
        update_cache: bool,
    ) -> torch.Tensor:
        """Run a full, Relative-RoPE KV window without reading cache cursors.

        This is a specialized continuation path for a cache already filled by
        the regular forward method. It keeps the sink prefix, rolls the tail
        by exactly one fixed input block when update_cache is true, and always
        writes the current block at the fixed tail position. Cursor tensors are
        only written with current_end; they are never read on the host, which
        makes this method suitable for CUDA graph capture.

        The caller must ensure every layer has a full local cache and that the
        input grid, batch, and block length stay fixed for a graph slot.
        """
        batch, tokens, _ = x.shape
        frames, height, width = grid_size
        frame_tokens = height * width
        if tokens != frames * frame_tokens:
            raise ValueError("ABot causal attention received inconsistent grid size")
        if not self.use_relative_rope:
            raise ValueError("ABot steady-state attention requires Relative-RoPE")
        if self.local_attn_size < 1:
            raise ValueError("ABot steady-state attention requires a finite local window")
        capacity = kv_cache["k"].shape[1]
        if capacity != self.local_attn_size * frame_tokens:
            raise ValueError("ABot steady-state cache capacity does not match its local window")
        if kv_cache["v"].shape != kv_cache["k"].shape:
            raise ValueError("ABot steady-state key and value cache shapes must match")
        if current_end.numel() != 1:
            raise ValueError("ABot steady-state current_end must be a scalar tensor")
        sink_tokens = self.sink_size * frame_tokens
        local_start = capacity - tokens
        rolled_tokens = local_start - sink_tokens
        if local_start < sink_tokens or rolled_tokens < 0:
            raise ValueError("ABot steady-state block does not fit in the rolling cache tail")
        scratch_shape = (batch, rolled_tokens, self.num_heads, self.head_dim)
        if roll_scratch_k.shape != scratch_shape or roll_scratch_v.shape != scratch_shape:
            raise ValueError("ABot steady-state rolling scratch has an incompatible shape")

        query = rearrange(self.norm_q(self.q(x)), "b s (h d) -> b s h d", h=self.num_heads)
        key = rearrange(self.norm_k(self.k(x)), "b s (h d) -> b s h d", h=self.num_heads)
        value = rearrange(self.v(x), "b s (h d) -> b s h d", h=self.num_heads)
        if update_cache:
            if rolled_tokens:
                roll_scratch_k.copy_(kv_cache["k"][:, sink_tokens + tokens : capacity])
                roll_scratch_v.copy_(kv_cache["v"][:, sink_tokens + tokens : capacity])
                kv_cache["k"][:, sink_tokens:local_start].copy_(roll_scratch_k)
                kv_cache["v"][:, sink_tokens:local_start].copy_(roll_scratch_v)
            kv_cache["global_end_index"].copy_(current_end.reshape_as(kv_cache["global_end_index"]))
            kv_cache["local_end_index"].fill_(capacity)
        kv_cache["k"][:, local_start:capacity].copy_(key.detach())
        kv_cache["v"][:, local_start:capacity].copy_(value)

        cache_indices = torch.arange(self.local_attn_size, device=x.device)
        query_start = self.local_attn_size - frames
        if query_start < 0:
            raise ValueError("ABot steady-state query block is larger than its local window")
        cached_key = _rope_apply_static(
            kv_cache["k"],
            (self.local_attn_size, height, width),
            freqs,
            cache_indices,
        )
        query = _rope_apply_static(
            query,
            grid_size,
            freqs,
            torch.arange(query_start, self.local_attn_size, device=x.device),
        )
        output = attention_fn(
            query,
            cached_key,
            kv_cache["v"],
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
        )
        return self.o(rearrange(output, "b s h d -> b s (h d)"))

    def forward(
        self,
        x: torch.Tensor,
        grid_size: tuple[int, int, int],
        freqs: torch.Tensor,
        kv_cache: dict[str, Any],
        current_start: int,
    ) -> torch.Tensor:
        batch, tokens, _ = x.shape
        frames, height, width = grid_size
        frame_tokens = height * width
        if tokens != frames * frame_tokens:
            raise ValueError("ABot causal attention received inconsistent grid size")
        query = rearrange(self.norm_q(self.q(x)), "b s (h d) -> b s h d", h=self.num_heads)
        key = rearrange(self.norm_k(self.k(x)), "b s (h d) -> b s h d", h=self.num_heads)
        value = rearrange(self.v(x), "b s (h d) -> b s h d", h=self.num_heads)
        local_start, local_end = self._update_cache(kv_cache, key, value, current_start, frame_tokens)
        # A non-zero sink occupies the prefix of the fixed KV allocation.  The
        # remainder has already been compacted by _update_cache into a rolling
        # tail, so both pieces must stay visible to attention.
        if self.local_attn_size == -1 or self.sink_size:
            visible_start = 0
        else:
            visible_start = max(0, local_end - self.local_attn_size * frame_tokens)
        visible_tokens = local_end - visible_start
        if visible_tokens % frame_tokens:
            raise RuntimeError("ABot causal KV cache lost frame alignment")
        visible_frames = visible_tokens // frame_tokens
        cached_key = kv_cache["k"][:, visible_start:local_end]
        cached_value = kv_cache["v"][:, visible_start:local_end]
        if self.use_relative_rope:
            if self.sink_size:
                # The cache layout is always [sink, rolling tail]. Keep all
                # positions inside this trained local window; raw K is rotated
                # afresh after each eviction.
                cache_indices = torch.arange(visible_frames, device=x.device)
                cached_key = _rope_apply(cached_key, (visible_frames, height, width), freqs, cache_indices)
                query_start = local_start // frame_tokens
                query = _rope_apply(
                    query, grid_size, freqs, torch.arange(query_start, query_start + frames, device=x.device)
                )
            else:
                cached_key = _rope_apply(
                    cached_key,
                    (visible_frames, height, width),
                    freqs,
                    torch.arange(visible_frames, device=x.device),
                )
                query_start = visible_frames - frames
                if query_start < 0:
                    raise RuntimeError("ABot query block is larger than its visible causal window")
                query = _rope_apply(
                    query,
                    grid_size,
                    freqs,
                    torch.arange(query_start, visible_frames, device=x.device),
                )
        else:
            start_frame = current_start // frame_tokens
            query = _rope_apply(
                query, grid_size, freqs, torch.arange(start_frame, start_frame + frames, device=x.device)
            )
        output = attention_fn(
            query,
            cached_key,
            cached_value,
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
        )
        return self.o(rearrange(output, "b s h d -> b s (h d)"))


class CausalWanCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.attention_config = AttentionConfig()

    def forward(self, x: torch.Tensor, context: torch.Tensor, cache: dict[str, Any]) -> torch.Tensor:
        query = rearrange(self.norm_q(self.q(x)), "b s (h d) -> b s h d", h=self.num_heads)
        if not bool(cache["is_init"]):
            key = rearrange(self.norm_k(self.k(context)), "b s (h d) -> b s h d", h=self.num_heads)
            value = rearrange(self.v(context), "b s (h d) -> b s h d", h=self.num_heads)
            if cache["k"].shape[1] < key.shape[1]:
                raise ValueError("ABot cross-attention cache is shorter than the text sequence")
            cache["k"][:, : key.shape[1]].copy_(key)
            cache["v"][:, : value.shape[1]].copy_(value)
            cache["sequence_length"] = key.shape[1]
            cache["is_init"] = True
        length = int(cache["sequence_length"])
        output = attention_fn(
            query,
            cache["k"][:, :length],
            cache["v"][:, :length],
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
        )
        return self.o(rearrange(output, "b s h d -> b s (h d)"))

    def forward_steady_state(
        self,
        x: torch.Tensor,
        cache: dict[str, Any],
        *,
        context_length: int,
    ) -> torch.Tensor:
        """Attend to an already-initialized fixed text cache without host reads."""
        if context_length < 1 or context_length > cache["k"].shape[1]:
            raise ValueError("ABot steady-state cross-attention context length is invalid")
        query = rearrange(self.norm_q(self.q(x)), "b s (h d) -> b s h d", h=self.num_heads)
        output = attention_fn(
            query,
            cache["k"][:, :context_length],
            cache["v"][:, :context_length],
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
        )
        return self.o(rearrange(output, "b s h d -> b s (h d)"))


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int,
        sink_size: int,
        eps: float,
        use_relative_rope: bool,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = CausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, eps, use_relative_rope=use_relative_rope
        )
        self.norm3 = LayerNorm(dim, eps=eps)
        self.cross_attn = CausalWanCrossAttention(dim, num_heads, eps)
        self.norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    @staticmethod
    def _expand(value: torch.Tensor, frames: int, frame_tokens: int) -> torch.Tensor:
        return value.unflatten(1, (frames, frame_tokens)).flatten(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        time_modulation: torch.Tensor,
        context: torch.Tensor,
        grid_size: tuple[int, int, int],
        freqs: torch.Tensor,
        kv_cache: dict[str, Any],
        crossattn_cache: dict[str, Any],
        current_start: int,
    ) -> torch.Tensor:
        frames, height, width = grid_size
        frame_tokens = height * width
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(device=x.device, dtype=x.dtype).unsqueeze(0) + time_modulation
        ).chunk(6, dim=2)
        normed = self.norm1(x).unflatten(1, (frames, frame_tokens))
        attention_input = (normed * (1 + scale_msa) + shift_msa).flatten(1, 2)
        attention = self.self_attn(attention_input, grid_size, freqs, kv_cache, current_start)
        x = x + (attention.unflatten(1, (frames, frame_tokens)) * gate_msa).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, crossattn_cache)
        normed = self.norm2(x).unflatten(1, (frames, frame_tokens))
        ffn_input = (normed * (1 + scale_mlp) + shift_mlp).flatten(1, 2)
        ffn_output = self.ffn(ffn_input)
        return x + (ffn_output.unflatten(1, (frames, frame_tokens)) * gate_mlp).flatten(1, 2)

    def forward_steady_state(
        self,
        x: torch.Tensor,
        time_modulation: torch.Tensor,
        grid_size: tuple[int, int, int],
        freqs: torch.Tensor,
        kv_cache: dict[str, Any],
        crossattn_cache: dict[str, Any],
        current_end: torch.Tensor,
        roll_scratch_k: torch.Tensor,
        roll_scratch_v: torch.Tensor,
        *,
        context_length: int,
        update_cache: bool,
    ) -> torch.Tensor:
        """Compose the fixed-window self and already-cached cross attention."""
        frames, height, width = grid_size
        frame_tokens = height * width
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(device=x.device, dtype=x.dtype).unsqueeze(0) + time_modulation
        ).chunk(6, dim=2)
        normed = self.norm1(x).unflatten(1, (frames, frame_tokens))
        attention_input = (normed * (1 + scale_msa) + shift_msa).flatten(1, 2)
        attention = self.self_attn.forward_steady_state(
            attention_input,
            grid_size,
            freqs,
            kv_cache,
            current_end,
            roll_scratch_k,
            roll_scratch_v,
            update_cache=update_cache,
        )
        x = x + (attention.unflatten(1, (frames, frame_tokens)) * gate_msa).flatten(1, 2)
        x = x + self.cross_attn.forward_steady_state(
            self.norm3(x),
            crossattn_cache,
            context_length=context_length,
        )
        normed = self.norm2(x).unflatten(1, (frames, frame_tokens))
        ffn_input = (normed * (1 + scale_mlp) + shift_mlp).flatten(1, 2)
        ffn_output = self.ffn(ffn_input)
        return x + (ffn_output.unflatten(1, (frames, frame_tokens)) * gate_mlp).flatten(1, 2)


class CausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        frames = time_embedding.shape[1]
        if x.shape[1] % frames:
            raise ValueError("ABot head requires an integral number of tokens per frame")
        tokens_per_frame = x.shape[1] // frames
        shift, scale = (self.modulation.to(device=x.device, dtype=x.dtype).unsqueeze(1) + time_embedding).chunk(
            2, dim=2
        )
        value = self.norm(x).unflatten(1, (frames, tokens_per_frame)) * (1 + scale) + shift
        return self.head(value)


class ABotWorldDiT(BaseModel):
    """Checkpoint-compatible ABot-World-0-5B-LF backbone."""

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 48,
        dim: int = 3072,
        ffn_dim: int = 14336,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 48,
        num_heads: int = 24,
        num_layers: int = 30,
        local_attn_size: int = 18,
        sink_size: int = 6,
        eps: float = 1e-6,
        use_relative_rope: bool = True,
        downscale_factor_control_adapter: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.use_relative_rope = use_relative_rope
        self.layer_name_list = ["blocks"]
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(dim, ffn_dim, num_heads, local_attn_size, sink_size, eps, use_relative_rope)
                for _ in range(num_layers)
            ]
        )
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        self.act_control_adapter = SimpleAdapter(
            32,
            dim,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
            downscale_factor=downscale_factor_control_adapter,
        )
        self.freqs = torch.cat(precompute_freqs_cis_3d(dim // num_heads), dim=1)
        self._freqs_by_device: dict[tuple[str, int | None], torch.Tensor] = {}

    def set_causal_attention_window(self, local_attn_size: int, sink_size: int = 0) -> None:
        if local_attn_size < 1:
            raise ValueError("ABot single-card inference requires a positive local_attn_size")
        if not 0 <= sink_size < local_attn_size:
            raise ValueError("ABot sink_size must be non-negative and smaller than local_attn_size")
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        for block in self.blocks:
            block.self_attn.local_attn_size = local_attn_size
            block.self_attn.sink_size = sink_size

    def set_attention_config(self, attention_config: AttentionConfig) -> None:
        for block in self.blocks:
            block.self_attn.attention_config = attention_config
            block.cross_attn.attention_config = attention_config

    def _frequencies(self, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index)
        if key not in self._freqs_by_device:
            self._freqs_by_device[key] = self.freqs.to(device=device)
        return self._freqs_by_device[key]

    def _unpatchify(self, x: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        frames, height, width = grid_size
        patch_t, patch_h, patch_w = self.patch_size
        if x.ndim == 4:
            # CausalHead retains [frame, tokens-per-frame] for frame-wise modulation.
            x = x.flatten(1, 2)
        return rearrange(
            x,
            "b (f h w) (pt ph pw c) -> b c (f pt) (h ph) (w pw)",
            f=frames,
            h=height,
            w=width,
            pt=patch_t,
            ph=patch_h,
            pw=patch_w,
            c=self.out_dim,
        )

    def forward_steady_state(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        act_context: torch.Tensor,
        kv_cache: list[dict[str, Any]],
        crossattn_cache: list[dict[str, Any]],
        current_end: torch.Tensor,
        roll_scratch_k: torch.Tensor,
        roll_scratch_v: torch.Tensor,
        *,
        update_cache: bool,
        act_context_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run a fixed-shape, full-cache Relative-RoPE continuation.

        This method is intentionally narrower than forward: callers must prime
        every self-attention cache to its full local window and initialize each
        text cross-attention cache through the regular path. It then avoids
        cache-cursor scalar reads and only writes the supplied device-resident
        current_end scalar. The normal dynamic forward path remains the
        fallback for first chunks, changing shapes, or stale caches.

        context is retained in the API to make equivalence explicit; only its
        fixed sequence length is used because cross-attention keys/values are
        already cached.
        """
        if x.ndim != 5 or timestep.ndim != 2:
            raise ValueError("ABot expects x=[B,C,F,H,W] and timestep=[B,F]")
        if x.shape[2] != timestep.shape[1] or x.shape[0] != timestep.shape[0]:
            raise ValueError("ABot timestep shape must match the latent batch and frame dimensions")
        if context.ndim != 3 or context.shape[0] != x.shape[0]:
            raise ValueError("ABot steady-state context must match the latent batch")
        if len(kv_cache) != self.num_layers or len(crossattn_cache) != self.num_layers:
            raise ValueError("ABot cache lists must contain one entry per transformer layer")
        if not self.use_relative_rope:
            raise ValueError("ABot steady-state forward requires Relative-RoPE")
        if current_end.device != x.device or current_end.dtype != torch.long:
            raise ValueError("ABot steady-state current_end must be a device-resident int64 tensor")
        if roll_scratch_k.device != x.device or roll_scratch_v.device != x.device:
            raise ValueError("ABot steady-state rolling scratch must share the latent device")

        embedded = self.patch_embedding(x)
        action = self.act_control_adapter(act_context.to(device=x.device, dtype=embedded.dtype))
        if action.shape != embedded.shape:
            raise ValueError(
                f"ABot action adapter output {tuple(action.shape)} does not match latent tokens {tuple(embedded.shape)}"
            )
        embedded = embedded + action * act_context_scale
        grid_size = tuple(int(value) for value in embedded.shape[2:])
        tokens = rearrange(embedded, "b c f h w -> b (f h w) c")
        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(tokens.dtype)
        )
        time_embedding = time_embedding.unflatten(0, timestep.shape)
        time_modulation = self.time_projection(time_embedding).unflatten(2, (6, self.dim))
        freqs = self._frequencies(tokens.device)
        context_length = context.shape[1]
        for index, block in enumerate(self.blocks):
            tokens = block.forward_steady_state(
                tokens,
                time_modulation,
                grid_size,
                freqs,
                kv_cache[index],
                crossattn_cache[index],
                current_end,
                roll_scratch_k,
                roll_scratch_v,
                context_length=context_length,
                update_cache=update_cache,
            )
        return self._unpatchify(self.head(tokens, time_embedding.unsqueeze(2)), grid_size)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        act_context: torch.Tensor,
        kv_cache: list[dict[str, Any]],
        crossattn_cache: list[dict[str, Any]],
        current_start: int,
        act_context_scale: float = 1.0,
    ) -> torch.Tensor:
        if x.ndim != 5 or timestep.ndim != 2:
            raise ValueError("ABot expects x=[B,C,F,H,W] and timestep=[B,F]")
        if x.shape[2] != timestep.shape[1] or x.shape[0] != timestep.shape[0]:
            raise ValueError("ABot timestep shape must match the latent batch and frame dimensions")
        if len(kv_cache) != self.num_layers or len(crossattn_cache) != self.num_layers:
            raise ValueError("ABot cache lists must contain one entry per transformer layer")
        embedded = self.patch_embedding(x)
        action = self.act_control_adapter(act_context.to(device=x.device, dtype=embedded.dtype))
        if action.shape != embedded.shape:
            raise ValueError(
                f"ABot action adapter output {tuple(action.shape)} does not match latent tokens {tuple(embedded.shape)}"
            )
        embedded = embedded + action * act_context_scale
        grid_size = tuple(int(value) for value in embedded.shape[2:])
        tokens = rearrange(embedded, "b c f h w -> b (f h w) c")
        time_embedding = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(tokens.dtype)
        )
        time_embedding = time_embedding.unflatten(0, timestep.shape)
        time_modulation = self.time_projection(time_embedding).unflatten(2, (6, self.dim))
        context = self.text_embedding(context)
        freqs = self._frequencies(tokens.device)
        for index, block in enumerate(self.blocks):
            tokens = block(
                tokens,
                time_modulation,
                context,
                grid_size,
                freqs,
                kv_cache[index],
                crossattn_cache[index],
                current_start,
            )
        return self._unpatchify(self.head(tokens, time_embedding.unsqueeze(2)), grid_size)

    @staticmethod
    def state_dict_converter() -> "ABotWorldDiTStateDictConverter":
        return ABotWorldDiTStateDictConverter()


class ABotWorldDiTStateDictConverter:
    """The published ABot safetensor already uses native parameter names."""

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return state_dict, {}

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return state_dict, {}
