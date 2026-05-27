from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh

from telefuser.core.base_model import BaseModel
from telefuser.distributed.parallel_shard import (
    sequence_parallel_shard,
    sequence_parallel_unshard,
)
from telefuser.models.video_projector import Causal_LQ4x_Proj
from telefuser.ops.attention import attention as attn_func
from telefuser.ops.attention.local_sparse_attn import (
    distributed_local_sparse_attention,
    local_sparse_attention,
)
from telefuser.ops.normalization import LayerNorm, RMSNorm, modulate
from telefuser.platforms import current_platform
from telefuser.utils.model_weight import hash_state_dict_keys


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Create sinusoidal position embeddings."""
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0) -> tuple:
    """Precompute 3D RoPE frequencies for video (frame, height, width)."""
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    """Precompute 1D RoPE frequencies using complex numbers."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply RoPE (Rotary Position Embedding) to input tensor."""
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class SelfAttention(nn.Module):
    """Self-attention with RoPE and local sparse attention support."""

    usp_flag = False

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.local_attn_mask = None

    def async_usp_forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        f: int | None = None,
        h: int | None = None,
        w: int | None = None,
        topk: int | None = None,
        kv_len: int | None = None,
        is_stream: bool = False,
        pre_cache_k: torch.Tensor | None = None,
        pre_cache_v: torch.Tensor | None = None,
        local_range: int = 9,
        win_size: tuple[int, int, int] = (2, 8, 8),
        device_mesh: DeviceMesh | None = None,
    ):
        """Async Ulysses-style sequence parallel forward."""
        from telefuser.distributed.device_mesh import get_ulysses_group, get_ulysses_world_size
        from telefuser.distributed.ulysses_comm import ulysses_gather_heads, ulysses_scatter_heads

        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f == 2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f == 6, "start f must be 6"

        sp_size = get_ulysses_world_size(device_mesh)
        assert self.num_heads % sp_size == 0, f"num heads {self.num_heads} cannot be divided by sp size {sp_size}"
        sp_group = get_ulysses_group(device_mesh=device_mesh)
        v = self.v(x)
        v_4d = rearrange(v, "b s (h d) -> b s h d", h=self.num_heads)
        v_wait = ulysses_scatter_heads(v_4d, sp_group)
        q = self.norm_q(self.q(x))
        q = rope_apply(q, freqs, self.num_heads)
        q_4d = rearrange(q, "b s (h d) -> b s h d", h=self.num_heads)
        q_wait = ulysses_scatter_heads(q_4d, sp_group)
        k = self.norm_k(self.k(x))
        k = rope_apply(k, freqs, self.num_heads)
        k_4d = rearrange(k, "b s (h d) -> b s h d", h=self.num_heads)
        k_wait = ulysses_scatter_heads(k_4d, sp_group)
        q_4d = q_wait()
        k_4d = k_wait()
        v_4d = v_wait()
        q = rearrange(q_4d, "b s h d -> b s (h d)")
        k = rearrange(k_4d, "b s h d -> b s (h d)")
        v = rearrange(v_4d, "b s h d -> b s (h d)")
        num_heads = self.num_heads // sp_size
        D = D // sp_size
        x, cache_k, cache_v = local_sparse_attention(
            q, k, v, B, f, h, w, D, topk, local_range, num_heads, kv_len, pre_cache_k, pre_cache_v, win=win_size
        )
        x_4d = rearrange(x, "b s (h d) -> b s h d", h=num_heads)
        x_wait = ulysses_gather_heads(x_4d, sp_group, num_heads=self.num_heads)
        x_4d = x_wait()
        x = rearrange(x_4d, "b s h d -> b s (h d)")
        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        f: int | None = None,
        h: int | None = None,
        w: int | None = None,
        topk: int | None = None,
        kv_len: int | None = None,
        is_stream: bool = False,
        pre_cache_k: torch.Tensor | None = None,
        pre_cache_v: torch.Tensor | None = None,
        local_range: int = 9,
        win_size: tuple[int, int, int] = (2, 8, 8),
        device_mesh: DeviceMesh | None = None,
    ):
        if self.usp_flag:
            return self.async_usp_forward(
                x, freqs, f, h, w, topk, kv_len, is_stream, pre_cache_k, pre_cache_v, local_range, win_size, device_mesh
            )
        return self.default_forward(
            x, freqs, f, h, w, topk, kv_len, is_stream, pre_cache_k, pre_cache_v, local_range, win_size, device_mesh
        )

    def default_forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        f: int | None = None,
        h: int | None = None,
        w: int | None = None,
        topk: int | None = None,
        kv_len: int | None = None,
        is_stream: bool = False,
        pre_cache_k: torch.Tensor | None = None,
        pre_cache_v: torch.Tensor | None = None,
        local_range: int = 9,
        win_size: tuple[int, int, int] = (2, 8, 8),
        device_mesh: DeviceMesh | None = None,
    ):
        B, L, D = x.shape
        if is_stream and pre_cache_k is not None and pre_cache_v is not None:
            assert f == 2, "f must be 2"
        if is_stream and (pre_cache_k is None or pre_cache_v is None):
            assert f == 6, "start f must be 6"

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if self.usp_flag:
            x, cache_k, cache_v = distributed_local_sparse_attention(
                q,
                k,
                v,
                B,
                f,
                h,
                w,
                D,
                topk,
                local_range,
                self.num_heads,
                kv_len,
                pre_cache_k,
                pre_cache_v,
                win_size,
                device_mesh,
            )
        else:
            x, cache_k, cache_v = local_sparse_attention(
                q, k, v, B, f, h, w, D, topk, local_range, self.num_heads, kv_len, pre_cache_k, pre_cache_v, win_size
            )
        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning with persistent KV cache."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.cache_k: torch.Tensor | None = None
        self.cache_v: torch.Tensor | None = None

    @torch.no_grad()
    def init_cache(self, ctx: torch.Tensor):
        """Initialize persistent KV cache from text embeddings.

        Args:
            ctx: [B, S_ctx, dim] context tensor from text embedding.
        """
        self.cache_k = self.norm_k(self.k(ctx))
        self.cache_v = self.v(ctx)

    def clear_cache(self):
        """Clear persistent KV cache."""
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        assert self.cache_k is not None and self.cache_v is not None
        k = self.cache_k
        v = self.cache_v
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        x = attn_func(q, k, v, input_layout="BSND", output_layout="BSND")
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        return self.o(x)


class GateModule(nn.Module):
    """Gated residual connection."""

    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual


class DiTBlock(nn.Module):
    """Diffusion Transformer block with self-attention, cross-attention, and FFN."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps)

        self.norm1 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def enable_usp(self):
        setattr(self, "use_usp", True)
        setattr(self.self_attn, "use_usp", True)

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        f: int | None = None,
        h: int | None = None,
        w: int | None = None,
        topk: int | None = None,
        kv_len: int | None = None,
        is_stream: bool = False,
        pre_cache_k: torch.Tensor | None = None,
        pre_cache_v: torch.Tensor | None = None,
        local_range: int = 9,
        win_size: tuple[int, int, int] = (2, 8, 8),
        device_mesh: DeviceMesh | None = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_attn_output, self_attn_cache_k, self_attn_cache_v = self.self_attn(
            input_x,
            freqs,
            f,
            h,
            w,
            topk,
            kv_len,
            is_stream,
            pre_cache_k,
            pre_cache_v,
            local_range,
            win_size,
            device_mesh,
        )

        x = self.gate(x, gate_msa, self_attn_output)
        x = x + self.cross_attn(self.norm3(x))
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        if is_stream:
            return x, self_attn_cache_k, self_attn_cache_v
        return x


class MLP(torch.nn.Module):
    """MLP with optional positional embedding."""

    def __init__(self, in_dim: int, out_dim: int, has_pos_emb: bool = False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    """Output head with modulation."""

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class FlashVSRModel(BaseModel):
    """FlashVSR DiT model for video super-resolution."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.freq_dim = freq_dim
        self.patch_size = patch_size

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([DiTBlock(dim, num_heads, ffn_dim, eps) for _ in range(num_layers)])
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        self._cross_kv_initialized = False
        self.LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
        self.ctx = nn.Parameter(torch.zeros((1, 512, 4096), dtype=torch.bfloat16))
        self.pre_cache_k = [None] * self.num_layers
        self.pre_cache_v = [None] * self.num_layers

    def clear_cross_kv(self):
        """Clear cross-attention KV cache."""
        for blk in self.blocks:
            blk.cross_attn.clear_cache()
        self._cross_kv_initialized = False

    def clear_kv_cache(self):
        """Clear self-attention KV cache."""
        self.pre_cache_k = [None] * self.num_layers
        self.pre_cache_v = [None] * self.num_layers
        current_platform.empty_cache()

    @torch.no_grad()
    def init_ctx_and_time_embedding(self):
        """Initialize text context and time embeddings."""
        ctx_txt = self.text_embedding(self.ctx)
        for blk in self.blocks:
            blk.cross_attn.init_cache(ctx_txt)
        self._cross_kv_initialized = True
        self.timestep = torch.tensor([1000.0], dtype=torch.bfloat16, device=self.ctx.device)
        self.t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, self.timestep))
        self.t_mod = self.time_projection(self.t).unflatten(1, (6, self.dim))

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def clean_LQ_proj_in_cache(self):
        """Clear LQ projector cache."""
        self.LQ_proj_in.clear_cache()

    def proj_LQ_video_clip(
        self,
        video_clip: torch.Tensor,
        tile: bool = False,
        tile_size: tuple[int, int] = (512, 512),
        tile_stride: tuple[int, int] = (256, 256),
    ):
        """Project low-quality video clip."""
        if tile:
            return self.LQ_proj_in.tile_stream_forward(
                video_clip=video_clip, tile_size=tile_size, tile_stride=tile_stride
            )
        return self.LQ_proj_in.stream_forward(video_clip=video_clip)

    def forward(
        self,
        x: torch.Tensor,
        LQ_latents: torch.Tensor | None = None,
        is_stream: bool = False,
        topk_ratio: float = 2.0,
        kv_ratio: float = 3.0,
        cur_process_idx: int = 0,
        local_range: int = 9,
        win_size: tuple[int, int, int] = (2, 8, 8),
        offload_kvcache: bool = False,
    ) -> torch.Tensor:
        if not self._cross_kv_initialized:
            self.init_ctx_and_time_embedding()
        x, (f, h, w) = self.patchify(x)
        window_size = h * w // (math.prod(win_size[1:]))
        square_num = window_size * window_size
        topk = int(square_num * topk_ratio) - 1
        kv_len = int(kv_ratio)

        # Build RoPE frequencies with frame offset for streaming
        if cur_process_idx == 0:
            freqs = (
                torch.cat(
                    [
                        self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                        self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                    ],
                    dim=-1,
                )
                .reshape(f * h * w, 1, -1)
                .to(x.device)
            )
        else:
            freqs = (
                torch.cat(
                    [
                        self.freqs[0][4 + cur_process_idx * 2 : 4 + cur_process_idx * 2 + f]
                        .view(f, 1, 1, -1)
                        .expand(f, h, w, -1),
                        self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                    ],
                    dim=-1,
                )
                .reshape(f * h * w, 1, -1)
                .to(x.device)
            )

        if self.usp_flag:
            sequence_parallel_shard(self.device_mesh, [x, freqs, LQ_latents[0]], seq_dims=[1, 0, 1])

        for block_id, block in enumerate(self.blocks):
            if LQ_latents is not None and block_id < len(LQ_latents):
                x = x + LQ_latents[block_id].to(x.device)
            x, last_pre_cache_k, last_pre_cache_v = block(
                x,
                self.t_mod,
                freqs,
                f,
                h,
                w,
                topk,
                kv_len=kv_len,
                is_stream=is_stream,
                pre_cache_k=self.pre_cache_k[block_id] if self.pre_cache_k is not None else None,
                pre_cache_v=self.pre_cache_v[block_id] if self.pre_cache_v is not None else None,
                local_range=local_range,
                win_size=win_size,
                device_mesh=self.device_mesh,
            )
            if self.pre_cache_k is not None:
                self.pre_cache_k[block_id] = last_pre_cache_k.cpu() if offload_kvcache else last_pre_cache_k
            if self.pre_cache_v is not None:
                self.pre_cache_v[block_id] = last_pre_cache_v.cpu() if offload_kvcache else last_pre_cache_v

        x = self.head(x, self.t)
        if self.usp_flag:
            (x,) = sequence_parallel_unshard(self.device_mesh, (x,), seq_dims=(1,), seq_lens=(f * h * w,))
        x = self.unpatchify(x, (f, h, w))
        return x

    def enable_usp(self):
        self.usp_flag = True
        SelfAttention.usp_flag = True

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()


class WanModelStateDictConverter:
    """State dict converter for FlashVSR model."""

    def __init__(self):
        pass

    def from_official(self, state_dict: dict) -> tuple[dict, dict]:
        print(f"model hash is {hash_state_dict_keys(state_dict)}")
        if hash_state_dict_keys(state_dict) == "0f889085aa6209c79f284d963d6cbe95":
            config = {
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict, config


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "0f889085aa6209c79f284d963d6cbe95", ["flashvsr_dit"], [FlashVSRModel], "official")
