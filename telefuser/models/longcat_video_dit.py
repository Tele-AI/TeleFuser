from __future__ import annotations

import gc
import math

import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from telefuser.core.base_model import BaseModel
from telefuser.distributed.device_mesh import get_ulysses_group
from telefuser.distributed.parallel_shard import (
    sequence_parallel_shard,
    sequence_parallel_unshard,
)
from telefuser.distributed.ulysses_comm import ulysses_gather_heads, ulysses_scatter_heads
from telefuser.offload import (
    AutoWrappedLinear,
    AutoWrappedModule,
    WanAutoCastLayerNorm,
    enable_sequential_cpu_offload,
)
from telefuser.offload.async_offload import AsyncOffloadManager
from telefuser.ops.attention import attention as attn_func
from telefuser.ops.attention.bsa import flash_attn_bsa_3d as _flash_attn_bsa_3d
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.utils.lora_network import decode_lora_module_name


class RMSNorm_FP32(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def broadcat(tensors: list[torch.Tensor], dim: int = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), (
        "invalid dimensions for broadcastable concatentation"
    )
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x: torch.Tensor):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int):
        """Rotary positional embedding for 3D
        Reference : https://blog.eleuther.ai/rotary-embeddings/
        Paper: https://arxiv.org/pdf/2104.09864.pdf
        Args:
            dim: Dimension of embedding
            base: Base value for exponential
        """
        super().__init__()
        self.head_dim = head_dim
        assert self.head_dim % 8 == 0, "Dim must be a multiply of 8 for 3D RoPE."
        # We take the assumption that the longest side of grid will not larger than 512,
        # i.e, 512 * 8 = 4098 input pixels
        self.base = 10000
        self.freqs_dict = {}

    def register_grid_size(self, grid_size: tuple[int, int, int]):
        if grid_size not in self.freqs_dict:
            self.freqs_dict.update({grid_size: self.precompute_freqs_cis_3d(grid_size)})

    def forward(self, grid_size: tuple[int, int, int]):
        self.register_grid_size(grid_size)
        return self.freqs_dict[grid_size]

    def precompute_freqs_cis_3d(self, grid_size: tuple[int, int, int]):
        num_frames, height, width = grid_size
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        grid_t = np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)
        grid_h = np.linspace(0, height, height, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(0, width, width, endpoint=False, dtype=np.float32)
        grid_t = torch.from_numpy(grid_t).float()
        grid_h = torch.from_numpy(grid_h).float()
        grid_w = torch.from_numpy(grid_w).float()
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)
        freqs = broadcat(
            (
                freqs_t[:, None, None, :],
                freqs_h[None, :, None, :],
                freqs_w[None, None, :, :],
            ),
            dim=-1,
        )
        # (T H W D)
        freqs = rearrange(freqs, "T H W D -> (T H W) D")
        return freqs


def rope_apply(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    """3D RoPE.

    Args:
        query: [B, head, seq, head_dim]
        key: [B, head, seq, head_dim]
    Returns:
        query and key with the same shape as input.
    """

    q_, k_ = q.float(), k.float()
    seq_num = freqs_cis.shape[0]
    freqs_cis = freqs_cis.float().to(q.device)
    cos, sin = freqs_cis.cos(), freqs_cis.sin()
    cos, sin = rearrange(cos, "n d -> 1 n 1 d"), rearrange(sin, "n d -> 1 n 1 d")
    q_[:, :seq_num] = (q_[:, :seq_num] * cos) + (rotate_half(q_[:, :seq_num]) * sin)
    k_[:, :seq_num] = (k_[:, :seq_num] * cos) + (rotate_half(k_[:, :seq_num]) * sin)

    return q_.type_as(q), k_.type_as(k)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: list[int] | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        shape: tuple[int, int, int] | None = None,
        num_cond_latents: int | None = None,
        return_kv: bool = False,
        attn_impl: str = "sdpa",
        use_usp: bool = False,
        device_mesh: object | None = None,
    ) -> torch.Tensor:
        """Multi-head self-attention with optional BSA and Ulysses sequence parallelism."""
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 1, 3, 4))  # [3, B, S, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if use_usp:
            group = get_ulysses_group(device_mesh)
            q_wait = ulysses_scatter_heads(q, group)
            k_wait = ulysses_scatter_heads(k, group)
            v_wait = ulysses_scatter_heads(v, group)
            q = q_wait()
            k = k_wait()
            v = v_wait()
        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()
        q, k = rope_apply(q, k, freqs_cis)
        # cond mode
        if num_cond_latents is not None and num_cond_latents > 0:
            num_cond_latents_thw = num_cond_latents * (N // shape[0])
            # process the condition tokens
            q_cond = q[:, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :num_cond_latents_thw].contiguous()
            x_cond = attn_func(
                q_cond,
                k_cond,
                v_cond,
                attn_impl=attn_impl,
                input_layout="BSND",
                output_layout="BSND",
            )
            # process the noise tokens
            q_noise = q[:, num_cond_latents_thw:].contiguous()
            x_noise = attn_func(
                q_noise,
                k,
                v,
                attn_impl=attn_impl,
                input_layout="BSND",
                output_layout="BSND",
            )
            # merge x_cond and x_noise
            x = torch.cat([x_cond, x_noise], dim=1).contiguous()
        else:
            if self.enable_bsa and shape is not None:
                # BSA: flash_attn_bsa_3d handles block reorder + gating + attention
                q_bhsd = rearrange(q, "b s n d -> b n s d")
                k_bhsd = rearrange(k, "b s n d -> b n s d")
                v_bhsd = rearrange(v, "b s n d -> b n s d")
                x = _flash_attn_bsa_3d(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    latent_shape_q=shape,
                    latent_shape_k=shape,
                    **self.bsa_params,
                )
                x = rearrange(x, "b n s d -> b s n d")
            else:
                x = attn_func(q, k, v, attn_impl=attn_impl, input_layout="BSND", output_layout="BSND")
        if use_usp:
            x_wait = ulysses_gather_heads(x, group, num_heads=self.num_heads)
            x = x_wait()
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        x = self.proj(x)

        if return_kv:
            return x, (k_cache, v_cache)
        else:
            return x

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        shape: tuple[int, int, int] | None = None,
        num_cond_latents: int | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_impl: str = "sdpa",
        use_usp: bool = False,
        device_mesh: object | None = None,
    ) -> torch.Tensor:
        """Multi-head attention with pre-computed KV cache for inference acceleration."""
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 1, 3, 4))  # [3, B, s, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if use_usp:
            group = get_ulysses_group(device_mesh)
            q_wait = ulysses_scatter_heads(q, group)
            k_wait = ulysses_scatter_heads(k, group)
            v_wait = ulysses_scatter_heads(v, group)
            q = q_wait()
            k = k_wait()
            v = v_wait()
        k_cache, v_cache = kv_cache
        assert k_cache.shape[0] == v_cache.shape[0] and k_cache.shape[0] in [1, B]
        if k_cache.shape[0] == 1:
            k_cache = k_cache.repeat(B, 1, 1, 1)
            v_cache = v_cache.repeat(B, 1, 1, 1)

        if num_cond_latents is not None and num_cond_latents > 0:
            k_full = torch.cat([k_cache, k], dim=1).contiguous()
            v_full = torch.cat([v_cache, v], dim=1).contiguous()
            q_padding = torch.cat([torch.empty_like(k_cache), q], dim=1).contiguous()
            q_padding, k_full = rope_apply(q_padding, k_full, freqs_cis)
            q = q_padding[:, -q.shape[1] :].contiguous()

        x = attn_func(
            q,
            k_full,
            v_full,
            attn_impl=attn_impl,
            input_layout="BSND",
            output_layout="BSND",
        )
        if use_usp:
            x_wait = ulysses_gather_heads(x, group, num_heads=self.num_heads)
            x = x_wait()
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        x = self.proj(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
    ):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, kv_seqlen: list[int], attn_impl: str = "sdpa"):
        """
        x: [B, N, C]
        cond: [B, M, C]
        """
        B, N, C = x.shape
        assert C == self.dim and cond.shape[2] == self.dim

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        q, k = self.q_norm(q), self.k_norm(k)

        x = attn_func(q, k, v, input_layout="BSND", output_layout="BSND", attn_impl=attn_impl)

        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        x = self.proj(x)
        return x


class LayerNorm_FP32(nn.LayerNorm):
    def __init__(self, dim: int, eps: float, elementwise_affine: bool):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

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


def modulate_fp32(norm_func: nn.Module, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    # Suppose x is (B, N, D), shift is (B, -1, D), scale is (B, -1, D)
    # ensure the modulation params be fp32
    assert shift.dtype == torch.float32, scale.dtype == torch.float32
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    x = x.to(dtype)
    return x


class FinalLayer_FP32(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size: int, num_patch: int, out_channels: int, adaln_tembed_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patch = num_patch
        self.out_channels = out_channels
        self.adaln_tembed_dim = adaln_tembed_dim

        self.norm_final = LayerNorm_FP32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, t: torch.Tensor, latent_shape: tuple[int, int, int]):
        # timestep shape: [B, T, C]
        assert t.dtype == torch.float32
        B, N, C = x.shape
        T, _, _ = latent_shape

        with amp.autocast("cuda", dtype=torch.float32):
            shift, scale = self.adaLN_modulation(t).unsqueeze(2).chunk(2, dim=-1)  # [B, T, 1, C]
            x = modulate_fp32(self.norm_final, x.view(B, T, -1, C), shift, scale).view(B, N, C)
            x = self.linear(x)
        return x


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, t_embed_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor, dtype: torch.dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    """

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, caption: torch.Tensor):
        caption = self.y_proj(caption)
        return caption


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (2, 4, 4),
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
        flatten: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        B, C, T, H, W = x.shape
        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class LongCatSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_bsa: bool = False,
        bsa_params: dict | None = None,
        cp_split_hw: list[int] | None = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # scale and gate modulation
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True))

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw,
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        y_seqlen: list[int],
        latent_shape: tuple[int, int, int],
        freqs_cis: torch.Tensor,
        num_cond_latents: int | None = None,
        return_kv: bool = False,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        skip_crs_attn: bool = False,
        attn_impl: str = "sdpa",
        use_usp: bool = False,
        device_mesh: object | None = None,
    ):
        """
        x: [B, N, C]
        y: [1, N_valid_tokens, C]
        t: [B, T, C_t]
        y_seqlen: [B]; type of a list
        latent_shape: latent shape of a single item
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape  # S != T*H*W in case of CP split on H*W.

        # compute modulation params in fp32
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
            )  # [B, T, 1, C]

        # self attn with modulation
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)

        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(
                x_m,
                shape=latent_shape,
                num_cond_latents=num_cond_latents,
                kv_cache=kv_cache,
                attn_impl=attn_impl,
                freqs_cis=freqs_cis,
                use_usp=use_usp,
                device_mesh=device_mesh,
            )
        else:
            attn_outputs = self.attn(
                x_m,
                shape=latent_shape,
                num_cond_latents=num_cond_latents,
                return_kv=return_kv,
                attn_impl=attn_impl,
                freqs_cis=freqs_cis,
                use_usp=use_usp,
                device_mesh=device_mesh,
            )

        if return_kv:
            x_s, kv_cache = attn_outputs
        else:
            x_s = attn_outputs

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            x = x + (gate_msa * x_s.view(B, -1, N // T, C)).view(B, -1, C)  # [B, N, C]
        x = x.to(x_dtype)

        # cross attn
        if not skip_crs_attn:
            # if use_usp and kv_cache is None:
            #     raise RuntimeError("usp should run with kv cache")
            if kv_cache is not None:
                num_cond_latents = 0
            x[:, num_cond_latents:] = x[:, num_cond_latents:] + self.cross_attn(
                self.pre_crs_attn_norm(x[:, num_cond_latents:]),
                y,
                y_seqlen,
                attn_impl=attn_impl,
            )

        # ffn with modulation
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N // T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_s = self.ffn(x_m)
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            x = x + (gate_mlp * x_s.view(B, -1, N // T, C)).view(B, -1, C)  # [B, N, C]
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        else:
            return x


class LongCatVideoTransformer3DModel(BaseModel):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        # default params
        patch_size: tuple[int] = (1, 2, 2),
        # attention config
        enable_bsa: bool = False,
        bsa_params: dict | None = None,
        cp_split_hw: list[int] | None = None,
        text_tokens_zero_pad: bool = True,
    ) -> None:
        super().__init__()

        if bsa_params is None:
            bsa_params = {"sparsity": 0.9375, "chunk_3d_shape_q": [4, 4, 4], "chunk_3d_shape_k": [4, 4, 4]}
        if cp_split_hw is None:
            cp_split_hw = [1, 1]

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(
            t_embed_dim=adaln_tembed_dim,
            frequency_embedding_size=frequency_embedding_size,
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer_FP32(
            hidden_size,
            np.prod(self.patch_size),
            out_channels,
            adaln_tembed_dim,
        )

        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

        self.lora_dict = {}
        self.active_loras = []
        # async swap
        self.weights_stream_mgr = None
        self.async_offload_manager = None
        self.layer_name_list = ["blocks"]
        self.clean_cuda_cache = True
        self.use_usp = False
        self.kv_cache_dict = {}
        self.rope_3d = RotaryPositionalEmbedding(hidden_size // num_heads)

    def enable_async_offload(self, device: torch.device, offload_config: object):
        logger.info("enable async offload for longcat video dit")
        self.async_offload_manager = AsyncOffloadManager(
            self.blocks,
            enabled=True,
            offload_ratio=offload_config.offload_ratio,
            prefetch_size=offload_config.prefetch_size,
            device=device,
            pin_cpu_memory=offload_config.pin_cpu_memory,
        )
        self.async_offload_flag = True

    def enable_sequential_cpu_offload(self, device: torch.device, torch_dtype: torch.dtype):
        """Enable sequential CPU offloading for memory efficiency."""
        dtype = next(iter(self.parameters())).dtype
        enable_sequential_cpu_offload(
            self,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm_FP32: AutoWrappedModule,
                LayerNorm_FP32: WanAutoCastLayerNorm,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=device,
                computation_dtype=torch_dtype,
                computation_device=device,
            ),
        )

    def cache_clean_latents(
        self,
        cond_latents: torch.Tensor,
        offload_kv_cache: bool,
        fsdp_forward_fn: object | None = None,
    ):
        """Pre-compute KV cache for conditioning latents.

        Args:
            cond_latents: Conditioning latents tensor
            offload_kv_cache: Whether to offload KV cache to CPU
            fsdp_forward_fn: Optional callable for FSDP-wrapped forward. When the model
                is wrapped with FSDP, self.forward() bypasses FSDP's allgather hooks.
                Pass the FSDP wrapper's forward to ensure proper parameter allgathering.
        """
        model_max_length = 512
        text_encoder_dim = 4096
        model_dtype = next(self.parameters()).dtype
        cond_latents = cond_latents.to(dtype=model_dtype)
        timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2]).to(
            device=cond_latents.device, dtype=cond_latents.dtype
        )
        # make null prompt tensor(skip_crs_attn=True, so tensors below will not be actually used)
        empty_embeds = torch.zeros(
            [cond_latents.shape[0], model_max_length, text_encoder_dim],
            device=cond_latents.device,
            dtype=cond_latents.dtype,
        )
        forward_fn = fsdp_forward_fn if fsdp_forward_fn is not None else self.forward
        _, kv_cache_dict = forward_fn(
            hidden_states=cond_latents,
            timestep=timestep,
            encoder_hidden_states=empty_embeds,
            return_kv=True,
            skip_crs_attn=True,
            offload_kv_cache=offload_kv_cache,
        )
        self.kv_cache_dict = kv_cache_dict

    def clear_cache(self):
        self.kv_cache_dict = {}

        gc.collect()
        current_platform.empty_cache()

    def enable_usp(self):
        logger.info("longcat video dit enable usp")
        self.use_usp = True

    def load_lora(
        self,
        lora_path: str,
        lora_key: str,
        multiplier: float = 1.0,
        lora_network_dim: int = 128,
        lora_network_alpha: float = 64,
    ):
        """Load a switchable LoRA from safetensors file.

        Unlike LoRALoader.apply_lora() which permanently merges weights,
        this stores the LoRA as a separate network that can be dynamically
        enabled/disabled via enable_loras()/disable_all_loras().

        Args:
            lora_path: Path to .safetensors file.
            lora_key: Key to identify this LoRA (e.g. "refinement_lora").
            multiplier: LoRA output multiplier.
            lora_network_dim: LoRA rank dimension.
            lora_network_alpha: LoRA alpha scaling factor.
        """
        from safetensors.torch import load_file

        from telefuser.utils.lora_network import create_lora_network

        lora_network_state_dict = load_file(lora_path, device="cpu")
        lora_network = create_lora_network(
            transformer=self,
            lora_network_state_dict_loaded=lora_network_state_dict,
            multiplier=multiplier,
            network_dim=lora_network_dim,
            network_alpha=lora_network_alpha,
        )
        lora_network.load_state_dict(lora_network_state_dict, strict=True)
        self.lora_dict[lora_key] = lora_network
        logger.info(f"Loaded switchable LoRA '{lora_key}' from {lora_path}")

    def enable_loras(self, lora_key_list: list[str] | None = None):
        self.disable_all_loras()
        if lora_key_list is None:
            return

        module_loras = {}  # {module_name: [lora1, lora2, ...]}
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        for lora_key in lora_key_list:
            if lora_key in self.lora_dict:
                for lora in self.lora_dict[lora_key].loras:
                    lora.to(model_device, dtype=model_dtype, non_blocking=True)
                    module_name = decode_lora_module_name(lora.lora_name)
                    if module_name not in module_loras:
                        module_loras[module_name] = []
                    module_loras[module_name].append(lora)
                self.active_loras.append(lora_key)

        for module_name, loras in module_loras.items():
            module = self._get_module_by_name(module_name)
            if not hasattr(module, "org_forward"):
                module.org_forward = module.forward
            module.forward = self._create_multi_lora_forward(module, loras)

    def _create_multi_lora_forward(self, module: nn.Module, loras: list[object]):
        def multi_lora_forward(x: torch.Tensor, *args, **kwargs):
            weight_dtype = x.dtype
            org_output = module.org_forward(x, *args, **kwargs)

            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
                    total_lora_output += lora_output

            return org_output + total_lora_output

        return multi_lora_forward

    def _get_module_by_name(self, module_name: str):
        try:
            module = self
            for part in module_name.split("."):
                module = getattr(module, part)
            return module
        except AttributeError as e:
            raise ValueError(f"Cannot find module: {module_name}, error: {e}")

    def disable_all_loras(self):
        for name, module in self.named_modules():
            if hasattr(module, "org_forward"):
                module.forward = module.org_forward
                delattr(module, "org_forward")

        for lora_key, lora_network in self.lora_dict.items():
            for lora in lora_network.loras:
                lora.to("cpu")

        self.active_loras.clear()

    def enable_bsa(
        self,
    ):
        for block in self.blocks:
            block.attn.enable_bsa = True

    def disable_bsa(
        self,
    ):
        for block in self.blocks:
            block.attn.enable_bsa = False

    def forward_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        t: torch.Tensor,
        y_seqlens: list[int],
        latent_shape: tuple[int, int, int],
        freqs_cis: torch.Tensor,
        num_cond_latents: int = 0,
        return_kv: bool = False,
        skip_crs_attn: bool = False,
        offload_kv_cache: bool = False,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass through transformer blocks only.

        This method is separated from forward() to allow targeted compilation
        of the compute-intensive block iteration loop.

        Args:
            hidden_states: Input hidden states [B, N, C]
            encoder_hidden_states: Text encoder hidden states [1, N_valid_tokens, C]
            t: Timestep embedding [B, T, C_t]
            y_seqlens: List of valid text token lengths per batch item
            latent_shape: Shape of latent grid (N_t, N_h, N_w)
            freqs_cis: Precomputed 3D RoPE frequencies
            num_cond_latents: Number of conditioning latent frames
            return_kv: Whether to return KV cache
            skip_crs_attn: Whether to skip cross attention
            offload_kv_cache: Whether to offload KV cache to CPU
            attn_impl: Attention implementation type

        Returns:
            If return_kv: tuple of (hidden_states, kv_cache_dict)
            Otherwise: hidden_states tensor
        """
        N_t, N_h, N_w = latent_shape
        kv_cache_dict_ret = {}

        if self.use_usp:
            sequence_parallel_shard(self.device_mesh, [hidden_states], [1], seq_divisions=[N_t])

        for i, block in enumerate(self.blocks):
            block_outputs = block(
                x=hidden_states,
                y=encoder_hidden_states,
                t=t,
                y_seqlen=y_seqlens,
                latent_shape=latent_shape,
                num_cond_latents=num_cond_latents,
                return_kv=return_kv,
                kv_cache=self.kv_cache_dict.get(i, None),
                skip_crs_attn=skip_crs_attn,
                attn_impl=attn_impl,
                freqs_cis=freqs_cis,
                use_usp=self.use_usp,
                device_mesh=self.device_mesh,
            )

            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (
                        kv_cache[0].contiguous(),
                        kv_cache[1].contiguous(),
                    )
            else:
                hidden_states = block_outputs

        if self.use_usp:
            (hidden_states,) = sequence_parallel_unshard(self.device_mesh, [hidden_states], (1,), (N_t * N_h * N_w,))

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        num_cond_latents: int = 0,
        return_kv: bool = False,
        skip_crs_attn: bool = False,
        offload_kv_cache: bool = False,
        attn_impl: str = "sdpa",
        device_mesh: object | None = None,
    ):
        B, _, T, H, W = hidden_states.shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]
        if self.kv_cache_dict:
            freqs_cis = self.rope_3d((N_t + num_cond_latents, N_h, N_w))
        else:
            freqs_cis = self.rope_3d((N_t, N_h, N_w))

        assert self.patch_size[0] == 1, "Currently, 3D x_embedder should not compress the temporal dimension."

        # expand the shape of timestep from [B] to [B, T]
        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t).clone()  # [B, T]
        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype)
        hidden_states = self.x_embedder(hidden_states)  # [B, N, C]

        with amp.autocast(device_type="cuda", dtype=torch.float32):
            t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)  # [B, T, C_t]
        encoder_hidden_states = self.y_embedder(encoder_hidden_states)  # [B, 1, N_token, C]
        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )  # [1, N_valid_tokens, C]
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()  # [B]
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

        # blocks
        blocks_output = self.forward_blocks(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            t=t,
            y_seqlens=y_seqlens,
            latent_shape=(N_t, N_h, N_w),
            freqs_cis=freqs_cis,
            num_cond_latents=num_cond_latents,
            return_kv=return_kv,
            skip_crs_attn=skip_crs_attn,
            offload_kv_cache=offload_kv_cache,
            attn_impl=attn_impl,
        )
        if return_kv:
            hidden_states, kv_cache_dict_ret = blocks_output
        else:
            hidden_states = blocks_output
        hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))  # [B, N, C=T_p*H_p*W_p*C_out]

        # if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
        #     hidden_states = context_parallel_util.gather_cp_2d(
        #                        hidden_states, shape=(N_t, N_h, N_w),
        #                        split_hw=self.cp_split_hw
        #    )

        hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)  # [B, C_out, H, W]

        # cast to float32 for better accuracy
        hidden_states = hidden_states.to(torch.float32)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        else:
            return hidden_states

    def unpatchify(self, x: torch.Tensor, N_t: int, N_h: int, N_w: int):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

    @staticmethod
    def state_dict_converter():
        return LongCatVideoTransformer3DModelDictConverter()

    def get_fsdp_module_names(self):
        return ["blocks"]

    def compile(self, mode: str = "blocks", **kwargs) -> None:
        """Compile model for better performance with torch.compile.

        Args:
            mode: Compilation mode:
                - "blocks": Compile only forward_blocks (default, most effective)
                - "full": Compile entire forward method
            **kwargs: Arguments passed to torch.compile()
        """
        # Import mark_static from torch._dynamo
        try:
            from torch._dynamo import mark_static

            # Mark module classes as static (instance attributes won't change after compile)
            mark_static(LongCatVideoTransformer3DModel)
            mark_static(LongCatSingleStreamBlock)
            mark_static(Attention)
            mark_static(MultiHeadCrossAttention)
            mark_static(FeedForwardSwiGLU)
            mark_static(FinalLayer_FP32)
            mark_static(RotaryPositionalEmbedding)
        except ImportError:
            logger.warning("torch._dynamo.mark_static not available, skipping static marking")

        # Compile based on mode
        if mode == "blocks":
            original_fn = self.forward_blocks
            self.forward_blocks = torch.compile(original_fn, **kwargs)
            logger.info(f"LongCatVideoTransformer3DModel compiled: mode={mode}")
        elif mode == "full":
            # Store original forward for fallback
            self._original_forward = self.forward
            self.forward = torch.compile(self.forward, **kwargs)
            logger.info(f"LongCatVideoTransformer3DModel compiled: mode={mode}")
        else:
            raise ValueError(f"Unknown compile mode: {mode}")


class LongCatVideoTransformer3DModelDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict: dict):
        return state_dict

    def from_official(self, state_dict: dict):
        return state_dict


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(
    None, "8b27900f680d7251ce44e2dc8ae1ffef", ["wan_video_dit"], [LongCatVideoTransformer3DModel], "official"
)
