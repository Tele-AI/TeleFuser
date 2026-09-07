"""Alignment heads for LingBot-VLA v2 multimodal task tokens.

Adapted from the Apache-2.0 licensed LingBot-VLA v2 implementation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


def _feed_forward(dim: int, mult: int = 4) -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def _reshape_tensor(x: torch.Tensor, heads: int) -> torch.Tensor:
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class _PerceiverAttention(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
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

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
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

        q = _reshape_tensor(q, self.heads)
        k = _reshape_tensor(k, self.heads)
        v = _reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(batch_size, latent_length, -1)

        return self.to_out(out)


class _TaskTokenResampler(nn.Module):
    def __init__(
        self,
        dim_in: int = 768,
        dim_mid: int = 1024,
        dim_head: int = 64,
        dim_out: int = 1024,
        num_layers: int = 8,
        num_queries: int = 8,
        num_heads: int = 16,
        ff_mult: int = 4,
    ) -> None:
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
                        _PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                        _feed_forward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
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
        proj_config: Mapping[str, int],
        llm_hidden_size: int = 4096,
    ) -> None:
        super().__init__()

        self.projector = _TaskTokenResampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            num_queries=proj_config["num_backbone_tokens"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        queries = self.projector(llm_feats, queries)
        return queries
