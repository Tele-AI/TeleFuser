"""
Concise re-implementation of
``https://github.com/openai/CLIP'' and
``https://github.com/mlfoundations/open_clip''.
"""

from __future__ import annotations

import math
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange

from telefuser.core.base_model import BaseModel
from telefuser.ops.attention import attention as attn_func

from .xlm_roberta import XLMRoberta


def pos_interpolate(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Interpolate position embeddings to target sequence length."""
    if pos.size(1) == seq_len:
        return pos
    else:
        src_grid = int(math.sqrt(pos.size(1)))
        tar_grid = int(math.sqrt(seq_len))
        n = pos.size(1) - src_grid * src_grid
        return torch.cat(
            [
                pos[:, :n],
                F.interpolate(
                    pos[:, n:].float().reshape(1, src_grid, src_grid, -1).permute(0, 3, 1, 2),
                    size=(tar_grid, tar_grid),
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .transpose(1, 2),
            ],
            dim=1,
        )


class QuickGELU(nn.Module):
    """Fast GELU approximation using sigmoid."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """LayerNorm that preserves input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self, dim: int, num_heads: int, causal: bool = False, attn_dropout: float = 0.0, proj_dropout: float = 0.0
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] input tensor.
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        q, k, v = self.to_qkv(x).view(b, s, 3, n, d).unbind(2)

        x = attn_func(q, k, v, input_layout="BSND", output_layout="BSND")
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        x = x.reshape(b, s, c)

        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class SwiGLU(nn.Module):
    """SwiGLU activation for improved efficiency."""

    def __init__(self, dim: int, mid_dim: int):
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class AttentionBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        post_norm: bool = False,
        causal: bool = False,
        activation: str = "quick_gelu",
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        assert activation in ["quick_gelu", "gelu", "swi_glu"]
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout, proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        if activation == "swi_glu":
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                QuickGELU() if activation == "quick_gelu" else nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim),
                nn.Dropout(proj_dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Module):
    """Attention-based pooling with CLS token."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        activation: str = "gelu",
        proj_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        self.norm_eps = norm_eps

        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == "quick_gelu" else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, C] input tensor.
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        q = self.to_q(self.cls_embedding).view(1, 1, n, d).expand(b, -1, -1, -1)
        k, v = self.to_kv(x).view(b, s, 2, n, d).unbind(2)

        x = attn_func(q, k, v, input_layout="BSND", output_layout="BSND")
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        x = x.reshape(b, 1, c)
        x = x.reshape(b, 1, c)

        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)

        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class VisionTransformer(nn.Module):
    """Vision Transformer for image encoding."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        mlp_ratio: int = 4,
        out_dim: int = 512,
        num_heads: int = 12,
        num_layers: int = 12,
        pool_type: str = "token",
        pre_norm: bool = True,
        post_norm: bool = False,
        activation: str = "quick_gelu",
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        if image_size % patch_size != 0:
            print("[WARNING] image_size is not divisible by patch_size", flush=True)
        assert pool_type in ("token", "token_fc", "attn_pool")
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.post_norm = post_norm
        self.norm_eps = norm_eps

        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=not pre_norm)
        if pool_type in ("token", "token_fc"):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(
            gain * torch.randn(1, self.num_patches + (1 if pool_type in ("token", "token_fc") else 0), dim)
        )
        self.dropout = nn.Dropout(embedding_dropout)

        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    dim, mlp_ratio, num_heads, post_norm, False, activation, attn_dropout, proj_dropout, norm_eps
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        if pool_type == "token":
            self.head = nn.Parameter(gain * torch.randn(dim, out_dim))
        elif pool_type == "token_fc":
            self.head = nn.Linear(dim, out_dim)
        elif pool_type == "attn_pool":
            self.head = AttentionPool(dim, mlp_ratio, num_heads, activation, proj_dropout, norm_eps)

    def forward(self, x: torch.Tensor, interpolation: bool = False, use_31_block: bool = False) -> torch.Tensor:
        b = x.size(0)

        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        if self.pool_type in ("token", "token_fc"):
            x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        if interpolation:
            e = pos_interpolate(self.pos_embedding, x.size(1))
        else:
            e = self.pos_embedding
        x = self.dropout(x + e)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        if use_31_block:
            x = self.transformer[:-1](x)
            return x
        else:
            x = self.transformer(x)
            return x


class XLMRobertaWithHead(XLMRoberta):
    """XLMRoberta with additional projection head."""

    def __init__(self, **kwargs):
        self.out_dim = kwargs.pop("out_dim")
        super().__init__(**kwargs)

        mid_dim = (self.dim + self.out_dim) // 2
        self.head = nn.Sequential(
            nn.Linear(self.dim, mid_dim, bias=False), nn.GELU(), nn.Linear(mid_dim, self.out_dim, bias=False)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = super().forward(ids)

        mask = ids.ne(self.pad_id).unsqueeze(-1).to(x)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        x = self.head(x)
        return x


class XLMRobertaCLIP(nn.Module):
    """CLIP-style model with XLMRoberta text encoder and Vision Transformer."""

    def __init__(
        self,
        embed_dim: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        vision_dim: int = 1280,
        vision_mlp_ratio: int = 4,
        vision_heads: int = 16,
        vision_layers: int = 32,
        vision_pool: str = "token",
        vision_pre_norm: bool = True,
        vision_post_norm: bool = False,
        activation: str = "gelu",
        vocab_size: int = 250002,
        max_text_len: int = 514,
        type_size: int = 1,
        pad_id: int = 1,
        text_dim: int = 1024,
        text_heads: int = 16,
        text_layers: int = 24,
        text_post_norm: bool = True,
        text_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_post_norm = text_post_norm
        self.norm_eps = norm_eps

        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps,
        )
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))

    def forward(self, imgs: torch.Tensor, txt_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: [B, 3, H, W] of torch.float32, normalized with:
                - mean: [0.48145466, 0.4578275, 0.40821073]
                - std: [0.26862954, 0.26130258, 0.27577711]
            txt_ids: [B, L] of torch.long, encoded by CLIPTokenizer.
        """
        xi = self.visual(imgs)
        return xi

    def param_groups(self):
        """Get parameter groups for optimizer (weight decay handling)."""
        groups = [
            {
                "params": [p for n, p in self.named_parameters() if "norm" in n or n.endswith("bias")],
                "weight_decay": 0.0,
            },
            {"params": [p for n, p in self.named_parameters() if not ("norm" in n or n.endswith("bias"))]},
        ]
        return groups


def _clip(
    pretrained_name: str | None = None,
    model_cls: Type[XLMRobertaCLIP] = XLMRobertaCLIP,
    dtype: torch.dtype = torch.float32,
    **kwargs,
):
    """Initialize CLIP model with transforms."""
    model = model_cls(**kwargs)
    model = model.to(dtype=dtype)
    output = (model,)

    if "siglip" in pretrained_name.lower():
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    transforms = T.Compose(
        [
            T.Resize((model.image_size, model.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    output += (transforms,)

    return output


def clip_xlm_roberta_vit_h_14(pretrained_name: str = "open-clip-xlm-roberta-large-vit-huge-14", **kwargs):
    """Create XLM-Roberta + ViT-H/14 CLIP model."""
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        activation="gelu",
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        text_post_norm=True,
        text_dropout=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
    )
    cfg.update(**kwargs)
    return _clip(pretrained_name, XLMRobertaCLIP, **cfg)


class WanImageEncoder(BaseModel):
    """CLIP-based image encoder for Wan video generation."""

    def __init__(self):
        super().__init__()
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(dtype=torch.float32)

    def encode_image(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Encode images to feature tensors."""
        size = (self.model.image_size,) * 2
        images = torch.cat(
            [
                F.interpolate(
                    u,
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                for u in images
            ]
        )
        images = self.transforms.transforms[-1](images.mul_(0.5).add_(0.5))

        out = self.model.visual(images, use_31_block=True)
        return out

    @staticmethod
    def state_dict_converter():
        return WanImageEncoderStateDictConverter()


class WanImageEncoderStateDictConverter:
    """State dict converter for Wan image encoder."""

    def __init__(self):
        pass

    def from_diffusers(self, state_dict: dict) -> dict:
        return state_dict

    def from_official(self, state_dict: dict) -> dict:
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.startswith("textual."):
                continue
            name = "model." + name
            state_dict_[name] = param
        return state_dict_


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(
    None, "5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "official"
)
