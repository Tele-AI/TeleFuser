"""Native Qwen-Image 2.1 diffusion transformer.

The 2.1 checkpoint is a single-stream transformer.  Text tokens form a causal
prefix and the target image tokens form one bidirectional block.  This module
keeps that computation in TeleFuser instead of delegating the model forward to
Diffusers.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.ops.attention import attention as attn_func
from telefuser.ops.normalization import RMSNorm
from telefuser.utils.model_weight import init_weights_on_device, load_state_dict


class QwenImage21TemporalTimesteps(nn.Module):
    """Cosine/sine timestep embedding used by the 2.1 checkpoint."""

    def __init__(self, timestep_dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        half = timestep_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        args = timestep.float()[:, None] * self.time_factor * self.freqs.to(timestep.device)[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding.to(timestep.dtype)


class QwenImage21TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps()
        self.timestep_embedder = QwenImage21TimestepEmbedding(embedding_dim)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return self.timestep_embedder(self.time_proj(timestep).to(dtype=dtype))


class QwenImage21TimestepEmbedding(nn.Module):
    """Named linear layers matching the Diffusers checkpoint layout."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, embedding_dim, bias=False)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(hidden_states)))


class ZeroCenterRMSNorm(nn.Module):
    """RMSNorm whose stored scale is centered at zero in the checkpoint."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        norm = torch.rsqrt(hidden_states.square().mean(dim=-1, keepdim=True) + self.eps)
        return (hidden_states * norm * (self.weight.float() + 1.0)).to(dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.text_norm = ZeroCenterRMSNorm(context_in_dim, eps)
        self.in_layer = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.act(self.in_layer(self.text_norm(hidden_states))))


class QwenImage21SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.gate_layer = nn.Linear(hidden_size, mlp_hidden_size, bias=False)
        self.out = nn.Linear(mlp_hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate_layer(hidden_states)) * self.proj(hidden_states))


def _rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
    freqs = torch.outer(
        index.float(), 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=index.device).float() / dim)
    )
    return torch.polar(torch.ones_like(freqs), freqs)


class QwenImage21Rope(nn.Module):
    """Three-axis rotary embedding for a text prefix and image token block."""

    def __init__(self, axes_dims: tuple[int, int, int] = (16, 56, 56), theta: int = 10000):
        super().__init__()
        self.axes_dims = axes_dims
        self.theta = theta

    def forward(
        self, img_shapes: list[tuple[int, int, int]], image_pad_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        mask = image_pad_mask.tolist()
        frame_index, height_index, width_index = [], [], []
        cursor = position = 0
        for frames, height, width in img_shapes:
            if frames != 1:
                raise ValueError("Qwen-Image 2.1 supports one frame per condition image")
            block_start = mask.index(True, cursor)
            text_indices = list(range(position, position + block_start - cursor))
            frame_index.extend(text_indices)
            height_index.extend(text_indices)
            width_index.extend(text_indices)
            position += block_start - cursor
            block_length = height * width
            frame_index.extend([position] * block_length)
            height_index.extend(h for h in range(-(height - height // 2), height // 2) for _ in range(width))
            width_index.extend(w for _ in range(height) for w in range(-(width - width // 2), width // 2))
            cursor = block_start + block_length
            position += max(height, width)
        trailing = list(range(position, position + len(mask) - cursor))
        frame_index.extend(trailing)
        height_index.extend(trailing)
        width_index.extend(trailing)
        if len(frame_index) != len(mask):
            raise ValueError("Image shapes do not match the expanded image-token mask")
        frame = torch.tensor(frame_index, device=device)
        height_index = torch.tensor(height_index, device=device)
        width_index = torch.tensor(width_index, device=device)
        return torch.cat(
            [
                _rope_params(frame, self.axes_dims[0], self.theta),
                _rope_params(height_index, self.axes_dims[1], self.theta),
                _rope_params(width_index, self.axes_dims[2], self.theta),
            ],
            dim=-1,
        )


def apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply complex rotary frequencies to ``(B, S, heads, head_dim)`` tensors."""

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = x_complex * freqs.to(x.device)[None, :, None]
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


class QwenImage21Attention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, heads * head_dim, bias=False)
        self.to_k = nn.Linear(dim, heads * head_dim, bias=False)
        self.to_v = nn.Linear(dim, heads * head_dim, bias=False)
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(heads * head_dim, dim, bias=False), nn.Dropout(0.0)])
        self.attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)

    def set_attention_config(self, config: AttentionConfig) -> None:
        self.attention_config = config

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        query = apply_rotary_emb(self.norm_q(query), rotary_emb)
        key = apply_rotary_emb(self.norm_k(key), rotary_emb)
        output = attn_func(
            query,
            key,
            value,
            attention_config=self.attention_config,
            attn_mask=attention_mask,
            input_layout="BSND",
            output_layout="BSND",
        ).flatten(-2)
        return self.to_out[1](self.to_out[0](output))


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, mlp_ratio: int, eps: float):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn = QwenImage21Attention(dim, heads, head_dim, eps)
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.img_mlp = QwenImage21SwiGLU(dim, dim * mlp_ratio)

    @staticmethod
    def _modulate(
        hidden_states: torch.Tensor, params: torch.Tensor, target_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = params.chunk(2, dim=-1)
        scale = _select_modulation(scale, target_mask)
        gate = _select_modulation(gate, target_mask)
        return hidden_states * (1 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        target_mask: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mod_attn, mod_mlp = modulation.chunk(2, dim=-1)
        normed, gate = self._modulate(self.img_norm1(hidden_states), mod_attn, target_mask)
        hidden_states = hidden_states + gate.tanh() * self.attn(normed, rotary_emb, attention_mask)
        normed, gate = self._modulate(self.img_norm2(hidden_states), mod_mlp, target_mask)
        hidden_states = hidden_states + gate.tanh() * self.img_mlp(normed)
        return hidden_states.clip(-65504, 65504) if hidden_states.dtype == torch.float16 else hidden_states


def _select_modulation(params: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    real = params[:-1].unsqueeze(1)
    zero = params[-1:].unsqueeze(0)
    return torch.where(target_mask.view(1, -1, 1), real, zero)


class QwenImage21AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(
        self, hidden_states: torch.Tensor, conditioning: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        scale = _select_modulation(self.linear(self.silu(conditioning)), target_mask)
        return self.norm(hidden_states) * (1 + scale)


class QwenImage21DiT(BaseModel):
    """Native single-stream Qwen-Image 2.1 DiT."""

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
    ):
        super().__init__()
        del patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_condition = causal_condition
        self.pos_embed = QwenImage21Rope(axes_dims_rope)
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(self.inner_dim)
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps)
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = QwenImage21AdaLayerNormContinuous(self.inner_dim, eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)
        self.attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)

    def set_attention_config(self, config: AttentionConfig) -> None:
        self.attention_config = config
        for block in self.transformer_blocks:
            block.attn.set_attention_config(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        encoder_hidden_states_mask: torch.Tensor | None = None,
        img_mask: torch.Tensor | None = None,
        return_dict: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        batch_size, target_tokens, _ = hidden_states.shape
        text_tokens = encoder_hidden_states.shape[1]
        if len(img_shapes) != batch_size or any(shapes != img_shapes[0] for shapes in img_shapes):
            raise ValueError("All batch items must share the same image-token layout")
        target_tokens = math.prod(img_shapes[0][-1])
        if sum(math.prod(shape) for shape in img_shapes[0]) != hidden_states.shape[1]:
            raise ValueError("Image shapes must match condition and target latent tokens")
        if img_mask is None:
            img_mask = torch.zeros(batch_size, text_tokens, dtype=torch.bool, device=hidden_states.device)
        target_mask = torch.ones(batch_size, target_tokens // 4, dtype=torch.bool, device=hidden_states.device)
        full_mask = torch.cat([img_mask, target_mask], dim=1)
        repeats = torch.where(full_mask[0], 4, 1)
        image_pad_mask = torch.repeat_interleave(full_mask[0], repeats)
        embedded_text = self.txt_in(encoder_hidden_states)
        joint = torch.cat(
            [embedded_text, embedded_text.new_zeros(batch_size, target_tokens // 4, self.inner_dim)], dim=1
        ).repeat_interleave(repeats, dim=1)
        joint[:, image_pad_mask] = self.img_in(hidden_states)
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        image_ids = torch.full((joint.shape[1],), -1, dtype=torch.long, device=hidden_states.device)
        cursor = 0
        for image_id, shape in enumerate(img_shapes[0]):
            length = math.prod(shape)
            image_ids[image_positions[cursor : cursor + length]] = image_id
            cursor += length
        rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, hidden_states.device)
        target_token_mask = torch.zeros(joint.shape[1], dtype=torch.bool, device=hidden_states.device)
        target_token_mask[-target_tokens:] = True
        valid_keys = torch.ones(batch_size, joint.shape[1], dtype=torch.bool, device=hidden_states.device)
        if encoder_hidden_states_mask is not None:
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            valid_keys[:, text_positions] = encoder_hidden_states_mask[:, ~img_mask[0]].bool()
        indices = torch.arange(joint.shape[1], device=hidden_states.device)
        same_image = (image_ids[:, None] == image_ids[None, :]) & (image_ids[:, None] >= 0)
        allowed = (indices[:, None] >= indices[None, :]) | same_image
        allowed = allowed[None, None] & valid_keys[:, None, None, :]
        timestep = timestep.to(hidden_states.dtype)
        if self.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        conditioning = self.time_text_embed(timestep, hidden_states.dtype)
        modulation = self.modulation(conditioning)
        for block in self.transformer_blocks:
            joint = block(joint, modulation, target_token_mask, rotary_emb, allowed)
        joint = self.norm_out(joint, conditioning, target_token_mask)
        output = self.proj_out(joint[:, -target_tokens:])
        return (output,) if not return_dict else output

    @staticmethod
    def state_dict_converter() -> "QwenImage21DiTStateDictConverter":
        return QwenImage21DiTStateDictConverter()

    @classmethod
    def from_pretrained(cls, path: str, torch_dtype: torch.dtype = torch.bfloat16) -> "QwenImage21DiT":
        config_path = os.path.join(path, "config.json")
        import json

        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        index_path = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")
        with init_weights_on_device("meta"):
            model = cls(**{key: value for key, value in config.items() if key in cls.__init__.__code__.co_varnames})
        state = load_state_dict(index_path, torch_dtype=torch_dtype)
        model.load_state_dict(state, assign=True)
        return model.to(dtype=torch_dtype).eval()


class QwenImage21DiTStateDictConverter:
    @staticmethod
    def from_diffusers(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return state_dict, {}

    @staticmethod
    def from_official(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return state_dict, {}
