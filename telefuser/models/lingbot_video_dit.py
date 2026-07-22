"""Checkpoint-compatible Dense LingBot-Video transformer modules.

Numerical behavior is adapted from the Apache-2.0 licensed upstream
LingBot-Video transformer implementation.
"""

from __future__ import annotations

from math import prod
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from telefuser.distributed.ulysses_comm import ulysses_all_to_all_split_cat
from telefuser.ops.attention import attention


class LingBotVideoPatchEmbed(nn.Module):
    """Patchify and restore ``[B,C,F,H,W]`` latent tensors."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.in_channels * prod(config.patch_size), config.hidden_size)

    def patchify(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5:
            raise ValueError("LingBot latent must have shape [B,C,F,H,W]")
        _, channels, frames, height, width = latent.shape
        pt, ph, pw = self.config.patch_size
        if channels != self.config.in_channels or frames % pt or height % ph or width % pw:
            raise ValueError("latent shape is incompatible with LingBot patch size")
        latent = latent.reshape(latent.shape[0], channels, frames // pt, pt, height // ph, ph, width // pw, pw)
        latent = latent.permute(0, 2, 4, 6, 3, 5, 7, 1)
        return latent.reshape(latent.shape[0], -1, pt * ph * pw * channels)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return projected patch tokens with shape ``[B,N,hidden_size]``."""
        return self.projection(self.patchify(latent))

    def unpatchify(self, tokens: torch.Tensor, *, frames: int, height: int, width: int) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape [B,N,D]")
        pt, ph, pw = self.config.patch_size
        grid = (frames // pt, height // ph, width // pw)
        expected = grid[0] * grid[1] * grid[2]
        if tokens.shape[1] != expected:
            raise ValueError(f"expected {expected} patch tokens, got {tokens.shape[1]}")
        values = tokens.reshape(tokens.shape[0], *grid, pt, ph, pw, self.config.in_channels)
        values = values.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return values.reshape(tokens.shape[0], self.config.in_channels, frames, height, width)


class LingBotVideoRMSNorm(nn.Module):
    """Checkpoint-compatible RMSNorm with fp32 accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (self.weight * normalized).to(input_dtype)


class LingBotVideoMLP(nn.Module):
    """Checkpoint-compatible SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def apply_lingbot_video_complex_rope(hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply the official complex64 RoPE representation to ``[B,S,H,D]`` tensors."""
    if hidden_states.ndim != 4 or hidden_states.shape[-1] % 2:
        raise ValueError("RoPE inputs must have shape [B,S,H,even_head_dim]")
    if freqs_cis.ndim == 2:
        freqs_cis = freqs_cis.unsqueeze(0)
    if freqs_cis.shape[-1] != hidden_states.shape[-1] // 2:
        raise ValueError("RoPE table does not match attention head dimension")
    complex_states = torch.view_as_complex(hidden_states.float().reshape(*hidden_states.shape[:-1], -1, 2))
    output = torch.view_as_real(complex_states * freqs_cis.unsqueeze(2)).flatten(3)
    return output.to(hidden_states.dtype)


class LingBotVideoTextEmbedder(nn.Module):
    """Checkpoint-compatible text feature projection."""

    def __init__(self, text_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(self.norm(hidden_states))))


class LingBotVideoAttention(nn.Module):
    """Native-SDPA equivalent of the official LingBot attention module."""

    def __init__(self, hidden_size: int, num_heads: int, norm_eps: float, qkv_bias: bool, out_bias: bool) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=out_bias)
        self.attention_config: object | None = None
        self.ulysses_group: dist.ProcessGroup | None = None

    def set_attention_config(self, attention_config: object) -> None:
        """Attach a runtime-selected TeleFuser attention implementation."""
        self.attention_config = attention_config

    def set_ulysses_group(self, group: dist.ProcessGroup | None) -> None:
        """Attach the Ulysses process group used for sequence-parallel attention."""
        self.ulysses_group = group

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_sequence_lengths: list[int] | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self.to_q(hidden_states).view(batch, sequence, self.num_heads, self.head_dim)
        key = self.to_k(hidden_states).view(batch, sequence, self.num_heads, self.head_dim)
        value = self.to_v(hidden_states).view(batch, sequence, self.num_heads, self.head_dim)
        query = apply_lingbot_video_complex_rope(self.norm_q(query), rotary_emb)
        key = apply_lingbot_video_complex_rope(self.norm_k(key), rotary_emb)
        group = self.ulysses_group
        use_ulysses = (
            group is not None and dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1
        )
        if use_ulysses:
            world_size = dist.get_world_size(group)
            local_heads = self.num_heads // world_size
            query = ulysses_all_to_all_split_cat(
                query.reshape(batch, sequence, self.num_heads * self.head_dim),
                group,
                scatter_dim=2,
                gather_dim=1,
            ).view(batch, sequence * world_size, local_heads, self.head_dim)
            key = ulysses_all_to_all_split_cat(
                key.reshape(batch, sequence, self.num_heads * self.head_dim),
                group,
                scatter_dim=2,
                gather_dim=1,
            ).view(batch, sequence * world_size, local_heads, self.head_dim)
            value = ulysses_all_to_all_split_cat(
                value.reshape(batch, sequence, self.num_heads * self.head_dim),
                group,
                scatter_dim=2,
                gather_dim=1,
            ).view(batch, sequence * world_size, local_heads, self.head_dim)
        output = attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attention_config=self.attention_config,
            attn_mask=attention_mask,
            sequence_lengths=packed_sequence_lengths,
            input_layout="BNSD",
            output_layout="BNSD",
        )
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("LingBot attention does not support log-sum-exp outputs")
        output = output.transpose(1, 2)
        if use_ulysses:
            output = ulysses_all_to_all_split_cat(
                output.reshape(batch, sequence * world_size, local_heads * self.head_dim),
                group,
                scatter_dim=1,
                gather_dim=2,
            ).view(batch, sequence, self.num_heads, self.head_dim)
        return self.to_out(output.reshape(batch, sequence, -1).to(hidden_states.dtype))


class LingBotVideoBlock(nn.Module):
    """Checkpoint-compatible Dense LingBot transformer block."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * hidden_size))
        self.norm1 = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.attn = LingBotVideoAttention(hidden_size, num_attention_heads, norm_eps, qkv_bias, out_bias)
        self.norm_post_attn = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.norm2 = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.ffn = LingBotVideoMLP(hidden_size, intermediate_size)
        self.norm_post_ffn = LingBotVideoRMSNorm(hidden_size, norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb6: torch.Tensor,
        rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_sequence_lengths: list[int] | None = None,
    ) -> torch.Tensor:
        batch, sequence, hidden_size = hidden_states.shape
        if temb6.shape != (batch, sequence, 6 * hidden_size):
            raise ValueError("temb6 must have shape [B,S,6*hidden_size]")
        modulation = temb6 + self.scale_shift_table.unsqueeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        attention_input = self.norm1(hidden_states) * (1.0 + scale_msa) + shift_msa
        attention_output = self.attn(
            attention_input.to(self.attn.to_q.weight.dtype),
            rotary_emb,
            attention_mask,
            packed_sequence_lengths,
        )
        hidden_states = hidden_states + (gate_msa.tanh() * self.norm_post_attn(attention_output)).to(
            hidden_states.dtype
        )
        mlp_input = self.norm2(hidden_states) * (1.0 + scale_mlp) + shift_mlp
        ffn_weight = getattr(getattr(self.ffn, "gate_proj", None), "weight", self.attn.to_q.weight)
        mlp_output = self.ffn(mlp_input.to(ffn_weight.dtype))
        return hidden_states + (gate_mlp.tanh() * self.norm_post_ffn(mlp_output)).to(hidden_states.dtype)


def make_lingbot_video_joint_position_ids(
    text_len: int, grid_t: int, grid_h: int, grid_w: int, device: torch.device
) -> torch.Tensor:
    """Create official [video; text] three-axis position IDs."""
    temporal = torch.arange(grid_t, device=device, dtype=torch.int32) + text_len + 1
    height = torch.arange(grid_h, device=device, dtype=torch.int32)
    width = torch.arange(grid_w, device=device, dtype=torch.int32)
    video = torch.stack(torch.meshgrid(temporal, height, width, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text = torch.stack((text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)), dim=-1)
    return torch.cat((video, text), dim=0)


def lingbot_video_complex_frequencies(
    position_ids: torch.Tensor, axes_dims: tuple[int, int, int], theta: float
) -> torch.Tensor:
    """Compute official multi-axis complex RoPE frequencies for position IDs."""
    if position_ids.ndim != 2 or position_ids.shape[1] != len(axes_dims):
        raise ValueError("position_ids must have shape [S,3]")
    position_ids_cpu = position_ids.detach().to(device="cpu")
    frequencies = []
    for axis, dimension in enumerate(axes_dims):
        values = torch.arange(0, dimension, 2, device="cpu", dtype=torch.float64)
        values = 1.0 / (theta ** (values / dimension))
        angles = (position_ids_cpu[:, axis].to(torch.float64).unsqueeze(1) * values.unsqueeze(0)).float()
        frequencies.append(torch.polar(torch.ones_like(angles), angles).to(torch.complex64))
    return torch.cat(frequencies, dim=-1).to(position_ids.device)


class LingBotVideoTimeEmbedder(nn.Module):
    """Parameter names compatible with Diffusers TimestepEmbedding."""

    def __init__(self, frequency_dim: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.linear_1 = nn.Linear(frequency_dim, hidden_size, bias=bias)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        exponent = (
            -torch.log(torch.tensor(10000.0, device=timesteps.device))
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        angles = timesteps.float().reshape(-1, 1) * exponent.exp().reshape(1, -1)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.frequency_dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return self.linear_2(F.silu(self.linear_1(embedding.to(self.linear_1.weight.dtype))))


class LingBotVideoTransformer3DModel(nn.Module):
    """Source-equivalent Dense LingBot-Video transformer native reference path."""

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        depth: int = 24,
        intermediate_size: int = 6144,
        text_dim: int = 2560,
        freq_dim: int = 256,
        norm_eps: float = 1e-6,
        rope_theta: float = 256.0,
        axes_dims: tuple[int, int, int] = (32, 48, 48),
        qkv_bias: bool = False,
        out_bias: bool = True,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size // num_attention_heads != sum(axes_dims):
            raise ValueError("head dimension must equal sum(axes_dims)")
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.axes_dims = axes_dims
        self.rope_theta = rope_theta
        self.patch_embedder = nn.Linear(in_channels * prod(patch_size), hidden_size, bias=patch_embed_bias)
        self.time_embedder = LingBotVideoTimeEmbedder(freq_dim, hidden_size, timestep_mlp_bias)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.text_embedder = LingBotVideoTextEmbedder(text_dim, hidden_size)
        self.blocks = nn.ModuleList(
            [
                LingBotVideoBlock(hidden_size, num_attention_heads, intermediate_size, norm_eps, qkv_bias, out_bias)
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj_out = nn.Linear(hidden_size, prod(patch_size) * out_channels)
        self.ulysses_group: dist.ProcessGroup | None = None

    def set_attention_config(self, attention_config: object) -> None:
        """Propagate a shared attention backend configuration to every DiT block."""
        for block in self.blocks:
            block.attn.set_attention_config(attention_config)

    def set_ulysses_group(self, group: dist.ProcessGroup | None) -> None:
        """Enable source-order Ulysses sequence parallelism for the joint token stream."""
        self.ulysses_group = group
        for block in self.blocks:
            block.attn.set_ulysses_group(group)

    def get_fsdp_module_names(self) -> list[str]:
        """Return block containers suitable for per-block FSDP wrapping."""
        return ["blocks"]

    def _ulysses_world_size(self) -> int:
        group = self.ulysses_group
        if group is None or not dist.is_available() or not dist.is_initialized():
            return 1
        return dist.get_world_size(group)

    def _ulysses_shard_joint(
        self,
        joint: torch.Tensor,
        rotary: torch.Tensor,
        temb_input: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Pad and shard the joint video/text sequence into equal Ulysses slices."""
        world_size = self._ulysses_world_size()
        if world_size == 1:
            return joint, rotary, temb_input, valid_mask, 0
        sequence = joint.shape[1]
        padding = (-sequence) % world_size
        if padding:
            joint = F.pad(joint, (0, 0, 0, padding))
            rotary = F.pad(rotary, (0, 0, 0, padding))
            temb_input = F.pad(temb_input, (0, 0, 0, padding))
            valid_mask = F.pad(valid_mask, (0, padding), value=False)
        local_sequence = joint.shape[1] // world_size
        rank = dist.get_rank(self.ulysses_group)
        start = rank * local_sequence
        end = start + local_sequence
        return joint[:, start:end], rotary[:, start:end], temb_input[:, start:end], valid_mask[:, start:end], padding

    def _ulysses_gather_sequence(self, local: torch.Tensor) -> torch.Tensor:
        """Gather equal local token slices in rank order without changing their layout."""
        if self._ulysses_world_size() == 1:
            return local
        gathered = [torch.empty_like(local) for _ in range(self._ulysses_world_size())]
        dist.all_gather(gathered, local.contiguous(), group=self.ulysses_group)
        return torch.cat(gathered, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, channels, frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.patch_size
        if channels * prod(self.patch_size) != self.patch_embedder.in_features:
            raise ValueError("latent channels do not match checkpoint configuration")
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError("latent geometry must be divisible by the checkpoint patch size")
        grid_t, grid_h, grid_w = frames // patch_t, height // patch_h, width // patch_w
        video_tokens = grid_t * grid_h * grid_w
        patches = hidden_states.reshape(batch, channels, grid_t, patch_t, grid_h, patch_h, grid_w, patch_w)
        patches = patches.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(batch, video_tokens, -1)
        video = self.patch_embedder(patches)
        encoded_text_length = encoder_hidden_states.shape[1]
        text_mask = (
            encoder_attention_mask.bool()
            if encoder_attention_mask is not None
            else torch.ones(batch, encoded_text_length, dtype=torch.bool, device=hidden_states.device)
        )
        text_lengths = [int(length) for length in text_mask.sum(dim=-1).detach().cpu().tolist()]
        packed_batch = batch > 1
        packed_attention = packed_batch or self._ulysses_world_size() > 1
        time_embedding = self.time_embedder(timestep)

        if packed_attention:
            joint_parts: list[torch.Tensor] = []
            rotary_parts: list[torch.Tensor] = []
            temb_parts: list[torch.Tensor] = []
            packed_sequence_lengths = [video_tokens + text_length for text_length in text_lengths]
            for index, text_length in enumerate(text_lengths):
                text = self.text_embedder(encoder_hidden_states[index : index + 1, :text_length])
                joint_parts.append(torch.cat((video[index : index + 1], text), dim=1))
                positions = make_lingbot_video_joint_position_ids(
                    text_length, grid_t, grid_h, grid_w, hidden_states.device
                )
                rotary_parts.append(lingbot_video_complex_frequencies(positions, self.axes_dims, self.rope_theta))
                temb_parts.append(
                    time_embedding[index : index + 1].unsqueeze(1).expand(1, video_tokens + text_length, -1)
                )
            joint = torch.cat(joint_parts, dim=1)
            rotary = torch.cat(rotary_parts).unsqueeze(0)
            temb_input = torch.cat(temb_parts, dim=1)
            valid_mask = torch.ones(1, joint.shape[1], dtype=torch.bool, device=hidden_states.device)
        else:
            text = self.text_embedder(encoder_hidden_states)
            joint = torch.cat((video, text), dim=1)
            positions = make_lingbot_video_joint_position_ids(
                encoded_text_length, grid_t, grid_h, grid_w, hidden_states.device
            )
            rotary = lingbot_video_complex_frequencies(positions, self.axes_dims, self.rope_theta).unsqueeze(0)
            temb_input = time_embedding.unsqueeze(1).expand(-1, joint.shape[1], -1)
            valid_mask = torch.cat(
                (torch.ones(batch, video_tokens, dtype=torch.bool, device=hidden_states.device), text_mask), dim=1
            )
            packed_sequence_lengths = None

        joint, rotary, temb_input, local_valid_mask, padding = self._ulysses_shard_joint(
            joint, rotary, temb_input, valid_mask
        )
        if packed_sequence_lengths is not None and padding:
            packed_sequence_lengths = [*packed_sequence_lengths, padding]
        temb6 = self.time_modulation(temb_input)
        attention_mask = None
        if packed_sequence_lengths is None and not bool(local_valid_mask.all()):
            attention_mask = local_valid_mask[:, None, None, :]
        for block in self.blocks:
            joint = block(joint, temb6, rotary, attention_mask, packed_sequence_lengths)
        final_modulation = self.norm_out_modulation(temb_input)
        shift, scale = final_modulation.chunk(2, dim=-1)
        projected = self.proj_out((self.norm_out(joint) * (1.0 + scale) + shift).to(self.proj_out.weight.dtype))
        if self._ulysses_world_size() > 1:
            projected = self._ulysses_gather_sequence(projected)
            if padding:
                projected = projected[:, :-padding]
        if packed_batch:
            projected = torch.cat(
                [part[:, :video_tokens] for part in torch.split(projected, packed_sequence_lengths[:batch], dim=1)]
            )
        else:
            projected = projected[:, :video_tokens]
        output = projected.reshape(batch, grid_t, grid_h, grid_w, patch_t, patch_h, patch_w, self.out_channels)
        return output.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(batch, self.out_channels, frames, height, width)
