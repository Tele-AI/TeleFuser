# Copyright 2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from pathlib import Path
from typing import Any

# Ported from Diffusers commit 843f1743e259e510f22065c0cb917d417b98d9f5.
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, apply_rotary_emb, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from diffusers.utils import logging

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig, AttnImplType, QuantConfig, QuantKernelBackend, QuantType
from telefuser.distributed.collectives import all_gather_cat, all_reduce_sum_
from telefuser.distributed.device_mesh import (
    get_ring_world_size,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    get_ulysses_group,
    get_ulysses_rank,
    get_ulysses_world_size,
)
from telefuser.distributed.ulysses_comm import ulysses_gather_heads_destination_major, ulysses_scatter_heads
from telefuser.ops.attention import SparseAttentionState, attention
from telefuser.ops.fp8_attention import apply_fp8_attention_output_correction, quantize_fp8_qkv_smoothed

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.models.transformers.transformer_flux._get_projections with FluxAttention->PAN2Attention
def _get_projections(attn: "PAN2Attention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


# Copied from diffusers.models.transformers.transformer_flux._get_fused_projections with FluxAttention->PAN2Attention
def _get_fused_projections(attn: "PAN2Attention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


# Copied from diffusers.models.transformers.transformer_flux._get_qkv_projections with FluxAttention->PAN2Attention
def _get_qkv_projections(attn: "PAN2Attention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class PAN2AttnProcessor:
    r"""
    Attention processor for PAN 2. Without `encoder_hidden_states` it runs plain self-attention (token refiner). With
    `encoder_hidden_states` it runs joint attention over `[video; text]`, where each stream has its own projections and
    only the video stream is rotated.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "PAN2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query).to(value.dtype)
        key = attn.norm_k(key).to(value.dtype)

        if rotary_emb is not None:
            query = apply_rotary_emb(query, rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, rotary_emb, sequence_dim=1)

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query).to(encoder_value.dtype)
            encoder_key = attn.norm_added_k(encoder_key).to(encoder_value.dtype)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        hidden_states = _parallel_attention(attn, query, key, value, attention_mask)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if encoder_hidden_states is None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            if attn.tp_group is not None:
                all_reduce_sum_([hidden_states], group=attn.tp_group)
            return hidden_states

        hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
            [hidden_states.shape[1] - encoder_hidden_states.shape[1], encoder_hidden_states.shape[1]], dim=1
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        if attn.tp_group is not None:
            all_reduce_sum_([hidden_states, encoder_hidden_states], group=attn.tp_group)
        return hidden_states, encoder_hidden_states


class PAN2Attention(nn.Module):
    _default_processor_cls = PAN2AttnProcessor
    _available_processors = [PAN2AttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        eps: float,
        bias: bool = True,
        added_kv_proj_dim: int | None = None,
        processor: PAN2AttnProcessor | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.fused_projections = False
        self.tp_group = None
        self.sp_group = None
        self.sp_communicator = None
        self.global_video_tokens = 0
        self.global_text_tokens = 0
        self.sparse_state = None
        self.attention_config = AttentionConfig()
        self.inner_dim = heads * dim_head
        self.use_bias = bias
        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = bias

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=bias), nn.Dropout(0.0)])

        if added_kv_proj_dim is not None:
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def set_processor(self, processor: Any) -> None:
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)


class PAN2RotaryPosEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: list[int],
        theta: float = 256.0,
        temporal_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta
        self.temporal_scale = temporal_scale

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        rope_sizes = [num_frames // self.patch_size_t, height // self.patch_size, width // self.patch_size]

        axes_grids = []
        for i in range(len(rope_sizes)):
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [T, H, W]
        grid = torch.stack(grid, dim=0)  # [3, T, H, W]
        # Only the frame positions are rescaled, to the frame rate the rotary embedding was trained at.
        grid[0] = grid[0] * self.temporal_scale

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (T * H * W, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (T * H * W, D / 2)
        return freqs_cos, freqs_sin


class PAN2TokenRefinerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = PAN2Attention(dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps, bias=False)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff = FeedForward(dim, inner_dim=int(dim * mlp_ratio), activation_fn="swiglu", bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        ff_output = self.ff(self.norm2(hidden_states))
        if self.attn.tp_group is not None:
            all_reduce_sum_([ff_output], group=self.attn.tp_group)
        hidden_states = hidden_states + ff_output
        return hidden_states


class PAN2TransformerBlock(nn.Module):
    r"""
    Joint-attention block. Video and text tokens attend to each other in a single attention call, but each stream has
    its own QKV / output projections and feed-forward network. The video stream is modulated by the timestep, the text
    stream by a learned, timestep-independent table.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        context_mlp_ratio: float,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = AdaLayerNormZero(dim, norm_type="layer_norm")
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.scale_shift_table_context = nn.Parameter(torch.zeros(6 * dim))

        self.attn = PAN2Attention(
            dim, heads=num_attention_heads, dim_head=attention_head_dim, eps=eps, added_kv_proj_dim=dim
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim, inner_dim=int(dim * mlp_ratio), activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = FeedForward(dim, inner_dim=int(dim * context_mlp_ratio), activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The norms, their shift/scale and the gated residuals run in float32, like the modulation.
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states.float(), emb=temb)
        norm_hidden_states = norm_hidden_states.type_as(hidden_states)
        (
            context_shift_msa,
            context_scale_msa,
            context_gate_msa,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.scale_shift_table_context.chunk(6, dim=-1)
        norm_encoder_hidden_states = (
            self.norm1_context(encoder_hidden_states.float()) * (1 + context_scale_msa) + context_shift_msa
        ).type_as(encoder_hidden_states)

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            rotary_emb=rotary_emb,
        )

        hidden_states = (hidden_states.float() + attn_output * gate_msa.unsqueeze(1)).type_as(hidden_states)
        encoder_hidden_states = (encoder_hidden_states.float() + context_attn_output * context_gate_msa).type_as(
            encoder_hidden_states
        )

        norm_hidden_states = (
            self.norm2(hidden_states.float()) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ).type_as(hidden_states)
        norm_encoder_hidden_states = (
            self.norm2_context(encoder_hidden_states.float()) * (1 + context_scale_mlp) + context_shift_mlp
        ).type_as(encoder_hidden_states)

        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        if self.attn.tp_group is not None:
            all_reduce_sum_([ff_output, context_ff_output], group=self.attn.tp_group)
        hidden_states = (hidden_states.float() + ff_output * gate_mlp.unsqueeze(1)).type_as(hidden_states)
        encoder_hidden_states = (encoder_hidden_states.float() + context_ff_output * context_gate_mlp).type_as(
            encoder_hidden_states
        )

        return hidden_states, encoder_hidden_states


class PAN2DiT(BaseModel, ConfigMixin):
    r"""
    The diffusion transformer of PAN 2.

    A stack of joint-attention blocks over the concatenation of the patchified video latents and the refined text
    tokens. The video input carries the noisy latents, the conditioning latents and a conditioning mask stacked along
    the channel axis, so the same model serves text-to-video and image-to-video.

    Args:
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The `(t, h, w)` patch size of the patch embedding.
        in_channels (`int`, defaults to `97`):
            The number of input channels: noisy latents, conditioning latents and one conditioning mask channel.
        out_channels (`int`, defaults to `48`):
            The number of output channels.
        num_attention_heads (`int`, defaults to `32`):
            The number of attention heads.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each attention head.
        num_layers (`int`, defaults to `44`):
            The number of joint-attention blocks.
        num_refiner_layers (`int`, defaults to `4`):
            The number of token refiner blocks applied to the text embeddings.
        mlp_ratio (`float`, defaults to `8.0`):
            The ratio of the video feed-forward inner dimension to the hidden dimension.
        context_mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the text feed-forward inner dimension to the hidden dimension.
        refiner_mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the token refiner feed-forward inner dimension to the hidden dimension.
        text_embed_dim (`int`, defaults to `4096`):
            The dimension of the text encoder hidden states.
        rope_axes_dim (`tuple[int, int, int]`, defaults to `(16, 56, 56)`):
            The number of channels of each `(t, h, w)` axis in the rotary embedding.
        rope_theta (`float`, defaults to `256.0`):
            The base of the rotary embedding.
        rope_temporal_scale (`float`, defaults to `16 / 24`):
            The factor the frame positions of the rotary embedding are multiplied with: the reference frame rate of the
            rotary embedding over the frame rate of the videos (16 over 24 fps).
    """

    config_name = "config.json"
    _supports_gradient_checkpointing = True
    # The timestep embedder and the modulation run in float32. `linear` is the modulation of every block
    # (`norm1.linear`) and of the output norm (`norm_out.linear`).
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table_context", "linear"]
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = ["PAN2TransformerBlock", "PAN2TokenRefinerBlock"]
    _repeated_blocks = ["PAN2TransformerBlock", "PAN2TokenRefinerBlock"]
    # Text is sharded before refinement; padded video/text shards are compacted into canonical joint order
    # for attention and restored before the inverse Ulysses all-to-all.

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 97,
        out_channels: int = 48,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 44,
        num_refiner_layers: int = 4,
        mlp_ratio: float = 8.0,
        context_mlp_ratio: float = 4.0,
        refiner_mlp_ratio: float = 4.0,
        text_embed_dim: int = 4096,
        rope_axes_dim: tuple[int, int, int] = (16, 56, 56),
        rope_theta: float = 256.0,
        rope_temporal_scale: float = 16 / 24,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        self.x_embedder = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.rope = PAN2RotaryPosEmbed(patch_size[1], patch_size[0], rope_axes_dim, rope_theta, rope_temporal_scale)

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

        self.context_embedder = nn.Sequential(RMSNorm(text_embed_dim, eps=1e-5), nn.Linear(text_embed_dim, inner_dim))
        self.context_refiner = nn.ModuleList(
            [
                PAN2TokenRefinerBlock(inner_dim, num_attention_heads, attention_head_dim, refiner_mlp_ratio, eps=1e-5)
                for _ in range(num_refiner_layers)
            ]
        )
        self.context_type_embedding = nn.Parameter(torch.zeros(inner_dim))

        self.transformer_blocks = nn.ModuleList(
            [
                PAN2TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, context_mlp_ratio)
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size[0] * patch_size[1] * patch_size[2] * out_channels)

        self.gradient_checkpointing = False
        self.layer_name_list = ["transformer_blocks"]
        self.sparse_attention_state = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        sparse_step_index: int = 0,
    ) -> torch.Tensor | Transformer2DModelOutput:
        r"""
        The [`PAN2Transformer3DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, in_channels, num_frames, height, width)`):
                The noisy latents, the conditioning latents and the conditioning mask, concatenated along channels.
            timestep (`torch.Tensor` of shape `(batch_size,)`):
                The denoising timestep in `[0, 1000]`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_seq_len, text_embed_dim)`):
                The text encoder hidden states.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary passed along to the `AttentionProcessor`; only used for the LoRA scale.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.modeling_outputs.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.modeling_outputs.Transformer2DModelOutput`] or `tuple`:
                The predicted velocity of shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        if attention_kwargs:
            raise ValueError("PAN2 LoRA attention kwargs are not supported by this inference adapter")
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. Rotary embedding, patch and timestep embeddings
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.x_embedder(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
        video_tokens, text_tokens = hidden_states.shape[1], encoder_hidden_states.shape[1]
        for module in self.modules():
            if isinstance(module, PAN2Attention):
                module.global_text_tokens = text_tokens
                module.global_video_tokens = 0
        for block in self.transformer_blocks:
            block.attn.global_video_tokens = video_tokens
        if self.usp_flag:
            world = get_ulysses_world_size(self.device_mesh)
            rank = get_ulysses_rank(self.device_mesh)
            hidden_states = _sequence_shard(hidden_states, world, rank)
            encoder_hidden_states = _sequence_shard(encoder_hidden_states, world, rank)
            rotary_emb = tuple(_sequence_shard(value.unsqueeze(0), world, rank).squeeze(0) for value in rotary_emb)
        if self.sparse_attention_state is not None:
            self.sparse_attention_state.update(numeral_timestep=sparse_step_index)

        timesteps_proj = self.time_proj(timestep).to(self.time_embedder.linear_1.weight.dtype)
        temb = self.time_embedder(timesteps_proj)

        # 2. Text embedding and refinement
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        for block in self.context_refiner:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states = self._gradient_checkpointing_func(block, encoder_hidden_states)
            else:
                encoder_hidden_states = block(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + self.context_type_embedding.to(encoder_hidden_states.device)

        # 3. Joint-attention blocks
        for index, block in enumerate(self.transformer_blocks):
            if self.sparse_attention_state is not None:
                self.sparse_attention_state.update(layer_idx=index)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, rotary_emb
                )
            else:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb, rotary_emb)

        # 4. Output norm, projection and unpatchify
        hidden_states = self.norm_out(hidden_states.type_as(temb), temb).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        if self.usp_flag:
            hidden_states = all_gather_cat(hidden_states, dim=1, group=get_ulysses_group(self.device_mesh))
            hidden_states = hidden_states[:, :video_tokens]

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p_h, p_w
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def set_attention_config(self, attention_config: AttentionConfig) -> None:
        super().set_attention_config(attention_config)
        if attention_config.attn_impl == AttnImplType.SOL_ATTN:
            if attention_config.sparse_config is None:
                raise ValueError("PAN2 Sol-Attn requires sparse configuration")
            self.sparse_attention_state = SparseAttentionState(
                config=attention_config.sparse_config, mask_map=None, model_type="pan2"
            )
        else:
            self.sparse_attention_state = None
        for block in self.context_refiner:
            block.attn.attention_config = (
                AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
                if attention_config.is_sparse()
                else attention_config
            )
        for block in self.transformer_blocks:
            block.attn.attention_config = attention_config
            block.attn.sparse_state = self.sparse_attention_state

    def to(self, *args: Any, **kwargs: Any) -> "PAN2DiT":
        def keep(name: str) -> bool:
            return (
                name.startswith("time_embedder.")
                or "scale_shift_table_context" in name
                or ".norm1.linear." in name
                or name.startswith("norm_out.linear.")
            )

        preserved = {
            name: parameter.detach().clone()
            for name, parameter in self.named_parameters()
            if keep(name) and not parameter.is_meta
        }
        result = super().to(*args, **kwargs)
        for name, parameter in result.named_parameters():
            if name in preserved:
                parameter.data = preserved[name].to(device=parameter.device, dtype=torch.float32)
            elif keep(name):
                parameter.data = parameter.data.float()
        return result

    def enable_usp(self, device_mesh: torch.distributed.device_mesh.DeviceMesh | None = None) -> None:
        self.device_mesh = device_mesh if device_mesh is not None else self.device_mesh
        if get_ring_world_size(self.device_mesh) != 1:
            raise NotImplementedError("PAN2 currently supports Ulysses sequence parallelism, not Ring")
        world = get_ulysses_world_size(self.device_mesh)
        if self.context_refiner[0].attn.heads % world:
            raise ValueError("Local PAN2 attention heads must divide the Ulysses degree")
        group = get_ulysses_group(self.device_mesh) if world > 1 else None
        communicator = self._configure_ulysses_communicator(group)
        for module in self.modules():
            if isinstance(module, PAN2Attention):
                module.sp_group = group
                module.sp_communicator = communicator
        self.usp_flag = world > 1

    def enable_tp(self, device_mesh: torch.distributed.device_mesh.DeviceMesh | None = None) -> None:
        self.device_mesh = device_mesh if device_mesh is not None else self.device_mesh
        world = get_tp_world_size(self.device_mesh)
        if world == 1:
            return
        if getattr(self, "tp_flag", False):
            raise RuntimeError("PAN2 tensor parallelism is already enabled")
        rank, group = get_tp_rank(self.device_mesh), get_tp_group(self.device_mesh)
        for module in self.modules():
            if not isinstance(module, PAN2Attention):
                continue
            if module.heads % world:
                raise ValueError("PAN2 attention heads must divide TP degree")
            for name in ["to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"]:
                if hasattr(module, name):
                    _column_shard(getattr(module, name), rank, world)
            _row_shard(module.to_out[0], rank, world)
            if hasattr(module, "to_add_out"):
                _row_shard(module.to_add_out, rank, world)
            module.heads //= world
            module.inner_dim //= world
            module.tp_group = group
        for block in self.context_refiner:
            _ffn_shard(block.ff, rank, world, gated=True)
        for block in self.transformer_blocks:
            _ffn_shard(block.ff, rank, world, gated=False)
            _ffn_shard(block.ff_context, rank, world, gated=False)
            _column_shard(block.norm1.linear, rank, world)
            block.norm1.linear.register_forward_hook(_ColumnGather(group))
        _column_shard(self.norm_out.linear, rank, world)
        self.norm_out.linear.register_forward_hook(_ColumnGather(group))
        self.tp_flag = True

    def enable_quant(self, quant_type: QuantConfig | QuantType | str | torch.dtype) -> None:
        if not isinstance(quant_type, QuantConfig):
            if quant_type not in (QuantType.FP8, "fp8", torch.float8_e4m3fn):
                raise ValueError("PAN2 supports tf-kernel FP8 quantization through QuantConfig")
            quant_type = QuantConfig(
                enabled=True, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL
            )
        if not quant_type.enabled:
            return
        if quant_type.quant_type != QuantType.FP8 or quant_type.kernel_backend not in {
            QuantKernelBackend.AUTO,
            QuantKernelBackend.TF_KERNEL,
        }:
            raise ValueError("PAN2 online quantization currently supports tf-kernel FP8")
        from telefuser.ops.fp8_gemm import FP8GemmOptions, count_linear_layers, enable_fp8_gemm

        def include(name: str, module: nn.Module) -> bool:
            selected = quant_type.quantize_modules or ("transformer_blocks.", "context_refiner.")
            return (
                any(part in name for part in selected)
                and ".norm1.linear" not in name
                and not any(part and part in name for part in quant_type.skip_modules)
            )

        self.tf_kernel_fp8_replaced_linear = count_linear_layers(self, module_filter=include)
        enable_fp8_gemm(
            self,
            options=FP8GemmOptions(
                fp16_weight_storage="keep" if quant_type.keep_fp16_weight else "discard",
                materialize_fp8_on_wrap=True,
            ),
            module_filter=include,
        )
        self.quant_type = quant_type.quant_type

    @staticmethod
    def state_dict_converter(config_path: str | Path | None = None) -> PAN2DiTStateDictConverter:
        return PAN2DiTStateDictConverter(config_path)


class PAN2DiTStateDictConverter:
    def __init__(self, config_path: str | Path | None) -> None:
        from pathlib import Path

        if config_path is None:
            raise ValueError("PAN2 requires the released transformer/config.json")
        self.config_path = Path(config_path)
        if self.config_path.is_dir():
            self.config_path = self.config_path / "config.json"

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        import inspect
        import json

        config = json.loads(self.config_path.read_text())
        accepted = inspect.signature(PAN2DiT.__init__).parameters
        return state_dict, {name: value for name, value in config.items() if name in accepted}

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return self.from_diffusers(state_dict)


def _sequence_shard(value, world, rank):
    length = value.shape[1]
    local = (length + world - 1) // world
    if local * world != length:
        shape = list(value.shape)
        shape[1] = local * world - length
        value = torch.cat((value, value.new_zeros(shape)), dim=1)
    return value[:, rank * local : (rank + 1) * local].contiguous()


def _replace_parameter(module, name, value):
    parameter = module.get_parameter(name)
    setattr(module, name, nn.Parameter(value.contiguous(), requires_grad=parameter.requires_grad))


def _column_shard(module, rank, world, sections=None):
    sections = sections or (module.out_features,)
    if any(size % world for size in sections):
        raise ValueError("PAN2 projection sections must divide TP degree")
    _replace_parameter(module, "weight", torch.cat([x.chunk(world, 0)[rank] for x in module.weight.split(sections)], 0))
    if module.bias is not None:
        _replace_parameter(module, "bias", torch.cat([x.chunk(world, 0)[rank] for x in module.bias.split(sections)], 0))
    module.out_features //= world


def _row_shard(module, rank, world):
    if module.in_features % world:
        raise ValueError("PAN2 row projection must divide TP degree")
    _replace_parameter(module, "weight", module.weight.chunk(world, 1)[rank])
    if module.bias is not None and rank != 0:
        _replace_parameter(module, "bias", torch.zeros_like(module.bias))
    module.in_features //= world


def _ffn_shard(module, rank, world, gated):
    projection = module.net[0].proj
    sections = (projection.out_features // 2,) * 2 if gated else None
    _column_shard(projection, rank, world, sections)
    _row_shard(module.net[2], rank, world)


class _ColumnGather:
    def __init__(self, group: dist.ProcessGroup) -> None:
        self.group = group

    def __call__(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        return all_gather_cat(output, dim=-1, group=self.group)


def _parallel_attention(module, query, key, value, attention_mask):
    group = module.sp_group
    use_sp = group is not None
    indices = None
    if use_sp:
        if attention_mask is not None:
            raise ValueError("PAN2 packed SP uses explicit token lengths, not arbitrary attention masks")
        world = dist.get_world_size(group)
        waits = [
            ulysses_scatter_heads(x, group, tag=tag, barrier=index == 2, communicator=module.sp_communicator)
            for index, (tag, x) in enumerate(zip(["q", "k", "v"], [query, key, value]))
        ]
        query, key, value = [wait() for wait in waits]
        video, text = module.global_video_tokens, module.global_text_tokens
        vlocal, tlocal = (video + world - 1) // world, (text + world - 1) // world
        joint = vlocal + tlocal
        valid = []
        for total, local, offset in [(video, vlocal, 0), (text, tlocal, vlocal)]:
            for rank in range(world):
                count = max(0, min(local, total - rank * local))
                valid.extend(range(rank * joint + offset, rank * joint + offset + count))
        indices = torch.tensor(valid, dtype=torch.long, device=query.device)
        original_shape = query.shape
        query, key, value = [x.index_select(1, indices) for x in [query, key, value]]
    config, state = module.attention_config, module.sparse_state
    scales = None
    correction = None
    if state is not None:
        if state.should_use_dense():
            config, state = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA), None
        elif config.sparse_config.sol_fp8:
            sparse = config.sparse_config
            if state.layer_idx >= sparse.sol_fp8_layer_start and (
                sparse.sol_fp8_layer_end is None or state.layer_idx < sparse.sol_fp8_layer_end
            ):
                query, key, value, qs, ks, vs, correction = quantize_fp8_qkv_smoothed(
                    query,
                    key,
                    value,
                    smoothing=sparse.sol_fp8_smoothing,
                    correct_v_bias=sparse.sol_fp8_v_bias_correction,
                )
                scales = (qs, ks, vs)
    output = attention(
        query,
        key,
        value,
        attention_config=config,
        sparse_state=state,
        attn_mask=attention_mask,
        q_scale=scales[0] if scales else None,
        k_scale=scales[1] if scales else None,
        v_scale=scales[2] if scales else None,
    )
    if correction is not None:
        output = apply_fp8_attention_output_correction(output.contiguous(), correction.contiguous())
    if use_sp:
        padded = output.new_zeros(original_shape)
        padded.index_copy_(1, indices, output)
        output = ulysses_gather_heads_destination_major(padded, group, num_heads=module.heads)()
    return output
