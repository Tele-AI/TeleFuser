"""Flux2 DiT model with optimized operations for single GPU.

This implementation follows the diffusers Flux2Transformer2DModel structure,
using TeleFuser internal ops for optimal performance on single GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig, AttnImplType, OffloadConfig
from telefuser.ops.activations import silu_and_mul
from telefuser.ops.attention import attention
from telefuser.ops.normalization import LayerNorm, RMSNorm, modulate
from telefuser.ops.rotary import apply_rotary_emb
from telefuser.utils.logging import logger


@dataclass
class Flux2DiTConfig:
    """Configuration for Flux2DiT model."""

    patch_size: int = 1
    in_channels: int = 128
    out_channels: int | None = None
    num_layers: int = 8
    num_single_layers: int = 24
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    joint_attention_dim: int = 12288
    timestep_guidance_channels: int = 256
    mlp_ratio: float = 3.0
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: int = 2000
    eps: float = 1e-6
    guidance_embeds: bool = False


# =============================================================================
# Helper Functions
# =============================================================================


def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    repeat_interleave_real: bool = True,
    use_real: bool = True,
    freqs_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 1D rotary positional embeddings.

    Args:
        dim: Embedding dimension (must be even).
        pos: Position indices tensor of shape [S].
        theta: Base frequency for rotary embeddings.
        repeat_interleave_real: Whether to repeat interleave real part.
        use_real: Whether to return real format (cos, sin).
        freqs_dtype: Data type for frequency computation.

    Returns:
        Tuple of (cos, sin) tensors of shape [S, D].
    """
    assert dim % 2 == 0

    # Compute frequency scale: [D/2]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))

    # Outer product: [S, D/2]
    freqs = torch.outer(pos.float(), freqs)

    if use_real and repeat_interleave_real:
        # Flux style: repeat_interleave to get [S, D]
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
        return freqs_cos, freqs_sin
    elif use_real:
        # Concat style
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()
        return freqs_cos, freqs_sin
    else:
        # Complex format
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


# =============================================================================
# Embedding Modules
# =============================================================================


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool = True, downscale_freq_shift: float = 0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal embeddings for timesteps."""
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = torch.exp(exponent).to(timesteps.dtype)
        emb = timesteps.unsqueeze(-1) * emb.unsqueeze(0)

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding with linear projection."""

    def __init__(self, in_channels: int, time_embed_dim: int, bias: bool = False):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=bias)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Flux2PosEmbed(nn.Module):
    """Positional embedding for RoPE in Flux2."""

    def __init__(self, theta: int, axes_dim: tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate rotary embeddings from position IDs.

        Args:
            ids: Position IDs of shape (S, 4) for 4D coordinates (T, H, W, L).

        Returns:
            Tuple of (cos, sin) tensors.
        """
        cos_out = []
        sin_out = []
        pos = ids.float()

        # Handle batch dimension
        if pos.ndim == 3:
            pos = pos[0]  # Take first batch

        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64

        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)

        return freqs_cos, freqs_sin


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    """Combined timestep and guidance embeddings."""

    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=in_channels, time_embed_dim=embedding_dim, bias=bias)

        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(in_channels=in_channels, time_embed_dim=embedding_dim, bias=bias)
        else:
            self.guidance_embedder = None

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor | None) -> torch.Tensor:
        """Compute timestep + guidance embeddings."""
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))

        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))
            return timesteps_emb + guidance_emb

        return timesteps_emb


class Flux2Modulation(nn.Module):
    """Modulation layer for shift/scale/gate parameters."""

    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> torch.Tensor:
        """Generate modulation parameters."""
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        return mod

    @staticmethod
    def split(mod: torch.Tensor, mod_param_sets: int) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Split modulation into (shift, scale, gate) tuples."""
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * mod_param_sets, dim=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(mod_param_sets))


# =============================================================================
# Feed-Forward Network
# =============================================================================


class Flux2SwiGLU(nn.Module):
    """SwiGLU activation for Flux2 FFN (no parameters)."""

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)


class Flux2FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 3.0,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


# =============================================================================
# Attention Modules
# =============================================================================


class Flux2Attention(nn.Module):
    """Attention module for Flux2 double-stream blocks.

    Uses optimized attention from ops.attention with Flash Attention support.
    """

    # Default attention config (can be overridden via set_attention_config)
    attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: int | None = None,
        bias: bool = False,
        eps: float = 1e-5,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool = True,
        out_bias: bool = True,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.head_dim = dim_head
        self.added_kv_proj_dim = added_kv_proj_dim

        # QKV projections
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim if added_kv_proj_dim is None else added_kv_proj_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim if added_kv_proj_dim is None else added_kv_proj_dim, self.inner_dim, bias=bias)

        # QK Norm with optimized RMSNorm
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Added KV projections for encoder hidden states
        if added_kv_proj_dim is not None:
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.to_add_out = nn.Linear(self.inner_dim, added_kv_proj_dim, bias=out_bias)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.out_dim, bias=out_bias),
            nn.Dropout(0.0),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for attention."""
        batch_size = hidden_states.shape[0]

        # QKV projections from image tokens
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        # Reshape and normalize
        query = query.view(batch_size, -1, self.heads, self.head_dim)
        key = key.view(batch_size, -1, self.heads, self.head_dim)
        value = value.view(batch_size, -1, self.heads, self.head_dim)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Handle encoder (text) tokens
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.view(batch_size, -1, self.heads, self.head_dim)
            encoder_key = encoder_key.view(batch_size, -1, self.heads, self.head_dim)
            encoder_value = encoder_value.view(batch_size, -1, self.heads, self.head_dim)

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            # Concatenate text + image tokens
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Apply RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Attention with optimized kernel (BSND layout)
        hidden_states: torch.Tensor = attention(
            query,
            key,
            value,
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
            return_lse=False,
        )
        hidden_states = hidden_states.reshape(batch_size, -1, self.inner_dim)

        # Split and apply output projections separately
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_len, hidden_states.shape[1] - text_seq_len], dim=1
            )
            # Apply to_add_out to text stream
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        # Apply to_out to image stream only
        hidden_states = self.to_out(hidden_states)

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            return hidden_states, encoder_hidden_states

        return hidden_states


class Flux2ParallelSelfAttention(nn.Module):
    """Parallel self-attention for Flux2 single-stream blocks.

    Fuses attention QKV with FFN input projections, and attention output with FFN output.
    Uses optimized attention and SwiGLU operations.
    """

    # Default attention config (can be overridden via set_attention_config)
    attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)

    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: int | None = None,
        bias: bool = False,
        eps: float = 1e-5,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.head_dim = dim_head

        self.mlp_hidden_dim = int(query_dim * mlp_ratio)
        self.mlp_mult_factor = 2

        # Fused QKV + MLP input projection
        self.to_qkv_mlp_proj = nn.Linear(
            query_dim,
            self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=bias,
        )
        self.mlp_act_fn = Flux2SwiGLU()

        # QK Norm with optimized RMSNorm
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # Fused output projection
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass for parallel self-attention."""
        batch_size = hidden_states.shape[0]

        # Fused projection
        qkv_mlp = self.to_qkv_mlp_proj(hidden_states)
        query, key, value, mlp_hidden = qkv_mlp.split(
            [
                self.inner_dim,
                self.inner_dim,
                self.inner_dim,
                self.mlp_hidden_dim * self.mlp_mult_factor,
            ],
            dim=-1,
        )

        # Reshape Q, K, V
        query = query.view(batch_size, -1, self.heads, self.head_dim)
        key = key.view(batch_size, -1, self.heads, self.head_dim)
        value = value.view(batch_size, -1, self.heads, self.head_dim)

        # QK Norm
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Attention with optimized kernel (BSND layout)
        attn_output: torch.Tensor = attention(
            query,
            key,
            value,
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
            return_lse=False,
        )
        attn_output = attn_output.reshape(batch_size, -1, self.inner_dim)

        # MLP with optimized SwiGLU
        mlp_output = self.mlp_act_fn(mlp_hidden)

        # Concatenate and output
        hidden_states = torch.cat([attn_output, mlp_output], dim=-1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states


# =============================================================================
# Transformer Blocks
# =============================================================================


class Flux2TransformerBlock(nn.Module):
    """Double-stream transformer block for Flux2.

    Processes image and text streams separately with cross-attention.
    Uses optimized LayerNorm and modulate operations.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        # Optimized LayerNorm with Triton kernel support
        self.norm1 = LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
        self.norm1_context = LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)

        self.attn = Flux2Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            eps=eps,
            added_kv_proj_dim=dim,
            added_proj_bias=bias,
            out_bias=bias,
        )

        self.norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

        self.norm2_context = LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for double-stream block."""
        # Modulation: (shift, scale, gate) x 2
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            temb_mod_txt, 2
        )

        # Image stream with optimized modulate
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = modulate(norm_hidden_states, shift_msa.squeeze(1), scale_msa.squeeze(1))

        # Text stream with optimized modulate
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = modulate(
            norm_encoder_hidden_states, c_shift_msa.squeeze(1), c_scale_msa.squeeze(1)
        )

        # Cross-attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Image stream update with optimized modulate
        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = modulate(norm_hidden_states, shift_mlp.squeeze(1), scale_mlp.squeeze(1))
        hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

        # Text stream update with optimized modulate
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = modulate(
            norm_encoder_hidden_states, c_shift_mlp.squeeze(1), c_scale_mlp.squeeze(1)
        )
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    """Single-stream transformer block for Flux2.

    Processes concatenated image and text tokens with parallel attention + FFN.
    Uses optimized LayerNorm and modulate operations.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        # Optimized LayerNorm with Triton kernel support
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False, bias=False)
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for single-stream block."""
        # Concatenate if encoder_hidden_states provided
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Modulation
        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]

        # Normalize and modulate with optimized ops
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = modulate(norm_hidden_states, mod_shift.squeeze(1), mod_scale.squeeze(1))

        # Parallel attention + FFN
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Residual with gate
        hidden_states = hidden_states + mod_gate * attn_output

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        # Split if requested
        if split_hidden_states and text_seq_len is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :text_seq_len],
                hidden_states[:, text_seq_len:],
            )
            return encoder_hidden_states, hidden_states

        return hidden_states


class AdaLayerNormContinuous(nn.Module):
    """Adaptive LayerNorm with continuous conditioning.

    Uses optimized LayerNorm from ops.normalization with Triton kernel support.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        # Use optimized LayerNorm
        self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine, bias=False)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = emb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


# =============================================================================
# Main Model
# =============================================================================


@dataclass
class Flux2DiTOutput:
    """Output of Flux2DiT forward pass."""

    sample: torch.Tensor


class Flux2DiT(BaseModel):
    """Flux2 DiT model inheriting from BaseModel.

    Optimized implementation for single GPU using TeleFuser internal ops:
    - Triton kernels for RoPE, LayerNorm, RMSNorm, and scale-shift operations
    - Flash Attention 2/3/4 and Sage Attention support
    - Optimized SwiGLU with tf_kernel support
    """

    def __init__(self, config: Flux2DiTConfig | None = None):
        super().__init__()
        self.config = config or Flux2DiTConfig()

        self.out_channels = self.config.out_channels or self.config.in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        # 1. Position embedding for RoPE
        self.pos_embed = Flux2PosEmbed(theta=self.config.rope_theta, axes_dim=self.config.axes_dims_rope)

        # 2. Timestep + guidance embeddings
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=self.config.timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=self.config.guidance_embeds,
        )

        # 3. Modulation layers
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(self.config.in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double-stream transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    mlp_ratio=self.config.mlp_ratio,
                    eps=self.config.eps,
                    bias=False,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # 6. Single-stream transformer blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    mlp_ratio=self.config.mlp_ratio,
                    eps=self.config.eps,
                    bias=False,
                )
                for _ in range(self.config.num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=self.config.eps,
            bias=False,
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.out_channels,
            bias=False,
        )

        # Set layer names for offloading
        self.layer_name_list = ["transformer_blocks", "single_transformer_blocks"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Flux2DiTOutput | tuple[torch.Tensor]:
        """Forward pass for Flux2DiT.

        Args:
            hidden_states: Image tokens of shape (B, seq_len, in_channels).
            encoder_hidden_states: Text tokens of shape (B, text_seq_len, joint_attention_dim).
            timestep: Timestep tensor.
            img_ids: Image position IDs of shape (B, seq_len, 4) or (seq_len, 4).
            txt_ids: Text position IDs of shape (B, text_seq_len, 4) or (text_seq_len, 4).
            guidance: Optional guidance tensor.
            return_dict: Whether to return a dict or tuple.

        Returns:
            Output tensor of shape (B, seq_len, out_channels).
        """
        # Handle position IDs batch dimension
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        # 1. Timestep embedding
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        # 2. Modulation
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        # 3. Input projections
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # 4. Rotary embeddings
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # 5. Double-stream blocks (use concat_rotary_emb for text + image tokens)
        encoder_hidden_states, hidden_states = self.forward_double_blocks(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_img=double_stream_mod_img,
            temb_mod_txt=double_stream_mod_txt,
            image_rotary_emb=concat_rotary_emb,
        )

        # 6. Single-stream blocks (concatenated)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        hidden_states = self.forward_single_blocks(
            hidden_states=hidden_states,
            temb_mod=single_stream_mod,
            image_rotary_emb=concat_rotary_emb,
        )

        # 7. Split back
        text_seq_len = encoder_hidden_states.shape[1]
        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )

        # 8. Output
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if return_dict:
            return Flux2DiTOutput(sample=output)

        return (output,)

    def forward_double_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through double-stream transformer blocks.

        This method is separated for torch.compile optimization.
        Compiling only the blocks avoids graph breaks from embedding operations.

        Args:
            hidden_states: Image tokens of shape (B, seq_len, inner_dim).
            encoder_hidden_states: Text tokens of shape (B, text_seq_len, inner_dim).
            temb_mod_img: Modulation for image stream.
            temb_mod_txt: Modulation for text stream.
            image_rotary_emb: Rotary embeddings tuple (cos, sin).

        Returns:
            Tuple of (encoder_hidden_states, hidden_states) after block processing.
        """
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_img=temb_mod_img,
                temb_mod_txt=temb_mod_txt,
                image_rotary_emb=image_rotary_emb,
            )
        return encoder_hidden_states, hidden_states

    def forward_single_blocks(
        self,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through single-stream transformer blocks.

        This method is separated for torch.compile optimization.
        Compiling only the blocks avoids graph breaks from embedding operations.

        Args:
            hidden_states: Concatenated tokens of shape (B, text_seq_len + img_seq_len, inner_dim).
            temb_mod: Modulation for single-stream blocks.
            image_rotary_emb: Rotary embeddings tuple (cos, sin).

        Returns:
            Hidden states after single-stream block processing.
        """
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod=temb_mod,
                image_rotary_emb=image_rotary_emb,
            )
        return hidden_states

    def get_fsdp_module_names(self) -> list[str]:
        """Get module names for FSDP sharding."""
        return ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]

    def set_attention_config(self, attention_config: AttentionConfig) -> None:
        """Set attention configuration for all attention modules.

        Args:
            attention_config: Attention configuration to apply.
        """
        super().set_attention_config(attention_config)
        # Propagate to all attention modules in transformer blocks
        for block in self.transformer_blocks:
            block.attn.attention_config = attention_config
        for block in self.single_transformer_blocks:
            block.attn.attention_config = attention_config
        logger.info(f"Set attention config to {attention_config.attn_impl.name} for Flux2DiT")

    def enable_async_offload(self, device: torch.device, offload_config: OffloadConfig):
        """Enable async CPU offloading for transformer blocks.

        Args:
            device: Device to run computation on.
            offload_config: Offload configuration.
        """
        from telefuser.distributed.async_offload import AsyncOffloadManager

        logger.info("Enable async offload for Flux2DiT")
        # Combine all transformer blocks for offloading
        all_blocks = list(self.transformer_blocks) + list(self.single_transformer_blocks)
        self.async_offload_manager = AsyncOffloadManager(
            all_blocks,
            enabled=True,
            offload_ratio=offload_config.offload_ratio,
            prefetch_size=offload_config.prefetch_size,
            device=device,
            pin_cpu_memory=offload_config.pin_cpu_memory,
        )
        self.async_offload_flag = True

    def compile(self, mode: str = "blocks", **kwargs) -> None:
        """Compile model for better performance with torch.compile.

        Args:
            mode: Compilation mode:
                - "blocks": Compile forward_double_blocks and forward_single_blocks (default, most effective)
                - "double_blocks": Compile only forward_double_blocks
                - "single_blocks": Compile only forward_single_blocks
                - "full": Compile entire forward method
            **kwargs: Arguments passed to torch.compile()
        """
        # Import mark_static from torch._dynamo
        try:
            from torch._dynamo import mark_static

            # Mark module classes as static (instance attributes won't change after compile)
            mark_static(Flux2DiT)
            mark_static(Flux2TransformerBlock)
            mark_static(Flux2SingleTransformerBlock)
            mark_static(Flux2Attention)
            mark_static(Flux2ParallelSelfAttention)
            mark_static(Flux2FeedForward)
            mark_static(AdaLayerNormContinuous)
        except ImportError:
            logger.warning("torch._dynamo.mark_static not available, skipping static marking")

        # Compile based on mode
        if mode == "blocks":
            # Compile both block forward methods
            original_double_fn = self.forward_double_blocks
            original_single_fn = self.forward_single_blocks
            self.forward_double_blocks = torch.compile(original_double_fn, **kwargs)
            self.forward_single_blocks = torch.compile(original_single_fn, **kwargs)
            logger.info(f"Flux2DiT compiled: mode={mode}")
        elif mode == "double_blocks":
            original_fn = self.forward_double_blocks
            self.forward_double_blocks = torch.compile(original_fn, **kwargs)
            logger.info(f"Flux2DiT compiled: mode={mode}")
        elif mode == "single_blocks":
            original_fn = self.forward_single_blocks
            self.forward_single_blocks = torch.compile(original_fn, **kwargs)
            logger.info(f"Flux2DiT compiled: mode={mode}")
        elif mode == "full":
            # Store original forward for fallback
            self._original_forward = self.forward
            self.forward = torch.compile(self.forward, **kwargs)
            logger.info(f"Flux2DiT compiled: mode={mode}")
        else:
            raise ValueError(f"Unknown compile mode: {mode}")

    @staticmethod
    def state_dict_converter():
        """Return state dict converter."""
        return Flux2DiTStateDictConverter()


class Flux2DiTStateDictConverter:
    """State dict converter for Flux2DiT.

    Since Flux2DiT follows the same structure as diffusers Flux2Transformer2DModel,
    the state dict is directly compatible without any key remapping.
    """

    def __init__(self):
        pass

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert from diffusers Flux2Transformer2DModel state dict.

        Args:
            state_dict: State dict from diffusers Flux2Transformer2DModel.

        Returns:
            Converted state dict for Flux2DiT.
        """
        # Direct compatibility - no key remapping needed
        return state_dict

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert from official BFL state dict format.

        Args:
            state_dict: State dict from official BFL format.

        Returns:
            Converted state dict for Flux2DiT.
        """
        # BFL format uses different key names, need to remap
        # Map BFL keys to diffusers/TeleFuser keys
        key_mapping = {
            # Embedders
            "img_in.weight": "x_embedder.weight",
            "txt_in.weight": "context_embedder.weight",
            "time_in.": "time_guidance_embed.",
            "vector_in.": "double_stream_modulation_img.linear.",
            "guidance_in.": "time_guidance_embed.guidance_embedder.",
            # Double stream modulation
            "double_blocks.": "transformer_blocks.",
            # Single stream modulation
            "single_blocks.": "single_transformer_blocks.",
            # Output
            "final_layer.": "norm_out.",
            "proj_out.weight": "proj_out.weight",
        }

        converted = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in key_mapping.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    break
            converted[new_key] = value

        return converted


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "39c6fc48f07bebecedbbaa971ff466c8", ["transformer"], [Flux2DiT], "diffusers")
