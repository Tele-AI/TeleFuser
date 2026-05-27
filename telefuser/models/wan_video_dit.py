from __future__ import annotations

import math
from typing import Any

import torch
import torch.amp as amp
import torch.nn as nn
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig, AttnImplType, OffloadConfig
from telefuser.distributed.device_mesh import (
    get_pp_group,
    get_pp_rank,
    get_pp_world_size,
    get_ulysses_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from telefuser.distributed.parallel_shard import (
    sequence_parallel_shard,
    sequence_parallel_unshard,
)
from telefuser.distributed.pp_comm import PipelineP2PComm
from telefuser.distributed.ulysses_comm import (
    ulysses_gather_heads,
    ulysses_scatter_heads,
)
from telefuser.offload import (
    AutoWrappedLinear,
    AutoWrappedModule,
    WanAutoCastLayerNorm,
    enable_sequential_cpu_offload,
)
from telefuser.offload.async_offload import AsyncOffloadManager
from telefuser.ops.attention import MaskMap, SparseAttentionState
from telefuser.ops.attention import attention as attn_func
from telefuser.ops.attention import long_context_attention as long_attn_func
from telefuser.ops.normalization import LayerNorm, RMSNorm, fused_scale_shift, modulate
from telefuser.ops.rotary import apply_rotary_emb
from telefuser.utils.logging import logger
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


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0) -> list:
    """Precompute 3D RoPE frequencies for video."""
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return [f_freqs_cis, h_freqs_cis, w_freqs_cis]


@amp.autocast("cuda", enabled=False)
@torch.compiler.disable()
def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> torch.Tensor:
    """Precompute 1D RoPE frequencies using complex numbers."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


@amp.autocast("cuda", enabled=False)
def rope_apply(x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply RoPE (Rotary Position Embedding).

    Uses ops.rotary.apply_rotary_emb which automatically selects the optimal
    implementation based on compile state and platform.

    Args:
        x: Input tensor [B, S, D] or [B, S, num_heads * head_dim]
        freqs_cos: Cosine frequencies [seq_len, 1, head_size//2] (real tensor, not complex)
        freqs_sin: Sine frequencies [seq_len, 1, head_size//2] (real tensor, not complex)
        num_heads: Number of attention heads

    Returns:
        Tensor with RoPE applied, same shape as input
    """
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    # apply_rotary_emb expects x: [B, S, H, D], freqs: (cos, sin)
    x_rope = apply_rotary_emb(x, (freqs_cos, freqs_sin))
    return x_rope.flatten(2)


class SelfAttention(nn.Module):
    """Self-attention with RoPE and sparse attention support."""

    attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)

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
        self.usp_flag = False

    def async_usp_forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        sparse_state: SparseAttentionState | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        """Async Ulysses-style sequence parallel forward."""
        group = get_ulysses_group(device_mesh)
        q = self.norm_q(self.q(x))
        q = rope_apply(q, freqs_cos, freqs_sin, self.num_heads)
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        q_wait = ulysses_scatter_heads(q, group)
        k = self.norm_k(self.k(x))
        k = rope_apply(k, freqs_cos, freqs_sin, self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        k_wait = ulysses_scatter_heads(k, group)
        v = self.v(x)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        v_wait = ulysses_scatter_heads(v, group)
        q = q_wait()
        k = k_wait()
        v = v_wait()
        if sparse_state is not None and sparse_state.config.sparse_impl == "radial":
            seqlen = q.shape[2]
            q = rearrange(q, "b n s d -> (b s) n d", s=seqlen, n=self.num_heads)
            k = rearrange(k, "b n s d -> (b s) n d", s=seqlen, n=self.num_heads)
            v = rearrange(v, "b n s d -> (b s) n d", s=seqlen, n=self.num_heads)

            from telefuser.core.config import AttentionConfig

            attention_config = AttentionConfig(
                attn_impl=AttnImplType.RADIAL_ATTN,
                sparse_config=sparse_state.config,
            )
            x = attn_func(
                q,
                k,
                v,
                attention_config=attention_config,
                sparse_state=sparse_state,
                input_layout="BSND",
                output_layout="BSND",
            )
        else:
            if self.attention_config.is_sparse():
                from telefuser.core.config import AttentionConfig

                dense_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
                x = attn_func(q, k, v, attention_config=dense_config, input_layout="BSND", output_layout="BSND")
            else:
                x = attn_func(
                    q, k, v, attention_config=self.attention_config, input_layout="BSND", output_layout="BSND"
                )
        out_wait = ulysses_gather_heads(x, group, num_heads=self.num_heads)
        out = out_wait()
        out = rearrange(out, "b s n d -> b s (n d)", n=self.num_heads)
        out = self.o(out)
        return out

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        sparse_state: SparseAttentionState | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        if self.usp_flag:
            return self.async_usp_forward(x, freqs_cos, freqs_sin, sparse_state, device_mesh)
        return self.default_forward(x, freqs_cos, freqs_sin, sparse_state)

    def default_forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        sparse_state: SparseAttentionState | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs_cos, freqs_sin, self.num_heads)
        k = rope_apply(k, freqs_cos, freqs_sin, self.num_heads)
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        if self.usp_flag:
            if sparse_state is not None and sparse_state.config.sparse_impl == "radial":
                from telefuser.core.config import AttentionConfig

                attention_config = AttentionConfig(
                    attn_impl=AttnImplType.RADIAL_ATTN, sparse_config=sparse_state.config
                )
                x = long_attn_func(
                    q,
                    k,
                    v,
                    attention_config=attention_config,
                    sparse_state=sparse_state,
                    input_layout="BSND",
                    output_layout="BSND",
                    device_mesh=device_mesh,
                )
            else:
                if self.attention_config.is_sparse():
                    from telefuser.core.config import AttentionConfig

                    dense_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
                    x = long_attn_func(
                        q,
                        k,
                        v,
                        input_layout="BSND",
                        output_layout="BSND",
                        device_mesh=device_mesh,
                        attention_config=dense_config,
                    )
                else:
                    x = long_attn_func(
                        q,
                        k,
                        v,
                        input_layout="BSND",
                        output_layout="BSND",
                        device_mesh=device_mesh,
                        attention_config=self.attention_config,
                    )
        elif sparse_state is not None and sparse_state.config.sparse_impl == "radial":
            seqlen = q.shape[2]
            q = rearrange(q, "b s n d -> (b s) n d", s=seqlen, n=self.num_heads)
            k = rearrange(k, "b s n d -> (b s) n d", s=seqlen, n=self.num_heads)
            v = rearrange(v, "b s n d -> (b s) n d", s=seqlen, n=self.num_heads)

            from telefuser.core.config import AttentionConfig

            attention_config = AttentionConfig(attn_impl=AttnImplType.RADIAL_ATTN, sparse_config=sparse_state.config)
            x = attn_func(
                q,
                k,
                v,
                attention_config=attention_config,
                sparse_state=sparse_state,
                input_layout="BSND",
                output_layout="BSND",
            )
        else:
            if self.attention_config.is_sparse():
                from telefuser.core.config import AttentionConfig

                dense_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
                x = attn_func(q, k, v, input_layout="BSND", output_layout="BSND", attention_config=dense_config)
            else:
                x = attn_func(
                    q, k, v, input_layout="BSND", output_layout="BSND", attention_config=self.attention_config
                )
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        return self.o(x)


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""

    attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
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
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        # CrossAttention always uses dense attention
        if self.attention_config.is_sparse():
            from telefuser.core.config import AttentionConfig

            dense_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
            x = attn_func(q, k, v, attention_config=dense_config, input_layout="BSND", output_layout="BSND")
        else:
            x = attn_func(q, k, v, attention_config=self.attention_config, input_layout="BSND", output_layout="BSND")
        x = rearrange(x, "b s n d -> b s (n d)", n=self.num_heads)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            k_img = rearrange(k_img, "b s (n d) -> b s n d", n=self.num_heads)
            v_img = rearrange(v_img, "b s (n d) -> b s n d", n=self.num_heads)
            if self.attention_config.is_sparse():
                from telefuser.core.config import AttentionConfig

                dense_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
                y = attn_func(q, k_img, v_img, attention_config=dense_config, input_layout="BSND", output_layout="BSND")
            else:
                y = attn_func(
                    q, k_img, v_img, attention_config=self.attention_config, input_layout="BSND", output_layout="BSND"
                )
            y = rearrange(y, "b s n d -> b s (n d)", n=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    """Gated residual connection."""

    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual


class DiTBlock(nn.Module):
    """Diffusion Transformer block with self-attention, cross-attention, and FFN."""

    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)

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

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        sparse_state: SparseAttentionState | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        # t_mod is expected to be preprocessed to [B, seq_len, 6, dim] format
        # by WanModel._prepare_modulation() to avoid dynamic dimension checks.
        # The caller ensures t_mod has shape [B, seq_len, 6, dim] where seq_len=1 for global.
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        t_mod_with_bias = modulation.unsqueeze(0) + t_mod  # [B, seq_len, 6, dim]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_mod_with_bias.chunk(6, dim=2)
        # Squeeze the modulation dimension: [B, seq_len, 1, dim] -> [B, seq_len, dim]
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)

        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        if sparse_state is not None:
            attn_output = self.self_attn(input_x, freqs_cos, freqs_sin, sparse_state=sparse_state)
        else:
            attn_output = self.self_attn(input_x, freqs_cos, freqs_sin, device_mesh=device_mesh)
        x = self.gate(x, gate_msa, attn_output)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
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

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t is expected to be preprocessed to [B, seq_len, dim] format
        # by WanModel._prepare_head_t() to avoid dynamic dimension checks.
        # The caller ensures t has shape [B, seq_len, dim] where seq_len=1 for global.
        modulation = self.modulation.to(dtype=t.dtype, device=t.device)
        t_mod = modulation.unsqueeze(0) + t.unsqueeze(2)  # [B, seq_len, 2, dim]
        shift, scale = t_mod.chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanModel(BaseModel):
    """Wan video generation DiT model with feature caching and sparse attention support."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.num_layers = num_layers

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, patch_size, eps)
        self.head_dim = dim // num_heads
        self.normal_freqs = precompute_freqs_cis_3d(self.head_dim)
        self.freqs = self.normal_freqs
        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        self.has_image_pos_emb = has_image_pos_emb

        self.dtype = torch.bfloat16
        self.y_camera = None
        self.async_offload_manager = None
        self.layer_name_list = ["blocks"]

    def check_y_camera_status(self):
        return self.y_camera

    def reset_y_camera_status(self):
        self.y_camera = None

    def enable_quant(self, quant_type: str | torch.dtype):
        """Enable quantization for transformer blocks."""
        if quant_type in [torch.float8_e4m3fn]:
            logger.info(f"loading weights with {quant_type}, start convert linear layer to {quant_type}")
            from telefuser.ops.quantized_linear import replace_linear_layers

            replace_linear_layers(self.blocks, quant_type)
            self.quant_type = quant_type

    def enable_async_offload(self, device: torch.device, offload_config: OffloadConfig):
        """Enable async CPU offloading for transformer blocks."""

        logger.info("enable async offload for wan video dit")
        self.async_offload_manager = AsyncOffloadManager(
            self.blocks,
            enabled=True,
            offload_ratio=offload_config.offload_ratio,
            prefetch_size=offload_config.prefetch_size,
            device=device,
            pin_cpu_memory=offload_config.pin_cpu_memory,
        )
        self.async_offload_flag = True

    def forward_blocks(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        sparse_state: SparseAttentionState | None = None,
    ) -> torch.Tensor:
        for block_id, block in enumerate(self.blocks):
            if sparse_state is not None:
                sparse_state.update(layer_idx=block_id)
            x = block(x, context, t_mod, freqs_cos, freqs_sin, sparse_state=sparse_state, device_mesh=self.device_mesh)
        return x

    def forward_blocks_pp(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        sparse_state: SparseAttentionState | None = None,
    ) -> torch.Tensor:
        """Forward through blocks with pipeline parallelism.

        Only processes the blocks assigned to this stage.
        """
        for block_id, block in enumerate(self.blocks[self.pp_start_idx : self.pp_end_idx]):
            if sparse_state is not None:
                sparse_state.update(layer_idx=self.pp_start_idx + block_id)
            x = block(x, context, t_mod, freqs_cos, freqs_sin, sparse_state=sparse_state, device_mesh=self.device_mesh)
        return x

    def enable_usp(self):
        """Enable Ulysses-style sequence parallelism."""
        logger.info("wan dit enable usp")
        self.usp_flag = True
        for block in self.blocks:
            block.self_attn.usp_flag = True

    def enable_pp(self):
        """Enable pipeline parallelism."""
        logger.info("wan dit enable pp")
        self.pp_flag = True
        pp_group = get_pp_group(self.device_mesh)
        self.pp_comm = PipelineP2PComm(pp_group)

        # Get stage indices for this rank
        pp_rank = get_pp_rank(self.device_mesh)
        pp_world_size = get_pp_world_size(self.device_mesh)

        # Calculate block range for this stage
        layers_per_stage = self.num_layers // pp_world_size
        remainder = self.num_layers % pp_world_size

        if pp_rank < remainder:
            self.pp_start_idx = pp_rank * (layers_per_stage + 1)
            self.pp_end_idx = self.pp_start_idx + layers_per_stage + 1
        else:
            self.pp_start_idx = remainder * (layers_per_stage + 1) + (pp_rank - remainder) * layers_per_stage
            self.pp_end_idx = self.pp_start_idx + layers_per_stage

        # Ensure last stage covers all remaining layers
        if pp_rank == pp_world_size - 1:
            self.pp_end_idx = self.num_layers

        self.is_pp_first_stage = is_pipeline_first_stage(self.device_mesh)
        self.is_pp_last_stage = is_pipeline_last_stage(self.device_mesh)

        logger.info(
            f"PP stage {pp_rank}/{pp_world_size}: "
            f"blocks [{self.pp_start_idx}:{self.pp_end_idx}], "
            f"is_first={self.is_pp_first_stage}, is_last={self.is_pp_last_stage}"
        )

    def _send_pp_metadata(self, grid_size: tuple, shape: tuple) -> None:
        """Send metadata (grid_size and shape) to next stage.

        Used by current stage to send metadata to the next stage via P2P.
        """
        # Send grid_size: (f, h, w)
        grid_tensor = torch.tensor(list(grid_size), dtype=torch.long, device="cuda")
        self.pp_comm.send_latent(grid_tensor.unsqueeze(0).float())  # Use float for send_latent

        # Send shape
        shape_tensor = torch.tensor(list(shape), dtype=torch.long, device="cuda")
        self.pp_comm.send_latent(shape_tensor.unsqueeze(0).float())

    def _recv_pp_metadata(self) -> tuple[tuple, tuple]:
        """Receive metadata (grid_size and shape) from previous stage.

        Returns:
            Tuple of (grid_size, shape)
        """
        # Receive grid_size
        grid_tensor = self.pp_comm.recv_latent(shape=(1, 3), dtype=torch.float32)
        grid_size = tuple(grid_tensor.squeeze(0).long().tolist())

        # Receive shape
        shape_tensor = self.pp_comm.recv_latent(shape=(1, 3), dtype=torch.float32)
        shape = tuple(shape_tensor.squeeze(0).long().tolist())

        return grid_size, shape

    def _broadcast_pp_shape(self, shape: tuple, src: int = 0) -> tuple:
        """Broadcast latent shape from first stage to all stages.

        DEPRECATED: Use _send_pp_metadata / _recv_pp_metadata instead.
        This function is kept for backward compatibility.
        """
        dist = torch.distributed

        pp_group = get_pp_group(self.device_mesh)
        if pp_group is None:
            return shape

        # Create shape tensor for broadcast
        shape_tensor = torch.tensor(list(shape), dtype=torch.long, device="cuda")
        dist.broadcast(shape_tensor, src=src, group=pp_group)

        return tuple(shape_tensor.tolist())

    def _broadcast_pp_grid_size(self, grid_size: tuple | None, src: int = 0) -> tuple:
        """Broadcast grid size (f, h, w) from first stage to all stages.

        DEPRECATED: Use _send_pp_metadata / _recv_pp_metadata instead.
        This function is kept for backward compatibility.
        """
        dist = torch.distributed

        pp_group = get_pp_group(self.device_mesh)
        if pp_group is None:
            return grid_size if grid_size else (1, 1, 1)

        # Create grid size tensor for broadcast
        if grid_size is not None:
            grid_tensor = torch.tensor(list(grid_size), dtype=torch.long, device="cuda")
        else:
            grid_tensor = torch.zeros(3, dtype=torch.long, device="cuda")

        dist.broadcast(grid_tensor, src=src, group=pp_group)

        return tuple(grid_tensor.tolist())

    def pp_forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        cn_latents: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        cond_flag: bool = True,
        add_condition: torch.Tensor | None = None,
        y_camera: torch.Tensor | None = None,
        sparse_state: SparseAttentionState | None = None,
        freq_start_idx: int = 0,
        freq_interval: int = 1,
    ) -> torch.Tensor | None:
        """Pipeline parallel forward pass.

        Each stage performs its portion of the computation:
        - First stage: embedding layers + assigned blocks -> send hidden states
        - Middle stages: recv hidden states -> assigned blocks -> send hidden states
        - Last stage: recv hidden states -> assigned blocks + head -> output

        Args:
            x: Input latent tensor (only used in first stage)
            timestep: Timestep tensor
            context: Text context embedding
            cn_latents: ControlNet latents (only used in first stage)
            clip_feature: CLIP image feature (only used in first stage)
            cond_flag: Condition flag for CFG
            add_condition: Additional condition tensor (only used in first stage)
            y_camera: Camera parameters (only used in first stage)
            sparse_state: Sparse attention state
            freq_start_idx: Start index for frequency
            freq_interval: Interval for frequency

        Returns:
            Output tensor (only valid on last stage), None on other stages
        """
        # Handle single GPU case (no PP)
        if self.pp_comm.world_size == 1:
            return self.forward(
                x=x,
                timestep=timestep,
                context=context,
                cn_latents=cn_latents,
                clip_feature=clip_feature,
                cond_flag=cond_flag,
                add_condition=add_condition,
                y_camera=y_camera,
                sparse_state=sparse_state,
                freq_start_idx=freq_start_idx,
                freq_interval=freq_interval,
            )

        # ========== First Stage: Embedding + First Blocks ==========
        if self.is_pp_first_stage:
            # Get grid size for seq_len calculation
            batch_size = x.shape[0]
            f, h, w = x.shape[2], x.shape[3] // self.patch_size[1], x.shape[4] // self.patch_size[2]
            seq_len = f * h * w

            # Time embedding
            with amp.autocast("cuda", dtype=torch.float32):
                if timestep.dim() == 1:
                    # Scalar timestep: compute global time embedding and modulation
                    t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
                    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
                    # Expand t_mod to [B, seq_len, 6, dim] for uniform processing
                    t_mod = t_mod.unsqueeze(1).expand(-1, seq_len, -1, -1)
                    # Expand t to [B, 1, dim] for uniform processing in Head
                    t = t.unsqueeze(1)
                else:
                    # Per-token timestep: compute per-token time embedding and modulation
                    t_flat = timestep.flatten()
                    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_flat))
                    e = e.unflatten(0, (batch_size, seq_len))
                    t_mod = self.time_projection(e).unflatten(2, (6, self.dim))
                    # t is already [B, seq_len, dim]
                    t = e
            t = t.to(self.dtype)
            t_mod = t_mod.to(self.dtype)

            # Text embedding
            context = self.text_embedding(context)

            # Image embedding (for I2V models)
            if self.has_image_input and clip_feature is not None:
                clip_embdding = self.img_emb(clip_feature)
                context = torch.cat([clip_embdding, context], dim=1)

            # Patch embedding
            x, (f, h, w) = self.patchify(x)

            # Add condition
            if add_condition is not None:
                x = add_condition + x

            # ControlNet conditioning
            if cn_latents is not None:
                condition = cn_latents
                condition = rearrange(condition, "b c f h w -> b (f h w) c").contiguous()
                mean_x, std_x = torch.mean(x, dim=(1, 2), keepdim=True), torch.std(x, dim=(1, 2), keepdim=True)
                mean_condition = torch.mean(condition, dim=(1, 2), keepdim=True)
                std_condition = torch.std(condition, dim=(1, 2), keepdim=True)
                condition = (condition - mean_condition) * (std_x / (std_condition + 1e-5)) + mean_x
                x = x + condition * 0.2

            # Compute frequency embeddings
            freqs = (
                torch.cat(
                    [
                        self.freqs[0][freq_start_idx : freq_interval * f + freq_start_idx : freq_interval]
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

            # Precompute cos/sin from complex freqs to avoid complex tensor in compiled region
            freqs_cos = freqs.real.contiguous()
            freqs_sin = freqs.imag.contiguous()

            # Sequence parallel shard if enabled
            if self.usp_flag:
                # Shard t_mod and t (if per-token) along seq_len dimension (dim 1)
                # For scalar timestep, t has seq_len=1 which broadcasts with any seq_len
                # For per-token timestep, t needs sharding to match x
                sequence_parallel_shard(self.device_mesh, [x, t_mod, t, freqs_cos, freqs_sin], seq_dims=[1, 1, 1, 0, 0])

            # Feature cache handling - step ID is managed internally
            ori_x = x
            if self.feature_cache.should_compute(cond_flag):
                x = self.forward_blocks_pp(x, context, t_mod, freqs_cos, freqs_sin, sparse_state=sparse_state)
                self.feature_cache.update(x, ori_x, cond_flag)
            else:
                x = self.feature_cache.approximate(x, cond_flag)

            # Save grid size for later use
            self._pp_grid_size = (f, h, w)
            self._pp_t = t

            # If this is also the last stage (shouldn't happen with proper PP setup), return output
            if self.is_pp_last_stage:
                # Head projection
                x = self.head(x, t)

                # Sequence parallel unshard if enabled
                if self.usp_flag:
                    (x,) = sequence_parallel_unshard(self.device_mesh, [x], seq_dims=[1], seq_lens=[f * h * w])

                # Unpatchify to video format
                x = self.unpatchify(x, (f, h, w))

                return x

            # Send metadata and latent to next stage via P2P
            # Use P2P instead of broadcast to handle multiple forwards correctly (e.g., CFG)
            self._send_pp_metadata((f, h, w), x.shape)
            self.pp_comm.send_latent(x)
            return None

        # ========== Middle / Last Stage: Receive + Process ==========
        else:
            # Receive metadata and hidden states from previous stage via P2P
            grid_size, latent_shape = self._recv_pp_metadata()
            f, h, w = grid_size
            seq_len = f * h * w

            # Receive hidden states from previous stage
            x = self.pp_comm.recv_latent(shape=latent_shape, dtype=self.dtype)

            # Time embedding (needed for modulation in blocks)
            batch_size = timestep.shape[0]
            with amp.autocast("cuda", dtype=torch.float32):
                if timestep.dim() == 1:
                    # Scalar timestep: compute global time embedding and modulation
                    t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
                    t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
                    # Expand t_mod to [B, seq_len, 6, dim] for uniform processing
                    t_mod = t_mod.unsqueeze(1).expand(-1, seq_len, -1, -1)
                    # Expand t to [B, 1, dim] for uniform processing in Head
                    t = t.unsqueeze(1)
                else:
                    # Per-token timestep: compute per-token time embedding and modulation
                    t_flat = timestep.flatten()
                    e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_flat))
                    e = e.unflatten(0, (batch_size, seq_len))
                    t_mod = self.time_projection(e).unflatten(2, (6, self.dim))
                    # t is already [B, seq_len, dim]
                    t = e
            t = t.to(self.dtype)
            t_mod = t_mod.to(self.dtype)

            # Text embedding (needed for cross attention)
            context = self.text_embedding(context)
            if self.has_image_input and clip_feature is not None:
                clip_embdding = self.img_emb(clip_feature)
                context = torch.cat([clip_embdding, context], dim=1)

            # Compute frequency embeddings using broadcasted grid_size
            freqs = (
                torch.cat(
                    [
                        self.freqs[0][freq_start_idx : freq_interval * f + freq_start_idx : freq_interval]
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

            # Precompute cos/sin from complex freqs to avoid complex tensor in compiled region
            freqs_cos = freqs.real.contiguous()
            freqs_sin = freqs.imag.contiguous()

            # Apply sequence parallel shard to freqs, t_mod and t if enabled (x is already sharded from previous stage)
            if self.usp_flag:
                sequence_parallel_shard(self.device_mesh, [t_mod, t, freqs_cos, freqs_sin], seq_dims=[1, 1, 0, 0])

            # Save grid size for unpatchify in last stage
            self._pp_grid_size = (f, h, w)

            # Process blocks for this stage
            x = self.forward_blocks_pp(x, context, t_mod, freqs_cos, freqs_sin, sparse_state=sparse_state)

            # Save t for head
            self._pp_t = t

        # ========== Last Stage: Head + Unpatchify ==========
        if self.is_pp_last_stage:
            f, h, w = self._pp_grid_size
            t = self._pp_t

            # Head projection
            x = self.head(x, t)

            # Sequence parallel unshard if enabled
            if self.usp_flag:
                (x,) = sequence_parallel_unshard(self.device_mesh, [x], seq_dims=[1], seq_lens=[f * h * w])

            # Unpatchify to video format
            x = self.unpatchify(x, (f, h, w))

            return x

        # ========== Middle Stage: Send to Next ==========
        else:
            # Send hidden states to next stage
            self.pp_comm.send_latent(x)
            return None

    def set_attention_config(self, attention_config: AttentionConfig):
        """Set attention implementation configuration."""
        logger.info(f"wan dit set attention config to {attention_config.attn_impl}")
        SelfAttention.attention_config = attention_config

    def enable_radial_attention(
        self,
        height: int,
        width: int,
        num_frames: int,
        dense_layers: int = 0,
        dense_timesteps: int = 40,
        decay_factor: float = 1.0,
        use_sage_attention: bool = False,
    ):
        """Enable radial attention for efficient video generation.

        Args:
            height: Video height.
            width: Video width.
            num_frames: Number of frames.
            dense_layers: Number of layers to use dense attention.
            dense_timesteps: Number of timesteps to use dense attention.
            decay_factor: Decay factor for attention window.
            use_sage_attention: Whether to use sage attention backend.
        """
        from telefuser.core.config import SparseAttentionConfig

        logger.info(f"Enabling radial attention: dense_layers={dense_layers}, dense_timesteps={dense_timesteps}")

        num_frames_padded = 1 + num_frames // (4 * self.patch_size[0])
        mod_value = 8 * self.patch_size[1]
        frame_size = int(height // mod_value) * int(width // mod_value)
        video_token_num = frame_size * num_frames_padded

        mask_map = MaskMap(video_token_num=video_token_num, num_frame=num_frames_padded)
        sparse_config = SparseAttentionConfig(
            sparse_impl="radial",
            dense_timesteps=dense_timesteps,
            dense_layers=dense_layers,
            decay_factor=decay_factor,
            use_sage_attention=use_sage_attention,
        )
        self.sparse_attention_state = SparseAttentionState(config=sparse_config, mask_map=mask_map, model_type="wan")
        logger.info(f"Radial attention initialized: video_token_num={video_token_num}")

    def create_sparse_state(self, numeral_timestep: int = 0, layer_idx: int = 0) -> SparseAttentionState | None:
        """Create/update sparse attention state for current step."""
        if not hasattr(self, "sparse_attention_state"):
            return None
        self.sparse_attention_state.update(numeral_timestep=numeral_timestep, layer_idx=layer_idx)
        return self.sparse_attention_state

    def enable_sequential_cpu_offload(
        self,
        device: torch.device,
        torch_dtype: torch.dtype,
        max_num_param: int | None = None,
        vram_limit: int | None = None,
    ):
        """Enable sequential CPU offloading for memory efficiency."""

        dtype = next(iter(self.parameters())).dtype
        enable_sequential_cpu_offload(
            self,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
                LayerNorm: WanAutoCastLayerNorm,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=device,
                computation_dtype=torch_dtype,
                computation_device=device,
            ),
            max_num_param=max_num_param,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=torch_dtype,
                computation_device=device,
            ),
            vram_limit=vram_limit,
        )

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

    def compile(self, mode: str = "blocks", **kwargs) -> None:
        """Compile model for better performance with torch.compile.

        Args:
            mode: Compilation mode:
                - "blocks": Compile only forward_blocks (default, most effective)
                - "full": Compile entire forward method
                - "blocks_pp": Compile forward_blocks_pp for pipeline parallelism
            **kwargs: Arguments passed to torch.compile()
        """
        # Import mark_static from torch._dynamo
        try:
            from torch._dynamo import mark_static

            # Mark module classes as static (instance attributes won't change after compile)
            mark_static(WanModel)
            mark_static(DiTBlock)
            mark_static(SelfAttention)
            mark_static(Head)
        except ImportError:
            logger.warning("torch._dynamo.mark_static not available, skipping static marking")

        # Compile based on mode
        if mode == "blocks":
            original_fn = self.forward_blocks
            self.forward_blocks = torch.compile(original_fn, **kwargs)
            logger.info(f"WanModel compiled: mode={mode}")
        elif mode == "blocks_pp":
            if not self.pp_flag:
                logger.warning("blocks_pp mode requires pp_flag to be enabled, falling back to blocks mode")
                original_fn = self.forward_blocks
                self.forward_blocks = torch.compile(original_fn, **kwargs)
            else:
                original_fn = self.forward_blocks_pp
                self.forward_blocks_pp = torch.compile(original_fn, **kwargs)
            logger.info(f"WanModel compiled: mode={mode}")
        elif mode == "full":
            # Store original forward for fallback
            self._original_forward = self.forward
            self.forward = torch.compile(self.forward, **kwargs)
            logger.info(f"WanModel compiled: mode={mode}")
        else:
            raise ValueError(f"Unknown compile mode: {mode}")

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        cn_latents: torch.Tensor | None = None,
        clip_feature: torch.Tensor | None = None,
        cond_flag: bool = True,
        add_condition: torch.Tensor | None = None,
        y_camera: torch.Tensor | None = None,
        sparse_state: SparseAttentionState | None = None,
        freq_start_idx: int = 0,
        freq_interval: int = 1,
    ) -> torch.Tensor:
        # Get grid size before patchify for seq_len calculation
        # x shape: [B, C, T, H, W]
        batch_size = x.shape[0]
        f, h, w = x.shape[2], x.shape[3] // self.patch_size[1], x.shape[4] // self.patch_size[2]
        seq_len = f * h * w

        # Handle per-token timestep for I2V conditioning
        # timestep can be:
        # - 1D tensor [B]: scalar timestep (T2V mode) -> global modulation
        # - 2D tensor [B, seq_len]: per-token timestep (I2V mode) -> per-token modulation
        with amp.autocast("cuda", dtype=torch.float32):
            if timestep.dim() == 1:
                # Scalar timestep: compute global time embedding and modulation
                t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
                t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
                # Expand t_mod to [B, seq_len, 6, dim] for uniform processing in DiTBlock
                t_mod = t_mod.unsqueeze(1).expand(-1, seq_len, -1, -1)
                # Expand t to [B, 1, dim] for uniform processing in Head
                t = t.unsqueeze(1)
            else:
                # Per-token timestep: compute per-token time embedding and modulation
                # timestep shape: [B, seq_len]
                t_flat = timestep.flatten()  # [B * seq_len]
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_flat))
                e = e.unflatten(0, (batch_size, seq_len))  # [B, seq_len, dim]
                t_mod = self.time_projection(e).unflatten(2, (6, self.dim))  # [B, seq_len, 6, dim]
                # t is already [B, seq_len, dim] for per-token mode
                t = e

        t = t.to(self.dtype)
        t_mod = t_mod.to(self.dtype)
        context = self.text_embedding(context)

        if self.has_image_input and clip_feature is not None:
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        x, (f, h, w) = self.patchify(x)
        if add_condition is not None:
            x = add_condition + x
        if cn_latents is not None:
            condition = cn_latents
            condition = rearrange(condition, "b c f h w -> b (f h w) c").contiguous()
            mean_x, std_x = torch.mean(x, dim=(1, 2), keepdim=True), torch.std(x, dim=(1, 2), keepdim=True)
            mean_condition = torch.mean(condition, dim=(1, 2), keepdim=True)
            std_condition = torch.std(condition, dim=(1, 2), keepdim=True)
            condition = (condition - mean_condition) * (std_x / (std_condition + 1e-5)) + mean_x
            x = x + condition * 0.2
        freqs = (
            torch.cat(
                [
                    self.freqs[0][freq_start_idx : freq_interval * f + freq_start_idx : freq_interval]
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

        # Precompute cos/sin from complex freqs to avoid complex tensor in compiled region
        # freqs is complex tensor, convert to real cos/sin tensors
        freqs_cos = freqs.real.contiguous()
        freqs_sin = freqs.imag.contiguous()

        if self.usp_flag:
            # Shard t_mod and t (if per-token) along seq_len dimension (dim 1)
            # For scalar timestep, t has seq_len=1 which broadcasts with any seq_len
            sequence_parallel_shard(self.device_mesh, [x, t_mod, t, freqs_cos, freqs_sin], seq_dims=[1, 1, 1, 0, 0])

        # Feature cache handling - step ID is managed internally
        ori_x = x
        if self.feature_cache.should_compute(cond_flag):
            x = self.forward_blocks(x, context, t_mod, freqs_cos, freqs_sin, sparse_state=sparse_state)
            self.feature_cache.update(x, ori_x, cond_flag)
        else:
            x = self.feature_cache.approximate(x, cond_flag)

        x = self.head(x, t)
        if self.usp_flag:
            (x,) = sequence_parallel_unshard(self.device_mesh, (x,), seq_dims=(1,), seq_lens=(f * h * w,))
        x = self.unpatchify(x, (f, h, w))
        return x

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()

    def get_fsdp_module_names(self) -> list[str]:
        return ["blocks"]

    def get_tp_plan(self):
        """Get tensor parallelism plan for the model."""
        from torch.distributed.tensor import Replicate, Shard
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            PrepareModuleInput,
            PrepareModuleOutput,
            RowwiseParallel,
        )

        tp_plan = {
            "text_embedding.0": ColwiseParallel(),
            "text_embedding.2": RowwiseParallel(),
            "time_embedding.0": ColwiseParallel(),
            "time_embedding.2": RowwiseParallel(),
            "time_projection.1": ColwiseParallel(output_layouts=Replicate()),
            "blocks.0": PrepareModuleInput(
                input_layouts=(Replicate(), None, None, None),
                desired_input_layouts=(Shard(1), None, None, None),
                use_local_output=True,
            ),
            "head": PrepareModuleOutput(
                output_layouts=Shard(1),
                desired_output_layouts=Replicate(),
                use_local_output=True,
            ),
        }
        for idx in range(len(self.blocks)):
            tp_plan.update(
                {
                    f"blocks.{idx}.self_attn": PrepareModuleInput(
                        input_layouts=(Shard(1), None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    f"blocks.{idx}.self_attn.q": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.self_attn.k": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.self_attn.v": ColwiseParallel(),
                    f"blocks.{idx}.self_attn.o": RowwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.self_attn.norm_q": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.self_attn.norm_k": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn": PrepareModuleInput(
                        input_layouts=(Shard(1), None),
                        desired_input_layouts=(Replicate(), None),
                    ),
                    f"blocks.{idx}.cross_attn.q": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.cross_attn.k": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.cross_attn.v": ColwiseParallel(),
                    f"blocks.{idx}.cross_attn.o": RowwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.cross_attn.norm_q": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.norm_k": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.cross_attn.k_img": ColwiseParallel(output_layouts=Shard(1)),
                    f"blocks.{idx}.cross_attn.v_img": ColwiseParallel(),
                    f"blocks.{idx}.cross_attn.norm_k_img": PrepareModuleOutput(
                        output_layouts=Shard(1),
                        desired_output_layouts=Shard(-1),
                    ),
                    f"blocks.{idx}.ffn": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    f"blocks.{idx}.ffn.0": ColwiseParallel(),
                    f"blocks.{idx}.ffn.2": RowwiseParallel(output_layouts=Shard(1)),
                }
            )
        return tp_plan


class WanModelStateDictConverter:
    """State dict converter for Wan video DiT."""

    def __init__(self):
        pass

    def from_diffusers(self, state_dict: dict) -> tuple[dict, dict]:
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        i2v_rename_dict = {
            "condition_embedder.image_embedder.ff.net.0.proj.bias": "img_emb.proj.1.bias",
            "condition_embedder.image_embedder.ff.net.0.proj.weight": "img_emb.proj.1.weight",
            "condition_embedder.image_embedder.ff.net.2.bias": "img_emb.proj.3.bias",
            "condition_embedder.image_embedder.ff.net.2.weight": "img_emb.proj.3.weight",
            "condition_embedder.image_embedder.norm1.bias": "img_emb.proj.0.bias",
            "condition_embedder.image_embedder.norm1.weight": "img_emb.proj.0.weight",
            "condition_embedder.image_embedder.norm2.bias": "img_emb.proj.4.bias",
            "condition_embedder.image_embedder.norm2.weight": "img_emb.proj.4.weight",
            "blocks.0.attn2.add_k_proj_bias": "blocks.0.cross_attn.k_img.bias",
            "blocks.0.attn2.add_k_proj.weight": "blocks.0.cross_attn.k_img.weight",
            "blocks.0.attn2.add_v_proj_bias": "blocks.0.cross_attn.v_img.bias",
            "blocks.0.attn2.add_v_proj.weight": "blocks.0.cross_attn.v_img.weight",
            "blocks.0.attn2.norm_added_k.weight": "blocks.0.cross_attn.norm_k_img.weight",
        }
        rename_dict.update(i2v_rename_dict)
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param

        # Model configurations based on hash
        weight_hash = hash_state_dict_keys(state_dict)
        config_map = {
            "cb104773c6c2cb6df4f9529ad5c60d0b": {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            },  # noqa: E501
            "7cf3a086b49216bded0728ce78d59687": {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            },  # noqa: E501
        }
        config = config_map.get(weight_hash, {})
        return state_dict_, config

    def from_official(self, state_dict: dict) -> tuple[dict, dict]:
        state_dict = {name: param for name, param in state_dict.items() if not name.startswith("vace")}
        state_dict_hash = hash_state_dict_keys(state_dict)

        # Model configurations based on hash
        config_map = {
            "9269f8db9040a9d860eaca435be61814": {
                "has_image_input": False,
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
            },  # noqa: E501
            "aafcfd9672c3a2456dc46e1cb6e52c70": {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            },  # noqa: E501
            "b3aba5f6fddb5e117640e751591db89f": {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            },  # noqa: E501
            "6bfcfb3b342cb286ce886889d519a77e": {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            },  # noqa: E501
            "6d6ccde6845b95ad9114ab993d917893": {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            },  # noqa: E501
            "349723183fc063b2bfc10bb2835cf677": {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6,
            },  # noqa: E501
            "efa44cddf936c70abd0ea28b6cbe946c": {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            },  # noqa: E501
            "3ef3b1f8e1dab83d5b71fd7b617f859f": {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True,
            },  # noqa: E501
            "1f5ab7703c6fc803fdded85ff040c316": {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
            },  # noqa: E501
            "5b013604280dd715f8457c6ed6d6a626": {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            },  # noqa: E501
            "4cf556355bc7e9b6545b38f4930f60b1": {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            },  # noqa: E501
            "9d0240d8e7650a9ec65b2b617cc9c357": {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
            },  # noqa: E501
            # Wan2.2 TI2V 5B - unified T2V/I2V model with blended latent approach
            "e1de6c02cdac79f8b739f4d3698cd216": {
                "has_image_input": False,  # Uses blended latent, no separate y input
                "patch_size": [1, 2, 2],
                "in_dim": 48,  # Blended latent (noise for uncond, image latent for cond)
                "dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "num_layers": 30,
                "eps": 1e-6,
            },  # noqa: E501
        }
        config = config_map.get(state_dict_hash, {})
        return state_dict, config


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

# Wan2.1 1.3B variants
register_model_config(None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "official")
register_model_config(None, "aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "official")
register_model_config(None, "6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "official")
register_model_config(None, "3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "official")
register_model_config(None, "b3aba5f6fddb5e117640e751591db89f", ["wan_video_dit"], [WanModel], "official")
register_model_config(None, "b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "official")
# Diffusers format
register_model_config(None, "cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers")
register_model_config(None, "7cf3a086b49216bded0728ce78d59687", ["wan_video_dit"], [WanModel], "diffusers")
# Wan2.2 A14B variants
register_model_config(None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "official")
register_model_config(None, "4cf556355bc7e9b6545b38f4930f60b1", ["wan_video_dit"], [WanModel], "official")
register_model_config(None, "47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "official")
register_model_config(None, "9d0240d8e7650a9ec65b2b617cc9c357", ["wan_video_dit"], [WanModel], "official")
# Wan2.2 5B TI2V
register_model_config(None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "official")
