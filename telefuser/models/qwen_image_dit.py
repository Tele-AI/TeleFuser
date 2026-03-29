from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig, AttnImplType, OffloadConfig
from telefuser.distributed.device_mesh import get_ulysses_group
from telefuser.distributed.parallel_shard import (
    sequence_parallel_shard,
    sequence_parallel_unshard,
)
from telefuser.distributed.ulysses_comm import (
    ulysses_gather_heads,
    ulysses_scatter_heads,
)
from telefuser.offload.async_offload import AsyncOffloadManager
from telefuser.ops.activations import get_activation
from telefuser.ops.attention import long_context_attention as long_attn_func
from telefuser.ops.ffn import FeedForward
from telefuser.ops.normalization import AdaLayerNormContinuous, LayerNorm, RMSNorm, modulate
from telefuser.utils.logging import logger

from ..ops.attention import attention as attn_func


class TimestepEmbedding(nn.Module):
    """Timestep embedding with optional conditioning projection."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (DDPM-style)."""
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor | tuple[torch.Tensor],
    use_real: bool = False,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensors.

    Args:
        x: Query or key tensor [B, S, H, D].
        freqs_cis: Precomputed frequency tensor.
        use_real: Use real-valued cos/sin instead of complex.
        use_real_unbind_dim: Dimension to unbind for real mode.
    """
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


class QwenTimestepProjEmbeddings(nn.Module):
    """Timestep projection with optional additional conditioning."""

    def __init__(self, embedding_dim: int, use_additional_t_cond: bool = False):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype, addition_t_cond: torch.Tensor | None = None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=dtype))

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    """3D RoPE embedding for video with positive/negative frequency support."""

    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        """Compute RoPE parameters for given index and dimension."""
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        txt_seq_lens: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_fhw: [frame, height, width] or list of shapes.
            txt_seq_lens: List of text sequence lengths.
            device: Computation device.
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=128)
    @torch.compiler.disable()
    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        """Compute video frequencies with LRU caching.

        Disabled for torch.compile to avoid graph breaks from dynamic shape computation.
        """
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenDoubleStreamAttention(nn.Module):
    """Dual-stream attention for image and text with RoPE support."""

    usp_flag = False
    attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_k = RMSNorm(head_dim, eps=1e-6)

        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm(head_dim, eps=1e-6)

        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def async_usp_forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor | None = None,
        encoder_hidden_states_mask: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Async Ulysses-style sequence parallel forward."""
        group = get_ulysses_group(device_mesh)
        seq_txt = text.shape[1]

        img_value = self.to_v(image)
        txt_value = self.add_v_proj(text)
        img_value = img_value.unflatten(-1, (self.num_heads, -1))
        txt_value = txt_value.unflatten(-1, (self.num_heads, -1))
        joint_value = torch.cat([txt_value, img_value], dim=1)
        joint_value_wait = ulysses_scatter_heads(joint_value, group)

        img_query = self.to_q(image)
        txt_query = self.add_q_proj(text)
        img_query = img_query.unflatten(-1, (self.num_heads, -1))
        txt_query = txt_query.unflatten(-1, (self.num_heads, -1))
        if self.norm_q is not None:
            img_query = self.norm_q(img_query)
        if self.norm_added_q is not None:
            txt_query = self.norm_added_q(txt_query)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs)
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_query_wait = ulysses_scatter_heads(joint_query, group)

        img_key = self.to_k(image)
        txt_key = self.add_k_proj(text)
        img_key = img_key.unflatten(-1, (self.num_heads, -1))
        txt_key = txt_key.unflatten(-1, (self.num_heads, -1))
        if self.norm_k is not None:
            img_key = self.norm_k(img_key)
        if self.norm_added_k is not None:
            txt_key = self.norm_added_k(txt_key)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_key = apply_rotary_emb_qwen(img_key, img_freqs)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_key_wait = ulysses_scatter_heads(joint_key, group)

        joint_value = joint_value_wait()
        joint_query = joint_query_wait()
        joint_key = joint_key_wait()

        out = attn_func(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
        )
        out_wait = ulysses_gather_heads(out, group, num_heads=self.num_heads)
        joint_hidden_states = out_wait()

        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        if len(self.to_out) > 1:
            img_attn_output = self.to_out[1](img_attn_output)

        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.FloatTensor | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if self.usp_flag:
            return self.async_usp_forward(image, text, image_rotary_emb, attention_mask, device_mesh)
        return self.default_forward(image, text, image_rotary_emb, attention_mask, device_mesh)

    def default_forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.FloatTensor | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = img_q.unflatten(-1, (self.num_heads, -1))
        img_k = img_k.unflatten(-1, (self.num_heads, -1))
        img_v = img_v.unflatten(-1, (self.num_heads, -1))

        txt_q = txt_q.unflatten(-1, (self.num_heads, -1))
        txt_k = txt_k.unflatten(-1, (self.num_heads, -1))
        txt_v = txt_v.unflatten(-1, (self.num_heads, -1))

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)

        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=1)
        joint_k = torch.cat([txt_k, img_k], dim=1)
        joint_v = torch.cat([txt_v, img_v], dim=1)
        if self.usp_flag:
            joint_attn_out = long_attn_func(
                joint_q,
                joint_k,
                joint_v,
                attn_mask=attention_mask,
                attention_config=self.attention_config,
                input_layout="BSND",
                output_layout="BSND",
                device_mesh=device_mesh,
            )
        else:
            joint_attn_out = attn_func(
                joint_q,
                joint_k,
                joint_v,
                attn_mask=attention_mask,
                attention_config=self.attention_config,
                input_layout="BSND",
                output_layout="BSND",
            )
        joint_attn_out = rearrange(joint_attn_out, "b s n d -> b s (n d)", n=joint_q.shape[2])

        txt_attn_output = joint_attn_out[:, :seq_txt, :]
        img_attn_output = joint_attn_out[:, seq_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    """Transformer block with dual-stream attention and gated MLP."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenDoubleStreamAttention(
            dim_a=dim,
            dim_b=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
        )
        self.img_norm2 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _modulate(self, x: torch.Tensor, mod_params: torch.Tensor) -> tuple:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return modulate(x, shift.unsqueeze(1), scale.unsqueeze(1)), gate.unsqueeze(1)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_mod_attn, img_mod_mlp = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod_attn, txt_mod_mlp = self.txt_mod(temb).chunk(2, dim=-1)

        img_normed = self.img_norm1(image)
        img_modulated, img_gate = self._modulate(img_normed, img_mod_attn)

        txt_normed = self.txt_norm1(text)
        txt_modulated, txt_gate = self._modulate(txt_normed, txt_mod_attn)
        img_attn_out, txt_attn_out = self.attn(
            image=img_modulated,
            text=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            device_mesh=device_mesh,
        )

        image = image + img_gate * img_attn_out
        text = text + txt_gate * txt_attn_out

        img_normed_2 = self.img_norm2(image)
        img_modulated_2, img_gate_2 = self._modulate(img_normed_2, img_mod_mlp)

        txt_normed_2 = self.txt_norm2(text)
        txt_modulated_2, txt_gate_2 = self._modulate(txt_normed_2, txt_mod_mlp)

        img_mlp_out = self.img_mlp(img_modulated_2)
        txt_mlp_out = self.txt_mlp(txt_modulated_2)

        image = image + img_gate_2 * img_mlp_out
        text = text + txt_gate_2 * txt_mlp_out

        return text, image


class QwenImageDiT(BaseModel):
    """Diffusion Transformer for Qwen-Image generation."""

    def __init__(self, num_layers: int = 60):
        super().__init__()

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=3072, use_additional_t_cond=False)

        self.txt_norm = RMSNorm(3584, eps=1e-6)

        self.img_in = nn.Linear(64, 3072)
        self.txt_in = nn.Linear(3584, 3072)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(3072, 3072, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(3072, 64)
        self.layer_name_list = ["transformer_blocks"]
        self.async_offload_manager = None

    def enable_quant(self, quant_type: str | torch.dtype):
        """Enable quantization for transformer blocks."""
        if quant_type in [torch.float8_e4m3fn]:
            logger.info(f"loading weights with {quant_type}, start convert linear layer to {quant_type}")
            from telefuser.ops.quantized_linear import replace_linear_layers

            replace_linear_layers(self.transformer_blocks, quant_type)
            self.quant_type = quant_type

    def get_fsdp_module_names(self) -> list[str]:
        return ["transformer_blocks"]

    def enable_usp(self):
        logger.info("enable usp for qwen image dit")
        self.usp_flag = True
        QwenDoubleStreamAttention.usp_flag = True

    def forward(
        self,
        latents: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        prompt_emb: torch.Tensor | None = None,
        prompt_emb_mask: torch.Tensor | None = None,
        edit_latents: torch.Tensor | None = None,
        cond_flag: bool = True,
    ) -> torch.Tensor:
        self.feature_cache_hook.mark_step_begin(cond_flag)

        img_shapes = [(1, latents.shape[2] // 2, latents.shape[3] // 2)]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        if prompt_emb_mask.dtype != torch.bool:
            prompt_emb_mask = prompt_emb_mask.to(torch.bool)
        prompt_emb = prompt_emb[:, : txt_seq_lens[0], :]
        timestep = timestep / 1000

        image = rearrange(
            latents,
            "B C (H P) (W Q) -> B (H W) (C P Q)",
            H=img_shapes[0][1],
            W=img_shapes[0][2],
            P=2,
            Q=2,
        )
        image_seq_len = image.shape[1]

        if edit_latents is not None:
            edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
            img_shapes += [(1, e.shape[2] // 2, e.shape[3] // 2) for e in edit_latents_list]
            edit_image = [
                rearrange(
                    e,
                    "B C (H P) (W Q) -> B (H W) (C P Q)",
                    H=e.shape[2] // 2,
                    W=e.shape[3] // 2,
                    P=2,
                    Q=2,
                )
                for e in edit_latents_list
            ]
            image = torch.cat([image] + edit_image, dim=1)
        image = self.img_in(image)
        conditioning = self.time_text_embed(timestep, image.dtype)
        text = self.txt_in(self.txt_norm(prompt_emb))
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        attention_mask = None
        img_freqs, txt_freqs = image_rotary_emb
        if self.usp_flag:
            sequence_parallel_shard(self.device_mesh, (image, text, img_freqs, txt_freqs), seq_dims=(1, 1, 0, 0))
        image_rotary_emb = (img_freqs, txt_freqs)

        ori_image = image
        cached_output = self.feature_cache_hook.pre_forward(image, cond_flag)
        if cached_output is None:
            image = self.forward_blocks(image, text, conditioning, image_rotary_emb, attention_mask)
            self.feature_cache_hook.post_forward(image, ori_image, cond_flag)
        else:
            image = cached_output

        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)
        if self.usp_flag:
            (image,) = sequence_parallel_unshard(self.device_mesh, (image,), seq_dims=(1,), seq_lens=(image_seq_len,))
        image = image[:, :image_seq_len]

        latents = rearrange(
            image,
            "B (H W) (C P Q) -> B C (H P) (W Q)",
            H=img_shapes[0][1],
            W=img_shapes[0][2],
            P=2,
            Q=2,
        )
        return latents

    @staticmethod
    def state_dict_converter():
        return QwenImageDitStateDictConverter()

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
            mark_static(QwenImageDiT)
            mark_static(QwenImageTransformerBlock)
            mark_static(QwenDoubleStreamAttention)
            mark_static(QwenEmbedRope)
        except ImportError:
            logger.warning("torch._dynamo.mark_static not available, skipping static marking")

        # Compile based on mode
        if mode == "blocks":
            original_fn = self.forward_blocks
            self.forward_blocks = torch.compile(original_fn, **kwargs)
            logger.info(f"QwenImageDiT compiled: mode={mode}")
        elif mode == "full":
            # Store original forward for fallback
            self._original_forward = self.forward
            self.forward = torch.compile(self.forward, **kwargs)
            logger.info(f"QwenImageDiT compiled: mode={mode}")
        else:
            raise ValueError(f"Unknown compile mode: {mode}")

    def forward_blocks(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        conditioning: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        attention_mask: torch.Tensor | None,
    ):
        for block in self.transformer_blocks:
            text, image = block(
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                device_mesh=self.device_mesh,
            )
        return image

    def enable_async_offload(self, device: torch.device, offload_config: OffloadConfig):
        logger.info("enable async offload for qwen image dit")
        self.async_offload_manager = AsyncOffloadManager(
            self.transformer_blocks,
            enabled=True,
            offload_ratio=offload_config.offload_ratio,
            prefetch_size=offload_config.prefetch_size,
            device=device,
            pin_cpu_memory=offload_config.pin_cpu_memory,
        )
        self.async_offload_flag = True


class QwenImageDitStateDictConverter:
    """State dict converter for Qwen-Image DiT."""

    def __init__(self):
        pass

    def from_diffusers(self, state_dict: dict) -> dict:
        return state_dict

    def from_official(self, state_dict: dict) -> dict:
        return state_dict
