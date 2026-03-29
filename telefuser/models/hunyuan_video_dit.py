# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

"""HunyuanVideo DiT model implementation for TeleFuser.

This implementation is based on the original HunyuanVideo-1.5 repository:
https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5
"""

from __future__ import annotations

import math
from functools import cache, partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.ops import LayerNorm, RMSNorm, modulate
from telefuser.ops.attention import attention as attn_func
from telefuser.utils.logging import logger

# =============================================================================
# Utility Functions
# =============================================================================


def to_2tuple(x):
    """Convert to 2-tuple."""
    if isinstance(x, tuple):
        return x
    return (x, x)


# =============================================================================
# Rotary Position Embedding (from posemb_layers.py)
# =============================================================================


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (torch.Tensor): [dim, ...]
    """
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)

    return grid


def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """Reshape frequency tensor for broadcasting it with another tensor.

    Args:
        freqs_cis: Frequency tensor to be reshaped.
        x: Target tensor for broadcasting compatibility.
        head_first: head dimension first (except batch dim) or not.

    Returns:
        Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using the given frequency tensor.

    Args:
        xq: Query tensor to apply rotary embeddings. [B, S, H, D]
        xk: Key tensor to apply rotary embeddings. [B, S, H, D]
        freqs_cis: Precomputed frequency tensor for complex exponential.
        head_first: head dimension first (except batch dim) or not.

    Returns:
        Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


@cache
def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Precompute the frequency tensor for complex exponential (cis).

    Args:
        dim: Dimension of the frequency tensor.
        pos: Position indices for the frequency tensor. [S] or scalar
        theta: Scaling factor for frequency computation. Defaults to 10000.0.
        use_real: If True, return real part and imaginary part separately.
        theta_rescale_factor: Rescale factor for theta. Defaults to 1.0.
        interpolation_factor: Interpolation factor for position. Defaults to 1.0.

    Returns:
        freqs_cis or (freqs_cos, freqs_sin)
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, [S, D/2]
        return freqs_cis


@cache
def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """N-d version of precompute_freqs_cis for tokens with n-d structure.

    Args:
        rope_dim_list: Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start: See get_meshgrid_nd.
        *args: See get_meshgrid_nd.
        theta: Scaling factor for frequency computation. Defaults to 10000.0.
        use_real: If True, return real part and imaginary part separately.
        theta_rescale_factor: Rescale factor for theta.
        interpolation_factor: Interpolation factor for position.

    Returns:
        pos_embed: [HW, D/2] or (cos, sin)
    """
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list)

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list)

    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)
        return emb


# =============================================================================
# Activation Layers
# =============================================================================


def get_activation_layer(act_type: str = "gelu_tanh"):
    """Get activation layer by type."""
    if act_type == "gelu_tanh":
        return lambda: nn.GELU(approximate="tanh")
    elif act_type == "gelu":
        return nn.GELU
    elif act_type == "silu":
        return nn.SiLU
    elif act_type == "relu":
        return nn.ReLU
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


# =============================================================================
# Normalization Layers
# =============================================================================


def get_norm_layer(norm_type: str = "rms"):
    """Get normalization layer by type."""
    if norm_type == "rms":
        return RMSNorm
    elif norm_type == "layer":
        return LayerNorm
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# =============================================================================
# Modulate Layers
# =============================================================================


class ModulateDiT(nn.Module):
    """Modulation layer for DiT."""

    def __init__(
        self,
        hidden_size: int,
        factor: int,
        act_layer: Callable = None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        if act_layer is None:
            act_layer = nn.SiLU
        self.act = act_layer() if not isinstance(act_layer, type) else act_layer()
        self.linear = nn.Linear(hidden_size, factor * hidden_size, bias=True, **factory_kwargs)
        # Zero-initialize the modulation
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))


def apply_gate(x, gate=None, tanh=False):
    """Apply gate to tensor."""
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)


# =============================================================================
# Embedding Layers
# =============================================================================


def timestep_embedding(t, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=t.device
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size,
        act_layer=None,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        if act_layer is None:
            act_layer = nn.SiLU

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer() if not isinstance(act_layer, type) else act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class PatchEmbed(nn.Module):
    """3D Patch Embedding for video latents."""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        is_reshape_temporal_channels=False,
        concat_condition=True,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        # Ensure patch_size is a tuple of 3 ints for 3D conv
        if isinstance(patch_size, (list, tuple)) and len(patch_size) == 3:
            self.patch_size = tuple(patch_size)
        elif isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
            self.patch_size = (1, patch_size[0], patch_size[1])
        elif isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size, patch_size)
        else:
            patch_size_tuple = to_2tuple(patch_size)
            self.patch_size = (1, patch_size_tuple[0], patch_size_tuple[1])
        self.flatten = flatten

        # Only support concat mode (multitask mask training)
        orig_in_chans = in_chans
        if concat_condition:
            if is_reshape_temporal_channels:
                in_chans = in_chans + in_chans // 2 + 1
            else:
                in_chans = in_chans * 2 + 1

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs,
        )

        nn.init.xavier_uniform_(
            self.proj.weight[:, :orig_in_chans].view(self.proj.weight[:, :orig_in_chans].size(0), -1)
        )
        # Special initialization for concat mode
        nn.init.zeros_(self.proj.weight[:, orig_in_chans:].view(self.proj.weight[:, orig_in_chans:].size(0), -1))

        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """Projects text embeddings."""

    def __init__(self, in_channels, hidden_size, act_layer=None, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        if act_layer is None:
            act_layer = nn.SiLU
        self.linear_1 = nn.Linear(in_features=in_channels, out_features=hidden_size, bias=True, **factory_kwargs)
        self.act_1 = act_layer() if not isinstance(act_layer, type) else act_layer()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True, **factory_kwargs)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class VisionProjection(nn.Module):
    """Vision embedding projection."""

    def __init__(self, input_dim, output_dim, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim, **factory_kwargs),
            nn.Linear(input_dim, input_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(input_dim, output_dim, **factory_kwargs),
            nn.LayerNorm(output_dim, **factory_kwargs),
        )

    def forward(self, vision_embeds):
        return self.proj(vision_embeds)


# =============================================================================
# ByT5 Mapper for Glyph Text Rendering
# =============================================================================


class ByT5Mapper(nn.Module):
    """Maps ByT5 encoder outputs to DiT embedding space.

    Args:
        in_dim: Input dimension (ByT5 output, 1472 for byt5-small)
        out_dim: Intermediate dimension
        hidden_dim: Hidden dimension for intermediate layer
        out_dim1: Final output dimension (DiT hidden size)
        use_residual: Whether to use residual connection
    """

    def __init__(
        self,
        in_dim: int = 1472,
        out_dim: int = 3072,
        hidden_dim: int = 12288,
        out_dim1: int = 3072,
        use_residual: bool = False,
    ):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim, "in_dim must equal out_dim for residual connection"
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim1)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., in_dim)

        Returns:
            Output tensor of shape (..., out_dim1)
        """
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x2 = self.act_fn(x)
        x2 = self.fc3(x2)
        if self.use_residual:
            x2 = x2 + residual
        return x2


# =============================================================================
# MLP Layers
# =============================================================================


class MLP(nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=None,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if act_layer is None:
            act_layer = nn.GELU
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_channels, hidden_channels, bias=bias[0], **factory_kwargs)
        self.act = act_layer() if not isinstance(act_layer, type) else act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_channels, **factory_kwargs) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_channels, out_features, bias=bias[1], **factory_kwargs)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LinearWarpforSingle(nn.Module):
    """Linear warp for single stream block output."""

    def __init__(self, in_dim: int, out_dim: int, bias=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias, **factory_kwargs)

    def forward(self, x, y):
        input = torch.cat([x.contiguous(), y.contiguous()], dim=2).contiguous()
        return self.fc(input)


class MLPEmbedder(nn.Module):
    """MLP embedder for vector inputs."""

    def __init__(self, in_dim: int, hidden_dim: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels, act_layer=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if act_layer is None:
            act_layer = nn.SiLU

        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        if isinstance(patch_size, int):
            self.linear = nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
                **factory_kwargs,
            )
        else:
            out_size = (
                patch_size[0] * patch_size[1] * patch_size[2] if len(patch_size) == 3 else patch_size[0] * patch_size[1]
            ) * out_channels
            self.linear = nn.Linear(
                hidden_size,
                out_size,
                bias=True,
            )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            act_layer() if not isinstance(act_layer, type) else act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


# =============================================================================
# Attention
# =============================================================================


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    attention_config: Optional[AttentionConfig] = None,
) -> torch.Tensor:
    """Compute attention using unified attention implementation.

    Args:
        q: Query tensor of shape [B, L, H, D]
        k: Key tensor of shape [B, L, H, D]
        v: Value tensor of shape [B, L, H, D]
        causal: Whether to apply causal masking.
        attention_config: Attention configuration. If None, uses FLASH_ATTN_2.

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    if attention_config is None:
        attention_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)

    x = attn_func(
        q,
        k,
        v,
        attention_config=attention_config,
        input_layout="BSND",
        output_layout="BSND",
        is_causal=causal,
    )
    b, s, h, d = x.shape
    out = x.reshape(b, s, -1)
    return out


def parallel_attention(
    q,
    k,
    v,
    attention_config: Optional[AttentionConfig] = None,
):
    """Parallel attention for image and text tokens using unified attention implementation.

    Args:
        q: Tuple of (query, encoder_query) tensors of shape [B, L, H, D]
        k: Tuple of (key, encoder_key) tensors of shape [B, L, H, D]
        v: Tuple of (value, encoder_value) tensors of shape [B, L, H, D]
        attention_config: Attention configuration. If None, uses FLASH_ATTN_2.

    Returns:
        Output tensor after attention of shape [B, L, H*D]
    """
    if attention_config is None:
        attention_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)

    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v

    # Concatenate image and text tokens
    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)

    hidden_states = attn_func(
        query,
        key,
        value,
        attention_config=attention_config,
        input_layout="BSND",
        output_layout="BSND",
        is_causal=False,
    )

    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)

    return hidden_states


# =============================================================================
# Token Refiner
# =============================================================================


class IndividualTokenRefinerBlock(nn.Module):
    """A single block for token refinement with self-attention and MLP."""

    attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        self.self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.self_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
            **factory_kwargs,
        )

        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q = self.self_attn_q_norm(q).to(v)
        k = self.self_attn_k_norm(k).to(v)
        # No mask applied - all tokens are valid
        attn = attention(q, k, v, attention_config=self.attention_config)
        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)
        return x


class IndividualTokenRefiner(nn.Module):
    """Stacks multiple IndividualTokenRefinerBlock modules."""

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        depth: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward pass."""
        for block in self.blocks:
            x = block(x, c)
        return x


class SingleTokenRefiner(nn.Module):
    """Single token refiner block for LLM text embedding refinement."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        heads_num: int,
        depth: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_embedder = nn.Linear(in_channels, hidden_size, bias=True, **factory_kwargs)
        act_layer = get_activation_layer(act_type)
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        self.c_embedder = TextProjection(in_channels, hidden_size, act_layer, **factory_kwargs)
        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            **factory_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward pass."""
        timestep_aware_representations = self.t_embedder(t)
        # Simple mean pooling
        context_aware_representations = x.mean(dim=1)
        context_aware_representations = self.c_embedder(context_aware_representations)
        c = timestep_aware_representations + context_aware_representations
        x = self.input_embedder(x)
        x = self.individual_token_refiner(x, c)
        return x


# =============================================================================
# DiT Blocks
# =============================================================================


class MMDoubleStreamBlock(nn.Module):
    """Double-stream attention block for image and text tokens."""

    attention_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num

        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        # Image stream
        self.img_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.img_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )

        # Text stream
        self.txt_mod = ModulateDiT(hidden_size, factor=6, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        block_idx=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Image modulation
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)

        # Text modulation
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Image attention
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)

        img_q = self.img_attn_q(img_modulated)
        img_k = self.img_attn_k(img_modulated)
        img_v = self.img_attn_v(img_modulated)
        img_q = rearrange(img_q, "B L (H D) -> B L H D", H=self.heads_num)
        img_k = rearrange(img_k, "B L (H D) -> B L H D", H=self.heads_num)
        img_v = rearrange(img_v, "B L (H D) -> B L H D", H=self.heads_num)
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
                f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            )
            img_q, img_k = img_qq, img_kk

        # Text attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_q = self.txt_attn_q(txt_modulated)
        txt_k = self.txt_attn_k(txt_modulated)
        txt_v = self.txt_attn_v(txt_modulated)
        txt_q = rearrange(txt_q, "B L (H D) -> B L H D", H=self.heads_num)
        txt_k = rearrange(txt_k, "B L (H D) -> B L H D", H=self.heads_num)
        txt_v = rearrange(txt_v, "B L (H D) -> B L H D", H=self.heads_num)
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Joint attention - no mask
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            attention_config=self.attention_config,
        )

        img_attn = attn[:, : img_q.shape[1]].contiguous()
        txt_attn = attn[:, img_q.shape[1] :].contiguous()

        # Image output
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )

        # Text output
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """Single-stream attention block."""

    attention_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # Separate Q, K, V, MLP projections (matching original implementation)
        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_k = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_v = nn.Linear(hidden_size, hidden_size, **factory_kwargs)
        self.linear1_mlp = nn.Linear(hidden_size, mlp_hidden_dim, **factory_kwargs)
        self.linear2 = LinearWarpforSingle(hidden_size + mlp_hidden_dim, hidden_size, bias=True, **factory_kwargs)

        act_layer = get_activation_layer(mlp_act_type)
        self.mlp_act = act_layer()

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)

        # Separate Q, K, V, MLP
        q = self.linear1_q(x_mod)
        k = self.linear1_k(x_mod)
        v = self.linear1_v(x_mod)
        mlp = self.linear1_mlp(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.heads_num)
        k = rearrange(k, "B L (H D) -> B L H D", H=self.heads_num)
        v = rearrange(v, "B L (H D) -> B L H D", H=self.heads_num)

        # Apply QK-Norm
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Split image and text for RoPE
        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]

        # Apply RoPE to image tokens only
        img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
        assert img_qq.shape == img_q.shape and img_kk.shape == img_k.shape, (
            f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        )
        img_q, img_k = img_qq, img_kk

        # Attention - no mask
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            attention_config=self.attention_config,
        )
        output = self.linear2(attn, self.mlp_act(mlp))

        return x + apply_gate(output, gate=mod_gate)


# =============================================================================
# Main DiT Model
# =============================================================================


class HunyuanVideoDiT(BaseModel):
    """HunyuanVideo Diffusion Transformer model.

    Args:
        patch_size: The size of the patch.
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hidden_size: The hidden size of the transformer backbone.
        heads_num: The number of attention heads.
        mlp_width_ratio: Width ratio for the transformer MLPs.
        mlp_act_type: Activation type for the transformer MLPs.
        mm_double_blocks_depth: Number of double-stream transformer blocks.
        mm_single_blocks_depth: Number of single-stream transformer blocks.
        rope_dim_list: Rotary embedding dim for t, h, w.
        qkv_bias: Use bias in qkv projection.
        qk_norm: Whether to use qk norm.
        qk_norm_type: Type of qk norm.
        guidance_embed: Use guidance embedding for distillation.
        text_projection: Text input projection. Default is "single_refiner".
        use_attention_mask: If to use attention mask.
        text_states_dim: Text encoder output dim.
        text_states_dim_2: Secondary text encoder output dim.
        text_pool_type: Type for text pooling.
        rope_theta: Rotary embedding theta parameter.
        glyph_byT5_v2: Use ByT5 glyph module.
        vision_projection: Vision condition embedding mode.
        vision_states_dim: Vision encoder states input dim.
        is_reshape_temporal_channels: For video VAE adaptation.
        use_cond_type_embedding: Use condition type embedding.
    """

    def __init__(
        self,
        patch_size: list = [1, 1, 1],
        in_channels: int = 32,
        concat_condition: bool = True,
        out_channels: int = None,
        hidden_size: int = 2048,
        heads_num: int = 16,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 54,
        mm_single_blocks_depth: int = 0,
        rope_dim_list: list = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,
        use_meanflow: bool = False,
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        text_states_dim: int = 3584,
        text_states_dim_2: int = None,
        text_pool_type: str = None,
        rope_theta: int = 256,
        glyph_byT5_v2: bool = True,
        vision_projection: str = "linear",
        vision_states_dim: int = 1152,
        is_reshape_temporal_channels: bool = False,
        use_cond_type_embedding: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection
        self.text_pool_type = text_pool_type
        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim
        self.glyph_byT5_v2 = glyph_byT5_v2
        self.use_cond_type_embedding = use_cond_type_embedding

        # ByT5 glyph support
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472, out_dim=hidden_size, hidden_dim=hidden_size, out_dim1=hidden_size, use_residual=False
            )
        else:
            self.byt5_in = None

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.img_in = PatchEmbed(
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            is_reshape_temporal_channels=is_reshape_temporal_channels,
            concat_condition=concat_condition,
            **factory_kwargs,
        )

        # Vision projection
        if vision_projection == "linear":
            self.vision_in = VisionProjection(
                input_dim=self.vision_states_dim, output_dim=self.hidden_size, **factory_kwargs
            )
        else:
            self.vision_in = None

        # Text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # Time modulation
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
        self.vector_in = (
            MLPEmbedder(text_states_dim_2, self.hidden_size, **factory_kwargs)
            if self.text_pool_type is not None
            else None
        )
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if guidance_embed
            else None
        )

        self.time_r_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs) if use_meanflow else None
        )

        # Double-stream blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # Single-stream blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        # Final layer
        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        # Condition type embedding
        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size)
            self.cond_type_embedding.weight.data.fill_(0)
            # 0: text_encoder feature
            # 1: byt5 feature
            # 2: vision_encoder feature
        else:
            self.cond_type_embedding = None

        self.dtype = torch.bfloat16
        self.layer_name_list = ["double_blocks", "single_blocks"]

    def enable_deterministic(self):
        """Enable deterministic mode for all blocks."""
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        """Disable deterministic mode for all blocks."""
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        """Get rotary position embeddings for given sizes.

        Args:
            rope_sizes: Tuple of (t, h, w) sizes.

        Returns:
            Tuple of (freqs_cos, freqs_sin) tensors.
        """
        target_ndim = 3
        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            tuple(rope_dim_list),
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        """Reorder text tokens by concatenating ByT5 and text encoder tokens.

        Args:
            byt5_txt: ByT5 text embeddings
            txt: Text encoder embeddings
            byt5_text_mask: ByT5 attention mask
            text_mask: Text encoder attention mask
            zero_feat: Whether to zero out padding features
            is_reorder: Whether to reorder tokens

        Returns:
            Tuple of (reordered embeddings, reordered mask)
        """
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    # When using block mask with approximate computation, set pad to zero to reduce error
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], pad_byt5, pad_text], dim=0
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [
                            byt5_txt_i[byt5_text_mask_i],
                            txt_i[text_mask_i],
                            byt5_txt_i[~byt5_text_mask_i],
                            txt_i[~text_mask_i],
                        ],
                        dim=0,
                    )
                reorder_mask_i = torch.cat(
                    [
                        byt5_text_mask_i[byt5_text_mask_i],
                        text_mask_i[text_mask_i],
                        byt5_text_mask_i[~byt5_text_mask_i],
                        text_mask_i[~text_mask_i],
                    ],
                    dim=0,
                )

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)

        return reorder_txt, reorder_mask

    def unpatchify(self, x, t, h, w):
        """Unpatchify a tensorized input back to frame format.

        Args:
            x: Input tensor of shape (N, T, patch_size**2 * C)
            t: Number of time steps
            h: Height in patch units
            w: Width in patch units

        Returns:
            Output tensor of shape (N, C, t * pt, h * ph, w * pw)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_states: torch.Tensor,
        text_states_2: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        timestep_r=None,
        vision_states: torch.Tensor = None,
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        guidance=None,
        mask_type="t2v",
        byt5_text_states: Optional[torch.Tensor] = None,
        byt5_text_mask: Optional[torch.Tensor] = None,
        cond_flag: bool = True,
    ) -> torch.Tensor:
        """Forward pass of HunyuanVideo DiT - simplified version.

        Args:
            hidden_states: Input latent tensor (B, C, T, H, W)
            timestep: Timestep tensor (B,)
            text_states: Text embeddings (B, L, D) - may contain padding
            text_states_2: Secondary text embeddings (B, D)
            encoder_attention_mask: Attention mask for text (B, L), 1 for valid, 0 for padding.
            timestep_r: Timestep for meanflow (optional)
            vision_states: Vision encoder states (optional)
            freqs_cos: Precomputed RoPE cos frequencies
            freqs_sin: Precomputed RoPE sin frequencies
            return_dict: Whether to return a dict
            guidance: Guidance scale for distilled models
            mask_type: Type of mask ("t2v" or "i2v")
            byt5_text_states: ByT5 embeddings for glyph text rendering (B, L, D)
            byt5_text_mask: ByT5 attention mask (B, L)
            cond_flag: True for conditional path, False for unconditional path

        Returns:
            Output tensor (B, C, T, H, W)
        """
        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)

        self.feature_cache_hook.mark_step_begin(cond_flag)

        img = hidden_states
        t = timestep
        txt = text_states
        text_mask = encoder_attention_mask

        # Extract valid tokens based on attention mask
        def extract_valid_tokens(embeddings, mask):
            """Extract valid tokens based on attention mask.

            Args:
                embeddings: (B, L, D) tensor
                mask: (B, L) tensor, 1 for valid, 0 for padding

            Returns:
                Tensor with only valid tokens, shape (B, L_valid, D)
            """
            if mask is None:
                return embeddings
            # For batch=1, simply index valid tokens
            if embeddings.shape[0] == 1:
                valid_indices = mask[0].nonzero(as_tuple=True)[0]
                return embeddings[:, valid_indices, :]
            # For batch > 1, we need to process each batch separately
            # This is simplified - assumes all batches have same valid length
            # In practice, batch>1 with variable lengths would need padding
            valid_indices = mask[0].nonzero(as_tuple=True)[0]
            return embeddings[:, valid_indices, :]

        # Extract valid text tokens
        if text_mask is not None:
            txt = extract_valid_tokens(txt, text_mask)

        bs, _, ot, oh, ow = img.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Get RoPE embeddings
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = self.get_rotary_pos_embed((tt, th, tw))

        # Patch embedding
        img = self.img_in(img)

        # Prepare modulation vectors
        vec = self.time_in(t)

        if text_states_2 is not None:
            vec_2 = self.vector_in(text_states_2)
            vec = vec + vec_2

        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(guidance)

        if timestep_r is not None:
            vec = vec + self.time_r_in(timestep_r)

        # Embed text tokens - no mask passed
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # Apply condition type embedding for text encoder features
        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
            txt = txt + cond_emb

        # Process ByT5 glyph features if enabled - direct concat, no reordering
        if self.glyph_byT5_v2 and self.byt5_in is not None and byt5_text_states is not None:
            # Extract valid ByT5 tokens if mask provided
            if byt5_text_mask is not None:
                byt5_text_states = extract_valid_tokens(byt5_text_states, byt5_text_mask)
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            # Ensure batch size matches txt before concatenation
            if byt5_txt.shape[0] != txt.shape[0]:
                if byt5_txt.shape[0] == 1 and txt.shape[0] > 1:
                    byt5_txt = byt5_txt.expand(txt.shape[0], -1, -1)
                elif byt5_txt.shape[0] > 1 and txt.shape[0] == 1:
                    byt5_txt = byt5_txt[:1]
            # Direct concatenation - no reorder_txt_token
            txt = torch.cat([byt5_txt, txt], dim=1)

        # Process vision features if enabled - direct concat, no reordering
        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states)
            # Skip vision tokens for t2v mode if vision_states is all zeros
            if mask_type == "t2v" and torch.all(vision_states == 0):
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
                # Don't concat zero vision tokens
            else:
                if self.cond_type_embedding is not None:
                    cond_emb = self.cond_type_embedding(
                        2
                        * torch.ones_like(
                            extra_encoder_hidden_states[:, :, 0],
                            dtype=torch.long,
                            device=extra_encoder_hidden_states.device,
                        )
                    )
                    extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb
                # Ensure batch size matches txt before concatenation
                if extra_encoder_hidden_states.shape[0] != txt.shape[0]:
                    if extra_encoder_hidden_states.shape[0] == 1 and txt.shape[0] > 1:
                        extra_encoder_hidden_states = extra_encoder_hidden_states.expand(txt.shape[0], -1, -1)
                    elif extra_encoder_hidden_states.shape[0] > 1 and txt.shape[0] == 1:
                        extra_encoder_hidden_states = extra_encoder_hidden_states[:1]
                # Direct concatenation - no reorder_txt_token
                txt = torch.cat([extra_encoder_hidden_states, txt], dim=1)

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        ori_img = img
        cached_output = self.feature_cache_hook.pre_forward(img, cond_flag)
        if cached_output is None:
            # Use compiled blocks if available
            if getattr(self, "_use_compiled_blocks", False):
                img, txt = self._compiled_forward_blocks(img, txt, vec, freqs_cis)
            else:
                # Pass through double-stream blocks - no mask passed
                for index, block in enumerate(self.double_blocks):
                    img, txt = block(
                        img=img,
                        txt=txt,
                        vec=vec,
                        freqs_cis=freqs_cis,
                        block_idx=index,
                    )

                txt_seq_len = txt.shape[1]
                img_seq_len = img.shape[1]

                # Merge image and text for single-stream blocks
                x = torch.cat((img, txt), 1)

                if len(self.single_blocks) > 0:
                    for index, block in enumerate(self.single_blocks):
                        x = block(
                            x=x,
                            vec=vec,
                            txt_len=txt_seq_len,
                            freqs_cis=(freqs_cos, freqs_sin),
                        )

                img = x[:, :img_seq_len, ...]
            self.feature_cache_hook.post_forward(img, ori_img, cond_flag)
        else:
            img = cached_output

        # Final Layer
        img = self.final_layer(img, vec)
        img = self.unpatchify(img, tt, th, tw)

        assert return_dict is False, "return_dict is not supported."

        return img

    def set_attention_config(self, attention_config: AttentionConfig):
        """Set attention implementation configuration."""
        logger.info(f"hunyuan video dit set attention config to {attention_config.attn_impl}")
        MMDoubleStreamBlock.attention_config = attention_config
        MMSingleStreamBlock.attention_config = attention_config
        IndividualTokenRefinerBlock.attention_config = attention_config

    def get_fsdp_module_names(self) -> list:
        """Return module names for FSDP sharding."""
        return ["double_blocks", "single_blocks"]

    def compile(self, mode: str = "blocks", **kwargs) -> None:
        """Compile model for better performance with torch.compile.

        Args:
            mode: Compilation mode:
                - "blocks": Compile only block processing (default, most effective)
                - "full": Compile entire forward method
            **kwargs: Arguments passed to torch.compile()
        """
        # Import mark_static from torch._dynamo
        try:
            from torch._dynamo import mark_static

            # Mark module classes as static (instance attributes won't change after compile)
            mark_static(HunyuanVideoDiT)
            mark_static(MMDoubleStreamBlock)
            mark_static(MMSingleStreamBlock)
            mark_static(IndividualTokenRefinerBlock)
            mark_static(IndividualTokenRefiner)
            mark_static(SingleTokenRefiner)
        except ImportError:
            logger.warning("torch._dynamo.mark_static not available, skipping static marking")

        # Compile based on mode
        if mode == "blocks":
            # Create a wrapper function for block processing
            def forward_blocks(
                img: torch.Tensor,
                txt: torch.Tensor,
                vec: torch.Tensor,
                freqs_cis: tuple,
            ) -> tuple:
                """Process through double and single stream blocks."""
                for index, block in enumerate(self.double_blocks):
                    img, txt = block(
                        img=img,
                        txt=txt,
                        vec=vec,
                        freqs_cis=freqs_cis,
                        block_idx=index,
                    )

                txt_seq_len = txt.shape[1]
                img_seq_len = img.shape[1]

                # Merge image and text for single-stream blocks
                x = torch.cat((img, txt), 1)

                if len(self.single_blocks) > 0:
                    freqs_cos, freqs_sin = freqs_cis if freqs_cis else (None, None)
                    for index, block in enumerate(self.single_blocks):
                        x = block(
                            x=x,
                            vec=vec,
                            txt_len=txt_seq_len,
                            freqs_cis=(freqs_cos, freqs_sin),
                        )

                img = x[:, :img_seq_len, ...]
                return img, txt

            self._compiled_forward_blocks = torch.compile(forward_blocks, **kwargs)
            self._use_compiled_blocks = True
            logger.info(f"HunyuanVideoDiT compiled: mode={mode}")
        elif mode == "full":
            # Store original forward for fallback
            self._original_forward = self.forward
            self.forward = torch.compile(self.forward, **kwargs)
            logger.info(f"HunyuanVideoDiT compiled: mode={mode}")
        else:
            raise ValueError(f"Unknown compile mode: {mode}")

    @staticmethod
    def state_dict_converter():
        """Return state dict converter."""
        return HunyuanVideoStateDictConverter()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ) -> "HunyuanVideoDiT":
        """Load HunyuanVideoDiT from pretrained checkpoint.

        This method loads a HunyuanVideo DiT from a pretrained checkpoint.
        It automatically detects the model configuration from the state dict.

        Args:
            pretrained_model_name_or_path: Path to the checkpoint file or directory
            torch_dtype: Data type for the model
            low_cpu_mem_usage: If True, use low memory mode for loading
            **kwargs: Additional arguments (ignored, for API compatibility)

        Returns:
            Loaded HunyuanVideoDiT model
        """
        import os

        from telefuser.utils.model_weight import load_state_dict

        # Load state dict from file or directory (try-except to avoid TOCTOU)
        if os.path.isfile(pretrained_model_name_or_path):
            state_dict = load_state_dict(pretrained_model_name_or_path)
            logger.info(f"Loaded state dict from {pretrained_model_name_or_path}")
        else:
            # Try common checkpoint file names
            checkpoint_names = [
                "diffusion_pytorch_model.safetensors",
                "model.safetensors",
                "diffusion_pytorch_model.bin",
            ]
            state_dict = None
            loaded_path = None
            for name in checkpoint_names:
                path = os.path.join(pretrained_model_name_or_path, name)
                try:
                    state_dict = load_state_dict(path)
                    loaded_path = path
                    break
                except FileNotFoundError:
                    continue
            if state_dict is None:
                raise FileNotFoundError(
                    f"No checkpoint found in {pretrained_model_name_or_path}. Searched for: {checkpoint_names}"
                )
            logger.info(f"Loaded state dict from {loaded_path}")

        # Get converter and detect config
        converter = cls.state_dict_converter()

        # Detect format from state dict keys and convert
        is_diffusers = any(k.startswith("transformer.") for k in state_dict.keys())

        if is_diffusers:
            converted_state_dict, config = converter.from_diffusers(state_dict)
        else:
            converted_state_dict, config = converter.from_official(state_dict)

        # Free memory: original state dict no longer needed
        del state_dict

        logger.info(
            f"Detected config: hidden_size={config['hidden_size']}, "
            f"double_blocks_depth={config['mm_double_blocks_depth']}, "
            f"single_blocks_depth={config['mm_single_blocks_depth']}"
        )

        # Initialize model with detected config
        if low_cpu_mem_usage:
            with torch.device("meta"):
                model = cls(**config)
        else:
            model = cls(**config)

        # Load weights
        model.load_state_dict(converted_state_dict, assign=True)
        model = model.to(dtype=torch_dtype)
        model.requires_grad_(False)
        model.eval()

        return model


class HunyuanVideoStateDictConverter:
    """State dict converter for HunyuanVideo DiT.

    Default configuration is for HunyuanVideo-1.5 720p T2V model.
    """

    # Default config for HunyuanVideo-1.5 720p T2V
    DEFAULT_CONFIG = {
        "patch_size": [1, 1, 1],
        "in_channels": 32,
        "concat_condition": True,
        "out_channels": 32,
        "hidden_size": 2048,
        "heads_num": 16,
        "mlp_width_ratio": 4.0,
        "mlp_act_type": "gelu_tanh",
        "mm_double_blocks_depth": 54,
        "mm_single_blocks_depth": 0,
        "rope_dim_list": [16, 56, 56],
        "rope_theta": 256,
        "qk_norm": True,
        "qk_norm_type": "rms",
        "qkv_bias": True,
        "guidance_embed": False,
        "use_meanflow": False,
        "text_projection": "single_refiner",
        "use_attention_mask": True,
        "text_states_dim": 3584,
        "text_states_dim_2": None,
        "text_pool_type": None,
        "glyph_byT5_v2": True,
        "vision_projection": "linear",
        "vision_states_dim": 1152,
        "is_reshape_temporal_channels": False,
        "use_cond_type_embedding": True,
    }

    def __init__(self):
        pass

    def _detect_config(self, state_dict: dict) -> dict:
        """Detect model configuration from state dict."""
        config = self.DEFAULT_CONFIG.copy()

        # Detect use_meanflow from presence of time_r_in module
        has_time_r_in = any("time_r_in" in name for name in state_dict.keys())
        if has_time_r_in:
            config["use_meanflow"] = True

        # Detect input channels from img_in.proj.weight shape
        # Shape is [out_channels, in_channels, kT, kH, kW]
        for name, param in state_dict.items():
            if "img_in.proj.weight" in name:
                input_channels = param.shape[1]
                # Normal model: in_channels * 2 + 1 (concat_condition=True)
                # SR model: in_channels * 3 + 2 (for SR pipeline)
                # We need to reverse-calculate the base in_channels
                # Try normal formula first: in_chans * 2 + 1
                if (input_channels - 1) % 2 == 0:
                    base_in_channels = (input_channels - 1) // 2
                    config["in_channels"] = base_in_channels
                    config["concat_condition"] = True
                # Try SR formula: in_chans * 3 + 2
                elif (input_channels - 2) % 3 == 0:
                    base_in_channels = (input_channels - 2) // 3
                    # For SR, we pass concat_condition=False with total input channels
                    config["concat_condition"] = False
                    config["in_channels"] = input_channels  # Use total channels directly
                    # Store base_in_channels for later out_channels detection
                    config["_base_in_channels"] = base_in_channels
                else:
                    # Unknown formula, use the input channels directly with concat_condition=False
                    config["in_channels"] = input_channels
                    config["concat_condition"] = False
                break

        # Detect output channels from final_layer.linear.weight
        # Shape: [patch_size^3 * out_channels, hidden_size] for 3D,
        # or [patch_size^2 * out_channels, hidden_size] for 2D
        for name, param in state_dict.items():
            if "final_layer.linear.weight" in name:
                # Get patch_size from config
                patch_size = config["patch_size"]
                if isinstance(patch_size, int):
                    patch_vol = patch_size**3
                else:
                    patch_vol = (
                        patch_size[0] * patch_size[1] * patch_size[2]
                        if len(patch_size) == 3
                        else patch_size[0] * patch_size[1]
                    )
                out_features = param.shape[0]
                # out_features = patch_vol * out_channels
                out_channels = out_features // patch_vol
                config["out_channels"] = out_channels
                break

        # Try to detect hidden_size and heads_num from state dict
        for name, param in state_dict.items():
            if "double_blocks.0.img_attn_q.weight" in name:
                hidden_size = param.shape[0]
                config["hidden_size"] = hidden_size
                # Infer heads_num: head_dim should be 128 (sum of rope_dim_list)
                # heads_num = hidden_size // head_dim
                head_dim = sum(config["rope_dim_list"])
                if hidden_size % head_dim == 0:
                    config["heads_num"] = hidden_size // head_dim
                break

        # Detect depth from state dict
        double_blocks_depth = 0
        single_blocks_depth = 0
        for name in state_dict.keys():
            if "double_blocks." in name:
                try:
                    idx = int(name.split("double_blocks.")[1].split(".")[0])
                    double_blocks_depth = max(double_blocks_depth, idx + 1)
                except (IndexError, ValueError):
                    pass
            if "single_blocks." in name:
                try:
                    idx = int(name.split("single_blocks.")[1].split(".")[0])
                    single_blocks_depth = max(single_blocks_depth, idx + 1)
                except (IndexError, ValueError):
                    pass

        if double_blocks_depth > 0:
            config["mm_double_blocks_depth"] = double_blocks_depth
        if single_blocks_depth > 0:
            config["mm_single_blocks_depth"] = single_blocks_depth

        # Clean up temporary keys
        config.pop("_base_in_channels", None)

        return config

    def from_diffusers(self, state_dict: dict) -> Tuple[dict, dict]:
        """Convert from diffusers format."""
        converted = {}

        # Basic mapping for diffusers format
        for name, param in state_dict.items():
            new_name = name
            # Handle common prefixes
            if name.startswith("transformer."):
                new_name = name[12:]
            converted[new_name] = param

        config = self._detect_config(state_dict)
        return converted, config

    def from_official(self, state_dict: dict) -> Tuple[dict, dict]:
        """Convert from official HunyuanVideo format (Civitai/official checkpoint)."""
        converted = {}

        for name, param in state_dict.items():
            new_name = name
            if name.startswith("model."):
                new_name = name[6:]
            elif name.startswith("module."):
                new_name = name[7:]
            converted[new_name] = param

        config = self._detect_config(state_dict)
        return converted, config
