"""HunyuanVideo VAE model implementation for TeleFuser.

Based on the official HunyuanVideo-1.5 VAE implementation.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from telefuser.core.base_model import BaseModel
from telefuser.utils.logging import logger


@dataclass
class AutoencoderKLOutput:
    """Output from VAE encode method."""

    latent_dist: "DiagonalGaussianDistribution"


@dataclass
class DecoderOutput:
    """Output from VAE decode method."""

    sample: torch.Tensor
    posterior: Optional["DiagonalGaussianDistribution"] = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for Channel-First or Last."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


def swish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    """Applies the swish activation function (SiLU) with optional inplace support."""
    return F.silu(x, inplace=inplace)


class CausalConv3d(nn.Module):
    """Causal Conv3d with configurable padding for temporal axis."""

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "replicate",
        disable_causal: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
            )
        else:
            # W, H, T padding
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)
        self.time_causal_padding = padding

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class AttnBlock(nn.Module):
    """Self-attention block for 3D video tensors."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = RMSNorm(in_channels, images=False)

        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()

        # Causal attention mask (2D: [seq_len, seq_len] broadcasts to [batch, 1, seq_len, seq_len])
        attention_mask = self._prepare_causal_attention_mask(f, h * w, h_.dtype, h_.device, batch_size=b)
        h_ = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def _prepare_causal_attention_mask(
        self, n_frame: int, n_hw: int, dtype: torch.dtype, device: torch.device, batch_size: int
    ) -> torch.Tensor:
        """Prepare a causal attention mask for 3D videos."""
        seq_len = n_frame * n_hw
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        for i in range(seq_len):
            i_frame = i // n_hw
            mask[i, : (i_frame + 1) * n_hw] = 0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """ResNet-style block for 3D video tensors."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = RMSNorm(in_channels, images=False)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = RMSNorm(out_channels, images=False)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = swish(h, inplace=True)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h, inplace=True)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    """Downsampling module with optional temporal downsampling."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = CausalConv3d(in_channels, out_channels // factor, kernel_size=3)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # Shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            B, C, T, H, W = x_first.shape
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = x_next.shape
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        return h + shortcut


class Upsample(nn.Module):
    """Hierarchical upsampling with temporal/ spatial support."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = CausalConv3d(in_channels, out_channels * factor, kernel_size=3)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # Shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut


class Encoder(nn.Module):
    """Hierarchical video encoder with temporal and spatial factorization."""

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        self.np = np

        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # Downsampling
        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= np.log2(ffactor_spatial / ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = Downsample(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # End
        self.norm_out = RMSNorm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        # Middle
        h = self.mid.block_1(h)  # type: ignore
        h = self.mid.attn_1(h)  # type: ignore
        h = self.mid.block_2(h)  # type: ignore

        # End
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h, inplace=True)
        h = self.conv_out(h)
        h = h + shortcut
        return h


class Decoder(nn.Module):
    """Hierarchical video decoder with upsampling factories."""

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.np = np

        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        block_in = block_out_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = Upsample(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # End
        self.norm_out = RMSNorm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z to block_in
        repeats = self.block_out_channels[0] // self.z_channels
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # Middle
        h = self.mid.block_1(h)  # type: ignore
        h = self.mid.attn_1(h)  # type: ignore
        h = self.mid.block_2(h)  # type: ignore

        # Upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # End
        h = self.norm_out(h)
        h = swish(h, inplace=True)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=parameters.device, dtype=parameters.dtype)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None) -> torch.Tensor:
        if other is None:
            return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])
        return 0.5 * torch.sum(
            (self.mean - other.mean).pow(2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=[1, 2, 3, 4],
        )

    def mode(self) -> torch.Tensor:
        return self.mean


def randn_tensor(
    shape: tuple,
    generator: Optional[torch.Generator] = None,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate random tensor with optional generator."""
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


class HunyuanVideoVAE(BaseModel):
    """HunyuanVideo VAE with KL regularization and 3D causal convolutions."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        ffactor_spatial: int = 8,
        ffactor_temporal: int = 4,
        scaling_factor: float = 0.476986,
        shift_factor: Optional[float] = None,
        sample_size: int = 256,
        sample_tsize: int = 64,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=tuple(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
        )

        self.use_slicing = False
        self.use_spatial_tiling = False

        # Tiling parameters
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> "HunyuanVideoVAE":
        """Load VAE from pretrained checkpoint.

        This method loads a HunyuanVideo VAE from a pretrained checkpoint directory.
        It reads the config.json and model weights from the specified path.

        Args:
            pretrained_model_name_or_path: Path to the pretrained model directory
            torch_dtype: Data type for the model
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Loaded HunyuanVideoVAE model
        """
        import json

        from telefuser.utils.model_weight import load_state_dict

        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model with config
        model = cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            latent_channels=config.get("latent_channels", 16),
            block_out_channels=tuple(config.get("block_out_channels", [128, 256, 512, 512])),
            layers_per_block=config.get("layers_per_block", 2),
            ffactor_spatial=config.get("ffactor_spatial", 8),
            ffactor_temporal=config.get("ffactor_temporal", 4),
            scaling_factor=config.get("scaling_factor", 0.476986),
            shift_factor=config.get("shift_factor", None),
            sample_size=config.get("sample_size", 256),
            sample_tsize=config.get("sample_tsize", 64),
        )

        # Load state dict (try-except to avoid TOCTOU)
        checkpoint_names = [
            "diffusion_pytorch_model.safetensors",
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
                f"Model weights not found in {pretrained_model_name_or_path}. Searched for: {checkpoint_names}"
            )

        # Load state dict
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            logger.warning(f"Missing keys when loading VAE: {result.missing_keys}")
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys when loading VAE: {result.unexpected_keys}")

        model = model.to(dtype=torch_dtype)
        logger.info(f"Loaded HunyuanVideoVAE from {loaded_path}")

        return model

    def set_tile_sample_min_size(self, sample_size: int, tile_overlap_factor: float = 0.2):
        """Set tile sample minimum size for tiling."""
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // self.ffactor_spatial
        self.tile_overlap_factor = tile_overlap_factor

    def enable_spatial_tiling(self, use_tiling: bool = True):
        """Enable spatial tiling for large resolution videos."""
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        """Disable spatial tiling."""
        self.use_spatial_tiling = False

    def enable_tiling(self, use_tiling: bool = True):
        """Enable tiling (spatial only for this VAE)."""
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        """Disable tiling."""
        self.disable_spatial_tiling()

    def enable_slicing(self):
        """Enable slicing for batch processing."""
        self.use_slicing = True

    def disable_slicing(self):
        """Disable slicing."""
        self.use_slicing = False

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend tensor b horizontally into a at blend_extent region."""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        """Blend tensor b vertically into a at blend_extent region."""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Tiled spatial encoding for large inputs via overlapping."""
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def spatial_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        """Tiled spatial decoding for large latents."""
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient processing."""
        original_use_slicing = self.use_slicing
        original_use_spatial_tiling = self.use_spatial_tiling

        self.enable_slicing()
        self.enable_tiling()
        yield
        self.use_slicing = original_use_slicing
        self.use_spatial_tiling = original_use_spatial_tiling

    def encode(self, x: torch.Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, Tuple]:
        """Encode video to latent distribution.

        Args:
            x: Video tensor [B, C, T, H, W]
            return_dict: Whether to return a dataclass output

        Returns:
            AutoencoderKLOutput with latent_dist attribute, or tuple
        """

        def _encode(x):
            if self.use_spatial_tiling and (
                x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
            ):
                return self.spatial_tiled_encode(x)
            return self.encoder(x)

        assert len(x.shape) == 5  # (B, C, T, H, W)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None) -> Union[DecoderOutput, Tuple]:
        """Decode latent to video.

        Args:
            z: Latent tensor
            return_dict: Whether to return a dataclass output
            generator: Optional random generator (unused, for API compatibility)

        Returns:
            DecoderOutput with sample attribute, or tuple
        """

        def _decode(z):
            if self.use_spatial_tiling and (
                z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_posterior: bool = True,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, Tuple]:
        """Forward autoencoder pass.

        Args:
            sample: Video tensor [B, C, T, H, W]
            sample_posterior: Whether to sample from posterior
            return_posterior: Whether to return posterior in output
            return_dict: Whether to return a dataclass output

        Returns:
            DecoderOutput with sample and optionally posterior
        """
        posterior = self.encode(sample).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec, posterior) if return_posterior else (dec,)

        return DecoderOutput(sample=dec, posterior=posterior if return_posterior else None)

    def get_latent_size(self, video_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get latent size from video size (T, H, W)."""
        t, h, w = video_size
        return (
            t // self.ffactor_temporal,
            h // self.ffactor_spatial,
            w // self.ffactor_spatial,
        )

    def get_video_size(self, latent_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get video size from latent size (T, H, W)."""
        t, h, w = latent_size
        return (
            t * self.ffactor_temporal,
            h * self.ffactor_spatial,
            w * self.ffactor_spatial,
        )

    @staticmethod
    def state_dict_converter():
        return HunyuanVideoVAEStateDictConverter()


class HunyuanVideoVAEStateDictConverter:
    """State dict converter for HunyuanVideo VAE."""

    def __init__(self):
        pass

    def from_official(self, state_dict: dict) -> Tuple[dict, dict]:
        """Convert from official HunyuanVideo format."""
        converted = {}

        for name, param in state_dict.items():
            new_name = name
            if name.startswith("vae."):
                new_name = name[4:]
            converted[new_name] = param

        config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 16,
            "block_out_channels": [128, 256, 512, 512],
            "layers_per_block": 2,
            "ffactor_spatial": 8,
            "ffactor_temporal": 4,
        }

        return converted, config
