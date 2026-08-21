"""LTX-2.5 latent spatial upsampler and checkpoint loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .checkpoint import inspect_checkpoint


class LTX25PerChannelStatistics(torch.nn.Module):
    """LTX video-latent normalization statistics stored in the VAE checkpoint."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(channels))
        self.register_buffer("mean-of-means", torch.zeros(channels))

    def un_normalize(self, latent: torch.Tensor) -> torch.Tensor:
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(latent)
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(latent)
        return latent * std + mean

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        std = self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(latent)
        mean = self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(latent)
        return (latent - mean) / std


class LTX25UpsamplerResBlock(torch.nn.Module):
    """The residual block used by the LTX-2.5 latent upsampler."""

    def __init__(self, channels: int, dims: int) -> None:
        super().__init__()
        conv = torch.nn.Conv2d if dims == 2 else torch.nn.Conv3d
        self.conv1 = conv(channels, channels, kernel_size=3, padding=1)
        self.norm1 = torch.nn.GroupNorm(32, channels)
        self.conv2 = conv(channels, channels, kernel_size=3, padding=1)
        self.norm2 = torch.nn.GroupNorm(32, channels)
        self.activation = torch.nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.activation(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        return self.activation(value + residual)


class LTX25PixelShuffleND(torch.nn.Module):
    """Channel-to-axis rearrangement matching the LTX-2.5 upsampler layout."""

    def __init__(self, dims: int) -> None:
        super().__init__()
        if dims not in (1, 2, 3):
            raise ValueError(f"Pixel shuffle dims must be 1, 2, or 3, got {dims}")
        self.dims = dims

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.dims == 2 and value.ndim == 4:
            batch, channels, height, width = value.shape
            if channels % 4:
                raise ValueError(f"2D pixel shuffle requires channels divisible by 4, got {channels}")
            value = value.reshape(batch, channels // 4, 2, 2, height, width)
            return value.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)
        if value.ndim != 5:
            raise ValueError(f"Pixel shuffle expects a 4D or 5D tensor, got {value.ndim}D")
        batch, channels, frames, height, width = value.shape
        if self.dims == 3:
            if channels % 8:
                raise ValueError(f"3D pixel shuffle requires channels divisible by 8, got {channels}")
            value = value.reshape(batch, channels // 8, 2, 2, 2, frames, height, width)
            return value.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(
                batch, channels // 8, frames * 2, height * 2, width * 2
            )
        if self.dims == 2:
            if channels % 4:
                raise ValueError(f"2D pixel shuffle requires channels divisible by 4, got {channels}")
            value = value.reshape(batch, channels // 4, 2, 2, frames, height, width)
            return value.permute(0, 1, 4, 5, 2, 6, 3).reshape(batch, channels // 4, frames, height * 2, width * 2)
        if channels % 2:
            raise ValueError(f"1D pixel shuffle requires channels divisible by 2, got {channels}")
        value = value.reshape(batch, channels // 2, 2, frames, height, width)
        return value.permute(0, 1, 3, 2, 4, 5).reshape(batch, channels // 2, frames * 2, height, width)


@dataclass(frozen=True, slots=True)
class LTX25SpatialUpsamplerConfig:
    """Architecture read from the spatial-upsampler checkpoint metadata."""

    in_channels: int
    mid_channels: int
    num_blocks_per_stage: int
    dims: int
    spatial_upsample: bool
    temporal_upsample: bool

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "LTX25SpatialUpsamplerConfig":
        config = metadata.get("config")
        if not isinstance(config, dict):
            raise ValueError("LTX-2.5 spatial upsampler is missing object config metadata")
        required = (
            "in_channels",
            "mid_channels",
            "num_blocks_per_stage",
            "dims",
            "spatial_upsample",
            "temporal_upsample",
        )
        missing = [field for field in required if field not in config]
        if missing:
            raise ValueError(f"LTX-2.5 spatial upsampler config is missing {missing}")
        return cls(**{field: config[field] for field in required})


class LTX25SpatialUpsampler(torch.nn.Module):
    """Faithful LTX-2.5 learned spatial latent upsampler."""

    def __init__(self, config: LTX25SpatialUpsamplerConfig) -> None:
        super().__init__()
        if not config.spatial_upsample or config.temporal_upsample or config.dims not in (2, 3):
            raise ValueError("LTX-2.5 distilled requires a spatial-only 2D or 3D latent upsampler")
        conv = torch.nn.Conv2d if config.dims == 2 else torch.nn.Conv3d
        self.config = config
        self.initial_conv = conv(config.in_channels, config.mid_channels, kernel_size=3, padding=1)
        self.initial_norm = torch.nn.GroupNorm(32, config.mid_channels)
        self.initial_activation = torch.nn.SiLU()
        self.res_blocks = torch.nn.ModuleList(
            [LTX25UpsamplerResBlock(config.mid_channels, config.dims) for _ in range(config.num_blocks_per_stage)]
        )
        self.upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(config.mid_channels, 4 * config.mid_channels, kernel_size=3, padding=1),
            LTX25PixelShuffleND(2),
        )
        self.post_upsample_res_blocks = torch.nn.ModuleList(
            [LTX25UpsamplerResBlock(config.mid_channels, config.dims) for _ in range(config.num_blocks_per_stage)]
        )
        self.final_conv = conv(config.mid_channels, config.in_channels, kernel_size=3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5:
            raise ValueError(f"LTX-2.5 latent upsampler expects [B, C, F, H, W], got {tuple(latent.shape)}")
        batch, _, frames, _, _ = latent.shape
        if self.config.dims == 2:
            value = latent.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, latent.shape[1], latent.shape[3], latent.shape[4]
            )
            value = self.initial_activation(self.initial_norm(self.initial_conv(value)))
            for block in self.res_blocks:
                value = block(value)
            value = self.upsampler(value)
            for block in self.post_upsample_res_blocks:
                value = block(value)
            value = self.final_conv(value)
            return value.reshape(batch, frames, value.shape[1], value.shape[2], value.shape[3]).permute(0, 2, 1, 3, 4)

        value = self.initial_activation(self.initial_norm(self.initial_conv(latent)))
        for block in self.res_blocks:
            value = block(value)
        value = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, value.shape[1], value.shape[3], value.shape[4])
        value = self.upsampler(value)
        value = value.reshape(batch, frames, value.shape[1], value.shape[2], value.shape[3]).permute(0, 2, 1, 3, 4)
        for block in self.post_upsample_res_blocks:
            value = block(value)
        return self.final_conv(value)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LTX25SpatialUpsampler":
        """Load an isolated LTX-2.5 upsampler with exact checkpoint-key coverage."""
        metadata = inspect_checkpoint(checkpoint_path).metadata
        model = cls(LTX25SpatialUpsamplerConfig.from_metadata(metadata))
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            state_dict = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise ValueError(f"LTX-2.5 upsampler load mismatch: missing={missing}, unexpected={unexpected}")
        return model.to(device=device, dtype=torch_dtype).eval()


def load_video_latent_statistics(video_vae_path: str | Path) -> LTX25PerChannelStatistics:
    """Load only the VAE normalization buffers needed by the spatial-upsample bridge."""
    source = Path(video_vae_path).expanduser().resolve()
    with safe_open(str(source), framework="pt", device="cpu") as checkpoint:
        state_dict = {
            key.removeprefix("per_channel_statistics."): checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key.startswith("per_channel_statistics.")
        }
    statistics = LTX25PerChannelStatistics(len(state_dict["std-of-means"]))
    missing, unexpected = statistics.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise ValueError(f"LTX-2.5 video-statistics load mismatch: missing={missing}, unexpected={unexpected}")
    return statistics.eval()


def upsample_video_latent(
    latent: torch.Tensor,
    upsampler: LTX25SpatialUpsampler,
    statistics: LTX25PerChannelStatistics,
) -> torch.Tensor:
    """Unnormalize, spatially upsample, then renormalize an LTX-2.5 video latent."""
    return statistics.normalize(upsampler(statistics.un_normalize(latent)))
