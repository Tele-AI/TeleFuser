from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from telefuser.core.base_model import BaseModel

# Import shared components from existing VAE
from .wan_video_vae import (
    CACHE_T,
    AttentionBlock,
    CausalConv3d,
    RMS_norm,
    ResidualBlock,
    Upsample,
    _convert_conv3d_to_channels_last_3d,
    _count_conv3d,
    check_is_instance,
)


class Resample(nn.Module):
    """2D/3D resampling module for Wan2.2 VAE.

    Note: Wan2.2 uses dim instead of dim//2 for upsample conv.
    Uses list-based feat_cache for torch.compile compatibility.
    """

    def __init__(self, dim: int, mode: str):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)),
            )
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        """Forward with list-based feature cache for torch.compile compatibility."""
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] != "Rep":
                        cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class Down_ResidualBlock(nn.Module):
    """Downsampling residual block with average pooling shortcut.

    Uses list-based feat_cache for torch.compile compatibility.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        x_copy = x.clone()
        for module in self.downsamples:
            if check_is_instance(module, ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x = module(x, feat_cache, feat_idx)
            elif check_is_instance(module, Resample) and feat_cache is not None and feat_idx is not None:
                x = module(x, feat_cache, feat_idx)
            else:
                x = module(x)
        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    """Upsampling residual block with duplicate upsample shortcut.

    Uses list-based feat_cache for torch.compile compatibility.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ):
        super().__init__()

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(
        self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None, first_chunk: bool = False
    ) -> torch.Tensor:
        x_main = x.clone()
        for module in self.upsamples:
            if check_is_instance(module, ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x_main = module(x_main, feat_cache, feat_idx)
            elif check_is_instance(module, Resample) and feat_cache is not None and feat_idx is not None:
                x_main = module(x_main, feat_cache, feat_idx)
            else:
                x_main = module(x_main)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class AvgDown3D(nn.Module):
    """Average pooling downsampling for 3D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):
    """Duplicate upsampling for 3D data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Patchify tensor with given patch size."""
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Unpatchify tensor with given patch size."""
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    return x


class Encoder3d(nn.Module):
    """3D encoder for Wan2.2 VAE with 12-channel input.

    Uses list-based feat_cache for torch.compile compatibility.
    """

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list = [],
        temperal_downsample: list[bool] = [True, True, False],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]

        # 12-channel input for patchified RGB (3 * 2 * 2)
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        """Forward with list-based feature cache."""
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if check_is_instance(layer, Down_ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    """3D decoder for Wan2.2 VAE with 12-channel output.

    Uses list-based feat_cache for torch.compile compatibility.
    """

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list = [],
        temperal_upsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)

        # 12-channel output for unpatchified RGB
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None, first_chunk: bool = False
    ) -> torch.Tensor:
        """Forward with list-based feature cache."""
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if check_is_instance(layer, Up_ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat([feat_cache[idx][:, :, -1:, :, :].to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class VideoVAE(nn.Module):
    """Full VAE model for Wan2.2 with encoder and decoder.

    Uses fixed-size list for feature cache (torch.compile friendly).
    """

    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 48,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list = [],
        temperal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def encode(self, x: torch.Tensor, scale: list) -> torch.Tensor:
        """Encode video to latent with fixed-size list feature cache.

        Fixed-size list [None] * conv_num is torch.compile friendly:
        - Static size known at compile time
        - No dynamic key lookups (vs dict with id(layer))
        - Better memory preallocation
        """
        conv_num = _count_conv3d(self.encoder)
        feat_cache = [None] * conv_num
        feat_idx = [0]

        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = None

        for i in range(iter_):
            feat_idx[0] = 0  # Reset index for each chunk
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], feat_cache=feat_cache, feat_idx=feat_idx
                )
                out = torch.cat([out, out_], 2)

        mu, _log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale, list):
            scale_t = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale_t[0].view(1, self.z_dim, 1, 1, 1)) * scale_t[1].view(1, self.z_dim, 1, 1, 1)
        else:
            scale_tensor = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale_tensor[0]) * scale_tensor[1]
        return mu

    def decode(self, z: torch.Tensor, scale: list) -> torch.Tensor:
        """Decode latent to video with fixed-size list feature cache.

        Fixed-size list is torch.compile friendly.
        """
        conv_num = _count_conv3d(self.decoder)
        feat_cache = [None] * conv_num
        feat_idx = [0]

        if isinstance(scale, list):
            scale_t = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale_t[1].view(1, self.z_dim, 1, 1, 1) + scale_t[0].view(1, self.z_dim, 1, 1, 1)
        else:
            scale_tensor = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale_tensor[1] + scale_tensor[0]

        iter_ = z.shape[2]
        x = self.conv2(z)
        out = None

        for i in range(iter_):
            feat_idx[0] = 0  # Reset index for each frame
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=True)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_cache, feat_idx=feat_idx)
                out = torch.cat([out, out_], 2)

        out = unpatchify(out, patch_size=2)
        return out


# Mean and std values for Wan2.2 VAE (48 channels)
WAN22_VAE_MEAN = [
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
]

WAN22_VAE_STD = [
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
]


class Wan22VideoVAE(BaseModel):
    """Wan2.2 Video VAE with 48 latent channels.

    Key differences from Wan2.1 VAE:
    - z_dim=48 (vs 16)
    - encoder dim=160, decoder dim=256 (vs 96)
    - 12-channel input/output (patchified RGB)
    - Uses list-based feat_cache for torch.compile compatibility
    - upsampling_factor=16 (vs 8)
    """

    # Predefined optimal 2D grid configurations
    # Format: (latent_h, latent_w, world_size) -> (grid_h, grid_w)
    GRID_TABLE = {
        # world_size = 2
        (30, 52, 2): (1, 2),
        (34, 60, 2): (1, 2),
        (45, 80, 2): (1, 2),
        (30, 30, 2): (1, 2),
        (36, 36, 2): (1, 2),
        (44, 44, 2): (1, 2),
        (60, 60, 2): (1, 2),
        (52, 30, 2): (2, 1),
        (60, 34, 2): (2, 1),
        (80, 45, 2): (2, 1),
        # world_size = 4
        (30, 52, 4): (2, 2),
        (34, 60, 4): (2, 2),
        (45, 80, 4): (2, 2),
        (30, 30, 4): (2, 2),
        (36, 36, 4): (2, 2),
        (44, 44, 4): (2, 2),
        (60, 60, 4): (2, 2),
        (52, 30, 4): (2, 2),
        (60, 34, 4): (2, 2),
        (80, 45, 4): (2, 2),
        # world_size = 8
        (30, 52, 8): (2, 4),
        (34, 60, 8): (2, 4),
        (45, 80, 8): (2, 4),
        (30, 30, 8): (2, 4),
        (36, 36, 8): (2, 4),
        (44, 44, 8): (2, 4),
        (60, 60, 8): (2, 4),
        (52, 30, 8): (4, 2),
        (60, 34, 8): (4, 2),
        (80, 45, 8): (4, 2),
    }

    def __init__(
        self,
        z_dim: int = 48,
        dim: int = 160,
        dec_dim: int = 256,
        parallelism: int = 1,
    ):
        super().__init__()

        self.mean = torch.tensor(WAN22_VAE_MEAN)
        self.std = torch.tensor(WAN22_VAE_STD)
        self.scale: list[torch.Tensor] = [self.mean, 1.0 / self.std]
        self._scale_device_cache: dict[str, torch.Tensor] = {}

        self.model = (
            VideoVAE(
                dim=dim,
                dec_dim=dec_dim,
                z_dim=z_dim,
            )
            .eval()
            .requires_grad_(False)
        )
        self.z_dim = z_dim
        self.upsampling_factor = 16  # Wan2.2 VAE: 16x spatial compression
        self.parallelism = parallelism

        # Persistent cache for streaming decode (list-based for torch.compile)
        self._feat_cache: list = []
        self._feat_idx: list = [0]

    def set_parallelism(self, parallelism: int):
        self.parallelism = parallelism

    def enable_channels_last_3d(self) -> int:
        """Enable channels_last_3d memory format for Conv3d weights.

        Must be called after load_state_dict, as weights need to be loaded first.

        Returns:
            Number of Conv3d layers converted.
        """
        return _convert_conv3d_to_channels_last_3d(self.model)

    # ==================== 2D Spatial Parallel Methods ====================

    def calculate_2d_grid(self, latent_h: int, latent_w: int, world_size: int) -> tuple[int, int]:
        """Calculate optimal 2D grid for spatial splitting."""
        key = (latent_h, latent_w, world_size)
        if key in self.GRID_TABLE:
            return self.GRID_TABLE[key]

        best_h, best_w = 1, world_size
        min_aspect_diff = float("inf")

        for h in range(1, world_size + 1):
            if world_size % h == 0:
                w = world_size // h
                if latent_h % h == 0 and latent_w % w == 0:
                    aspect_diff = abs((latent_h / h) - (latent_w / w))
                    if aspect_diff < min_aspect_diff:
                        min_aspect_diff = aspect_diff
                        best_h, best_w = h, w

        return best_h, best_w

    def _compute_padded_slice(
        self,
        rank: int,
        world_size: int,
        total_size: int,
        chunk_size: int,
        padding: int,
    ) -> tuple[int, int]:
        """Compute slice indices with proper padding."""
        if world_size == 1:
            return 0, total_size

        if rank == 0:
            return 0, chunk_size + 2 * padding
        elif rank == world_size - 1:
            return total_size - (chunk_size + 2 * padding), total_size
        else:
            return rank * chunk_size - padding, (rank + 1) * chunk_size + padding

    def _remove_latent_padding(
        self,
        tensor: torch.Tensor,
        rank_h: int,
        rank_w: int,
        world_size_h: int,
        world_size_w: int,
        chunk_h: int,
        chunk_w: int,
    ) -> torch.Tensor:
        """Remove padding from latent tensor.

        Uses LightX2V approach: directly keep the core chunk region
        (chunk_h x chunk_w) instead of calculating padding to remove.
        """
        if world_size_h == 1:
            h_start, h_end = 0, tensor.shape[3]
        elif rank_h == 0:
            h_start, h_end = 0, chunk_h
        elif rank_h == world_size_h - 1:
            h_start, h_end = tensor.shape[3] - chunk_h, tensor.shape[3]
        else:
            # Middle ranks: remove padding from both sides
            padding = (tensor.shape[3] - chunk_h) // 2
            h_start, h_end = padding, tensor.shape[3] - padding

        if world_size_w == 1:
            w_start, w_end = 0, tensor.shape[4]
        elif rank_w == 0:
            w_start, w_end = 0, chunk_w
        elif rank_w == world_size_w - 1:
            w_start, w_end = tensor.shape[4] - chunk_w, tensor.shape[4]
        else:
            # Middle ranks: remove padding from both sides
            padding = (tensor.shape[4] - chunk_w) // 2
            w_start, w_end = padding, tensor.shape[4] - padding

        return tensor[:, :, :, h_start:h_end, w_start:w_end].contiguous()

    def _reconstruct_2d(
        self,
        chunks: list[torch.Tensor],
        world_size_h: int,
        world_size_w: int,
        dim: int,
    ) -> torch.Tensor:
        """Reconstruct full tensor from 2D gathered chunks."""
        rows = []
        for h_idx in range(world_size_h):
            cols = [chunks[h_idx * world_size_w + w_idx] for w_idx in range(world_size_w)]
            rows.append(torch.cat(cols, dim=dim + 1))
        return torch.cat(rows, dim=dim)

    def encode_dist_2d(
        self,
        video: torch.Tensor,
        world_size_h: int,
        world_size_w: int,
        cur_rank_h: int,
        cur_rank_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode video with true 2D spatial splitting."""
        import torch.distributed as dist

        spatial_ratio = self.upsampling_factor  # 16 for Wan2.2
        padding_latent = 1

        latent_h = video.shape[3] // spatial_ratio
        latent_w = video.shape[4] // spatial_ratio
        chunk_h = latent_h // world_size_h
        chunk_w = latent_w // world_size_w

        chunk_h_input = chunk_h * spatial_ratio
        chunk_w_input = chunk_w * spatial_ratio
        padding_input = padding_latent * spatial_ratio

        h_start, h_end = self._compute_padded_slice(
            cur_rank_h, world_size_h, video.shape[3], chunk_h_input, padding_input
        )
        w_start, w_end = self._compute_padded_slice(
            cur_rank_w, world_size_w, video.shape[4], chunk_w_input, padding_input
        )

        video_chunk = video[:, :, :, h_start:h_end, w_start:w_end].contiguous().to(device)
        scale = self._get_scale_on_device(device, video_chunk.dtype)
        encoded_chunk = self.model.encode(video_chunk, scale)

        encoded_chunk = self._remove_latent_padding(
            encoded_chunk, cur_rank_h, cur_rank_w, world_size_h, world_size_w, chunk_h, chunk_w
        )

        world_size_total = world_size_h * world_size_w
        full_encoded = [torch.empty_like(encoded_chunk) for _ in range(world_size_total)]
        dist.all_gather(full_encoded, encoded_chunk)

        encoded = self._reconstruct_2d(full_encoded, world_size_h, world_size_w, dim=3)
        return encoded.squeeze(0).cpu()

    def decode_dist_2d(
        self,
        latent: torch.Tensor,
        world_size_h: int,
        world_size_w: int,
        cur_rank_h: int,
        cur_rank_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Decode latent with true 2D spatial splitting."""
        import torch.distributed as dist

        spatial_ratio = self.upsampling_factor  # 16
        padding_latent = 2

        latent_h = latent.shape[3]
        latent_w = latent.shape[4]
        chunk_h = latent_h // world_size_h
        chunk_w = latent_w // world_size_w

        h_start, h_end = self._compute_padded_slice(cur_rank_h, world_size_h, latent_h, chunk_h, padding_latent)
        w_start, w_end = self._compute_padded_slice(cur_rank_w, world_size_w, latent_w, chunk_w, padding_latent)

        latent_chunk = latent[:, :, :, h_start:h_end, w_start:w_end].contiguous().to(device)
        scale = self._get_scale_on_device(device, latent_chunk.dtype)
        decoded_chunk = self.model.decode(latent_chunk, scale)

        # Calculate output chunk size (latent chunk * spatial_ratio)
        chunk_h_output = chunk_h * spatial_ratio
        chunk_w_output = chunk_w * spatial_ratio

        decoded_chunk = self._remove_latent_padding(
            decoded_chunk, cur_rank_h, cur_rank_w, world_size_h, world_size_w, chunk_h_output, chunk_w_output
        )

        world_size_total = world_size_h * world_size_w
        full_decoded = [torch.empty_like(decoded_chunk) for _ in range(world_size_total)]
        dist.all_gather(full_decoded, decoded_chunk)

        decoded = self._reconstruct_2d(full_decoded, world_size_h, world_size_w, dim=3)
        return decoded.squeeze(0).cpu().clamp_(-1, 1)

    def encode_parallel(
        self,
        videos: list[torch.Tensor],
        device: torch.device,
        method: str = "2d_split",
        world_size_h: int | None = None,
        world_size_w: int | None = None,
    ) -> torch.Tensor:
        """Encode videos with parallel processing across GPUs.

        Args:
            videos: List of video tensors [C, T, H, W]
            device: Computation device
            method: Parallel method ("2d_split" or "tile_dist")
            world_size_h: Manual H grid size (optional)
            world_size_w: Manual W grid size (optional)

        Returns:
            Encoded latents tensor [B, C, T_latent, H_latent, W_latent]
        """
        import torch.distributed as dist

        world_size = dist.get_world_size()
        cur_rank = dist.get_rank()

        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            latent_h = video.shape[3] // self.upsampling_factor
            latent_w = video.shape[4] // self.upsampling_factor

            if method == "2d_split":
                if world_size_h is None or world_size_w is None:
                    world_size_h, world_size_w = self.calculate_2d_grid(latent_h, latent_w, world_size)
                cur_rank_h = cur_rank // world_size_w
                cur_rank_w = cur_rank % world_size_w

                hidden_state = self.encode_dist_2d(video, world_size_h, world_size_w, cur_rank_h, cur_rank_w, device)
            else:
                tile_size = (34 * 16, 34 * 16)
                tile_stride = (18 * 16, 16 * 16)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride).squeeze(0)

            hidden_states.append(hidden_state)

        return torch.stack(hidden_states)

    def decode_parallel(
        self,
        hidden_states: torch.Tensor,
        device: torch.device,
        method: str = "2d_split",
        world_size_h: int | None = None,
        world_size_w: int | None = None,
    ) -> torch.Tensor:
        """Decode latents with parallel processing across GPUs.

        Args:
            hidden_states: Latent tensor [B, C, T, H, W] or [C, T, H, W]
            device: Computation device
            method: Parallel method ("2d_split" or "tile_dist")
            world_size_h: Manual H grid size (optional)
            world_size_w: Manual W grid size (optional)

        Returns:
            Decoded video tensor on CPU [B, C, T_out, H_out, W_out]
        """
        import torch.distributed as dist

        world_size = dist.get_world_size()
        cur_rank = dist.get_rank()

        if hidden_states.dim() == 4:
            hidden_states = hidden_states.unsqueeze(0)

        batch_size = hidden_states.shape[0]
        videos = []

        for i in range(batch_size):
            latent = hidden_states[i : i + 1]
            latent_h = latent.shape[3]
            latent_w = latent.shape[4]

            if method == "2d_split":
                if world_size_h is None or world_size_w is None:
                    world_size_h, world_size_w = self.calculate_2d_grid(latent_h, latent_w, world_size)
                cur_rank_h = cur_rank // world_size_w
                cur_rank_w = cur_rank % world_size_w

                video = self.decode_dist_2d(latent, world_size_h, world_size_w, cur_rank_h, cur_rank_w, device)
            else:
                video = self.tiled_decode(latent, device, (34, 34), (18, 16))

            videos.append(video)

        return torch.stack(videos)

    # ==================== Original Methods ====================

    def _get_scale_on_device(self, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        """Get scale tensors cached on specific device."""
        cache_key = f"{device}_{dtype}"
        if cache_key not in self._scale_device_cache:
            self._scale_device_cache[cache_key] = [
                self.mean.to(dtype=dtype, device=device),
                (1.0 / self.std).to(dtype=dtype, device=device),
            ]
        return self._scale_device_cache[cache_key]

    def build_1d_mask(self, length: int, left_bound: bool, right_bound: bool, border_width: int) -> torch.Tensor:
        """Build 1D mask with linear blending at boundaries."""
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(
        self, data: torch.Tensor, is_bound: tuple[bool, bool, bool, bool], border_width: tuple[int, int]
    ) -> torch.Tensor:
        """Build 2D mask for tile blending."""
        from einops import rearrange, repeat

        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_encode(
        self, video: torch.Tensor, device: torch.device, tile_size: tuple[int, int], tile_stride: tuple[int, int]
    ) -> torch.Tensor:
        """Encode with tiling for memory efficiency."""
        import torch.distributed as dist
        from tqdm import tqdm

        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = device if self.parallelism > 1 else "cpu"
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros(
            (1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, self.z_dim, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )

        scale = self._get_scale_on_device(computation_device, video.dtype)
        hide_progress_bar = self.parallelism > 1 and dist.is_initialized() and dist.get_rank() != 0
        for i, (h, h_, w, w_) in enumerate(tqdm(tasks, desc="VAE ENCODING", disable=hide_progress_bar)):
            if self.parallelism > 1 and dist.is_initialized() and (i % dist.get_world_size() != dist.get_rank()):
                continue
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.upsampling_factor,
                    (size_w - stride_w) // self.upsampling_factor,
                ),
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            target_h_end = target_h + hidden_states_batch.shape[3]
            target_w_end = target_w + hidden_states_batch.shape[4]
            values[:, :, :, target_h:target_h_end, target_w:target_w_end] += hidden_states_batch * mask
            weight[:, :, :, target_h:target_h_end, target_w:target_w_end] += mask

        if self.parallelism > 1 and dist.is_initialized():
            dist.all_reduce(values)
            dist.all_reduce(weight)
        values = values / weight
        values = values.float()
        return values

    def tiled_decode(
        self,
        hidden_states: torch.Tensor,
        device: torch.device,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> torch.Tensor:
        """Decode with tiling for memory efficiency."""
        import torch.distributed as dist
        from tqdm import tqdm

        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_, w_ = h + size_h, w + size_w
                tasks.append((h, h_, w, w_))

        data_device = device if self.parallelism > 1 else "cpu"
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros(
            (1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=data_device,
        )
        # Output is RGB (3 channels) after unpatchify in VideoVAE.decode()
        values = torch.zeros(
            (1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=data_device,
        )

        scale = self._get_scale_on_device(computation_device, hidden_states.dtype)
        hide_progress_bar = self.parallelism > 1 and dist.is_initialized() and dist.get_rank() != 0
        for i, (h, h_, w, w_) in enumerate(tqdm(tasks, desc="VAE DECODING", disable=hide_progress_bar)):
            if self.parallelism > 1 and dist.is_initialized() and (i % dist.get_world_size() != dist.get_rank()):
                continue
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, scale).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                ),
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            target_h_end = target_h + hidden_states_batch.shape[3]
            target_w_end = target_w + hidden_states_batch.shape[4]
            values[:, :, :, target_h:target_h_end, target_w:target_w_end] += hidden_states_batch * mask
            weight[:, :, :, target_h:target_h_end, target_w:target_w_end] += mask

        if self.parallelism > 1 and dist.is_initialized():
            dist.all_reduce(values)
            dist.all_reduce(weight)
        values = values / weight
        values = values.cpu()
        # unpatchify is already called in VideoVAE.decode(), output is already RGB
        values = values.float().clamp_(-1, 1)
        return values

    def single_encode(self, video: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Encode single video without parallelism."""
        video = video.to(device)
        scale = self._get_scale_on_device(device, video.dtype)
        x = self.model.encode(video, scale)
        return x.float()

    def single_decode(self, hidden_state: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Decode single latent without parallelism."""
        hidden_state = hidden_state.to(device)
        scale = self._get_scale_on_device(device, hidden_state.dtype)
        video = self.model.decode(hidden_state, scale)
        # unpatchify is already called in VideoVAE.decode()
        return video.float().clamp_(-1, 1)

    def encode(
        self,
        videos: list[torch.Tensor],
        device: torch.device,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        """Encode videos to latent space.

        Automatically uses parallel processing if parallelism > 1 and distributed is initialized.
        - Multi-GPU: tiled=True → tile_dist, tiled=False → 2d_split (default)
        - Single GPU: tiled=True → tiled_encode, tiled=False → direct encode

        Args:
            videos: List of video tensors [C, T, H, W]
            device: Computation device
            tiled: Controls processing mode:
                   - Single GPU: True = tiled processing, False = direct encode
                   - Multi-GPU: True = tile_dist method, False = 2d_split method (default)
            tile_size: Tile size for tiled processing
            tile_stride: Tile stride for tiled processing

        Returns:
            Encoded latents tensor [B, C, T_latent, H_latent, W_latent]
        """
        import torch.distributed as dist

        # Auto-detect parallel mode
        if self.parallelism > 1 and dist.is_initialized():
            # tiled=True → tile_dist, tiled=False → 2d_split
            method = "tile_dist" if tiled else "2d_split"
            return self.encode_parallel(videos, device, method=method)

        # Single GPU processing
        videos = [video.to("cpu") for video in videos]
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size_scaled = (tile_size[0] * 16, tile_size[1] * 16)  # 16x for Wan2.2
                tile_stride_scaled = (tile_stride[0] * 16, tile_stride[1] * 16)
                hidden_state = self.tiled_encode(video, device, tile_size_scaled, tile_stride_scaled)
            else:
                hidden_state = self.single_encode(video, device)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(
        self,
        hidden_states: list[torch.Tensor],
        device: torch.device,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        """Decode latents to video frames.

        Automatically uses parallel processing if parallelism > 1 and distributed is initialized.
        - Multi-GPU: tiled=True → tile_dist, tiled=False → 2d_split (default)
        - Single GPU: tiled=True → tiled_decode, tiled=False → direct decode

        Args:
            hidden_states: List of latent tensors [C, T, H, W]
            device: Computation device
            tiled: Controls processing mode:
                   - Single GPU: True = tiled processing, False = direct decode
                   - Multi-GPU: True = tile_dist method, False = 2d_split method (default)
            tile_size: Tile size for tiled processing
            tile_stride: Tile stride for tiled processing

        Returns:
            Decoded video tensor on CPU [B, C, T_out, H_out, W_out]
        """
        import torch.distributed as dist

        # Auto-detect parallel mode
        if self.parallelism > 1 and dist.is_initialized():
            # tiled=True → tile_dist, tiled=False → 2d_split
            method = "tile_dist" if tiled else "2d_split"
            hidden_states_tensor = torch.stack(hidden_states)
            return self.decode_parallel(hidden_states_tensor, device, method=method)

        # Single GPU processing
        hidden_states = [hidden_state.to("cpu") for hidden_state in hidden_states]
        videos = []
        for hidden_state in hidden_states:
            hidden_state = hidden_state.unsqueeze(0)
            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.single_decode(hidden_state, device)
            video = video.cpu()
            videos.append(video)
        videos = torch.cat(videos, dim=0)
        return videos

    def cached_decode_withflag(
        self,
        hidden_state: torch.Tensor,
        device: torch.device,
        is_first_clip: bool,
        is_last_clip: bool,
    ) -> torch.Tensor:
        """Decode with persistent feature cache for streaming generation.

        Maintains intermediate feature cache across decode calls for efficiency.
        Uses fixed-size list for torch.compile compatibility.

        Args:
            hidden_state: Latent tensor [C, T, H, W] or [1, C, T, H, W]
            device: Target device
            is_first_clip: If True, initialize cache (first segment)
            is_last_clip: If True, clear cache after decoding (last segment)

        Returns:
            Decoded video tensor [C, T_out, H_out, W_out]
        """
        # Initialize cache on first clip
        if is_first_clip:
            conv_num = _count_conv3d(self.model.decoder)
            self._feat_cache = [None] * conv_num
            self._feat_idx = [0]

        # Add batch dimension if needed
        if hidden_state.dim() == 4:
            hidden_state = hidden_state.unsqueeze(0)

        hidden_state = hidden_state.to(device)

        # Apply scaling
        scale = self._get_scale_on_device(device, hidden_state.dtype)
        z = hidden_state / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)

        # Decode frame-by-frame with cache
        iter_ = z.shape[2]
        x = self.model.conv2(z)

        for i in range(iter_):
            self._feat_idx[0] = 0  # Reset index for each frame
            if i == 0:
                out = self.model.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_cache,
                    feat_idx=self._feat_idx,
                    first_chunk=True,
                )
            else:
                out_ = self.model.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_cache,
                    feat_idx=self._feat_idx,
                )
                out = torch.cat([out, out_], 2)

        # Clear cache on last clip
        if is_last_clip:
            self._feat_cache = []
            self._feat_idx = [0]

        video = out.clamp_(-1, 1)

        # Unpatchify
        video = unpatchify(video, patch_size=2)

        # Remove batch dimension if single item
        if video.shape[0] == 1:
            video = video.squeeze(0)

        return video

    @staticmethod
    def state_dict_converter():
        return Wan22VideoVAEStateDictConverter()

    def enable_sequential_cpu_offload(self, device: torch.device, torch_dtype: torch.dtype):
        """Enable sequential CPU offloading for memory efficiency."""
        from telefuser.offload import (
            AutoWrappedLinear,
            AutoWrappedModule,
            enable_sequential_cpu_offload,
        )

        dtype = next(iter(self.parameters())).dtype
        enable_sequential_cpu_offload(
            self.model,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=device,
                computation_dtype=torch_dtype,
                computation_device=device,
            ),
        )


class Wan22VideoVAEStateDictConverter:
    """State dict converter for Wan2.2 VAE."""

    def from_official(self, state_dict: dict) -> tuple[dict, dict]:
        """Convert from official Wan2.2 checkpoint format."""
        state_dict_ = {}
        cfg_dict_ = {}
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        for name in state_dict:
            state_dict_["model." + name] = state_dict[name]
        return state_dict_, cfg_dict_


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [Wan22VideoVAE], "official")
