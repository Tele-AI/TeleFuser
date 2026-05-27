from __future__ import annotations

import torch
from torch import nn

from telefuser.core.base_model import BaseModel

CACHE_T = 2


class QwenImageCausalConv3d(torch.nn.Conv3d):
    """Causal 3D convolution with feature caching for temporal consistency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Reorder padding for causal convolution: (left, right, top, bottom, front, back)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = torch.nn.functional.pad(x, padding)
        return super().forward(x)


class QwenImageRMS_norm(nn.Module):
    """RMS normalization with channel-first/last support."""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.nn.functional.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma
            + self.bias
        )


class QwenImageResidualBlock(nn.Module):
    """Residual block with causal 3D convolutions and feature caching."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = torch.nn.SiLU()

        self.norm1 = QwenImageRMS_norm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImageCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        h = self.conv_shortcut(x)

        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


class QwenImageAttentionBlock(nn.Module):
    """Single-head causal self-attention block."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)

        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class QwenImageUpsample(nn.Upsample):
    """Upsample that preserves input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    """2D/3D resampling with causal temporal convolutions."""

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImageCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
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
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
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


class QwenImageMidBlock(nn.Module):
    """Middle block with residual and attention layers."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim

        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        x = self.resnets[0](x, feat_cache, feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache, feat_idx)
        return x


class QwenImageEncoder3d(nn.Module):
    """3D encoder with causal convolutions."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temperal_downsample: list[bool] = [True, True, False],
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = torch.nn.SiLU()

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv_in = QwenImageCausalConv3d(3, dims[0], 3, padding=1)

        self.down_blocks = torch.nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(QwenImageResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0

        self.mid_block = QwenImageMidBlock(out_dim, dropout, non_linearity, num_layers=1)

        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        x = self.mid_block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    """Upsampling block with residual layers."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImageResample(out_dim, mode=upsample_mode)])

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(nn.Module):
    """3D decoder with causal convolutions."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temperal_upsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = torch.nn.SiLU()

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)

        self.mid_block = QwenImageMidBlock(dims[0], dropout, non_linearity, num_layers=1)

        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                in_dim = in_dim // 2

            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            up_block = QwenImageUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            if upsample_mode is not None:
                scale *= 2.0

        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, 3, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache, feat_idx)

        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageVAE(BaseModel):
    """VAE for Qwen-Image with encoder and decoder."""

    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: tuple[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: list[float] = [],
        temperal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
    ):
        super().__init__()

        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = QwenImageEncoder3d(
            base_dim,
            z_dim * 2,
            list(dim_mult),
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)

        self.decoder = QwenImageDecoder3d(
            base_dim,
            z_dim,
            list(dim_mult),
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

        # Pre-computed normalization constants
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean).view(1, 16, 1, 1, 1)
        self.std = 1 / torch.tensor(std).view(1, 16, 1, 1, 1)

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.unsqueeze(2)
        x = self.encoder(x)
        x = self.quant_conv(x)
        x = x[:, :16]
        mean, std = (
            self.mean.to(dtype=x.dtype, device=x.device),
            self.std.to(dtype=x.dtype, device=x.device),
        )
        x = (x - mean) * std
        x = x.squeeze(2)
        return x

    def decode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.unsqueeze(2)
        mean, std = (
            self.mean.to(dtype=x.dtype, device=x.device),
            self.std.to(dtype=x.dtype, device=x.device),
        )
        x = x / std + mean
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        x = x.squeeze(2)
        return x

    @staticmethod
    def state_dict_converter():
        return QwenImageVAEStateDictConverter()


class QwenImageVAEStateDictConverter:
    """State dict converter for Qwen-Image VAE."""

    def __init__(self):
        pass

    def from_diffusers(self, state_dict: dict) -> dict:
        return state_dict

    def from_official(self, state_dict: dict) -> dict:
        return state_dict


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "ed4ea5824d55ec3107b09815e318123a", ["qwen_image_vae"], [QwenImageVAE], "official")
