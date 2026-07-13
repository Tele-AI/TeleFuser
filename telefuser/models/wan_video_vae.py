from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from telefuser.core.base_model import BaseModel
from telefuser.utils.logging import logger
from telefuser.utils.model_weight import hash_state_dict_keys

CACHE_T = 2


@dataclass
class WanVideoVAEStreamingDecodeState:
    """Session-owned temporal feature cache for incremental VAE decoding."""

    feat_cache: list[object] = field(default_factory=list)
    feat_idx: list[int] = field(default_factory=lambda: [0])


def _count_conv3d(model: nn.Module) -> int:
    """Count Conv3d layers in a model (for feat_cache initialization)."""
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


def _convert_conv3d_to_channels_last_3d(module: nn.Module) -> int:
    """Convert all Conv3d weights to channels_last_3d format if cuDNN is available.

    Eliminates NCHW<->NHWC format conversion overhead in cuDNN (~9% faster).

    Args:
        module: Module to convert (typically VAE encoder/decoder)

    Returns:
        Number of Conv3d layers converted (0 if cuDNN unavailable).
    """
    try:
        if not torch.backends.cudnn.is_available():
            return 0
    except Exception:
        return 0

    count = 0
    for child in module.children():
        if isinstance(child, nn.Conv3d):
            if not child.weight.data.is_contiguous(memory_format=torch.channels_last_3d):
                child.weight.data = child.weight.data.to(memory_format=torch.channels_last_3d)
                count += 1
        count += _convert_conv3d_to_channels_last_3d(child)
    return count


def check_is_instance(model: nn.Module, module_class: type) -> bool:
    """Check if model is instance of module_class (handles DataParallel)."""
    if isinstance(model, module_class):
        return True
    if hasattr(model, "module") and isinstance(model.module, module_class):
        return True
    return False


def block_causal_mask(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Create block causal mask for attention."""
    b, n, s, _, device = *x.size(), x.device
    assert s % block_size == 0
    num_blocks = s // block_size

    mask = torch.zeros(b, n, s, s, dtype=torch.bool, device=device)
    for i in range(num_blocks):
        mask[:, :, i * block_size : (i + 1) * block_size, : (i + 1) * block_size] = 1
    return mask


class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution for temporal consistency."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        x = F.pad(x, padding)
        x = x.contiguous(memory_format=torch.channels_last_3d)
        return super().forward(x)


class RMS_norm(nn.Module):
    """RMS normalization with channel-first/last support."""

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


class Upsample(nn.Upsample):
    """Upsample that fixes bfloat16 support for nearest neighbor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    """2D/3D resampling module with upsampling and downsampling."""

    def __init__(self, dim: int, mode: str):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
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
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
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


class ResidualBlock(nn.Module):
    """Residual block with causal 3D convolutions."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head causal self-attention block."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


class Encoder3d(nn.Module):
    """3D VAE encoder with causal convolutions."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list = [],
        temperal_downsample: list[bool] = [True, True, False],
        dropout: float = 0.0,
        pruning_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        dims = [dim * u for u in [1] + dim_mult]
        dims = [int(d * (1 - pruning_rate)) for d in dims]
        scale = 1.0

        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
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
        """Forward pass with optional feature caching for chunked encode."""
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if check_is_instance(layer, ResidualBlock) and feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            elif check_is_instance(layer, Resample) and feat_cache is not None and feat_idx is not None:
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
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    """3D VAE decoder with causal convolutions."""

    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list = [],
        temperal_upsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        pruning_rate: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        dims = [int(d * (1 - pruning_rate)) for d in dims]

        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, feat_cache: list | None = None, feat_idx: list | None = None) -> torch.Tensor:
        """Forward pass with list-based feature caching."""
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
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
            if feat_cache is not None and feat_idx is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if check_is_instance(layer, CausalConv3d) and feat_cache is not None and feat_idx is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class VideoVAE(nn.Module):
    """Full VAE with encoder and decoder for video."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: list = [],
        temperal_downsample: list[bool] = [False, True, True],
        dropout: float = 0.0,
        encode_pruning_rate: float = 0.0,
        decode_pruning_rate: float = 0.0,
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
            pruning_rate=encode_pruning_rate,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
            pruning_rate=decode_pruning_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x: torch.Tensor, scale: list) -> torch.Tensor:
        """Encode video to latent with chunked processing for memory efficiency.

        Processes video in chunks (1 frame + 4-frame segments) to avoid OOM on long videos.
        Uses feature cache for causal convolutions across chunks.
        """
        conv_num = _count_conv3d(self.encoder)
        feat_cache = [None] * conv_num
        feat_idx = [0]

        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4  # Number of chunks: first frame + rest in 4-frame chunks

        for i in range(iter_):
            feat_idx[0] = 0  # Reset index for each chunk
            if i == 0:
                # First chunk: single frame
                out = self.encoder(x[:, :, :1, :, :], feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                # Subsequent chunks: 4 frames each
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :], feat_cache=feat_cache, feat_idx=feat_idx
                )
                out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=mu.dtype, device=mu.device)
            mu = (mu - scale[0]) * scale[1]
        return mu

    def decode(self, z: torch.Tensor, scale: list) -> torch.Tensor:
        """Decode latent to video frames using list-based feature cache."""
        conv_num = _count_conv3d(self.decoder)
        feat_cache = [None] * conv_num
        feat_idx = [0]

        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            scale = scale.to(dtype=z.dtype, device=z.device)
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            feat_idx[0] = 0  # Reset index for each frame
            if i == 0:
                out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=feat_cache, feat_idx=feat_idx)
                out = torch.cat([out, out_], 2)
        return out

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)


class WanVideoVAE(BaseModel):
    """Video VAE wrapper with tiled encoding/decoding and 2D spatial parallel support."""

    # Predefined optimal 2D grid configurations
    # Format: (latent_h, latent_w, world_size) -> (grid_h, grid_w)
    GRID_TABLE = {
        # world_size = 2
        (60, 104, 2): (1, 2),
        (68, 120, 2): (1, 2),
        (90, 160, 2): (1, 2),
        (60, 60, 2): (1, 2),
        (72, 72, 2): (1, 2),
        (88, 88, 2): (1, 2),
        (120, 120, 2): (1, 2),
        (104, 60, 2): (2, 1),
        (120, 68, 2): (2, 1),
        (160, 90, 2): (2, 1),
        # world_size = 4
        (60, 104, 4): (2, 2),
        (68, 120, 4): (2, 2),
        (90, 160, 4): (2, 2),
        (60, 60, 4): (2, 2),
        (72, 72, 4): (2, 2),
        (88, 88, 4): (2, 2),
        (120, 120, 4): (2, 2),
        (104, 60, 4): (2, 2),
        (120, 68, 4): (2, 2),
        (160, 90, 4): (2, 2),
        # world_size = 8
        (60, 104, 8): (2, 4),
        (68, 120, 8): (2, 4),
        (90, 160, 8): (2, 4),
        (60, 60, 8): (2, 4),
        (72, 72, 8): (2, 4),
        (88, 88, 8): (2, 4),
        (120, 120, 8): (2, 4),
        (104, 60, 8): (4, 2),
        (120, 68, 8): (4, 2),
        (160, 90, 8): (4, 2),
    }

    def __init__(
        self,
        z_dim: int = 16,
        parallelism: int = 1,
        encode_pruning_rate: float = 0.0,
        decode_pruning_rate: float = 0.0,
    ):
        super().__init__()
        self.z_dim = z_dim  # Expose z_dim for external access

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
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.scale = [self.mean, 1.0 / self.std]

        self.model = (
            VideoVAE(
                z_dim=z_dim,
                encode_pruning_rate=encode_pruning_rate,
                decode_pruning_rate=decode_pruning_rate,
            )
            .eval()
            .requires_grad_(False)
        )

        self.upsampling_factor = 8
        self.parallelism = parallelism

        # Feature cache for streaming decode (list-based)
        self._feat_cache = []
        self._feat_idx = [0]

    def enable_channels_last_3d(self) -> int:
        """Enable channels_last_3d memory format for Conv3d weights.

        Must be called after load_state_dict, as weights need to be loaded first.

        Returns:
            Number of Conv3d layers converted.
        """
        return _convert_conv3d_to_channels_last_3d(self.model)

    def set_parallelism(self, parallelism: int):
        self.parallelism = parallelism

    # ==================== 2D Spatial Parallel Methods ====================

    def calculate_2d_grid(self, latent_h: int, latent_w: int, world_size: int) -> tuple[int, int]:
        """Calculate optimal 2D grid for spatial splitting.

        Args:
            latent_h: Latent height
            latent_w: Latent width
            world_size: Number of GPUs

        Returns:
            (grid_h, grid_w): Number of splits along H and W dimensions
        """
        key = (latent_h, latent_w, world_size)
        if key in self.GRID_TABLE:
            return self.GRID_TABLE[key]

        # Find optimal grid: minimize aspect ratio difference
        best_h, best_w = 1, world_size
        min_aspect_diff = float("inf")

        for h in range(1, world_size + 1):
            if world_size % h == 0:
                w = world_size // h
                # Check if divisible
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
        """Compute slice indices with proper padding for boundary consistency.

        Args:
            rank: Current rank in this dimension
            world_size: Number of splits in this dimension
            total_size: Total size of the dimension
            chunk_size: Core chunk size (without padding)
            padding: Padding size in input space

        Returns:
            (start, end): Slice indices with padding
        """
        if world_size == 1:
            return 0, total_size

        if rank == 0:
            # First rank: padding only on the right
            start = 0
            end = chunk_size + 2 * padding
        elif rank == world_size - 1:
            # Last rank: padding only on the left
            start = total_size - (chunk_size + 2 * padding)
            end = total_size
        else:
            # Middle ranks: padding on both sides
            start = rank * chunk_size - padding
            end = (rank + 1) * chunk_size + padding

        return start, end

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
        """Remove padding from latent tensor after encode/decode.

        Uses LightX2V approach: directly keep the core chunk region
        (chunk_h x chunk_w) instead of calculating padding to remove.

        Args:
            tensor: Latent tensor with padding [B, C, T, H, W]
            rank_h: Rank in H dimension
            rank_w: Rank in W dimension
            world_size_h: Number of splits in H
            world_size_w: Number of splits in W
            chunk_h: Core chunk size in H dimension (latent space)
            chunk_w: Core chunk size in W dimension (latent space)

        Returns:
            Tensor with padding removed, shape [B, C, T, chunk_h, chunk_w]
        """
        # Remove H padding - keep core chunk_h region
        if world_size_h == 1:
            h_start, h_end = 0, tensor.shape[3]
        elif rank_h == 0:
            h_start = 0
            h_end = chunk_h
        elif rank_h == world_size_h - 1:
            h_start = tensor.shape[3] - chunk_h
            h_end = tensor.shape[3]
        else:
            # Middle ranks: remove padding from both sides
            padding = (tensor.shape[3] - chunk_h) // 2
            h_start = padding
            h_end = tensor.shape[3] - padding

        # Remove W padding - keep core chunk_w region
        if world_size_w == 1:
            w_start, w_end = 0, tensor.shape[4]
        elif rank_w == 0:
            w_start = 0
            w_end = chunk_w
        elif rank_w == world_size_w - 1:
            w_start = tensor.shape[4] - chunk_w
            w_end = tensor.shape[4]
        else:
            # Middle ranks: remove padding from both sides
            padding = (tensor.shape[4] - chunk_w) // 2
            w_start = padding
            w_end = tensor.shape[4] - padding

        return tensor[:, :, :, h_start:h_end, w_start:w_end].contiguous()

    def _reconstruct_2d(
        self,
        chunks: list[torch.Tensor],
        world_size_h: int,
        world_size_w: int,
        dim: int,
    ) -> torch.Tensor:
        """Reconstruct full tensor from 2D gathered chunks.

        Args:
            chunks: List of chunk tensors from all_gather
            world_size_h: Number of splits along H
            world_size_w: Number of splits along W
            dim: Dimension to concatenate along (3 for H, 4 for W)

        Returns:
            Reconstructed full tensor
        """
        rows = []
        for h_idx in range(world_size_h):
            cols = []
            for w_idx in range(world_size_w):
                chunk_idx = h_idx * world_size_w + w_idx
                cols.append(chunks[chunk_idx])
            # Concatenate along W dimension (dim 4)
            rows.append(torch.cat(cols, dim=dim + 1))
        # Concatenate along H dimension (dim 3)
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
        """Encode video with true 2D spatial splitting.

        Each GPU processes a fixed spatial region with proper padding for
        boundary consistency. Uses single all_gather for communication.

        Args:
            video: Input video tensor [1, C, T, H, W]
            world_size_h: Number of splits along H dimension
            world_size_w: Number of splits along W dimension
            cur_rank_h: Current rank's position in H grid
            cur_rank_w: Current rank's position in W grid
            device: Computation device

        Returns:
            Encoded latent tensor on CPU [C, T, H_latent, W_latent]
        """
        spatial_ratio = self.upsampling_factor  # 8
        padding_latent = 1  # 1 latent pixel padding for encode

        # Calculate chunk dimensions in latent space
        latent_h = video.shape[3] // spatial_ratio
        latent_w = video.shape[4] // spatial_ratio
        chunk_h = latent_h // world_size_h
        chunk_w = latent_w // world_size_w

        # Convert to input space
        chunk_h_input = chunk_h * spatial_ratio
        chunk_w_input = chunk_w * spatial_ratio
        padding_input = padding_latent * spatial_ratio  # 8 pixels

        # Compute slices with padding in input space
        h_start, h_end = self._compute_padded_slice(
            cur_rank_h, world_size_h, video.shape[3], chunk_h_input, padding_input
        )
        w_start, w_end = self._compute_padded_slice(
            cur_rank_w, world_size_w, video.shape[4], chunk_w_input, padding_input
        )

        # Extract video chunk
        video_chunk = video[:, :, :, h_start:h_end, w_start:w_end].contiguous()
        video_chunk = video_chunk.to(device)

        # Encode the chunk
        encoded_chunk = self.model.encode(video_chunk, self.scale)

        # Remove padding from encoded result
        encoded_chunk = self._remove_latent_padding(
            encoded_chunk, cur_rank_h, cur_rank_w, world_size_h, world_size_w, chunk_h, chunk_w
        )

        # Gather all chunks
        world_size_total = world_size_h * world_size_w
        full_encoded = [torch.empty_like(encoded_chunk) for _ in range(world_size_total)]
        dist.all_gather(full_encoded, encoded_chunk)

        # Reconstruct full latent
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
        """Decode latent with true 2D spatial splitting.

        Each GPU processes a fixed spatial region with proper padding for
        boundary consistency. Uses single all_gather for communication.

        Args:
            latent: Input latent tensor [1, C, T, H, W]
            world_size_h: Number of splits along H dimension
            world_size_w: Number of splits along W dimension
            cur_rank_h: Current rank's position in H grid
            cur_rank_w: Current rank's position in W grid
            device: Computation device

        Returns:
            Decoded video tensor on CPU [C, T_out, H_out, W_out]
        """
        spatial_ratio = self.upsampling_factor  # 8
        padding_latent = 2  # 2 latent pixels padding for decode (larger kernel)

        # Calculate chunk dimensions in latent space
        latent_h = latent.shape[3]
        latent_w = latent.shape[4]
        chunk_h = latent_h // world_size_h
        chunk_w = latent_w // world_size_w

        # Compute slices with padding in latent space
        h_start, h_end = self._compute_padded_slice(cur_rank_h, world_size_h, latent_h, chunk_h, padding_latent)
        w_start, w_end = self._compute_padded_slice(cur_rank_w, world_size_w, latent_w, chunk_w, padding_latent)

        # Extract latent chunk
        latent_chunk = latent[:, :, :, h_start:h_end, w_start:w_end].contiguous()
        latent_chunk = latent_chunk.to(device)

        # Decode the chunk
        decoded_chunk = self.model.decode(latent_chunk, self.scale)

        # Calculate output chunk size (latent chunk * spatial_ratio)
        chunk_h_output = chunk_h * spatial_ratio
        chunk_w_output = chunk_w * spatial_ratio

        # Remove padding from decoded result
        decoded_chunk = self._remove_latent_padding(
            decoded_chunk, cur_rank_h, cur_rank_w, world_size_h, world_size_w, chunk_h_output, chunk_w_output
        )

        # Gather all chunks
        world_size_total = world_size_h * world_size_w
        full_decoded = [torch.empty_like(decoded_chunk) for _ in range(world_size_total)]
        dist.all_gather(full_decoded, decoded_chunk)

        # Reconstruct full video
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
            method: Parallel method, options:
                - "2d_split": True 2D spatial splitting (recommended, fastest)
                - "tile_dist": Tile task distribution (original method)
            world_size_h: Manual H grid size (optional, auto-calculated if None)
            world_size_w: Manual W grid size (optional, auto-calculated if None)

        Returns:
            Encoded latents tensor [B, C, T_latent, H_latent, W_latent]
        """
        world_size = dist.get_world_size()
        cur_rank = dist.get_rank()

        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)  # [1, C, T, H, W]
            latent_h = video.shape[3] // self.upsampling_factor
            latent_w = video.shape[4] // self.upsampling_factor

            if method == "2d_split":
                # Calculate 2D grid
                if world_size_h is None or world_size_w is None:
                    world_size_h, world_size_w = self.calculate_2d_grid(latent_h, latent_w, world_size)
                cur_rank_h = cur_rank // world_size_w
                cur_rank_w = cur_rank % world_size_w

                hidden_state = self.encode_dist_2d(video, world_size_h, world_size_w, cur_rank_h, cur_rank_w, device)
            else:  # "tile_dist"
                # Original tile distribution method
                tile_size = (34 * 8, 34 * 8)
                tile_stride = (18 * 8, 16 * 8)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
                hidden_state = hidden_state.squeeze(0)

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
            method: Parallel method, options:
                - "2d_split": True 2D spatial splitting (recommended, fastest)
                - "tile_dist": Tile task distribution (original method)
            world_size_h: Manual H grid size (optional, auto-calculated if None)
            world_size_w: Manual W grid size (optional, auto-calculated if None)

        Returns:
            Decoded video tensor on CPU [B, C, T_out, H_out, W_out]
        """
        world_size = dist.get_world_size()
        cur_rank = dist.get_rank()

        # Handle single tensor
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.unsqueeze(0)

        batch_size = hidden_states.shape[0]
        videos = []

        for i in range(batch_size):
            latent = hidden_states[i : i + 1]  # [1, C, T, H, W]
            latent_h = latent.shape[3]
            latent_w = latent.shape[4]

            if method == "2d_split":
                # Calculate 2D grid
                if world_size_h is None or world_size_w is None:
                    world_size_h, world_size_w = self.calculate_2d_grid(latent_h, latent_w, world_size)
                cur_rank_h = cur_rank // world_size_w
                cur_rank_w = cur_rank % world_size_w

                video = self.decode_dist_2d(latent, world_size_h, world_size_w, cur_rank_h, cur_rank_w, device)
            else:  # "tile_dist"
                video = self.tiled_decode(latent, device, (34, 34), (18, 16))

            videos.append(video)

        return torch.stack(videos)

    # ==================== Original Methods ====================

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
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def tiled_decode(
        self,
        hidden_states: torch.Tensor,
        device: torch.device,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> torch.Tensor:
        """Decode with tiling for memory efficiency."""
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

        data_device = device
        computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros(
            (1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=data_device,
        )

        hide_progress_bar = self.parallelism > 1 and dist.get_rank() != 0
        for i, (h, h_, w, w_) in enumerate(tqdm(tasks, desc="VAE DECODING", disable=hide_progress_bar)):
            if self.parallelism > 1 and (i % dist.get_world_size() != dist.get_rank()):
                continue
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale).to(data_device)

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

        if self.parallelism > 1:
            dist.all_reduce(values)
            dist.all_reduce(weight)
        values = values / weight
        # Move to CPU to reduce VRAM usage (video output is large)
        values = values.cpu().clamp_(-1, 1)
        return values

    def tiled_encode(
        self, video: torch.Tensor, device: torch.device, tile_size: tuple[int, int], tile_stride: tuple[int, int]
    ) -> torch.Tensor:
        """Encode with tiling for memory efficiency."""
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

        data_device = device
        computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros(
            (1, 1, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (1, 16, out_T, H // self.upsampling_factor, W // self.upsampling_factor),
            dtype=video.dtype,
            device=data_device,
        )

        hide_progress_bar = self.parallelism > 1 and dist.get_rank() != 0
        for i, (h, h_, w, w_) in enumerate(tqdm(tasks, desc="VAE ENCODING", disable=hide_progress_bar)):
            if self.parallelism > 1 and (i % dist.get_world_size() != dist.get_rank()):
                continue
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale).to(data_device)

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

        if self.parallelism > 1:
            dist.all_reduce(values)
            dist.all_reduce(weight)
        values = values / weight
        return values

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
        # Auto-detect parallel mode
        if self.parallelism > 1 and dist.is_initialized():
            # tiled=True → tile_dist, tiled=False → 2d_split
            method = "tile_dist" if tiled else "2d_split"
            result = self.encode_parallel(videos, device, method=method)
            # Parallel encode returns CPU tensors for memory efficiency
            # Move to target device for consistency with single-GPU behavior
            return result.to(device)

        # Single GPU processing
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size_scaled = (tile_size[0] * 8, tile_size[1] * 8)
                tile_stride_scaled = (tile_stride[0] * 8, tile_stride[1] * 8)
                hidden_state = self.tiled_encode(video, device, tile_size_scaled, tile_stride_scaled)
            else:
                video = video.to(device)
                hidden_state = self.model.encode(video, self.scale)
            hidden_state = hidden_state.squeeze(0)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states)
        return hidden_states

    def decode(
        self,
        hidden_states: torch.Tensor,
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
            hidden_states: Latent tensor [B, C, T, H, W] or [C, T, H, W]
            device: Target device
            tiled: Controls processing mode:
                   - Single GPU: True = tiled processing, False = direct decode
                   - Multi-GPU: True = tile_dist method, False = 2d_split method (default)
            tile_size: Tile size for tiled processing
            tile_stride: Tile stride for tiled processing

        Returns:
            Decoded video tensor on CPU [B, C, T_out, H_out, W_out]
        """
        # Auto-detect parallel mode
        if self.parallelism > 1 and dist.is_initialized():
            # tiled=True → tile_dist, tiled=False → 2d_split
            method = "tile_dist" if tiled else "2d_split"
            return self.decode_parallel(hidden_states, device, method=method)

        # Single GPU processing
        # Handle single tensor [C, T, H, W] by adding batch dimension
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.unsqueeze(0)

        batch_size = hidden_states.shape[0]
        videos = []

        for i in range(batch_size):
            hidden_state = hidden_states[i : i + 1].to(device)  # [1, C, T, H, W]

            if tiled:
                video = self.tiled_decode(hidden_state, device, tile_size, tile_stride)
            else:
                video = self.model.decode(hidden_state, self.scale)
                # Move to CPU to reduce VRAM usage
                video = video.cpu().clamp_(-1, 1)

            videos.append(video)

        videos = torch.cat(videos, dim=0)
        return videos

    def cached_decode_withflag(
        self,
        hidden_state: torch.Tensor,
        device: torch.device,
        is_first_clip: bool,
        is_last_clip: bool,
        decode_state: WanVideoVAEStreamingDecodeState | None = None,
    ) -> torch.Tensor:
        """Decode with persistent feature cache for streaming generation.

        Maintains intermediate feature cache across decode calls for efficiency.
        This is critical for streaming VAE decode where segments share cached features.

        Args:
            hidden_state: Latent tensor [C, T, H, W] or [1, C, T, H, W]
            device: Target device
            is_first_clip: If True, clear cache before decoding (first segment)
            is_last_clip: If True, clear cache after decoding (last segment)
            decode_state: Optional session-owned cache. The legacy model-owned cache
                is used when omitted.

        Returns:
            Decoded video tensor [C, T_out, H_out, W_out]
        """
        feat_cache = self._feat_cache if decode_state is None else decode_state.feat_cache
        feat_idx = self._feat_idx if decode_state is None else decode_state.feat_idx

        # Clear cache on first clip
        if is_first_clip:
            conv_num = _count_conv3d(self.model.decoder)
            feat_cache = [None] * conv_num
            feat_idx = [0]
            if decode_state is None:
                self._feat_cache = feat_cache
                self._feat_idx = feat_idx
            else:
                decode_state.feat_cache = feat_cache
                decode_state.feat_idx = feat_idx

        # Add batch dimension if needed
        if hidden_state.dim() == 4:
            hidden_state = hidden_state.unsqueeze(0)

        hidden_state = hidden_state.to(device)

        # Apply scaling
        scale = self.scale
        if isinstance(scale[0], torch.Tensor):
            scale = [s.to(dtype=hidden_state.dtype, device=device) for s in scale]
            z = hidden_state / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = hidden_state / scale[1] + scale[0]

        # Decode frame-by-frame with cache
        iter_ = z.shape[2]
        x = self.model.conv2(z)

        for i in range(iter_):
            feat_idx[0] = 0  # Reset index for each frame
            if i == 0:
                out = self.model.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=feat_idx,
                )
            else:
                out_ = self.model.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=feat_idx,
                )
                out = torch.cat([out, out_], 2)

        # Clear cache on last clip
        if is_last_clip:
            if decode_state is None:
                self._feat_cache = []
                self._feat_idx = [0]
            else:
                decode_state.feat_cache = []
                decode_state.feat_idx = [0]

        video = out.clamp_(-1, 1)

        # Remove batch dimension if single item
        if video.shape[0] == 1:
            video = video.squeeze(0)

        return video

    @staticmethod
    def state_dict_converter():
        return WanVideoVAEStateDictConverter()

    def enable_sequential_cpu_offload(self, device: torch.device, torch_dtype: torch.dtype):
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


class WanVideoVAEStateDictConverter:
    """State dict converter for Wan video VAE."""

    def from_official(self, state_dict: dict) -> tuple[dict, dict]:
        state_dict_ = {}
        cfg_dict_ = {}
        weight_hash = hash_state_dict_keys(state_dict)
        if weight_hash == "19560d299104e665df05de9a03074ed5":
            cfg_dict_["encode_pruning_rate"] = 0.75
            cfg_dict_["decode_pruning_rate"] = 0.75
        if weight_hash == "e9addbd0c9d54bc1827116b98e0dd1a0":
            cfg_dict_["decode_pruning_rate"] = 0.75
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        for name in state_dict:
            state_dict_["model." + name] = state_dict[name]
        return state_dict_, cfg_dict_


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "official")
register_model_config(None, "19560d299104e665df05de9a03074ed5", ["wan_video_vae"], [WanVideoVAE], "official")
register_model_config(None, "ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "official")
register_model_config(None, "e9addbd0c9d54bc1827116b98e0dd1a0", ["wan_video_vae"], [WanVideoVAE], "official")
