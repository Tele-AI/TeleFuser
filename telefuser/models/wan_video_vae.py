from __future__ import annotations

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
    """Video VAE wrapper with tiled encoding/decoding support."""

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

    def set_parallelism(self, parallelism: int):
        self.parallelism = parallelism

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
        hidden_states = []
        for video in videos:
            video = video.unsqueeze(0)
            if tiled:
                tile_size = (tile_size[0] * 8, tile_size[1] * 8)
                tile_stride = (tile_stride[0] * 8, tile_stride[1] * 8)
                hidden_state = self.tiled_encode(video, device, tile_size, tile_stride)
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
        """Unified decode for batched latent tensors.

        Accepts tensor directly [B, C, T, H, W], loops over batch dimension.
        Uses tiled_decode for large latents or model.decode for small latents.

        Args:
            hidden_states: Latent tensor [B, C, T, H, W] or [C, T, H, W]
            device: Target device
            tiled: If True, use tiled_decode for memory efficiency.
                   If False, use model.decode directly.
            tile_size: Tile size for tiled processing
            tile_stride: Tile stride for tiled processing

        Returns:
            Decoded video tensor on CPU [B, C, T_out, H_out, W_out]
        """
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
    ) -> torch.Tensor:
        """Decode with persistent feature cache for streaming generation.

        Maintains intermediate feature cache across decode calls for efficiency.
        This is critical for streaming VAE decode where segments share cached features.

        Args:
            hidden_state: Latent tensor [C, T, H, W] or [1, C, T, H, W]
            device: Target device
            is_first_clip: If True, clear cache before decoding (first segment)
            is_last_clip: If True, clear cache after decoding (last segment)

        Returns:
            Decoded video tensor [C, T_out, H_out, W_out]
        """
        # Clear cache on first clip
        if is_first_clip:
            conv_num = _count_conv3d(self.model.decoder)
            self._feat_cache = [None] * conv_num
            self._feat_idx = [0]

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
            self._feat_idx[0] = 0  # Reset index for each frame
            if i == 0:
                out = self.model.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_cache,
                    feat_idx=self._feat_idx,
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
