from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from telefuser.distributed.collectives import all_reduce_sum_

CACHE_T = 2


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


class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution for temporal consistency."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reorder padding: (left, right, top, bottom, front, back)
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
        x = F.pad(x, padding, mode="replicate")
        return super().forward(x)


class PixelShuffle3d(nn.Module):
    """3D pixel shuffle for video feature rearrangement."""

    def __init__(self, ff: int, hh: int, ww: int):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, H, W)
        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
            ff=self.ff,
            hh=self.hh,
            ww=self.ww,
        )


class Buffer_LQ4x_Proj(nn.Module):
    """Buffer-based low-quality 4x projection for video features."""

    def __init__(self, in_dim: int, out_dim: int, layer_num: int = 30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )  # f -> f/2
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(
            self.hidden_dim1,
            self.hidden_dim2,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )  # f -> f/2
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])

        self.clip_idx = 0

    def forward(self, video: torch.Tensor) -> list[torch.Tensor]:
        self.clear_cache()
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        # Pad with first frame repeated
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:, :, i * 4 : (i + 1) * 4, :, :])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv1"] = cache1_x
            x = self.conv1(x, self.cache["conv1"])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv2"] = cache2_x
            if i == 0:
                continue
            x = self.conv2(x, self.cache["conv2"])
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim=2)
        out_x = rearrange(out_x, "b c f h w -> b (f h w) c")
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs

    def clear_cache(self):
        self.cache = {}
        self.cache["conv1"] = None
        self.cache["conv2"] = None
        self.clip_idx = 0

    def stream_forward(self, video_clip: torch.Tensor) -> list[torch.Tensor] | None:
        """Streaming forward for video clips."""
        if self.clip_idx == 0:
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv1"] = cache1_x
            x = self.conv1(x, self.cache["conv1"])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv2"] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv1"] = cache1_x
            x = self.conv1(x, self.cache["conv1"])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv2"] = cache2_x
            x = self.conv2(x, self.cache["conv2"])
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, "b c f h w -> b (f h w) c")
            outputs = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs


class Causal_LQ4x_Proj(nn.Module):
    """Causal low-quality 4x projection with tile support for video features."""

    def __init__(self, in_dim: int, out_dim: int, layer_num: int = 30, parallelism: int = 1):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num
        self.out_dim = out_dim

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)

        self.conv1 = CausalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(
            self.hidden_dim1,
            self.hidden_dim2,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])

        self.clip_idx = 0
        self.parallelism = parallelism
        self.cache = {}
        self.cache["conv1"] = None
        self.cache["conv2"] = None
        self.tile_tasks = None

    def forward(self, video: torch.Tensor) -> list[torch.Tensor]:
        self.clear_cache()
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)

        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:, :, i * 4 : (i + 1) * 4, :, :])
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            if i == 0:
                self.cache["conv2"] = cache2_x
                continue
            x = self.conv2(x, self.cache["conv2"])
            self.cache["conv2"] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim=2)
        out_x = rearrange(out_x, "b c f h w -> b (f h w) c")
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs

    def clear_cache(self):
        self.cache = {}
        self.cache["conv1"] = None
        self.cache["conv2"] = None
        self.clip_idx = 0
        self.tile_tasks = None

    def set_parallelism(self, parallelism: int):
        self.parallelism = parallelism

    def tile_stream_forward(
        self,
        video_clip: torch.Tensor,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
        device: torch.device | None = None,
    ) -> list[torch.Tensor] | None:
        """Tile-based stream forward with overlapping tiles and averaging.

        Args:
            video_clip: [1, 3, 4, h, w] or [1, 3, 1, h, w] for first frame.
            tile_size: (height, width) of tiles.
            tile_stride: (height, width) stride between tiles.
            device: Computation device.
        """
        if device is None:
            device = video_clip.device
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride
        _, _, T, H, W = video_clip.shape

        # Split tasks into overlapping tiles
        tasks = []
        for h in range(0, H, stride_h):
            if h - stride_h >= 0 and h - stride_h + size_h >= H:
                continue
            for w in range(0, W, stride_w):
                if w - stride_w >= 0 and w - stride_w + size_w >= W:
                    continue
                h_end = min(h + size_h, H)
                w_end = min(w + size_w, W)
                tasks.append((h, h_end, w, w_end))

        out_T = (T + 3) // 4
        output_H = H // 16
        output_W = W // 16

        # Initialize weight and values buffers for each layer
        weights = []
        all_values = []
        for _ in range(self.layer_num):
            weights.append(torch.zeros((1, 1, out_T, output_H, output_W), dtype=video_clip.dtype, device=device))
            all_values.append(
                torch.zeros(
                    (1, 1, out_T, output_H, output_W, self.out_dim),
                    dtype=video_clip.dtype,
                    device=device,
                )
            )
        hide_progress_bar = self.parallelism > 1 and dist.get_rank() != 0

        # Process each tile
        for i, (h, h_end, w, w_end) in enumerate(tqdm(tasks, desc="Processing tiles", disable=hide_progress_bar)):
            if self.parallelism > 1 and (i % dist.get_world_size() != dist.get_rank()):
                continue
            tile_clip = video_clip[:, :, :, h:h_end, w:w_end]

            conv1_cache_key = f"conv1-{h}-{w}"
            conv2_cache_key = f"conv2-{h}-{w}"
            if self.clip_idx == 0:
                first_frame = tile_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
                tile_clip = torch.cat([first_frame, tile_clip], dim=2)
                x = self.pixel_shuffle(tile_clip)
                cache1_x = x[:, :, -CACHE_T:, :, :].clone()
                self.cache[conv1_cache_key] = cache1_x
                x = self.conv1(x, self.cache[conv1_cache_key])
                x = self.norm1(x)
                x = self.act1(x)
                cache2_x = x[:, :, -CACHE_T:, :, :].clone()
                self.cache[conv2_cache_key] = cache2_x
                tile_outputs = None
            else:
                x = self.pixel_shuffle(tile_clip)
                cache1_x = x[:, :, -CACHE_T:, :, :].clone()
                x = self.conv1(x, self.cache[conv1_cache_key])
                self.cache[conv1_cache_key] = cache1_x
                x = self.norm1(x)
                x = self.act1(x)
                cache2_x = x[:, :, -CACHE_T:, :, :].clone()
                x = self.conv2(x, self.cache[conv2_cache_key])
                self.cache[conv2_cache_key] = cache2_x
                x = self.norm2(x)
                x = self.act2(x)
                output_shape = x.shape
                out_x = rearrange(x, "b c f h w -> b (f h w) c")
                tile_outputs = []
                for i in range(self.layer_num):
                    tile_outputs.append(self.linear_layers[i](out_x))

            if tile_outputs is not None:
                target_h = h // 16
                target_w = w // 16
                tile_h = (h_end - h) // 16
                tile_w = (w_end - w) // 16

                mask = self.build_mask(
                    output_shape,
                    is_bound=(h == 0, h_end >= H, w == 0, w_end >= W),
                    border_width=((size_h - stride_h) // 16, (size_w - stride_w) // 16),
                ).to(device)

                for layer_idx, tile_output in enumerate(tile_outputs):
                    spatial_output = tile_output.view(1, 1, out_T, tile_h, tile_w, -1)
                    all_values[layer_idx][:, :, :, target_h : target_h + tile_h, target_w : target_w + tile_w, :] += (
                        spatial_output * mask.unsqueeze(-1)
                    )
                    weights[layer_idx][:, :, :, target_h : target_h + tile_h, target_w : target_w + tile_w] += mask

        if self.clip_idx == 0:
            self.clip_idx += 1
            return None

        # Average overlapping regions
        final_outputs = []
        for layer_idx in range(self.layer_num):
            weight = weights[layer_idx]
            all_value = all_values[layer_idx]
            if self.parallelism > 1:
                all_reduce_sum_((all_value, weight))
            weight[weight == 0] = 1
            averaged = all_value / weight.unsqueeze(-1)
            averaged = averaged.view(1, -1, self.linear_layers[0].out_features).cpu()
            final_outputs.append(averaged)
        self.clip_idx += 1
        return final_outputs

    def build_1d_mask(self, length: int, left_bound: bool, right_bound: bool, border_width: int) -> torch.Tensor:
        """Build 1D mask with linear blending at boundaries."""
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + 1) / border_width, dims=(0,))
        return x

    def build_mask(
        self,
        data_shape: tuple,
        is_bound: tuple[bool, bool, bool, bool],
        border_width: tuple[int, int],
    ) -> torch.Tensor:
        """Build 2D mask from 1D masks for tile blending."""
        _, _, _, H, W = data_shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def stream_forward(self, video_clip: torch.Tensor) -> list[torch.Tensor] | None:
        """Streaming forward for video clips."""
        if self.clip_idx == 0:
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv2"] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv2(x, self.cache["conv2"])
            self.cache["conv2"] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, "b c f h w -> b (f h w) c")
            outputs = []
            for i in range(self.layer_num):
                outputs.append(self.linear_layers[i](out_x))
            self.clip_idx += 1
            return outputs
