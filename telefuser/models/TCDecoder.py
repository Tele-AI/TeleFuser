"""
Tiny AutoEncoder for Hunyuan Video (Decoder-only, pruned)
- Encoder removed
- Transplant/widening helpers removed
- Deepening (IdentityConv2d+ReLU) is now built into the decoder structure itself
"""

from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from tqdm.auto import tqdm

from telefuser.utils.model_weight import hash_state_dict_keys

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


class IdentityConv2d(nn.Conv2d):
    """Same-shape Conv2d initialized to identity (Dirac)."""

    def __init__(self, C: int, kernel_size: int = 3, bias: bool = False):
        pad = kernel_size // 2
        super().__init__(C, C, kernel_size, padding=pad, bias=bias)
        with torch.no_grad():
            init.dirac_(self.weight)
            if self.bias is not None:
                self.bias.zero_()


def conv(n_in: int, n_out: int, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    """Clamp activations to [-3, 3] using tanh."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Memory block with skip connection for temporal consistency."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    """Temporal pooling: reduces frame dimension by stride factor."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    """Temporal growth: expands frame dimension by stride factor."""

    def __init__(self, n_f: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


class PixelShuffle3d(nn.Module):
    """3D pixel shuffle for video upsampling."""

    def __init__(self, ff: int, hh: int, ww: int):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, H, W)
        B, C, F, H, W = x.shape
        if F % self.ff != 0:
            # Pad frames by repeating first frame
            first_frame = x[:, :, 0:1, :, :].repeat(1, 1, self.ff - F % self.ff, 1, 1)
            x = torch.cat([first_frame, x], dim=2)
        return rearrange(
            x,
            "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
            ff=self.ff,
            hh=self.hh,
            ww=self.ww,
        ).transpose(1, 2)


def apply_model_with_memblocks(
    model: nn.Sequential,
    x: torch.Tensor,
    parallel: bool,
    show_progress_bar: bool,
    mem: list | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, list]:
    """Apply a sequential model with memblocks to the given input.

    Args:
        model: nn.Sequential of blocks to apply.
        x: Input data, dimensions NTCHW (batch, time, channels, height, width).
        parallel: If True, parallelize over timesteps (fast but O(T) memory).
                  If False, sequential processing (slow but O(1) memory).
        show_progress_bar: Enable tqdm progress bar.
        mem: Optional memory state for MemBlocks.
        device: Device for computation.

    Returns:
        NTCHW tensor of output data and updated memory state.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N * T, C, H, W)
        for b in tqdm(model, disable=not show_progress_bar, desc="decode video"):
            if isinstance(b, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                _x = x.reshape(N, T, C, H, W)
                # Create memory by padding and selecting previous frames
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = b(x, mem)
            else:
                x = b(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        out = []
        # Initialize work queue with first block for each timestep
        work_queue = [TWorkItem(xt, 0) for t, xt in enumerate(x.reshape(N, T * C, H, W).chunk(T, dim=1))]
        progress_bar = tqdm(range(T), disable=not show_progress_bar, desc="decode video")
        while work_queue:
            xt, i = work_queue.pop(0)
            xt = xt.to(device)
            if i == 0:
                progress_bar.update(1)
            if i == len(model):
                out.append(xt)
            else:
                b = model[i]
                if isinstance(b, MemBlock):
                    if mem[i] is None:
                        xt_new = b(xt, xt * 0)
                        mem[i] = xt
                    else:
                        xt_new = b(xt, mem[i])
                        mem[i].copy_(xt)
                    work_queue.insert(0, TWorkItem(xt_new, i + 1))
                elif isinstance(b, TPool):
                    if mem[i] is None:
                        mem[i] = []
                    mem[i].append(xt)
                    if len(mem[i]) > b.stride:
                        raise ValueError("TPool internal state invalid.")
                    elif len(mem[i]) == b.stride:
                        N_, C_, H_, W_ = xt.shape
                        xt = b(torch.cat(mem[i], 1).view(N_ * b.stride, C_, H_, W_))
                        mem[i] = []
                        work_queue.insert(0, TWorkItem(xt, i + 1))
                elif isinstance(b, TGrow):
                    xt = b(xt)
                    NT, C_, H_, W_ = xt.shape
                    # Split expanded frames and add to queue in reverse order
                    for xt_next in reversed(xt.view(N, b.stride * C_, H_, W_).chunk(b.stride, 1)):
                        work_queue.insert(0, TWorkItem(xt_next, i + 1))
                else:
                    xt = b(xt)
                    work_queue.insert(0, TWorkItem(xt, i + 1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x, mem


class TAEHV(nn.Module):
    """Tiny AutoEncoder for Hunyuan Video - decoder only variant."""

    image_channels = 3

    def __init__(
        self,
        decoder_time_upscale: tuple[bool, bool] = (True, True),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
        channels: list[int] = [512, 256, 128, 128],
        latent_channels: int = 16,
    ):
        """Initialize TAEHV (decoder-only) with built-in deepening after every ReLU.

        Deepening config: how_many_each=1, k=3 (fixed).
        """
        super().__init__()
        self.latent_channels = latent_channels
        n_f = channels
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        # Build the decoder "skeleton"
        base_decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            conv(n_f[3], TAEHV.image_channels),
        )

        # Inline deepening: insert (IdentityConv2d(k=3) + ReLU) after every ReLU
        self.decoder = self._apply_identity_deepen(base_decoder, how_many_each=1, k=3)

        self.pixel_shuffle = PixelShuffle3d(4, 8, 8)

        # Initialize decoder mem state
        self.mem = [None] * len(self.decoder)

    @staticmethod
    def _apply_identity_deepen(decoder: nn.Sequential, how_many_each: int = 1, k: int = 3) -> nn.Sequential:
        """Return new Sequential where every nn.ReLU is followed by how_many_each*(IdentityConv2d(k)+ReLU)."""
        new_layers = []
        for b in decoder:
            new_layers.append(b)
            if isinstance(b, nn.ReLU):
                # Deduce channel count from preceding layer
                C = None
                if len(new_layers) >= 2 and isinstance(new_layers[-2], nn.Conv2d):
                    C = new_layers[-2].out_channels
                elif len(new_layers) >= 2 and isinstance(new_layers[-2], MemBlock):
                    C = new_layers[-2].conv[-1].out_channels
                if C is not None:
                    for _ in range(how_many_each):
                        new_layers.append(IdentityConv2d(C, kernel_size=k, bias=False))
                        new_layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*new_layers)

    def patch_tgrow_layers(self, sd: dict) -> dict:
        """Patch TGrow layers to use a smaller kernel if needed (decoder-only)."""
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0] :]
        return sd

    def decode(
        self,
        x: torch.Tensor,
        parallel: bool = True,
        show_progress_bar: bool = True,
        cond: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Decode a sequence of frames from latents.

        Args:
            x: NTCHW latent tensor.
            parallel: Use parallel processing if True.
            show_progress_bar: Show progress bar.
            cond: Optional conditioning tensor.
            device: Device for computation.

        Returns:
            NTCHW RGB tensor in ~[0, 1].
        """
        if device is None:
            device = x.device
        trim_flag = self.mem[-8] is None

        if cond is not None:
            x = torch.cat([self.pixel_shuffle(cond), x], dim=2)

        x, self.mem = apply_model_with_memblocks(
            self.decoder, x, parallel, show_progress_bar, mem=self.mem, device=device
        )

        if trim_flag:
            return x[:, self.frames_to_trim :]
        return x

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Decoder-only model: call decode(...) instead.")

    def clean_mem(self):
        """Clear decoder memory state."""
        self.mem = [None] * len(self.decoder)

    @staticmethod
    def state_dict_converter():
        return TAEHVStateDictConverter()


class TAEHVStateDictConverter:
    """State dict converter for TAEHV."""

    def __init__(self):
        pass

    def from_official(self, state_dict: dict) -> tuple[dict, dict]:
        if hash_state_dict_keys(state_dict) == "4c3523c69fb7b24cf2db147a715b277f":
            return state_dict, dict(latent_channels=16 + 768)
        return state_dict, {}


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "4c3523c69fb7b24cf2db147a715b277f", ["wan_video_decoder"], [TAEHV], "official")
