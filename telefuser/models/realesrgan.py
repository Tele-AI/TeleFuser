# SPDX-License-Identifier: Apache-2.0
"""
Real-ESRGAN model for image super-resolution.

Real-ESRGAN model code is adapted from:
  - https://github.com/xinntao/Real-ESRGAN  (BSD-3-Clause License)
  Copyright (c) 2021 xinntao

Model weights download:
  - Default: https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth
  - Official releases: https://github.com/xinntao/Real-ESRGAN/releases
  - Alternative: https://huggingface.co/speechbrain/Real-ESRGAN

Supported architectures:
  - SRVGGNetCompact: Lightweight model (realesr-animevideov3, realesr-general-x4v3)
  - RRDBNet: Heavier model with higher quality (RealESRGAN_x4plus)
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from telefuser.core.base_model import BaseModel


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style network for super resolution.

    Corresponds to ``realesr-animevideov3`` and ``realesr-general-x4v3``.
    Reference: xinntao/Real-ESRGAN (BSD-3-Clause).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # first activation
        self.body.append(self._make_act(act_type, num_feat))
        # body convs + activations
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._make_act(act_type, num_feat))
        # last conv: maps to out_ch * upscale^2 for pixel shuffle
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _make_act(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        elif act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # residual addition with nearest upsampled input
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDB (RealESRGAN_x4plus)."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDB network for RealESRGAN_x4plus (heavier, higher quality for photos)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        in_ch = num_in_ch
        if scale == 2:
            in_ch = num_in_ch * 4
        elif scale == 1:
            in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def _build_net_from_state_dict(state_dict: dict) -> nn.Module:
    """Detect architecture from checkpoint keys and return an unloaded network."""
    if "conv_first.weight" in state_dict:
        # RRDBNet (e.g., RealESRGAN_x4plus)
        num_feat = state_dict["conv_first.weight"].shape[0]
        num_block = sum(1 for k in state_dict if k.startswith("body.") and k.endswith(".rdb1.conv1.weight"))
        num_grow_ch = state_dict["body.0.rdb1.conv1.weight"].shape[0]
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
        )
    else:
        # SRVGGNetCompact (e.g., realesr-animevideov3)
        num_feat = state_dict["body.0.weight"].shape[0]
        # body layout: [first_conv, first_act, (conv, act)*num_conv, last_conv]
        # count 4-D weight tensors = first_conv + loop_convs + last_conv = num_conv + 2
        conv_keys = sorted(
            [k for k in state_dict if k.startswith("body.") and k.endswith(".weight") and state_dict[k].dim() == 4],
            key=lambda k: int(k.split(".")[1]),
        )
        num_conv = len(conv_keys) - 2  # subtract first and last
        # upscale from last conv output channels: out_ch = num_out_ch * upscale^2
        last_out_ch = state_dict[conv_keys[-1]].shape[0]
        upscale = int(math.sqrt(last_out_ch / 3))
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=num_feat,
            num_conv=num_conv,
            upscale=upscale,
            act_type="prelu",
        )


class RealESRGAN(BaseModel):
    """Real-ESRGAN model for image super-resolution.

    Supports both SRVGGNetCompact (lightweight) and RRDBNet (heavier) architectures.
    Architecture is auto-detected from checkpoint weights.

    Model weights download:
      - Default: https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth
      - Official releases: https://github.com/xinntao/Real-ESRGAN/releases
    """

    def __init__(
        self,
        model_path: str,
        scale: int = 4,
    ):
        super().__init__()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                "Please download from: "
                "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth"
            )
        self.model_path = model_path
        self.target_scale = scale
        self._net: Optional[nn.Module] = None
        self._native_scale: int = 4

    @property
    def net(self) -> nn.Module:
        """Lazy-loaded network."""
        self._ensure_loaded()
        assert self._net is not None
        return self._net

    @property
    def native_scale(self) -> int:
        """Native scale of the loaded model."""
        return self._native_scale

    def _ensure_loaded(self) -> None:
        """Load model weights if not already loaded."""
        if self._net is not None:
            return

        state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)

        # Some checkpoints wrap weights under a 'params' or 'params_ema' key
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        self._net = _build_net_from_state_dict(state_dict)
        self._net.load_state_dict(state_dict, strict=True)
        self._net.eval()

        # Detect the model's native scale from network architecture
        if isinstance(self._net, SRVGGNetCompact):
            self._native_scale = self._net.upscale
        elif isinstance(self._net, RRDBNet):
            self._native_scale = self._net.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor [B, C, H, W] in range [0, 1].

        Returns:
            Upscaled tensor [B, C, H*scale, W*scale].
        """
        return self.net(x)

    def upscale_frame(
        self,
        frame: torch.Tensor,
        outscale: Optional[float] = None,
    ) -> torch.Tensor:
        """Upscale a single frame.

        Args:
            frame: Input tensor [B, C, H, W] in range [0, 1].
            outscale: Desired final upscaling factor. If different from the
                      model's native scale, a cheap resize is applied after
                      the network output.

        Returns:
            Upscaled tensor in range [0, 1].
        """
        h, w = frame.shape[2:4]

        with torch.no_grad():
            out = self.net(frame)

        # If the desired outscale differs from the model's native scale,
        # resize to (h * outscale, w * outscale).
        if outscale is not None and outscale != self._native_scale:
            target_h = int(h * outscale)
            target_w = int(w * outscale)
            out = F.interpolate(out, size=(target_h, target_w), mode="bicubic", align_corners=False)

        return out.clamp(0.0, 1.0)

    def upscale_frames(
        self,
        images: torch.Tensor,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Upscale a batch of frames.

        Args:
            images: Input tensor [N, H, W, C] in range [0, 1].
            device: Device for computation.

        Returns:
            Upscaled tensor [N, H*scale, W*scale, C] in range [0, 1].
        """
        # Convert [N, H, W, C] to [N, C, H, W]
        x = images.permute(0, 3, 1, 2).to(device)
        outscale = self.target_scale if self.target_scale != self._native_scale else None

        with torch.no_grad():
            out = self.upscale_frame(x, outscale=outscale)

        # Convert back to [N, H, W, C]
        return out.permute(0, 2, 3, 1).cpu()

    @staticmethod
    def state_dict_converter():
        return RealESRGANStateDictConverter()


class RealESRGANStateDictConverter:
    """State dict converter for RealESRGAN."""

    def __init__(self):
        pass

    def from_official(self, state_dict: dict) -> dict:
        # No conversion needed for standard Real-ESRGAN checkpoints
        return state_dict


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "d93074599b84fa126cbc6fb8bf61ea6e", ["upscaler_model"], [RealESRGAN], "official")
