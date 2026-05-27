from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from telefuser.core.base_model import BaseModel

backwarp_tenGrid: dict = {}


def _calculate_target_frame_positions(
    source_fps: float, target_fps: float, total_source_frames: int
) -> List[Tuple[int, int, float]]:
    """Calculate which frames need to be generated for the target frame rate.

    Returns:
        List of (source_frame_index1, source_frame_index2, interpolation_factor) tuples.
    """
    frame_positions = []
    duration = (total_source_frames - 1) / source_fps
    total_target_frames = int(duration * target_fps) + 1

    for target_idx in range(total_target_frames):
        target_time = target_idx / target_fps
        source_position = target_time * source_fps
        source_idx1 = int(source_position)
        source_idx2 = min(source_idx1 + 1, total_source_frames - 1)

        if source_idx1 == source_idx2:
            interpolation_factor = 0.0
        else:
            interpolation_factor = source_position - source_idx1

        frame_positions.append((source_idx1, source_idx2, interpolation_factor))

    return frame_positions


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Warp input tensor using optical flow."""
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        # Create coordinate grid for warping
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3])
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2])
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1)

    # Normalize flow to grid coordinates [-1, 1]
    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (backwarp_tenGrid[k].to(tenInput.device) + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def conv(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


def conv_bn(
    in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1
):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    """Feature extraction head for RIFE."""

    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, feat: bool = False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    """Residual convolution block with learnable scaling."""

    def __init__(self, c: int, dilation: int = 1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """Intermediate Flow Estimation block for multi-scale processing."""

    def __init__(self, in_planes: int, c: int = 64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor, flow: torch.Tensor | None = None, scale: float = 1) -> tuple:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) * 1.0 / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class IFNet(BaseModel):
    """RIFE (Real-Time Intermediate Flow Estimation) for video frame interpolation."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=96)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block4 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: list = [8, 4, 2, 1],
        training: bool = False,
        fastmode: bool = True,
        ensemble: bool = False,
    ) -> tuple:
        if not training:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i in range(5):
            if flow is None:
                flow, mask, feat = block[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None,
                    scale=scale_list[i],
                )
                if ensemble:
                    print("warning: ensemble is not supported since RIFEv4.21")
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )
                if ensemble:
                    print("warning: ensemble is not supported since RIFEv4.21")
                else:
                    mask = m0
                flow = flow + fd
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        mask = torch.sigmoid(mask)
        merged[4] = warped_img0 * mask + warped_img1 * (1 - mask)
        return flow_list, mask_list[4], merged

    def interpolate_frame(self, img0: torch.Tensor, img1: torch.Tensor, timestep: float = 0.5, scale: float = 1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged = self.__call__(imgs, timestep, scale_list)
        return merged[-1]

    def interpolate_frames(
        self,
        images: torch.Tensor,
        source_fps: float,
        target_fps: float,
        scale: float = 1.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Interpolate frames from source FPS to target FPS.

        Args:
            images: ComfyUI Image tensor [N, H, W, C] in range [0, 1].
            source_fps: Source frame rate.
            target_fps: Target frame rate.
            scale: Scale factor for processing.
            device: Device for computation.

        Returns:
            Interpolated tensor [M, H, W, C] in range [0, 1].
        """
        assert images.dim() == 4 and images.shape[-1] == 3, "Input must be [N, H, W, C] with C=3"

        if source_fps == target_fps:
            return images

        total_source_frames = images.shape[0]
        height, width = images.shape[1:3]

        # Calculate padding for model (must be multiple of 128/scale)
        tmp = max(128, int(128 / scale))
        ph = ((height - 1) // tmp + 1) * tmp
        pw = ((width - 1) // tmp + 1) * tmp
        padding = (0, pw - width, 0, ph - height)

        frame_positions = _calculate_target_frame_positions(source_fps, target_fps, total_source_frames)
        output_frames = []

        for source_idx1, source_idx2, interp_factor in frame_positions:
            if interp_factor == 0.0 or source_idx1 == source_idx2:
                output_frames.append(images[source_idx1].cpu())
            else:
                frame1 = images[source_idx1]
                frame2 = images[source_idx2]

                # Convert ComfyUI format [H, W, C] to model format [1, C, H, W]
                I0 = frame1.permute(2, 0, 1).unsqueeze(0).to(device)
                I1 = frame2.permute(2, 0, 1).unsqueeze(0).to(device)
                I0 = F.pad(I0, padding)
                I1 = F.pad(I1, padding)

                with torch.no_grad():
                    interpolated = self.interpolate_frame(I0, I1, timestep=interp_factor, scale=scale)

                # Convert back to ComfyUI format
                interpolated_frame = interpolated[0, :, :height, :width].permute(1, 2, 0).cpu()
                output_frames.append(interpolated_frame)

        return torch.stack(output_frames, dim=0)

    @staticmethod
    def state_dict_converter():
        return IFNetStateDictConverter()


class IFNetStateDictConverter:
    """State dict converter for IFNet."""

    def __init__(self):
        pass

    def from_official(self, state_dict: dict) -> dict:
        state_dict_ = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
            if "module." in k and "teacher" not in k and "caltime" not in k
        }
        return state_dict_


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(None, "47ac734bfc54bc2f75ff43e8e016588d", ["vfi_model"], [IFNet], "official")
