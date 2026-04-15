from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics
from telefuser.models.wan_video_vae import WanVideoVAE, _convert_conv3d_to_channels_last_3d
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


class VAEStage(BaseStage):
    """VAE encoding/decoding stage for Wan video.

    Handles video encoding/decoding with support for tiled processing
    and frame conditioning for image-to-video generation.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        # Try Wan2.2 VAE first (48 channels), fallback to Wan2.1 VAE (16 channels)
        self.vae: WanVideoVAE = module_manager.fetch_module("wan_video_vae")
        self.model_names = ["vae"]

        # Convert Conv3d weights to channels_last_3d for cuDNN optimization
        # Must be done after load_state_dict (after fetch_module)
        conv_count = _convert_conv3d_to_channels_last_3d(self.vae.model)
        if conv_count > 0:
            logger.info(f"VAE: Converted {conv_count} Conv3d weights to channels_last_3d format")

        # Apply torch.compile to VAE encode/decode for better performance
        compile_config = model_runtime_config.compile_config
        if compile_config.enabled:
            # Compile the unified decode method (used by all decode paths)
            self.vae.decode = torch.compile(self.vae.decode)
            logger.info("✓ torch.compile enabled for VAE decode")

    @ProfilingContext4Debug("vae encode image")
    def encode_image(
        self,
        image: torch.Tensor,
        end_image: torch.Tensor | None,
        num_frames: int,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
        concat_mask: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, int] | tuple[tuple[torch.Tensor, torch.Tensor], int]:
        """Encode input image(s) to latent for I2V.

        Supports two modes:
        - Wan2.1 I2V style (concat_mask=True): Encode full video sequence with zeros,
          concatenate mask as first channel. DiT has in_dim=17 (1+16).
        - Wan2.2 TI2V style (concat_mask=False): Only encode single frame,
          blend happens in denoising stage. DiT has in_dim=48.

        Args:
            image: Input image tensor [1, C, H, W] from preprocess_image
            end_image: Optional end image tensor [1, C, H, W]
            num_frames: Number of frames in output video
            tiled: Enable tiled VAE processing
            tile_size: Tile size for tiled processing
            tile_stride: Tile stride for tiled processing
            concat_mask: If True, use Wan2.1 I2V style (encode full video, concat mask).
                        If False, use Wan2.2 TI2V style (encode single frame only).

        Returns:
            If concat_mask=True (Wan2.1 I2V): latent with mask [1, 1+z_dim, T, H, W]
            If concat_mask=False (Wan2.2 TI2V): tuple of (latent [z_dim, 1, H, W], num_frames)
        """
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            # image is [1, C, H, W] from preprocess_image
            height = image.shape[2]
            width = image.shape[3]
            image = image.to(self.device).squeeze(0)  # [C, H, W]

            if concat_mask:
                # Wan2.1 I2V style: Encode full video sequence (image + zeros)
                if end_image is not None:
                    end_image = end_image.to(self.device).squeeze(0)  # [C, H, W]
                    vae_input = torch.concat(
                        [
                            image.unsqueeze(1),  # [C, 1, H, W]
                            torch.zeros(3, num_frames - 2, height, width, device=self.device),
                            end_image.unsqueeze(1),
                        ],
                        dim=1,
                    )
                else:
                    vae_input = torch.concat(
                        [
                            image.unsqueeze(1),
                            torch.zeros(3, num_frames - 1, height, width, device=self.device),
                        ],
                        dim=1,
                    )

                # Encode to latent
                y = self.vae.encode(
                    [vae_input], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                )[0]

                # Create mask: 0 for conditioned frames, 1 for noise frames
                # Wan2.1/Wan2.2 I2V uses a 4-channel mask
                latent_t = y.shape[1]
                latent_h = y.shape[2]
                latent_w = y.shape[3]

                # Create 4-channel mask following Wan2.2 I2V implementation
                # First frame is conditioned (mask=1), rest are noise (mask=0)
                msk = torch.zeros(4, latent_t, latent_h, latent_w, device=self.device)
                msk[:, 0] = 1  # First frame is conditioned
                if end_image is not None:
                    msk[:, -1] = 1  # Last frame is also conditioned

                y = torch.concat([msk, y]).unsqueeze(0)  # [1, 4+z_dim, T, H, W]
                return y
            else:
                # Wan2.2 TI2V style: Only encode single frame
                image_input = image.unsqueeze(1)  # [C, 1, H, W]
                start_latent = self.vae.encode(
                    [image_input], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                )[0]

                if end_image is not None:
                    end_input = end_image.to(self.device).squeeze(0).unsqueeze(1)
                    end_latent = self.vae.encode(
                        [end_input], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
                    )[0]
                    return (start_latent, end_latent), num_frames
                else:
                    return start_latent, num_frames

    @ProfilingContext4Debug("vae encode video")
    def encode_video(
        self,
        input_video: list[torch.Tensor],
        tiled: bool = True,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        """Encode video frames to latents."""
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            latents = self.vae.encode(
                input_video,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            latents = latents.to(dtype=self.torch_dtype, device=self.device)
        return latents

    @ProfilingContext4Debug("vae decode video")
    def decode_video(
        self,
        latents: torch.Tensor,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        """Decode latents to video frames.

        Args:
            latents: Latent tensor [B, C, T, H, W] or [C, T, H, W]
            tiled: Enable tiled processing
            tile_size: Tile size for tiled processing
            tile_stride: Tile stride for tiled processing

        Returns:
            Decoded video frames tensor on CPU
        """
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            # Unified decode() handles both single and batched tensors
            frames = self.vae.decode(
                latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
        return frames

    @ProfilingContext4Debug("vae decode video cached")
    def decode_video_cached(
        self,
        latents: torch.Tensor,
        is_first_clip: bool,
        is_last_clip: bool,
    ) -> torch.Tensor:
        """Decode latents to video frames with persistent feature cache.

        Uses cached intermediate features for streaming generation efficiency.
        Critical for LiveAct streaming decode where segments share cached features.

        Args:
            latents: Latent tensor [C, T, H, W] or [1, C, T, H, W]
            is_first_clip: If True, clear cache before decoding (first segment)
            is_last_clip: If True, clear cache after decoding (last segment)

        Returns:
            Decoded video frames tensor
        """
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            frames = self.vae.cached_decode_withflag(
                latents,
                device=self.device,
                is_first_clip=is_first_clip,
                is_last_clip=is_last_clip,
            )
        return frames

    def parallel_models(self):
        """Configure tensor parallelism for VAE."""
        self.vae.set_parallelism(self.model_runtime_config.parallel_config.world_size)

    @with_model_offload(["vae"])
    @torch.inference_mode()
    @with_metrics
    def process(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to specified VAE method."""
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{method} is not supported in wan vae")
