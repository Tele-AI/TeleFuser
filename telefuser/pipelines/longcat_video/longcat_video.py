from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.common.rift_vfi import RiftVFIStage
from telefuser.pipelines.wan_video.vae import VAEStage
from telefuser.utils.func import auto_async_call
from telefuser.utils.logging import logger
from telefuser.utils.lora_network import REFINE_LORA_KEY
from telefuser.utils.video import VideoData
from telefuser.worker.parallel_worker import ParallelWorker

from .dit_denoising import LongCatDitDenoisingStage
from .refine_denoise import LongCatRefineDenoisingStage
from .text_encoding import LongCatTextEncodingStage

# Official LongCat 720p bucket table for BSA mode (scale_factor_spatial=64).
# All entries are divisible by 64 (vae=8 * patchify=2 * chunk_q=4).
# Source: longcat_video/utils/bukcet_config.py ASPECT_RATIO_960_F64
_ASPECT_RATIO_960_F64: dict[float, tuple[int, int]] = {
    0.22: (448, 2048),
    0.29: (512, 1792),
    0.36: (576, 1600),
    0.45: (640, 1408),
    0.55: (704, 1280),
    0.63: (768, 1216),
    0.76: (832, 1088),
    0.88: (896, 1024),
    1.00: (960, 960),
    1.14: (1024, 896),
    1.31: (1088, 832),
    1.50: (1152, 768),
    1.58: (1216, 768),
    1.82: (1280, 704),
    1.91: (1344, 704),
    2.20: (1408, 640),
    2.30: (1472, 640),
    2.67: (1536, 576),
    2.89: (1664, 576),
    3.62: (1856, 512),
    3.75: (1920, 512),
}


def _bsa_align_resolution(height: int, width: int) -> tuple[int, int]:
    """Find the nearest BSA-compatible 720p resolution by aspect ratio.

    Uses the official ASPECT_RATIO_960_F64 bucket table where all resolutions
    are divisible by 64 (vae=8 * patchify=2 * bsa_chunk_q=4).

    Args:
        height: Requested pixel height.
        width: Requested pixel width.

    Returns:
        (height, width) from the nearest bucket entry.
    """
    ratio = height / width
    nearest = min(_ASPECT_RATIO_960_F64.keys(), key=lambda r: abs(r - ratio))
    return _ASPECT_RATIO_960_F64[nearest]


@dataclass
class LongCatVideoPipelineConfig:
    """Configuration for LongCat video generation pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    vfi_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_vfi: bool = False
    enable_metrics: bool = False
    enable_refine: bool = False
    refine_dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    refine_num_steps: int = 50
    refine_t_thresh: float = 0.5
    refine_lora_path: str = ""
    enable_refine_bsa: bool = False
    refine_num_frames: int | None = None
    spatial_refine_only: bool = False


class LongCatVideoPipeline(BasePipeline):
    """LongCat video generation pipeline supporting T2V, I2V, and video continuation."""

    def __init__(
        self,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.num_ref_video_frames = 13
        self.base_fps = 15

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        stages = [self.vae_stage, self.denoise_stage, self.text_encoding_stage]
        if hasattr(self, "vfi_stage"):
            stages.append(self.vfi_stage)
        if hasattr(self, "refine_denoise_stage"):
            stages.append(self.refine_denoise_stage)
        return stages

    def init(
        self,
        module_manager: ModuleManager,
        config: LongCatVideoPipelineConfig,
    ):
        self.config = config
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        if config.enable_vae_parallel:
            logger.info("enable parallel worker for vae")
            self.vae_stage = ParallelWorker(self.vae_stage)
        if config.sample_solver == "euler":
            self.scheduler = FlowMatchEulerDiscreteScheduler(
                shift=12.0,
                use_dynamic_shifting=False,
                invert_sigmas=False,
                time_shift_type="linear",
            )
        else:
            raise NotImplementedError(f"solver {config.sample_solver} is not supported")
        self.denoise_stage = LongCatDitDenoisingStage(
            "denoise",
            module_manager,
            config.dit_config,
            self.scheduler,
        )
        if config.enable_denoising_parallel:
            logger.info("enable parallel worker for denoising")
            self.denoise_stage = ParallelWorker(self.denoise_stage)
        self.text_encoding_stage = LongCatTextEncodingStage(
            "text_encoding",
            module_manager,
            config.text_encoding_config,
        )
        if config.enable_vfi:
            self.vfi_stage = RiftVFIStage("vfi", module_manager, config.vfi_config)

        if config.enable_refine:
            self.refine_scheduler = FlowMatchEulerDiscreteScheduler(
                shift=12.0,
                use_dynamic_shifting=False,
                invert_sigmas=False,
                time_shift_type="linear",
            )
            # Pre-load refinement LoRA as switchable (not permanently merged)
            if config.refine_lora_path:
                dit = module_manager.fetch_module("wan_video_dit")
                dit.load_lora(config.refine_lora_path, REFINE_LORA_KEY)
            self.refine_denoise_stage = LongCatRefineDenoisingStage(
                "refine_denoise",
                module_manager,
                config.refine_dit_config,
                self.refine_scheduler,
            )
            if config.enable_denoising_parallel:
                logger.info("enable parallel worker for refine denoising")
                self.refine_denoise_stage = ParallelWorker(self.refine_denoise_stage)

        # Auto-enable metrics if configured
        if config.enable_metrics:
            self.enable_metrics()

    def enable_cpu_offload(self):
        if not self.config.enable_denoising_parallel:
            if hasattr(self.denoise_stage, "enable_cpu_offload"):
                self.denoise_stage.enable_cpu_offload()
        if not self.config.enable_vae_parallel:
            if hasattr(self.vae_stage, "enable_cpu_offload"):
                self.vae_stage.enable_cpu_offload()
        if hasattr(self.text_encoding_stage, "enable_cpu_offload"):
            self.text_encoding_stage.enable_cpu_offload()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        input_image: Image.Image | None = None,
        input_video: VideoData | torch.Tensor | None = None,
        seed: int | None = None,
        rand_device: str | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        sigma_shift: float = 12.0,
        use_kv_cache: bool = False,
        use_cfg_zero_star: bool = True,
        attn_impl: str = "sdpa",
        return_latents: bool = False,
        need_encode: bool = True,
        use_distill: bool = False,
        enhance_hf: bool = False,
        target_fps: int | None = None,
        enable_refine: bool = False,
        refine_height: int | None = None,
        refine_width: int | None = None,
        refine_num_steps: int | None = None,
        refine_t_thresh: float | None = None,
        refine_num_frames: int | None = None,
        enable_refine_bsa: bool | None = None,
        spatial_refine_only: bool | None = None,
        progress_bar_cmd: Callable = tqdm,
    ):
        if use_kv_cache and input_image is None and input_video is None:
            logger.warning("use_kv_cache=True requires input_image or input_video, falling back to False")
            use_kv_cache = False

        origin_width, origin_height = width, height
        num_noise_frames_added = 0
        num_cond_frames_added = 0

        has_input_video = input_video is not None

        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        logger.info(f"start generate {num_frames} {width}x{height} frames")

        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            logger.info(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        # Tiler parameters
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        # Initialize noise
        if rand_device is None:
            rand_device = self.device
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        latents = noise.to(dtype=self.torch_dtype, device=self.device)

        # Encode prompts
        prompt_list = [prompt]
        if cfg_scale > 1:
            prompt_list.append(negative_prompt)

        prompt_emb_list_handler = auto_async_call(self.text_encoding_stage.process, prompt_list)

        # Encode image
        ref_latent = None
        num_cond_latents = 0

        if input_video is not None or input_image is not None:
            if input_video is not None:
                # vc
                if need_encode:
                    # Encode video
                    current_fps = input_video.fps()
                    stride = max(1, round(current_fps / self.base_fps))
                    input_video_length = len(input_video)

                    if input_video_length < self.num_ref_video_frames * stride:
                        logger.error(
                            f"input video length is {input_video_length} \
                            < {self.num_ref_video_frames} with stride {stride}"
                        )
                        return [], None

                    ref_frames = input_video.raw_data()[::stride][-self.num_ref_video_frames :]
                    input_video_processed = self.preprocess_images(ref_frames, height, width)
                    input_video_data = torch.stack(input_video_processed, dim=2).to(
                        dtype=self.torch_dtype, device=self.device
                    )

                    ref_latent_handler = auto_async_call(
                        self.vae_stage.process,
                        "encode_video",
                        input_video_data,
                        **tiler_kwargs,
                    )
                    ref_latent = ref_latent_handler()
                    ref_latent = ref_latent.to(dtype=self.torch_dtype, device=self.device)
                    num_cond_latents = ref_latent.shape[2]
                else:
                    # input_video is latent
                    num_cond_latents = (self.num_ref_video_frames - 1) // 4
                    ref_latent = input_video[:, :, -num_cond_latents:].to(dtype=self.torch_dtype, device=self.device)
            else:
                # i2v
                input_image_data = self.preprocess_image(input_image, height, width).unsqueeze(2)

                ref_latent_handler = auto_async_call(
                    self.vae_stage.process,
                    "encode_video",
                    input_image_data,
                    **tiler_kwargs,
                )
                ref_latent = ref_latent_handler()
                ref_latent = ref_latent.to(dtype=self.torch_dtype, device=self.device)
                num_cond_latents = ref_latent.shape[2]

        if ref_latent is not None:
            latents[:, :, : ref_latent.shape[2]] = ref_latent

        prompt_emb_list = prompt_emb_list_handler()
        prompt_embeds, prompt_attention_mask = prompt_emb_list[0]

        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        if cfg_scale > 1:
            negative_prompt_embeds, negative_prompt_attention_mask = prompt_emb_list[1]

        # denoise
        latents_handler = auto_async_call(
            self.denoise_stage.process,
            latents,
            num_inference_steps,
            num_cond_latents,
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            cfg_scale,
            sigma_shift,
            use_kv_cache,
            use_cfg_zero_star,
            attn_impl,
            use_distill,
            enhance_hf,
            progress_bar_cmd,
        )
        latents = latents_handler()

        if use_kv_cache and ref_latent is not None:
            latents = torch.cat([ref_latent, latents], dim=2)

        # Official LongCat Refinement (LoRA-based)
        if enable_refine and hasattr(self, "refine_denoise_stage"):
            _ref_height = refine_height if refine_height is not None else height * 2
            _ref_width = refine_width if refine_width is not None else width * 2
            _ref_height, _ref_width = self.check_resize_height_width(_ref_height, _ref_width)
            _enable_bsa = enable_refine_bsa if enable_refine_bsa is not None else self.config.enable_refine_bsa
            if _enable_bsa:
                # Use official bucket table to find nearest BSA-compatible resolution by aspect ratio.
                # All bucket entries are divisible by vae(8)*patchify(2)*chunk_q(4)=64.
                orig_h, orig_w = _ref_height, _ref_width
                _ref_height, _ref_width = _bsa_align_resolution(_ref_height, _ref_width)
                if _ref_height != orig_h or _ref_width != orig_w:
                    logger.info(
                        f"BSA enabled: adjusted refine resolution from {orig_w}x{orig_h} "
                        f"to {_ref_width}x{_ref_height} (nearest BSA-compatible bucket)"
                    )
            _ref_steps = refine_num_steps if refine_num_steps is not None else self.config.refine_num_steps
            _ref_thresh = refine_t_thresh if refine_t_thresh is not None else self.config.refine_t_thresh
            _ref_frames = refine_num_frames if refine_num_frames is not None else self.config.refine_num_frames
            _spatial_only = spatial_refine_only if spatial_refine_only is not None else self.config.spatial_refine_only

            # 1. Decode base latents to pixels
            base_frames_handler = auto_async_call(self.vae_stage.process, "decode_video", latents, **tiler_kwargs)
            base_frames = base_frames_handler()  # (B, C, T, H, W)
            b, c, t, h, w = base_frames.shape

            # Determine target frame count for temporal refine
            if _ref_frames is None and not _spatial_only:
                # Default: temporal + spatial refine (frame count doubles)
                _ref_frames = t * 2
            new_frame_count = _ref_frames if _ref_frames is not None and _ref_frames != t else t
            is_temporal_refine = new_frame_count != t

            logger.info(
                f"Refine: {width}x{height} -> {_ref_width}x{_ref_height}, "
                f"frames {t}->{new_frame_count}, steps={_ref_steps}, "
                f"t_thresh={_ref_thresh}, bsa={_enable_bsa}"
            )

            # 2. Upsample pixels to target resolution (and optionally temporal)
            if is_temporal_refine:
                upsampled = F.interpolate(
                    base_frames, size=(new_frame_count, _ref_height, _ref_width), mode="trilinear", align_corners=True
                )
            else:
                base_4d = base_frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
                up_4d = F.interpolate(base_4d, size=(_ref_height, _ref_width), mode="bilinear", align_corners=True)
                upsampled = up_4d.reshape(b, t, c, _ref_height, _ref_width).permute(0, 2, 1, 3, 4)

            # 3. Prepare condition frames for I2V/VC refine
            refine_num_cond_frames = 0
            refine_cond_pixels = None
            if input_image is not None:
                # I2V refine: use original image as 1 condition frame at refine resolution
                refine_num_cond_frames = 1
                cond_img = self.preprocess_image(input_image, _ref_height, _ref_width).unsqueeze(2)
                refine_cond_pixels = cond_img.to(dtype=self.torch_dtype, device=self.device)
            elif has_input_video and need_encode:
                # VC refine: use original condition frames at refine resolution
                base_cond_frames = self.num_ref_video_frames
                refine_num_cond_frames = base_cond_frames * 2 if is_temporal_refine else base_cond_frames
                cond_imgs = self.preprocess_images(ref_frames, _ref_height, _ref_width)
                cond_data = torch.stack(cond_imgs, dim=2).to(dtype=self.torch_dtype, device=self.device)
                if is_temporal_refine:
                    # Temporal upsample condition frames to double count
                    cond_data = F.interpolate(
                        cond_data,
                        size=(refine_num_cond_frames, _ref_height, _ref_width),
                        mode="trilinear",
                        align_corners=True,
                    )
                refine_cond_pixels = cond_data

            # 4. BSA padding alignment (official uses bsa_latent_granularity=4)
            vae_temporal_factor = 4
            bsa_latent_granularity = 4
            num_cond_frames_added = 0
            num_noise_frames_added = 0

            if _enable_bsa:
                num_noise_frames = upsampled.shape[2] - refine_num_cond_frames

                if refine_cond_pixels is not None:
                    # Align num_cond_latents to bsa_latent_granularity
                    raw_cond_latents = (
                        1 + math.ceil((refine_num_cond_frames - 1) / vae_temporal_factor)
                        if refine_num_cond_frames > 0
                        else 0
                    )
                    aligned_cond_latents = (
                        math.ceil(raw_cond_latents / bsa_latent_granularity) * bsa_latent_granularity
                        if raw_cond_latents > 0
                        else 0
                    )
                    num_cond_frames_added = (
                        (1 + (aligned_cond_latents - 1) * vae_temporal_factor - refine_num_cond_frames)
                        if aligned_cond_latents > 0
                        else 0
                    )
                    refine_num_cond_frames += num_cond_frames_added

                # Align num_noise_latents to bsa_latent_granularity (always, not just when cond present)
                raw_noise_latents = math.ceil(num_noise_frames / vae_temporal_factor)
                aligned_noise_latents = math.ceil(raw_noise_latents / bsa_latent_granularity) * bsa_latent_granularity
                num_noise_frames_added = aligned_noise_latents * vae_temporal_factor - num_noise_frames

                # Pad front (condition) and back (noise)
                if num_cond_frames_added > 0:
                    pad_front = upsampled[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
                    upsampled = torch.cat([pad_front, upsampled], dim=2)
                if num_noise_frames_added > 0:
                    pad_back = upsampled[:, :, -1:].repeat(1, 1, num_noise_frames_added, 1, 1)
                    upsampled = torch.cat([upsampled, pad_back], dim=2)

                logger.info(
                    f"BSA padding: cond_frames_added={num_cond_frames_added}, "
                    f"noise_frames_added={num_noise_frames_added}"
                )

            # 5. Prepend condition frames if present
            refine_num_cond_latents = 0
            if refine_cond_pixels is not None:
                # Prepend condition pixels (already at refine resolution)
                upsampled = torch.cat([refine_cond_pixels, upsampled], dim=2)
                refine_num_cond_latents = 1 + (refine_num_cond_frames - 1) // vae_temporal_factor

            # 6. Encode upsampled pixels back to latent space
            hr_latents_handler = auto_async_call(self.vae_stage.process, "encode_video", upsampled, **tiler_kwargs)
            hr_latents = hr_latents_handler()
            hr_latents = hr_latents.to(dtype=self.torch_dtype, device=self.device)

            # 7. Refine denoise (no CFG, with refinement LoRA, optional BSA)
            refine_latents_handler = auto_async_call(
                self.refine_denoise_stage.process,
                hr_latents,
                _ref_steps,
                refine_num_cond_latents,
                prompt_embeds,
                prompt_attention_mask,
                _ref_thresh,
                attn_impl,
                seed,
                progress_bar_cmd,
                _enable_bsa,
            )
            latents = refine_latents_handler()

            # 8. Track BSA padding for post-decode trim
            origin_width, origin_height = _ref_width, _ref_height

        # Decode
        frames_handler = auto_async_call(
            self.vae_stage.process,
            "decode_video",
            latents,
            **tiler_kwargs,
        )
        frames = frames_handler()

        frames = self.tensor2video(frames[0], width=origin_width, height=origin_height)

        # Trim BSA padding: remove cond-aligned frames from front and noise-aligned frames from back.
        # Matches official: output_video[:, num_cond_frames_added: new_frame_size+num_cond_frames_added]
        if num_cond_frames_added > 0 or num_noise_frames_added > 0:
            end = len(frames) - num_noise_frames_added if num_noise_frames_added > 0 else len(frames)
            frames = frames[num_cond_frames_added:end]

        if has_input_video:
            frames = frames[self.num_ref_video_frames :]

        # VFI (Video Frame Interpolation)
        if self.config.enable_vfi and target_fps is not None:
            logger.info(f"Interpolating video from {self.base_fps} fps to {target_fps} fps")
            frames_handler = auto_async_call(self.vfi_stage.process, frames, self.base_fps, target_fps)
            frames = frames_handler()
            logger.info(f"VFI complete, total frames: {len(frames)}")

        if return_latents:
            return frames, latents
        return frames, None

    def __del__(self):
        for attr in ("vae_stage", "denoise_stage", "text_encoding_stage", "vfi_stage", "refine_denoise_stage"):
            if hasattr(self, attr):
                delattr(self, attr)
