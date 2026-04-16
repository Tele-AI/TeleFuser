"""LiveAct Pipeline: Audio-conditioned Image-to-Video Generation.

Generates talking head videos from an input image and audio using
a diffusion transformer with audio cross-attention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as transforms
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.utils.func import auto_async_call
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker

from .audio_encoding import AudioEncodingStage
from .denoising import KVCacheConfig, LiveActDenoisingStage


def center_rescale_crop_keep_ratio(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Center crop image while keeping aspect ratio."""
    if isinstance(target_size, int):
        target_h = target_w = target_size
    else:
        target_h, target_w = target_size

    w, h = image.size
    scale = max(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h
    image = image.crop((left, top, right, bottom))

    return image


@dataclass
class LiveActPipelineConfig:
    """Configuration for LiveAct pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    clip_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    audio_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)

    # Audio parameters
    audio_window: int = 5
    stream_audio: bool = True  # True = re-encode per iteration (precise), False = pre-encode (faster)

    # Generation parameters
    block_sizes: tuple[int, int] = (6, 8)

    # KV cache parameters
    fp8_kv_cache: bool = False
    offload_cache: bool = False
    mean_memory: bool = False

    # Parallel processing
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False

    # Feature flags
    enable_metrics: bool = False


# Model architecture constants (hardcoded, not configurable)
VAE_STRIDE = (4, 8, 8)  # temporal, height, width
PATCH_SIZE = (1, 2, 2)  # temporal, height, width
LATENT_CHANNELS = 16


class LiveActPipeline(BasePipeline):
    """LiveAct: Audio-conditioned Image-to-Video Generation Pipeline.

    Generates talking head videos from an input image and audio.
    Uses a Wan-style diffusion transformer with audio cross-attention.

    Features:
    - Audio-conditioned video generation
    - Streaming generation with KV cache
    - Optional audio CFG for enhanced lip-sync
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16

    def preprocess_image(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        """Preprocess image with center crop to maintain aspect ratio."""

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_rescale_crop_keep_ratio(pil_image, (height, width))),
                transforms.ToTensor(),
                transforms.Resize((height, width)),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        return transform(image).unsqueeze(0)

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        return [
            self.audio_encoding_stage,
            self.denoise_stage,
            self.text_encoding_stage,
            self.clip_encoding_stage,
            self.vae_stage,
        ]

    def init(self, module_manager: ModuleManager, config: LiveActPipelineConfig) -> None:
        """Initialize pipeline stages."""
        self._model_info = module_manager.get_model_info()
        self.config = config

        from telefuser.pipelines.wan_video.clip_encoding import ClipEncodingStage
        from telefuser.pipelines.wan_video.text_encoding import TextEncodingStage
        from telefuser.pipelines.wan_video.vae import VAEStage

        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        self.clip_encoding_stage = ClipEncodingStage("clip_encoding", module_manager, config.clip_config)
        self.text_encoding_stage = TextEncodingStage("text_encoding", module_manager, config.text_encoding_config)
        self.audio_encoding_stage = AudioEncodingStage(
            "audio_encoding",
            module_manager,
            config.audio_config,
            audio_window=config.audio_window,
        )

        self.denoise_stage = LiveActDenoisingStage(
            "denoise",
            module_manager,
            config.dit_config,
            kv_cache_config=KVCacheConfig(
                fp8_kv_cache=config.fp8_kv_cache,
                offload_cache=config.offload_cache,
            ),
            mean_memory=config.mean_memory,
        )

        # Wrap with ParallelWorker for distributed inference
        if config.enable_denoising_parallel:
            self.denoise_stage = ParallelWorker(self.denoise_stage)
        if config.enable_vae_parallel:
            self.vae_stage = ParallelWorker(self.vae_stage)

        if config.enable_metrics:
            self.enable_metrics()

    def prepare_vae_latent(self) -> None:
        """Placeholder - VAE latent preparation is now inline in __call__."""
        pass

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        input_image: Image.Image,
        audio_path: str | None = None,
        audio: torch.Tensor | None = None,
        audio_sr: int = 16000,
        seed: int | None = None,
        height: int = 480,
        width: int = 832,
        fps: int = 24,
        num_inference_steps: int = 3,
        audio_cfg: float = 1.0,
    ) -> List[Image.Image]:
        """Generate talking head video from image and audio.

        Args:
            prompt: Text prompt for video generation
            input_image: Input image for I2V
            audio_path: Path to audio file
            audio: Audio tensor (alternative to audio_path)
            audio_sr: Sample rate of audio tensor
            seed: Random seed
            height: Video height
            width: Video width
            fps: Target fps
            num_inference_steps: Number of denoising steps
            audio_cfg: Audio CFG scale

        Returns:
            List of PIL Images representing the generated video frames
        """
        logger.info(f"Generating LiveAct video: {width}x{height}, fps={fps}")
        height, width = self.check_resize_height_width(height, width)

        input_image_tensor = self.preprocess_image(input_image, height, width)

        prompt_emb = self.text_encoding_stage.process([prompt])[0]

        clip_fea = self.clip_encoding_stage.process(input_image_tensor)

        stream_audio = self.config.stream_audio
        _, audio_duration = self.audio_encoding_stage.process(
            audio_path=audio_path,
            audio=audio,
            sr=audio_sr,
            fps=fps,
            stream_audio=stream_audio,
        )

        if stream_audio:
            logger.info("Using stream_audio mode: re-encode audio per iteration")
        else:
            logger.info("Using pre-encoded mode: slice embedding per iteration")

        blksz_lst = self.config.block_sizes
        frame_num_per_iter = (sum(blksz_lst) - 1) * VAE_STRIDE[0] + 1
        iter_total_num = int(audio_duration / (VAE_STRIDE[0] * blksz_lst[-1] / fps)) + 1

        logger.info(f"Audio duration: {audio_duration:.2f}s, iterations: {iter_total_num}")

        # Encode image using auto_async_call pattern for ParallelWorker compatibility
        y_handler = auto_async_call(
            self.vae_stage.process,
            "encode_image",
            input_image_tensor,
            None,
            frame_num_per_iter,
            tiled=False,
            concat_mask=True,
        )
        y = y_handler()

        # Calculate tokens per frame for DiT
        tokens_per_frame = (height // (PATCH_SIZE[1] * VAE_STRIDE[1])) * (width // (PATCH_SIZE[2] * VAE_STRIDE[2]))

        gen_video_list = []
        pre_latent = None
        total_start_time = None

        if seed is not None:
            torch.manual_seed(seed)

        for iteration in range(iter_total_num):
            iter_start_time = time.time()

            logger.info(f"Generating segment {iteration + 1}/{iter_total_num}")

            audio_start_idx, audio_end_idx = 0, frame_num_per_iter
            if (iteration - 1) * blksz_lst[-1] * VAE_STRIDE[0] > 0:
                audio_start_idx += (iteration - 1) * blksz_lst[-1] * VAE_STRIDE[0]
                audio_end_idx += (iteration - 1) * blksz_lst[-1] * VAE_STRIDE[0]

            if stream_audio:
                audio_emb_for_dit = self.audio_encoding_stage.process_stream_audio_segment(
                    audio_start_idx=audio_start_idx,
                    audio_end_idx=audio_end_idx,
                    fps=fps,
                    frame_num=frame_num_per_iter,
                )
            else:
                audio_emb_for_dit = self.audio_encoding_stage.process_pre_encoded_segment(
                    audio_start_idx=audio_start_idx,
                    audio_end_idx=audio_end_idx,
                )

            f = iteration if iteration <= 1 else 1
            latent_shape = (LATENT_CHANNELS, blksz_lst[f], height // VAE_STRIDE[1], width // VAE_STRIDE[2])
            latent = torch.randn(latent_shape, dtype=self.torch_dtype, device=self.device)

            latent_handler = auto_async_call(
                self.denoise_stage.process,
                latent=latent,
                context=prompt_emb,
                clip_fea=clip_fea,
                audio_embedding=audio_emb_for_dit,
                y=y,
                tokens_per_frame=tokens_per_frame,
                audio_cfg=audio_cfg,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
            latent = latent_handler()

            if f == 0:
                vae_decode_handler = auto_async_call(self.vae_stage.process, "decode_video", latent)
                videos = vae_decode_handler()[0]
            else:
                latent_to_decode = torch.concat([pre_latent[:, -3:], latent], dim=1)
                vae_decode_handler = auto_async_call(self.vae_stage.process, "decode_video", latent_to_decode)
                videos = vae_decode_handler()[0, :, 9:]

            pre_latent = latent
            gen_video_list.append(videos.cpu())

            torch.cuda.synchronize()
            iter_end_time = time.time()

            generated_frames = blksz_lst[f] * VAE_STRIDE[0]
            generated_duration_ms = generated_frames / fps * 1000
            iter_cost_ms = (iter_end_time - iter_start_time) * 1000

            logger.info(
                f"Done Block {iteration}: duration {generated_duration_ms:.0f}ms video cost {iter_cost_ms:.2f}ms"
            )

            if total_start_time is None:
                total_start_time = iter_start_time

        torch.cuda.synchronize()
        total_end_time = time.time()
        total_cost_s = total_end_time - total_start_time if total_start_time else 0

        total_frames = blksz_lst[0] * VAE_STRIDE[0] + (iter_total_num - 1) * blksz_lst[1] * VAE_STRIDE[0]
        total_duration_s = total_frames / fps

        logger.info(
            f"Total: generated {total_duration_s:.2f}s video, cost {total_cost_s:.2f}s ({total_cost_s * 1000:.0f}ms)"
        )

        rtf = total_cost_s / total_duration_s if total_duration_s > 0 else 0
        logger.info(f"Real-time factor: {rtf:.2f}x ({'real-time' if rtf < 1.0 else 'slower than real-time'})")

        videos = (torch.concat(gen_video_list, dim=1).permute(1, 2, 3, 0) + 1.0) / 2
        videos_np = (videos.float().cpu().numpy() * 255).clip(0, 255).astype("uint8")

        return [Image.fromarray(frame) for frame in videos_np]
