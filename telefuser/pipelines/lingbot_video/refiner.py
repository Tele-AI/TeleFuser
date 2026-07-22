"""Low-noise refiner schedule and latent handoff primitives."""
# Training-aligned frame selection below is adapted from the Apache-2.0
# licensed upstream LingBot-Video ``utils.py`` implementation.

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from telefuser.platforms import current_platform

from .denoising import LingBotVideoDenoisingStage, reinject_ti2v_condition
from .vae import (
    LingBotVideoVAEDecodeStage,
    LingBotVideoVAEEncodeStage,
    first_frame_condition_mask,
)


def compute_refiner_sigmas(
    *,
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    t_thresh: float | None,
    tail_steps: int = 0,
) -> np.ndarray | None:
    """Match the official refiner's low-noise sigma-tail construction."""
    if t_thresh is None:
        return None
    if not 0 < t_thresh <= 1 or num_inference_steps < 1 or tail_steps < 0:
        raise ValueError("invalid refiner sigma configuration")
    base = np.linspace(sigma_max, sigma_min, num_inference_steps + 1)[:-1]
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    sigmas = shifted[shifted <= t_thresh + 1e-6]
    if not len(sigmas) or abs(float(sigmas[0]) - t_thresh) > 1e-6:
        sigmas = np.concatenate([[t_thresh], sigmas])
    if tail_steps:
        tail = np.linspace(float(sigmas[-1]), min(sigma_min, float(sigmas[-1])), tail_steps + 2)[1:-1]
        sigmas = np.concatenate([sigmas, tail])
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas < 0) or np.any(sigmas > 1) or np.any(np.diff(sigmas) >= 0):
        raise ValueError("refiner sigma schedule must be finite, descending, and within [0,1]")
    return sigmas.astype(np.float32)


def prepare_refiner_latent(x_up: torch.Tensor, noise: torch.Tensor, t_thresh: float | torch.Tensor) -> torch.Tensor:
    """Mix encoded low-resolution video with noise at the refiner threshold."""
    if x_up.shape != noise.shape:
        raise ValueError("x_up and noise must have matching shapes")
    threshold = torch.as_tensor(t_thresh, device=x_up.device, dtype=x_up.dtype)
    while threshold.ndim < x_up.ndim:
        threshold = threshold.unsqueeze(-1)
    return (1.0 - threshold) * x_up + threshold * noise


def compute_training_frame_budget(
    num_source_frames: int, source_fps: float, *, sample_fps: int = 24, vae_tc: int = 4
) -> tuple[int, float, int]:
    """Return the upstream training-aligned frame and temporal-latent budgets."""
    if num_source_frames <= 0:
        return 1, 0.0, 1
    if source_fps <= 0 or sample_fps <= 0 or vae_tc <= 0:
        raise ValueError("source_fps, sample_fps, and vae_tc must be positive")
    raw_frames = int(num_source_frames / source_fps * sample_fps) if source_fps > sample_fps else num_source_frames
    sample_frames = max(((raw_frames - 1) // vae_tc) * vae_tc + 1, 1)
    vae_fps = sample_frames / num_source_frames * source_fps
    return int(sample_frames), float(vae_fps), (sample_frames - 1) // vae_tc + 1


def compute_training_aligned_indices(num_source_frames: int, sample_frames: int) -> torch.Tensor:
    """Select training-aligned indices and pad short inputs with their last frame."""
    if sample_frames <= 0:
        return torch.empty(0, dtype=torch.long)
    if num_source_frames <= 0:
        return torch.zeros(sample_frames, dtype=torch.long)
    if num_source_frames >= sample_frames:
        return torch.from_numpy(np.linspace(0, num_source_frames - 1, sample_frames, dtype=int)).long()
    return torch.cat(
        (torch.arange(num_source_frames), torch.full((sample_frames - num_source_frames,), num_source_frames - 1))
    )


def prepare_refiner_video(
    video: torch.Tensor,
    *,
    source_fps: float,
    height: int,
    width: int,
    sample_fps: int = 24,
    vae_tc: int = 4,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, dict[str, int | float | bool | None]]:
    """Sample and bicubically resize an in-memory RGB video for the refiner VAE."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B,3,F,H,W]")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when set")
    total_frames = video.shape[2]
    sample_frames, vae_fps, temporal_latents = compute_training_frame_budget(
        total_frames, source_fps, sample_fps=sample_fps, vae_tc=vae_tc
    )
    uncapped_frames = sample_frames
    truncated = max_frames is not None and sample_frames > max_frames
    if truncated:
        sample_frames = int(max_frames)
        vae_fps = sample_frames / total_frames * source_fps
        temporal_latents = (sample_frames - 1) // vae_tc + 1
    indices = compute_training_aligned_indices(total_frames, sample_frames).to(video.device)
    sampled = video.index_select(2, indices)
    batch, channels, frames, input_height, input_width = sampled.shape
    flat = sampled.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, input_height, input_width)
    resized = F.interpolate(flat, size=(height, width), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    prepared = resized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
    return prepared, {
        "src_fps": float(source_fps),
        "sample_frame": int(sample_frames),
        "sample_frame_uncapped": int(uncapped_frames),
        "max_frames": max_frames,
        "truncated_by_max_frames": bool(truncated),
        "vae_fps": float(vae_fps),
        "t_vae": int(temporal_latents),
        "num_source_frames": int(total_frames),
        "align_to_training": True,
    }


def load_refiner_video_file(
    path: str | Path,
    *,
    height: int,
    width: int,
    sample_fps: int = 24,
    vae_tc: int = 4,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, dict[str, int | float | bool | None]]:
    """Decode an MP4 with PyAV and apply the upstream refiner sampling contract."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("MP4 refiner handoff requires the optional PyAV dependency") from exc

    container = av.open(str(path))
    try:
        stream = next(iter(container.streams.video), None)
        if stream is None:
            raise ValueError(f"video has no video stream: {path}")
        source_fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    finally:
        container.close()
    if not frames:
        raise ValueError(f"video has no decoded frames: {path}")
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).float().div(255.0)
    return prepare_refiner_video(
        video,
        source_fps=source_fps,
        height=height,
        width=width,
        sample_fps=sample_fps,
        vae_tc=vae_tc,
        max_frames=max_frames,
    )


def load_refiner_first_frame(
    path: str | Path,
    *,
    target_height: int,
    target_width: int,
    geometry_height: int,
    geometry_width: int,
) -> torch.Tensor:
    """Load the TI2V refiner frame zero with the upstream low-resolution geometry."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image_width, image_height = image.size
    geometry_aspect = float(geometry_width) / float(geometry_height)
    image_aspect = float(image_width) / float(image_height)
    if image_aspect > geometry_aspect:
        crop_height = image_height
        crop_width = max(1, int(round(crop_height * geometry_aspect)))
        left, top = int(round((image_width - crop_width) / 2.0)), 0
    else:
        crop_width = image_width
        crop_height = max(1, int(round(crop_width / geometry_aspect)))
        left, top = 0, int(round((image_height - crop_height) / 2.0))
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    resized = crop.resize((target_width, target_height), resample=Image.BICUBIC)
    pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return pixels.permute(1, 0, 2, 3).unsqueeze(0).contiguous()


class LingBotVideoRefinerStage:
    """In-memory low-noise refiner runtime independent of the base DiT stage."""

    def __init__(
        self,
        *,
        denoising_stage: LingBotVideoDenoisingStage,
        vae_encode_stage: LingBotVideoVAEEncodeStage,
        vae_decode_stage: LingBotVideoVAEDecodeStage,
        scheduler: object,
    ) -> None:
        self.denoising_stage = denoising_stage
        self.vae_encode_stage = vae_encode_stage
        self.vae_decode_stage = vae_decode_stage
        self.scheduler = scheduler

    @staticmethod
    def _resolve_stage_result(value: object) -> torch.Tensor:
        """Resolve deferred ParallelWorker outputs while accepting direct stage tensors."""
        result = value() if callable(value) else value
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"LingBot refiner stage returned {type(result).__name__}, expected a tensor")
        return result

    def close(self) -> None:
        """Close an owned distributed denoising worker after the refiner stage completes."""
        close = getattr(self.denoising_stage, "close", None)
        if callable(close):
            close()

    def _offload_vae_between_stages(self) -> None:
        """Release VAE weights and allocator blocks before high-resolution DiT execution."""
        for stage in (self.vae_encode_stage, self.vae_decode_stage):
            offload_models = getattr(stage, "offload_models", None)
            if callable(offload_models):
                offload_models()
        current_platform.empty_cache()

    def refine(
        self,
        lowres_video: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        positive_attention_mask: torch.Tensor | None,
        negative_attention_mask: torch.Tensor | None,
        *,
        num_inference_steps: int,
        guidance_scale: float,
        shift: float,
        t_thresh: float,
        tail_steps: int = 2,
        clean_first_frame: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        before_decode: Callable[[], None] | None = None,
    ) -> torch.Tensor:
        """Refine an in-memory RGB video using official low-noise initialization."""
        x_up = self.vae_encode_stage.encode(lowres_video, generator=generator)
        condition = None
        if clean_first_frame is not None:
            if clean_first_frame.ndim == 4:
                clean_first_frame = clean_first_frame.unsqueeze(2)
            clean_latent = self.vae_encode_stage.encode(clean_first_frame, generator=generator)
            condition = clean_latent[:, :, :1].contiguous()
            x_up = x_up.clone()
            x_up[:, :, :1] = condition.to(x_up.dtype)
        # A 1088p VAE encode may leave several GiB of allocator blocks on
        # rank zero.  The SP refiner is a separate stage, so it must not share
        # that residency with the long-sequence MoE denoising worker.
        self._offload_vae_between_stages()
        if noise is None:
            noise = torch.randn(x_up.shape, dtype=x_up.dtype, device=x_up.device, generator=generator)
        elif noise.shape != x_up.shape:
            raise ValueError("injected refiner noise must match the encoded low-resolution latent")
        else:
            noise = noise.to(device=x_up.device, dtype=x_up.dtype)
        latent = prepare_refiner_latent(x_up, noise, t_thresh)
        sigmas = compute_refiner_sigmas(
            sigma_max=float(self.scheduler.sigma_max),
            sigma_min=float(self.scheduler.sigma_min),
            num_inference_steps=num_inference_steps,
            shift=shift,
            t_thresh=t_thresh,
            tail_steps=tail_steps,
        )
        if sigmas is None:
            self.scheduler.set_timesteps(num_inference_steps, device=latent.device, shift=shift)
        else:
            self.scheduler.set_timesteps(len(sigmas), device=latent.device, sigmas=sigmas, shift=1.0)
        condition_mask = (
            first_frame_condition_mask(latent.shape[2], device=latent.device) if condition is not None else None
        )
        for timestep in self.scheduler.timesteps:
            if condition is not None:
                latent = reinject_ti2v_condition(latent, condition, condition_mask)
            prediction = self._resolve_stage_result(
                self.denoising_stage.predict_noise_with_cfg(
                    latent,
                    timestep.expand(latent.shape[0]),
                    positive_prompt_embeds,
                    negative_prompt_embeds,
                    positive_attention_mask,
                    negative_attention_mask,
                    guidance_scale,
                )
            )
            latent = self.scheduler.step(prediction, timestep, latent)
        if condition is not None:
            latent = reinject_ti2v_condition(latent, condition, condition_mask)
        # The native SP worker owns the refiner weights.  Release it before
        # loading the VAE decoder, otherwise the two high-memory stages can
        # overlap on rank zero.
        self.close()
        if before_decode is not None:
            before_decode()
        return self._resolve_stage_result(self.vae_decode_stage.decode(latent))
