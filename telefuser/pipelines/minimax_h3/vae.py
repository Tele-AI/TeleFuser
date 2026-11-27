# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 media preparation and modality-specific VAE stages."""

from __future__ import annotations

import contextlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_tp_group
from telefuser.platforms import current_platform

from .canvas import (
    minimax_h3_prepare_keyframe_canvas,
    minimax_h3_stretch_keyframe_canvas,
)
from .material_io import MiniMaxH3MaterialFacts
from .packed_tokens import minimax_h3_patchify_video_latent
from .resolved_plan import (
    MiniMaxH3MaterialPlanItem,
    MiniMaxH3ResolvedPlan,
    minimax_h3_resolve_spatial_shape,
)

MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048


@dataclass(frozen=True)
class MiniMaxH3PreparedCondition:
    material: MiniMaxH3MaterialPlanItem
    kind: str
    image: Image.Image | None = None
    video_frames: torch.Tensor | None = None
    visual_rows: torch.Tensor | None = None
    audio_rows: torch.Tensor | None = None
    latent_t: int = 0
    latent_h: int = 0
    latent_w: int = 0
    ref_audio_t: int = 0
    has_audio: bool = False

    def as_denoising_payload(self) -> dict[str, Any]:
        """Drop source media while preserving denoising inputs for tensor transport."""
        return {
            "material": self.material,
            "kind": self.kind,
            "visual_rows": self.visual_rows,
            "audio_rows": self.audio_rows,
            "latent_t": self.latent_t,
            "latent_h": self.latent_h,
            "latent_w": self.latent_w,
            "ref_audio_t": self.ref_audio_t,
            "has_audio": self.has_audio,
        }


@contextlib.contextmanager
def _scoped_seed(seed: int, device: torch.device):
    if device.type == "npu":
        cpu_state = torch.get_rng_state()
        npu_state = torch.npu.get_rng_state(device)
        try:
            torch.manual_seed(seed)
            torch.npu.manual_seed(seed)
            yield
        finally:
            torch.set_rng_state(cpu_state)
            torch.npu.set_rng_state(npu_state, device)
        return
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield


def _reference_image_size(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or width > 4 * height or height > 4 * width:
        raise ValueError("reference image ratio must be within the inclusive range 1:4 to 4:1")
    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    target_width = max(32, int(round(width * scale / 32)) * 32)
    target_height = max(32, int(round(height * scale / 32)) * 32)
    return target_width, target_height


def _decode_video(
    path: Path,
    *,
    width: int,
    height: int,
    frame_count: int,
    start_time_seconds: float,
) -> torch.Tensor:
    command = ["ffmpeg", "-v", "error"]
    if start_time_seconds > 0:
        command += ["-ss", f"{start_time_seconds:.9g}"]
    command += [
        "-i",
        str(path),
        "-an",
        "-vf",
        f"fps=24,scale={width}:{height}:flags=lanczos,setsar=1",
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    payload = subprocess.run(command, check=True, capture_output=True).stdout
    frame_bytes = width * height * 3
    if not payload or len(payload) % frame_bytes:
        raise ValueError(f"ffmpeg returned invalid reference video bytes for {path}")
    array = np.frombuffer(payload, dtype=np.uint8).reshape(-1, height, width, 3).copy()
    return torch.from_numpy(array)


def _decode_audio(
    path: Path,
    *,
    source_rate: int,
    start_time_seconds: float,
    max_duration_seconds: float,
) -> torch.Tensor:
    command = ["ffmpeg", "-v", "error"]
    if start_time_seconds > 0:
        command += ["-ss", f"{start_time_seconds:.9g}"]
    command += [
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "2",
        "-ar",
        str(source_rate),
        "-t",
        f"{max_duration_seconds:.9g}",
        "-f",
        "f32le",
        "pipe:1",
    ]
    payload = subprocess.run(command, check=True, capture_output=True).stdout
    if not payload or len(payload) % (2 * torch.float32.itemsize):
        raise ValueError(f"ffmpeg returned invalid reference audio bytes for {path}")
    waveform = torch.from_numpy(np.frombuffer(payload, dtype=np.float32).reshape(-1, 2).T.copy())
    if source_rate != 32_000:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, source_rate, 32_000)
    return waveform


class MiniMaxH3VideoVAEStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__("minimax_h3_video_vae", model_runtime_config)
        self.video_vae = module_manager.fetch_module("minimax_h3_video_vae")
        if self.video_vae is None:
            raise ValueError("ModuleManager must contain MiniMax H3 video VAE")
        self.model_names = ["video_vae"]

    def parallel_models(self) -> None:
        parallel_config = self.model_runtime_config.parallel_config
        unsupported = {
            "cfg_degree": parallel_config.cfg_degree,
            "sp_ring_degree": parallel_config.sp_ring_degree,
            "sp_ulysses_degree": parallel_config.sp_ulysses_degree,
            "pp_degree": parallel_config.pp_degree,
            "dp_degree": parallel_config.dp_degree,
        }
        invalid = {name: degree for name, degree in unsupported.items() if degree != 1}
        if invalid:
            raise NotImplementedError(f"MiniMax H3 video VAE does not support these parallel degrees: {invalid}")
        if parallel_config.tp_degree <= 1:
            raise NotImplementedError("MiniMax H3 video VAE parallel tiling requires a multi-rank TP group")
        if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
            raise ValueError("MiniMax H3 video VAE parallel tiling cannot be combined with model CPU offload")
        device_mesh = create_device_mesh_from_config(parallel_config)
        group = get_tp_group(device_mesh)
        if group is None:
            raise RuntimeError("MiniMax H3 video VAE parallel tiling requires a TP process group")
        self.video_vae.enable_parallel_tiling(group)
        self.video_vae.to(self.device)
        self.onload_models_flag = True
        current_platform.empty_cache()

    @staticmethod
    def prepare_media(
        plan: MiniMaxH3ResolvedPlan,
        paths: dict[int, Path],
        facts: dict[int, MiniMaxH3MaterialFacts],
    ) -> list[MiniMaxH3PreparedCondition]:
        prepared: list[MiniMaxH3PreparedCondition] = []
        target_width = int(plan.shape["width"])
        target_height = int(plan.shape["height"])
        for material_index, material in enumerate(plan.materials):
            path = paths[int(material.condition_index)]
            item_facts = facts[int(material.condition_index)]
            if material.material_chain == "image.target_canvas":
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                image = (
                    minimax_h3_stretch_keyframe_canvas(
                        image,
                        target_width=target_width,
                        target_height=target_height,
                    )
                    if material_index == 0
                    else minimax_h3_prepare_keyframe_canvas(
                        image,
                        target_width=target_width,
                        target_height=target_height,
                        allow_upscale=True,
                    )
                )
                prepared.append(MiniMaxH3PreparedCondition(material, "image", image=image))
            elif material.material_chain == "image.reference_preserve":
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = _reference_image_size(image.width, image.height)
                image = image.resize((width, height), Image.Resampling.LANCZOS)
                prepared.append(MiniMaxH3PreparedCondition(material, "image", image=image))
            elif material.material_chain in {"video.reference_preserve", "video_audio.reference_preserve"}:
                if item_facts.width is None or item_facts.height is None:
                    raise ValueError("reference video probe is missing display dimensions")
                resolved = minimax_h3_resolve_spatial_shape(width=item_facts.width, height=item_facts.height)
                frames = _decode_video(
                    path,
                    width=int(resolved["width"]),
                    height=int(resolved["height"]),
                    frame_count=int(plan.shape["frame_count"]),
                    start_time_seconds=material.start_time_seconds,
                )
                prepared.append(
                    MiniMaxH3PreparedCondition(
                        material,
                        "video_audio" if material.material_chain == "video_audio.reference_preserve" else "video",
                        video_frames=frames,
                        has_audio=item_facts.has_audio,
                    )
                )
            elif material.material_chain == "audio":
                prepared.append(MiniMaxH3PreparedCondition(material, "audio", has_audio=True))
            else:
                raise ValueError(f"unsupported material chain {material.material_chain!r}")
        return prepared

    @with_model_offload(["video_vae"])
    @torch.inference_mode()
    def encode_visual(self, conditions: list[MiniMaxH3PreparedCondition]) -> list[MiniMaxH3PreparedCondition]:
        return self._encode_visual(conditions, keep_rows_on_device=False)

    @with_model_offload(["video_vae"])
    @torch.inference_mode()
    def encode_visual_for_denoising(self, conditions: list[MiniMaxH3PreparedCondition]) -> list[dict[str, Any]]:
        """Encode channel-friendly condition rows without staging them through CPU."""
        encoded = self._encode_visual(conditions, keep_rows_on_device=True)
        return [condition.as_denoising_payload() for condition in encoded]

    def _encode_visual(
        self,
        conditions: list[MiniMaxH3PreparedCondition],
        *,
        keep_rows_on_device: bool,
    ) -> list[MiniMaxH3PreparedCondition]:
        output: list[MiniMaxH3PreparedCondition] = []
        parameter = next(self.video_vae.parameters())
        for condition in conditions:
            if condition.image is None and condition.video_frames is None:
                output.append(condition)
                continue
            with _scoped_seed(42, parameter.device):
                if condition.image is not None:
                    latent = self.video_vae.encode_images(condition.image, use_fp16_latent=True)[0]
                else:
                    assert condition.video_frames is not None
                    frames = condition.video_frames.cpu().numpy()
                    latent = self.video_vae.encode_videos(frames, use_fp16_latent=True)[0]
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            mean = latent.new_tensor(self.video_vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = latent.new_tensor(self.video_vae.config.latents_std).view(1, -1, 1, 1, 1)
            normalized = latent.float().sub(mean).div(std)
            rows = minimax_h3_patchify_video_latent(normalized, patch_size=(1, 2, 2))
            if not keep_rows_on_device:
                rows = rows.cpu()
            output.append(
                MiniMaxH3PreparedCondition(
                    **{
                        **condition.__dict__,
                        "visual_rows": rows,
                        "latent_t": int(normalized.shape[2]),
                        "latent_h": int(normalized.shape[3]),
                        "latent_w": int(normalized.shape[4]),
                    }
                )
            )
        return output

    @with_model_offload(["video_vae"])
    @torch.inference_mode()
    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        device = next(self.video_vae.parameters()).device
        use_fp16_autocast = device.type in {"cuda", "npu"}
        if use_fp16_autocast:
            self.video_vae.prepare_decoder_autocast_weights(torch.float16)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16_autocast):
            frames = self.video_vae.decode_normalized(latent.to(device))
        # Keep the large decoded tensor on the producing device for CUDA IPC;
        # on NPU (host-queue transport), move to CPU immediately so decoder
        # output buffers and cached workspace segments are released before the
        # next request onloads the text encoder.
        result = frames.permute(0, 2, 3, 4, 1).float().contiguous()
        if device.type == "npu":
            result = result.cpu()
            del frames
            current_platform.empty_cache()
        return result


class MiniMaxH3AudioVAEStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__("minimax_h3_audio_vae", model_runtime_config)
        self.audio_vae = module_manager.fetch_module("minimax_h3_audio_vae")
        if self.audio_vae is None:
            raise ValueError("ModuleManager must contain MiniMax H3 audio VAE")
        self.model_names = ["audio_vae"]

    @with_model_offload(["audio_vae"])
    @torch.inference_mode()
    def encode_audio(
        self,
        conditions: list[MiniMaxH3PreparedCondition],
        paths: dict[int, Path],
        facts: dict[int, MiniMaxH3MaterialFacts],
        duration_seconds: float,
    ) -> list[MiniMaxH3PreparedCondition]:
        output: list[MiniMaxH3PreparedCondition] = []
        device = next(self.audio_vae.parameters()).device
        for condition in conditions:
            if not condition.has_audio:
                output.append(condition)
                continue
            index = int(condition.material.condition_index)
            is_video = condition.material.material_chain in {
                "video.reference_preserve",
                "video_audio.reference_preserve",
            }
            source_rate = 44_100 if is_video else facts[index].sample_rate
            if source_rate is None:
                source_rate = 44_100
            waveform = _decode_audio(
                paths[index],
                source_rate=source_rate,
                start_time_seconds=condition.material.start_time_seconds,
                max_duration_seconds=duration_seconds,
            )
            latent = self.audio_vae.encode_mean(waveform.unsqueeze(1).to(device), 32_000)
            rows = latent.permute(0, 2, 1).reshape(-1, int(latent.shape[1])).float().cpu()
            output.append(
                MiniMaxH3PreparedCondition(
                    **{
                        **condition.__dict__,
                        "audio_rows": rows,
                        "ref_audio_t": int(latent.shape[-1]),
                    }
                )
            )
        return output

    @with_model_offload(["audio_vae"])
    @torch.inference_mode()
    def decode_audio(self, latent: torch.Tensor) -> torch.Tensor:
        device = next(self.audio_vae.parameters()).device
        return self.audio_vae.decode_normalized(latent.to(device)).float().cpu()


__all__ = [
    "MiniMaxH3PreparedCondition",
    "MiniMaxH3AudioVAEStage",
    "MiniMaxH3VideoVAEStage",
]
