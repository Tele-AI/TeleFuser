from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import AttentionConfig, ModelRuntimeConfig, ParallelConfig
from telefuser.models.lingbot_world_fast_dit import LingBotWorldFastDiT
from telefuser.models.t5_tokenizer import HuggingfaceTokenizer
from telefuser.models.wan_video_text_encoder import WanTextEncoder
from telefuser.models.wan_video_vae import WanVideoVAE
from telefuser.utils.logging import logger
from telefuser.utils.model_weight import load_state_dict
from telefuser.utils.profiler import ProfilingContext4Debug
from telefuser.worker.parallel_worker import ParallelWorker

from .control import LingBotWorldFastControlBuilder, LingBotWorldFastControlContext
from .denoising import LingBotWorldFastDenoisingStage
from .session import (
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionStatus,
    resolve_lingbot_frame_count,
)
from .vae_stage import LingBotWorldFastVAEDecodeStage, LingBotWorldFastVAEEncodeStage

if TYPE_CHECKING:
    from .streaming import LingBotWorldFastStreamingRuntime


@dataclass
class LingBotWorldFastPipelineConfig:
    checkpoint_dir: str = ""
    fast_checkpoint_path: str = "lingbot_world_fast"
    vae_config: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(torch_dtype=torch.float32))
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_torch_dtype: torch.dtype = torch.bfloat16
    control_type: str = "cam"
    orig_height: int = 480
    orig_width: int = 832
    max_area: int = 480 * 832
    local_attn_size: int = -1
    sink_size: int = 0
    timestep_indices: tuple[int, ...] = (0, 179, 358, 679)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    vae_parallel_config: ParallelConfig = field(default_factory=lambda: ParallelConfig(device_ids=[0]))
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)


class LingBotWorldFastPipeline(BasePipeline):
    """Pipeline wrapper for LingBot-World-Fast chunked causal generation."""

    clear_memory_after_call = False

    def __init__(self, device: str, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16
        self._next_cache_handle = 0
        self._cache_handle_lock = threading.Lock()
        self._streaming_runtime: LingBotWorldFastStreamingRuntime | None = None
        self._streaming_runtime_lock = threading.Lock()

    def _get_stages(self) -> list:
        return [self.denoise_stage] if hasattr(self, "denoise_stage") else []

    @staticmethod
    def _notify_progress(
        progress_callback: Callable[..., None] | None,
        stage: str,
        **data: object,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, **data)
        except Exception as exc:
            logger.warning(f"LingBot progress callback failed at stage={stage}: {exc}")

    def _runtime_device(self, runtime_config: ModelRuntimeConfig) -> torch.device:
        if runtime_config.device_type is None:
            return torch.device(self.device)
        if runtime_config.device_type == "cuda":
            return torch.device(f"cuda:{runtime_config.device_id}")
        return torch.device(runtime_config.device_type)

    def init(self, config: LingBotWorldFastPipelineConfig) -> None:
        if config.control_type not in {"cam", "act"}:
            raise ValueError(f"Unsupported LingBot control_type: {config.control_type!r}")
        self.config = config
        checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve()
        self._model_info = [{"name": "lingbot_world_fast", "path": str(checkpoint_root)}]
        self.text_device = self._runtime_device(config.text_encoding_config)
        self.vae_device = self._runtime_device(config.vae_config)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.load_state_dict(load_state_dict(str(checkpoint_root / "models_t5_umt5-xxl-enc-bf16.pth")))
        self.text_encoder = self.text_encoder.to(
            device=self.text_device, dtype=config.text_encoding_config.torch_dtype
        ).eval()
        tokenizer_path = checkpoint_root / "google" / "umt5-xxl"
        self.tokenizer = HuggingfaceTokenizer(str(tokenizer_path), 512, "whitespace")

        vae_state_dict, vae_cfg = WanVideoVAE.state_dict_converter().from_official(
            load_state_dict(str(checkpoint_root / "Wan2.1_VAE.pth"))
        )
        self.vae = WanVideoVAE(**vae_cfg)
        self.vae.load_state_dict(vae_state_dict, strict=False)
        self.vae.eval()
        self._validate_vae_parallel_config(config.vae_parallel_config)
        vae_runtime_config = ModelRuntimeConfig(
            device_type=config.vae_config.device_type,
            device_id=config.vae_config.device_id,
            torch_dtype=config.vae_config.torch_dtype,
            parallel_config=config.vae_parallel_config,
        )
        vae_encode_stage = LingBotWorldFastVAEEncodeStage("lingbot_world_fast_vae_encode", self.vae, vae_runtime_config)
        vae_decode_stage = LingBotWorldFastVAEDecodeStage("lingbot_world_fast_vae_decode", self.vae, vae_runtime_config)
        self.vae_encode_worker = ParallelWorker(vae_encode_stage)
        self.vae_decode_worker = ParallelWorker(vae_decode_stage)

        fast_path = checkpoint_root / config.fast_checkpoint_path
        dit_device = "cpu" if config.parallel_config.world_size > 1 else self.device
        self.dit = LingBotWorldFastDiT.from_pretrained(
            str(fast_path),
            torch_dtype=config.dit_torch_dtype,
            control_type=config.control_type,
            config=self._build_dit_config(config),
        ).to(dit_device)
        self.dit.eval().requires_grad_(False)

        pipeline_device = torch.device(self.device)
        dit_runtime_config = ModelRuntimeConfig(
            device_type=pipeline_device.type,
            device_id=pipeline_device.index or 0,
            torch_dtype=config.dit_torch_dtype,
            attention_config=config.attention_config,
            parallel_config=config.parallel_config,
        )
        denoise_stage = LingBotWorldFastDenoisingStage("lingbot_world_fast_denoise", self.dit, dit_runtime_config)
        self.denoise_stage = ParallelWorker(denoise_stage) if config.parallel_config.world_size > 1 else denoise_stage

    @staticmethod
    def _build_dit_config(config: LingBotWorldFastPipelineConfig) -> dict[str, object]:
        return {
            "patch_size": (1, 2, 2),
            "text_len": 512,
            "control_type": config.control_type,
            "local_attn_size": int(config.local_attn_size),
            "sink_size": int(config.sink_size),
        }

    @staticmethod
    def _validate_vae_parallel_config(parallel_config: ParallelConfig) -> None:
        """Restrict the VAE stage to one configured GPU until VAE model parallelism exists."""
        parallel_config.validate()
        if parallel_config.world_size != 1:
            raise ValueError("LingBot VAE stage currently requires exactly one GPU")

    @staticmethod
    def _resolve_self_kv_size(
        *,
        frame_tokens: int,
        latent_frames: int,
        config: LingBotWorldFastPipelineConfig,
    ) -> int:
        if int(config.local_attn_size) > -1:
            return int(frame_tokens) * int(config.local_attn_size)
        return int(frame_tokens) * int(latent_frames)

    @torch.inference_mode()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.text_device)
        mask = mask.to(self.text_device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb.to(self.device)

    @staticmethod
    def _best_output_size(w: int, h: int, expected_area: int, dw: int = 16, dh: int = 16) -> tuple[int, int]:
        """Match the LingBot source by independently flooring both dimensions."""
        aspect_ratio = h / w
        output_height = int(math.sqrt(expected_area * aspect_ratio) // dh * dh)
        output_width = int(math.sqrt(expected_area / aspect_ratio) // dw * dw)
        return output_width, output_height

    def control_context(self, session_config: LingBotWorldFastSessionConfig) -> LingBotWorldFastControlContext:
        """Return control geometry without allocating a generation runtime."""
        self._validate_session_config(session_config)
        width, height = self._best_output_size(
            session_config.image.width,
            session_config.image.height,
            self.config.max_area,
        )
        height, width = self.check_resize_height_width(height, width)
        _, latent_frames = resolve_lingbot_frame_count(
            session_config.frame_num,
            session_config.chunk_size,
            session_config.frame_policy,
        )
        if session_config.intrinsics is None:
            focal = float(max(self.config.orig_width, self.config.orig_height))
            intrinsics = torch.tensor(
                [focal, focal, self.config.orig_width * 0.5, self.config.orig_height * 0.5],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            intrinsics = torch.as_tensor(session_config.intrinsics, dtype=torch.float32, device=self.device)
            if intrinsics.ndim == 2:
                if intrinsics.shape[0] < 1 or intrinsics.shape[1] != 4:
                    raise ValueError(
                        f"Session intrinsics must have shape (4,) or (frames, 4), got {tuple(intrinsics.shape)}"
                    )
                intrinsics = intrinsics[0]
            if intrinsics.shape != (4,):
                raise ValueError(
                    f"Session intrinsics must have shape (4,) or (frames, 4), got {tuple(intrinsics.shape)}"
                )
        return LingBotWorldFastControlContext(
            control_type=self.config.control_type,
            device=self.device,
            control_dtype=torch.float32,
            orig_height=self.config.orig_height,
            orig_width=self.config.orig_width,
            height=height,
            width=width,
            latent_h=height // 8,
            latent_w=width // 8,
            latent_frames=latent_frames,
            chunk_size=session_config.chunk_size,
            intrinsics=intrinsics,
        )

    def _validate_session_config(self, session_config: LingBotWorldFastSessionConfig) -> None:
        if session_config.control_mode != self.config.control_type:
            raise ValueError(
                f"Session control_mode {session_config.control_mode!r} does not match "
                f"pipeline control_type {self.config.control_type!r}"
            )
        if not isinstance(session_config.chunk_size, int) or isinstance(session_config.chunk_size, bool):
            raise ValueError(f"chunk_size must be a positive integer, got {session_config.chunk_size!r}")
        if session_config.chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer, got {session_config.chunk_size}")
        effective_frame_num, _ = resolve_lingbot_frame_count(
            session_config.frame_num,
            session_config.chunk_size,
            session_config.frame_policy,
        )
        session_config.frame_num = effective_frame_num

    def _validate_control(self, session: LingBotWorldFastGenerationSession, control: torch.Tensor) -> None:
        expected_device = torch.device(self.device)
        if control.device.type != expected_device.type or (
            expected_device.index is not None and control.device.index != expected_device.index
        ):
            raise ValueError(f"Control device must be compatible with {self.device}, got {control.device}")
        if control.dtype != torch.float32:
            raise ValueError(f"Control dtype must be torch.float32, got {control.dtype}")
        channels_per_pixel = 7 if self.config.control_type == "act" else 6
        height_factor = max(1, session.height // session.latent_h)
        width_factor = max(1, session.width // session.latent_w)
        expected_channels = channels_per_pixel * height_factor * width_factor
        expected_shape = (1, expected_channels, session.chunk_size, session.latent_h, session.latent_w)
        if tuple(control.shape) != expected_shape:
            raise ValueError(f"Control shape must be {expected_shape}, got {tuple(control.shape)}")

    def _prepare_image_tensor(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        """Normalize and resize the image for the dedicated VAE worker."""
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).sub(0.5).div(0.5)
        tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(height, width), mode="bicubic").squeeze(0)
        return tensor.to("cpu", dtype=self.config.vae_config.torch_dtype)

    def _initialize_vae_session(self, session: LingBotWorldFastGenerationSession) -> None:
        """Register session-owned VAE caches in the dedicated worker."""
        if session.cache_handle is None or session.condition_image is None:
            raise RuntimeError("VAE session initialization requires an image and cache handle")
        self.vae_encode_worker.initialize_cache(
            cache_handle=session.cache_handle, condition_image=session.condition_image, sync=True
        )
        self.vae_decode_worker.initialize_cache(cache_handle=session.cache_handle, sync=True)

    def _release_vae_session_cache(self, session: LingBotWorldFastGenerationSession) -> bool:
        if not hasattr(self, "vae_encode_worker") or not hasattr(self, "vae_decode_worker"):
            return True
        if session.cache_handle is None:
            return True
        try:
            self.vae_encode_worker.release_cache(session.cache_handle, sync=True)
            self.vae_decode_worker.release_cache(session.cache_handle, sync=True)
        except Exception as exc:
            logger.error(f"Failed to release LingBot VAE cache handle {session.cache_handle}: {exc}")
            return False
        return True

    def _release_session_cache(self, session: LingBotWorldFastGenerationSession) -> bool:
        cache_handle = session.cache_handle
        if cache_handle is None:
            return True
        try:
            if isinstance(self.denoise_stage, ParallelWorker):
                self.denoise_stage.release_cache(cache_handle, sync=True)
            else:
                self.denoise_stage.release_cache(cache_handle)
        except Exception as exc:
            logger.error(f"Failed to release LingBot cache handle {cache_handle}: {exc}")
            return False
        session.cache_handle = None
        return True

    def release_session(self, session: LingBotWorldFastGenerationSession) -> None:
        """Idempotently release cache and decoder state owned by a session."""
        with session.lifecycle_lock:
            vae_cache_released = self._release_vae_session_cache(session)
            cache_released = self._release_session_cache(session)
            session.condition_image = None
            if not cache_released or not vae_cache_released:
                session.status = LingBotWorldFastSessionStatus.POISONED
                session.poisoned_reason = f"Failed to release cache handle {session.cache_handle}"
            elif session.status != LingBotWorldFastSessionStatus.POISONED:
                session.status = LingBotWorldFastSessionStatus.RELEASED

    def close(self) -> None:
        """Deterministically close the multi-process denoising worker group."""
        with self._streaming_runtime_lock:
            streaming_runtime = self._streaming_runtime
            self._streaming_runtime = None
        if streaming_runtime is not None:
            streaming_runtime.close()
        denoise_stage = getattr(self, "denoise_stage", None)
        if isinstance(denoise_stage, ParallelWorker):
            denoise_stage.close()
        for vae_worker in (getattr(self, "vae_encode_worker", None), getattr(self, "vae_decode_worker", None)):
            if isinstance(vae_worker, ParallelWorker):
                vae_worker.close()

    def _get_streaming_runtime(self) -> LingBotWorldFastStreamingRuntime:
        """Return the one actor graph owned by this pipeline instance."""
        with self._streaming_runtime_lock:
            if self._streaming_runtime is None:
                from .streaming import LingBotWorldFastStreamingRuntime

                self._streaming_runtime = LingBotWorldFastStreamingRuntime(self)
            return self._streaming_runtime

    @torch.inference_mode()
    def warmup(self, session_config: LingBotWorldFastSessionConfig) -> None:
        """Run and release first and subsequent chunks to initialize runtime state."""
        expected_frame_num = 4 * (2 * session_config.chunk_size - 1) + 1
        if session_config.frame_num != expected_frame_num:
            raise ValueError(
                "LingBot warmup requires exactly two complete chunks: "
                f"expected frame_num={expected_frame_num} for chunk_size={session_config.chunk_size}, "
                f"got {session_config.frame_num}"
            )

        control_context = self.control_context(session_config)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], control_context.chunk_size, axis=0)
        action: dict[str, object] = {"poses": poses}
        if control_context.control_type == "act":
            action["action"] = np.zeros((control_context.chunk_size, 4), dtype=np.float32)

        control_builder = LingBotWorldFastControlBuilder(control_context)
        controls = [control_builder.defer(action) for _ in range(2)]
        self.generate_video(session_config, controls)

    def __del__(self) -> None:
        """Best-effort fallback for callers that do not explicitly close the pipeline."""
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _resolve_control(control: torch.Tensor | Callable[[], torch.Tensor]) -> torch.Tensor:
        resolved = control() if callable(control) else control
        if not isinstance(resolved, torch.Tensor):
            raise TypeError("Deferred control factory must return a torch.Tensor")
        return resolved

    @ProfilingContext4Debug("initialize_session")
    @torch.inference_mode()
    def _create_initialized_session(
        self,
        session_config: LingBotWorldFastSessionConfig,
        progress_callback: Callable[..., None] | None = None,
    ) -> LingBotWorldFastGenerationSession:
        """Allocate model state for the first chunk; callers never invoke this directly."""
        control_context = self.control_context(session_config)
        self._notify_progress(progress_callback, "encoding_prompt", device=str(self.text_device))
        prompt_emb = self.encode_prompt(session_config.prompt)
        self._notify_progress(progress_callback, "prompt_encoded")

        width = control_context.width
        height = control_context.height

        self._notify_progress(progress_callback, "preparing_image", width=width, height=height)
        image_tensor = self._prepare_image_tensor(session_config.image, height, width)

        lat_h = control_context.latent_h
        lat_w = control_context.latent_w
        lat_f = control_context.latent_frames
        patch_area = self.dit.patch_size[1] * self.dit.patch_size[2]
        frame_tokens = (lat_h * lat_w) // patch_area
        kv_size = self._resolve_self_kv_size(
            frame_tokens=frame_tokens,
            latent_frames=lat_f,
            config=self.config,
        )
        max_attention_size = (
            kv_size if session_config.max_attention_size is None else int(session_config.max_attention_size)
        )

        self._notify_progress(
            progress_callback,
            "allocating_runtime",
            latent_frames=lat_f,
            latent_height=lat_h,
            latent_width=lat_w,
            total_chunks=lat_f // session_config.chunk_size,
        )
        noise_generator = torch.Generator(device=self.device)
        noise_generator.manual_seed(int(session_config.seed))
        denoise_generator = torch.Generator(device=self.device)
        denoise_seed = (int(session_config.seed) ^ 0x51A7E5EED) & 0x7FFF_FFFF_FFFF_FFFF
        denoise_generator.manual_seed(denoise_seed)
        with self._cache_handle_lock:
            cache_handle = self._next_cache_handle
            self._next_cache_handle += 1
        session = LingBotWorldFastGenerationSession(
            prompt_emb=prompt_emb,
            config=session_config,
            condition_image=image_tensor,
            latent_h=lat_h,
            latent_w=lat_w,
            latent_f=lat_f,
            height=height,
            width=width,
            frame_tokens=frame_tokens,
            chunk_size=session_config.chunk_size,
            max_attention_size=max_attention_size,
            cache_handle=cache_handle,
        )
        try:
            self._initialize_vae_session(session)
            initialize_cache_kwargs = dict(
                cache_handle=cache_handle,
                batch_size=1,
                kv_size=kv_size,
                max_sequence_length=session_config.max_sequence_length,
                sample_shift=session_config.sample_shift,
                generator_state=denoise_generator.get_state().tolist(),
                noise_generator_state=noise_generator.get_state().tolist(),
                noise_shape=(1, 16, session_config.chunk_size, lat_h, lat_w),
                timestep_indices=getattr(self.config, "timestep_indices", (0, 179, 358, 679)),
            )
            if isinstance(self.denoise_stage, ParallelWorker):
                self.denoise_stage.initialize_cache(**initialize_cache_kwargs, sync=True)
            else:
                self.denoise_stage.initialize_cache(**initialize_cache_kwargs)
        except Exception:
            self._release_session_cache(session)
            self._release_vae_session_cache(session)
            raise
        if session_config.world_kv_binding is not None:
            session.world_kv_binding = session_config.world_kv_binding
            try:
                session.world_kv_binding.on_runtime_created(session, session_config)
                if session.world_kv_cached_latents:
                    logger.info(f"world_kv: fast-forward {len(session.world_kv_cached_latents)} chunks (decode-only)")
            except Exception as exc:
                logger.warning(f"world_kv on_runtime_created failed; falling back to cold run: {exc}")
                session.world_kv_cached_latents = {}
        self._notify_progress(progress_callback, "runtime_created", width=width, height=height, latent_frames=lat_f)
        logger.info(f"LingBot runtime created: {width}x{height}, latent={lat_f}x{lat_h}x{lat_w}")
        session.status = LingBotWorldFastSessionStatus.READY
        return session

    @torch.inference_mode()
    def generate_video(
        self,
        session_config: LingBotWorldFastSessionConfig,
        controls: list[torch.Tensor | Callable[[], torch.Tensor]],
        progress_callback: Callable[..., None] | None = None,
        timeout: float = 300.0,
    ) -> list[Image.Image]:
        """Generate all chunks through the bounded three-stage actor scheduler.

        This is an offline adapter over the pipeline-owned streaming runtime. It
        keeps every tensor edge bounded and returns decoded batches in chunk order.
        """
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        runtime = self._create_initialized_session(session_config, progress_callback)
        streaming_runtime: LingBotWorldFastStreamingRuntime
        session = None
        control_iterator = iter(controls)
        frames: list[Image.Image] = []
        pending_frames: dict[int, list[Image.Image]] = {}
        submitted = 0
        deadline = time.monotonic() + timeout
        try:
            streaming_runtime = self._get_streaming_runtime()
            session = streaming_runtime.create_session(runtime, progress_callback)
            while runtime.current_chunk_index < runtime.chunk_count:
                error = streaming_runtime.error(session)
                if error is not None:
                    raise RuntimeError("LingBot streaming scheduler failed") from error
                while submitted < runtime.chunk_count and streaming_runtime.can_submit_chunk(session):
                    try:
                        control = self._resolve_control(next(control_iterator))
                    except StopIteration as exc:
                        raise ValueError("Control sequence ended before the generation session completed") from exc
                    self._validate_control(runtime, control)
                    if not streaming_runtime.try_submit_chunk(session, submitted, control):
                        raise RuntimeError("LingBot streaming ingress became unavailable after capacity check")
                    submitted += 1

                error = streaming_runtime.error(session)
                if error is not None:
                    raise RuntimeError("LingBot streaming scheduler failed") from error
                for chunk_index, chunk_frames in streaming_runtime.poll_frames(session):
                    if chunk_index in pending_frames:
                        raise RuntimeError(f"Streaming scheduler emitted chunk {chunk_index} twice")
                    pending_frames[chunk_index] = chunk_frames
                while runtime.current_chunk_index in pending_frames:
                    chunk_frames = pending_frames.pop(runtime.current_chunk_index)
                    frames.extend(chunk_frames)
                    runtime.current_chunk_index += 1
                    runtime.emitted_frames += len(chunk_frames)
                if runtime.current_chunk_index >= runtime.chunk_count:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for LingBot streaming scheduler output")
                streaming_runtime.wait_until_idle(session, timeout=min(remaining, 0.05))
        finally:
            if session is None:
                self.release_session(runtime)
            else:
                try:
                    for interval in streaming_runtime.stage_idle_intervals(session, "denoise"):
                        logger.info(
                            "LingBot streaming DiT interval "
                            f"{interval.previous_sequence_id}->{interval.sequence_id}: "
                            f"idle={interval.idle_seconds:.3f}s reason={interval.reason} "
                            f"missing_inputs={interval.missing_inputs}"
                        )
                finally:
                    streaming_runtime.close_session(session)
        return frames
