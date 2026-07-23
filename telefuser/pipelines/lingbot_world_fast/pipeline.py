from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

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

from .async_vae import AsyncVAEChunkHandle, AsyncVAEDecodeTask, AsyncVAEManager
from .control import LingBotWorldFastControlBuilder, LingBotWorldFastControlContext
from .denoising import LingBotWorldFastDenoisingStage
from .session import (
    LingBotWorldFastChunkRequest,
    LingBotWorldFastChunkResult,
    LingBotWorldFastConditionPrefetch,
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionStatus,
    resolve_lingbot_frame_count,
)


@dataclass
class LingBotWorldFastPipelineConfig:
    checkpoint_dir: str = ""
    fast_checkpoint_path: str = "lingbot_world_fast"
    vae_config: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(torch_dtype=torch.float32))
    async_vae_config: ModelRuntimeConfig | None = None
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
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    enable_async_vae: bool = False
    vae_queue_size: int = 1
    enable_condition_prefetch: bool = False


class LingBotWorldFastPipeline(BasePipeline):
    """Pipeline wrapper for LingBot-World-Fast chunked causal generation."""

    clear_memory_after_call = False

    def __init__(self, device: str, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16
        self._async_vae_manager: AsyncVAEManager | None = None
        self._async_vae_runtimes: dict[int, LingBotWorldFastGenerationSession] = {}
        self._next_async_vae_generation_id = 1
        self.async_vae: WanVideoVAE | None = None
        self.async_vae_device: torch.device | None = None

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

    @property
    def async_vae_enabled(self) -> bool:
        return bool(getattr(getattr(self, "config", None), "enable_async_vae", False))

    @property
    def condition_prefetch_enabled(self) -> bool:
        return bool(getattr(getattr(self, "config", None), "enable_condition_prefetch", False))

    def _ensure_async_vae_manager(self) -> AsyncVAEManager:
        if not self.async_vae_enabled:
            raise RuntimeError("LingBot async VAE is disabled")
        manager = getattr(self, "_async_vae_manager", None)
        if manager is None:
            queue_size = int(getattr(self.config, "vae_queue_size", 1))
            manager = AsyncVAEManager(self, queue_size=queue_size)
            self._async_vae_manager = manager
        return manager

    def _assign_async_vae_generation(self, session: LingBotWorldFastGenerationSession) -> int:
        generation_id = session.async_vae_generation_id
        if generation_id is None:
            generation_id = self._next_async_vae_generation_id
            self._next_async_vae_generation_id += 1
            session.async_vae_generation_id = generation_id
            self._async_vae_runtimes[generation_id] = session
        return generation_id

    def _async_vae_runtime(self, generation_id: int) -> LingBotWorldFastGenerationSession:
        try:
            return self._async_vae_runtimes[generation_id]
        except KeyError as exc:
            raise RuntimeError(f"Unknown LingBot async VAE generation_id={generation_id}") from exc

    def _cancel_async_vae_generation(self, session: LingBotWorldFastGenerationSession) -> None:
        generation_id = session.async_vae_generation_id
        if generation_id is None:
            return
        manager = getattr(self, "_async_vae_manager", None)
        if manager is not None:
            manager.cancel_generation(generation_id)
        self._async_vae_runtimes.pop(generation_id, None)
        session.async_vae_generation_id = None

    def _record_latent_ready_event(self, latent: torch.Tensor) -> torch.cuda.Event | None:
        if not latent.is_cuda or not torch.cuda.is_available():
            return None
        with torch.cuda.device(latent.device):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(latent.device))
            return event

    @staticmethod
    def _build_vae(
        vae_state_dict: dict[str, torch.Tensor],
        vae_cfg: dict[str, object],
        *,
        device: torch.device,
        torch_dtype: torch.dtype,
    ) -> WanVideoVAE:
        vae = WanVideoVAE(**vae_cfg)
        vae.load_state_dict(vae_state_dict, strict=False)
        return vae.to(device=device, dtype=torch_dtype).eval()

    @property
    def has_separate_async_vae(self) -> bool:
        return getattr(self, "async_vae", None) is not None and self.async_vae is not self.vae

    def init(self, config: LingBotWorldFastPipelineConfig) -> None:
        if config.control_type not in {"cam", "act"}:
            raise ValueError(f"Unsupported LingBot control_type: {config.control_type!r}")
        self.config = config
        checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve()
        self._model_info = [{"name": "lingbot_world_fast", "path": str(checkpoint_root)}]
        self.text_device = self._runtime_device(config.text_encoding_config)
        self.vae_device = self._runtime_device(config.vae_config)
        async_vae_config = config.async_vae_config
        self.async_vae_device = (
            self._runtime_device(async_vae_config) if async_vae_config is not None else self.vae_device
        )

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
        self.vae = self._build_vae(
            vae_state_dict,
            vae_cfg,
            device=self.vae_device,
            torch_dtype=config.vae_config.torch_dtype,
        )
        if async_vae_config is None:
            self.async_vae = self.vae
        else:
            self.async_vae = self._build_vae(
                vae_state_dict,
                vae_cfg,
                device=self.async_vae_device,
                torch_dtype=async_vae_config.torch_dtype,
            )
            logger.info(
                "LingBot async VAE decoder loaded separately: "
                f"encode_device={self.vae_device} decode_device={self.async_vae_device}"
            )
        del vae_state_dict

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
        self._next_cache_handle = 0
        self._async_vae_manager = None
        self._async_vae_runtimes = {}
        self._next_async_vae_generation_id = 1

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

    @torch.inference_mode()
    def decode_video_cached(
        self,
        session: LingBotWorldFastGenerationSession,
        latents: torch.Tensor,
        is_first_clip: bool,
        is_last_clip: bool,
    ) -> torch.Tensor:
        return self.vae.cached_decode_withflag(
            latents,
            device=self.vae_device,
            is_first_clip=is_first_clip,
            is_last_clip=is_last_clip,
            decode_state=session.decoder_state,
        )

    @torch.inference_mode()
    def decode_video_cached_async(
        self,
        session: LingBotWorldFastGenerationSession,
        latents: torch.Tensor,
        is_first_clip: bool,
        is_last_clip: bool,
    ) -> torch.Tensor:
        async_vae = self.async_vae or self.vae
        async_vae_device = self.async_vae_device or self.vae_device
        return async_vae.cached_decode_withflag(
            latents,
            device=async_vae_device,
            is_first_clip=is_first_clip,
            is_last_clip=is_last_clip,
            decode_state=session.decoder_state,
        )

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
        """Normalize before applying the source tensor-space bicubic resize."""
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).sub(0.5).div(0.5)
        tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(height, width), mode="bicubic").squeeze(0)
        return tensor.to(self.vae_device, dtype=self.config.vae_config.torch_dtype)

    def _encode_condition_chunk(
        self,
        session: LingBotWorldFastGenerationSession,
        *,
        chunk_index: int | None = None,
    ) -> torch.Tensor:
        """Encode only the current image-conditioning chunk with a persistent VAE cache."""
        chunk_index = session.current_chunk_index if chunk_index is None else int(chunk_index)
        is_first_chunk = chunk_index == 0
        is_last_chunk = chunk_index == session.chunk_count - 1
        pixel_frames = 1 + 4 * (session.chunk_size - 1) if is_first_chunk else 4 * session.chunk_size
        video = torch.zeros(
            (3, pixel_frames, session.height, session.width),
            device=self.vae_device,
            dtype=self.config.vae_config.torch_dtype,
        )
        if is_first_chunk:
            if session.condition_image is None:
                raise RuntimeError("The first condition chunk requires the session image tensor")
            video[:, 0] = session.condition_image

        latent = self.vae.cached_encode_withflag(
            video,
            device=self.vae_device,
            is_first_clip=is_first_chunk,
            is_last_clip=is_last_chunk,
            encode_state=session.encoder_state,
        ).to(self.device)
        if latent.shape[1] != session.chunk_size:
            raise RuntimeError(
                f"VAE condition chunk has {latent.shape[1]} latent frames, expected {session.chunk_size}"
            )

        mask = torch.zeros(
            (4, session.chunk_size, latent.shape[2], latent.shape[3]),
            device=latent.device,
            dtype=latent.dtype,
        )
        if is_first_chunk:
            mask[:, 0] = 1
            session.condition_image = None
        return torch.cat([mask, latent], dim=0).unsqueeze(0)

    def _encode_condition_chunk_profiled(
        self,
        session: LingBotWorldFastGenerationSession,
        *,
        chunk_index: int,
        progress_callback: Callable[..., None] | None = None,
        condition_prefetched: bool = False,
        synchronize: bool = False,
    ) -> tuple[torch.Tensor, float]:
        self._notify_progress(
            progress_callback,
            "encoding_condition_chunk",
            index=chunk_index,
            condition_prefetched=condition_prefetched,
        )
        condition_start_ns = time.perf_counter_ns()
        condition_chunk = self._encode_condition_chunk(session, chunk_index=chunk_index)
        if synchronize and self.vae_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.vae_device)
        condition_end_ns = time.perf_counter_ns()
        condition_encode_ms = (condition_end_ns - condition_start_ns) / 1_000_000.0
        self._notify_progress(
            progress_callback,
            "condition_chunk_encoded",
            index=chunk_index,
            condition_encode_ms=condition_encode_ms,
            condition_prefetched=condition_prefetched,
        )
        return condition_chunk, condition_encode_ms

    def _wait_for_condition_prefetch(self, prefetch: LingBotWorldFastConditionPrefetch, timeout: float = 0.1) -> None:
        while not prefetch.done.wait(timeout=timeout):
            continue

    def _consume_or_encode_condition_chunk(
        self,
        session: LingBotWorldFastGenerationSession,
        *,
        chunk_index: int,
        progress_callback: Callable[..., None] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        prefetch = session.condition_prefetch
        if prefetch is None:
            condition_chunk, condition_encode_ms = self._encode_condition_chunk_profiled(
                session,
                chunk_index=chunk_index,
                progress_callback=progress_callback,
            )
            return condition_chunk, {
                "condition_encode_ms": condition_encode_ms,
                "condition_prefetched": False,
                "condition_prefetch_wait_ms": 0.0,
            }
        if prefetch.chunk_index != chunk_index:
            raise RuntimeError(
                f"LingBot condition prefetch order violation: expected chunk {chunk_index}, "
                f"got prefetched chunk {prefetch.chunk_index}"
            )

        wait_start_ns = time.perf_counter_ns()
        self._wait_for_condition_prefetch(prefetch)
        wait_ms = (time.perf_counter_ns() - wait_start_ns) / 1_000_000.0
        session.condition_prefetch = None
        if prefetch.exception is not None:
            raise RuntimeError(
                f"LingBot condition prefetch failed for chunk {chunk_index}: {prefetch.exception}"
            ) from prefetch.exception
        if prefetch.condition_chunk is None:
            raise RuntimeError(f"LingBot condition prefetch for chunk {chunk_index} produced no tensor")
        self._notify_progress(
            progress_callback,
            "condition_prefetch_consumed",
            index=chunk_index,
            condition_encode_ms=prefetch.condition_encode_ms,
            condition_prefetch_wait_ms=wait_ms,
        )
        return prefetch.condition_chunk, {
            "condition_encode_ms": prefetch.condition_encode_ms,
            "condition_prefetched": True,
            "condition_prefetch_wait_ms": wait_ms,
        }

    def _start_condition_prefetch(
        self,
        session: LingBotWorldFastGenerationSession,
        *,
        chunk_index: int,
        progress_callback: Callable[..., None] | None = None,
    ) -> None:
        if not self.condition_prefetch_enabled or chunk_index >= session.chunk_count:
            return
        if session.condition_prefetch is not None:
            raise RuntimeError(
                f"LingBot condition prefetch already in flight for chunk {session.condition_prefetch.chunk_index}"
            )

        prefetch = LingBotWorldFastConditionPrefetch(chunk_index=chunk_index)
        session.condition_prefetch = prefetch

        def worker() -> None:
            prefetch.start_ns = time.perf_counter_ns()
            self._notify_progress(progress_callback, "condition_prefetch_start", index=chunk_index)
            try:
                condition_chunk, condition_encode_ms = self._encode_condition_chunk_profiled(
                    session,
                    chunk_index=chunk_index,
                    progress_callback=progress_callback,
                    condition_prefetched=True,
                    synchronize=True,
                )
                prefetch.condition_chunk = condition_chunk
                prefetch.condition_encode_ms = condition_encode_ms
            except BaseException as exc:
                prefetch.exception = exc
            finally:
                prefetch.end_ns = time.perf_counter_ns()
                self._notify_progress(
                    progress_callback,
                    "condition_prefetch_done",
                    index=chunk_index,
                    condition_encode_ms=prefetch.condition_encode_ms,
                    failed=prefetch.exception is not None,
                )
                prefetch.done.set()

        threading.Thread(target=worker, daemon=True, name=f"lingbot-condition-prefetch-{chunk_index}").start()

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
        with session.transaction_lock:
            self._cancel_async_vae_generation(session)
            if session.condition_prefetch is not None:
                self._wait_for_condition_prefetch(session.condition_prefetch, timeout=0.1)
                session.condition_prefetch = None
            cache_released = self._release_session_cache(session)
            session.decoder_state.feat_cache = []
            session.decoder_state.feat_idx = [0]
            session.encoder_state.feat_cache = []
            session.encoder_state.feat_idx = [0]
            session.condition_image = None
            session.noise_generator = None
            if not cache_released:
                session.status = LingBotWorldFastSessionStatus.POISONED
                session.poisoned_reason = f"Failed to release cache handle {session.cache_handle}"
            elif session.status != LingBotWorldFastSessionStatus.POISONED:
                session.status = LingBotWorldFastSessionStatus.RELEASED

    def close(self) -> None:
        """Deterministically close async VAE and multi-process denoising workers."""
        async_vae_manager = getattr(self, "_async_vae_manager", None)
        if async_vae_manager is not None:
            async_vae_manager.close()
            self._async_vae_manager = None
        denoise_stage = getattr(self, "denoise_stage", None)
        if isinstance(denoise_stage, ParallelWorker):
            denoise_stage.close()

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

        session = LingBotWorldFastGenerationSession(config=session_config)
        try:
            self(
                session,
                LingBotWorldFastChunkRequest(
                    chunk_index=0,
                    session_id="warmup",
                    control=LingBotWorldFastControlBuilder(control_context).defer(action),
                ),
            )
            self(
                session,
                LingBotWorldFastChunkRequest(
                    chunk_index=1,
                    session_id="warmup",
                    control=LingBotWorldFastControlBuilder(control_context).defer(action),
                ),
            )
        finally:
            self.release_session(session)

    def __del__(self) -> None:
        """Best-effort fallback for callers that do not explicitly close the pipeline."""
        try:
            self.close()
        except Exception:
            pass

    @torch.inference_mode()
    def __call__(
        self,
        session: LingBotWorldFastGenerationSession,
        request: LingBotWorldFastChunkRequest,
        progress_callback: Callable[..., None] | None = None,
    ) -> LingBotWorldFastChunkResult:
        """Initialize a session on first use and generate one controlled chunk."""
        if not session.transaction_lock.acquire(blocking=False):
            raise RuntimeError("LingBot session already has a chunk in progress")
        try:
            self._validate_chunk_request(session, request)
            resolved_control: torch.Tensor | None = None
            if session.status == LingBotWorldFastSessionStatus.NEW:
                try:

                    def materialize_first_control() -> None:
                        nonlocal resolved_control
                        resolved_control = self._resolve_control(request.control)

                    initialized = self._create_initialized_session(
                        session.config,
                        progress_callback,
                        before_cache=materialize_first_control,
                    )
                    session.__dict__.update(
                        (key, value) for key, value in initialized.__dict__.items() if key != "transaction_lock"
                    )
                except Exception as exc:
                    session.status = LingBotWorldFastSessionStatus.POISONED
                    session.poisoned_reason = f"{type(exc).__name__}: {exc}"
                    self.release_session(session)
                    raise
            if resolved_control is None:
                resolved_control = self._resolve_control(request.control)
            self._validate_control(session, resolved_control)
            return self._generate_session_chunk(session, request, resolved_control, progress_callback)
        finally:
            session.transaction_lock.release()

    @staticmethod
    def _validate_chunk_request(
        session: LingBotWorldFastGenerationSession,
        request: LingBotWorldFastChunkRequest,
    ) -> None:
        if session.status == LingBotWorldFastSessionStatus.POISONED:
            raise RuntimeError(f"Cannot continue poisoned LingBot session: {session.poisoned_reason}")
        if session.status == LingBotWorldFastSessionStatus.RUNNING:
            raise RuntimeError("LingBot session already has a chunk in progress")
        if session.status == LingBotWorldFastSessionStatus.RELEASED:
            raise RuntimeError("Cannot generate a chunk from an inactive LingBot session")
        if request.chunk_index != session.current_chunk_index:
            raise ValueError(
                f"Chunk request index {request.chunk_index} does not match session index {session.current_chunk_index}"
            )

    @staticmethod
    def _resolve_control(control: torch.Tensor | Callable[[], torch.Tensor]) -> torch.Tensor:
        resolved = control() if callable(control) else control
        if not isinstance(resolved, torch.Tensor):
            raise TypeError("Deferred control factory must return a torch.Tensor")
        return resolved

    def _generate_session_chunk(
        self,
        session: LingBotWorldFastGenerationSession,
        request: LingBotWorldFastChunkRequest,
        control: torch.Tensor,
        progress_callback: Callable[..., None] | None,
    ) -> LingBotWorldFastChunkResult:
        """Execute a chunk while the caller holds the session transaction lock."""
        session.status = LingBotWorldFastSessionStatus.RUNNING
        try:
            chunk_frames = self.generate_next_chunk(
                session,
                control=control,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            session.status = LingBotWorldFastSessionStatus.POISONED
            session.poisoned_reason = f"{type(exc).__name__}: {exc}"
            self.release_session(session)
            raise

        session.status = LingBotWorldFastSessionStatus.COMMITTED
        done = session.current_chunk_index >= session.chunk_count
        if done:
            self.release_session(session)
            if session.status == LingBotWorldFastSessionStatus.POISONED:
                raise RuntimeError(session.poisoned_reason or "Final chunk cleanup failed")
        logger.info(
            f"Generated LingBot chunk {request.chunk_index + 1}/{session.chunk_count}: {len(chunk_frames)} frames"
        )
        return LingBotWorldFastChunkResult(
            chunk_index=request.chunk_index,
            frames=chunk_frames,
            emitted_frames=session.emitted_frames,
            done=done,
            session_id=request.session_id,
        )

    @torch.inference_mode()
    def generate_video(
        self,
        session_config: LingBotWorldFastSessionConfig,
        controls: list[torch.Tensor | Callable[[], torch.Tensor]],
        progress_callback: Callable[..., None] | None = None,
    ) -> list[Image.Image]:
        """Drain externally prepared controls through the single-chunk API."""
        session = LingBotWorldFastGenerationSession(config=session_config)
        frames: list[Image.Image] = []
        completed = False
        try:
            for chunk_index, control in enumerate(controls):
                result = self(
                    session,
                    LingBotWorldFastChunkRequest(
                        chunk_index=chunk_index,
                        control=control,
                    ),
                    progress_callback=progress_callback,
                )
                frames.extend(result.frames)
                if result.done:
                    completed = True
                    break
            if not completed and session.status != LingBotWorldFastSessionStatus.RELEASED:
                raise ValueError("Control sequence ended before the generation session completed")
        finally:
            self.release_session(session)
        return frames

    @ProfilingContext4Debug("initialize_session")
    @torch.inference_mode()
    def _create_initialized_session(
        self,
        session_config: LingBotWorldFastSessionConfig,
        progress_callback: Callable[..., None] | None = None,
        before_cache: Callable[[], None] | None = None,
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
        if before_cache is not None:
            before_cache()
        cache_handle = self._next_cache_handle
        self._next_cache_handle += 1
        session = LingBotWorldFastGenerationSession(
            prompt_emb=prompt_emb,
            config=session_config,
            condition_image=image_tensor,
            noise_generator=noise_generator,
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
            initialize_cache_kwargs = dict(
                cache_handle=cache_handle,
                batch_size=1,
                kv_size=kv_size,
                max_sequence_length=session_config.max_sequence_length,
                sample_shift=session_config.sample_shift,
                generator_state=denoise_generator.get_state().tolist(),
                timestep_indices=getattr(self.config, "timestep_indices", (0, 179, 358, 679)),
            )
            if isinstance(self.denoise_stage, ParallelWorker):
                self.denoise_stage.initialize_cache(**initialize_cache_kwargs, sync=True)
            else:
                self.denoise_stage.initialize_cache(**initialize_cache_kwargs)
        except Exception:
            self._release_session_cache(session)
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

    def _next_noise_chunk(self, session: LingBotWorldFastGenerationSession) -> torch.Tensor:
        if session.noise_generator is None:
            raise RuntimeError("The session noise generator has already been released")
        return torch.randn(
            (1, 16, session.chunk_size, session.latent_h, session.latent_w),
            generator=session.noise_generator,
            device=self.device,
            dtype=torch.float32,
        )

    def _generate_chunk_latent(
        self,
        runtime: LingBotWorldFastGenerationSession,
        control: torch.Tensor,
        progress_callback: Callable[..., None] | None = None,
        *,
        return_profile: bool = False,
    ) -> tuple[int, torch.Tensor, dict[str, object]]:
        idx = runtime.current_chunk_index
        chunk_start_ns = time.perf_counter_ns()
        latent_chunk = self._next_noise_chunk(runtime)
        condition_chunk, condition_profile = self._consume_or_encode_condition_chunk(
            runtime,
            chunk_index=idx,
            progress_callback=progress_callback,
        )
        self._start_condition_prefetch(runtime, chunk_index=idx + 1, progress_callback=progress_callback)
        control_chunk = control

        self._notify_progress(progress_callback, "dit_start", index=idx)
        current_start = idx * runtime.chunk_size * runtime.frame_tokens
        cached_latent = runtime.world_kv_cached_latents.pop(idx, None) if runtime.world_kv_cached_latents else None
        profile: dict[str, object] = {
            "chunk_id": idx,
            "chunk_start_ns": chunk_start_ns,
            "cached_latent": cached_latent is not None,
        }
        profile.update(condition_profile)
        if cached_latent is not None:
            self._notify_progress(progress_callback, "decoding_cached_chunk", index=idx)
            denoised = cached_latent.to(device=self.device, dtype=self.torch_dtype)
            profile["dit_total_ms"] = 0.0
            profile["kv_update_ms"] = 0.0
        else:
            with ProfilingContext4Debug("denoise_chunk"):
                denoise_kwargs = dict(
                    cache_handle=runtime.cache_handle,
                    latent_chunk=latent_chunk,
                    condition_chunk=condition_chunk,
                    prompt_emb=runtime.prompt_emb,
                    control_chunk=control_chunk,
                    current_start=current_start,
                    max_attention_size=runtime.max_attention_size,
                    chunk_id=idx,
                    return_profile=return_profile,
                )
                self._notify_progress(progress_callback, "kv_update_start", index=idx)
                dit_start_ns = time.perf_counter_ns()
                if isinstance(self.denoise_stage, ParallelWorker):
                    denoise_result = self.denoise_stage.denoise_and_update_cache(**denoise_kwargs, sync=True)
                else:
                    denoise_result = self.denoise_stage.denoise_and_update_cache(**denoise_kwargs)
                dit_end_ns = time.perf_counter_ns()

            if return_profile and isinstance(denoise_result, tuple):
                denoised, worker_profile = denoise_result
                profile.update(worker_profile)
            else:
                denoised = denoise_result
            profile.setdefault("dit_total_ms", (dit_end_ns - dit_start_ns) / 1_000_000.0)
            profile.setdefault("kv_update_ms", profile.get("kv_update_gpu_ms", 0.0))
            self._notify_progress(progress_callback, "dit_end", index=idx, profile=profile)
            self._notify_progress(progress_callback, "kv_update_end", index=idx, profile=profile)

            if runtime.world_kv_binding is not None:
                try:
                    runtime.world_kv_binding.on_chunk_finalized(runtime, idx, denoised)
                except Exception as exc:
                    logger.warning(f"world_kv on_chunk_finalized failed at chunk {idx}: {exc}")

        profile["chunk_period_ms"] = (time.perf_counter_ns() - chunk_start_ns) / 1_000_000.0
        return idx, denoised, profile

    def _decode_latent_to_images(
        self,
        runtime: LingBotWorldFastGenerationSession,
        idx: int,
        denoised: torch.Tensor,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[Image.Image]:
        self._notify_progress(progress_callback, "decoding_chunk", index=idx, device=str(self.vae_device))
        with ProfilingContext4Debug("vae_decode"):
            frames = self.decode_video_cached(
                runtime,
                denoised,
                is_first_clip=(idx == 0),
                is_last_clip=(idx == runtime.chunk_count - 1),
            )
            images = self.tensor2video(frames)
        self._notify_progress(progress_callback, "chunk_decoded", index=idx, frames=len(images))
        return images

    @torch.inference_mode()
    def generate_next_chunk(
        self,
        runtime: LingBotWorldFastGenerationSession,
        control: torch.Tensor,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[Image.Image]:
        if runtime.current_chunk_index >= runtime.chunk_count:
            return []

        with ProfilingContext4Debug("generate_next_chunk"):
            idx, denoised, _profile = self._generate_chunk_latent(
                runtime,
                control=control,
                progress_callback=progress_callback,
            )
            images = self._decode_latent_to_images(runtime, idx, denoised, progress_callback)
            runtime.current_chunk_index += 1
            runtime.emitted_frames += len(images)
            return images

    @torch.inference_mode()
    def submit_async_vae_chunk(
        self,
        session: LingBotWorldFastGenerationSession,
        request: LingBotWorldFastChunkRequest,
        progress_callback: Callable[..., None] | None = None,
    ) -> AsyncVAEChunkHandle:
        """Generate one latent chunk, enqueue VAE decode, and return its completion handle."""
        manager = self._ensure_async_vae_manager()
        manager.raise_if_failed()
        if not session.transaction_lock.acquire(blocking=False):
            raise RuntimeError("LingBot session already has a chunk in progress")
        try:
            self._validate_chunk_request(session, request)
            resolved_control: torch.Tensor | None = None
            if session.status == LingBotWorldFastSessionStatus.NEW:
                try:

                    def materialize_first_control() -> None:
                        nonlocal resolved_control
                        resolved_control = self._resolve_control(request.control)

                    initialized = self._create_initialized_session(
                        session.config,
                        progress_callback,
                        before_cache=materialize_first_control,
                    )
                    session.__dict__.update(
                        (key, value) for key, value in initialized.__dict__.items() if key != "transaction_lock"
                    )
                except Exception as exc:
                    session.status = LingBotWorldFastSessionStatus.POISONED
                    session.poisoned_reason = f"{type(exc).__name__}: {exc}"
                    self.release_session(session)
                    raise
            if resolved_control is None:
                resolved_control = self._resolve_control(request.control)
            self._validate_control(session, resolved_control)

            session.status = LingBotWorldFastSessionStatus.RUNNING
            try:
                idx, denoised, profile = self._generate_chunk_latent(
                    session,
                    control=resolved_control,
                    progress_callback=progress_callback,
                    return_profile=True,
                )
                generation_id = self._assign_async_vae_generation(session)
                enqueue_ns = time.perf_counter_ns()
                handle = AsyncVAEChunkHandle(
                    session_id=request.session_id,
                    generation_id=generation_id,
                    chunk_id=idx,
                    is_last_clip=(idx == session.chunk_count - 1),
                    enqueue_ns=enqueue_ns,
                    denoise_profile=profile,
                )
                task = AsyncVAEDecodeTask(
                    session_id=request.session_id,
                    generation_id=generation_id,
                    chunk_id=idx,
                    latent=denoised,
                    latent_ready_event=self._record_latent_ready_event(denoised),
                    is_first_clip=(idx == 0),
                    is_last_clip=(idx == session.chunk_count - 1),
                    handle=handle,
                )
                self._notify_progress(progress_callback, "vae_enqueue", index=idx, generation_id=generation_id)
                manager.enqueue(task)
                self._notify_progress(
                    progress_callback,
                    "vae_enqueued",
                    index=idx,
                    generation_id=generation_id,
                    vae_queue_wait_ms=handle.queue_wait_ms,
                    vae_queue_depth=handle.queue_depth_after_enqueue,
                )
            except Exception as exc:
                session.status = LingBotWorldFastSessionStatus.POISONED
                session.poisoned_reason = f"{type(exc).__name__}: {exc}"
                self.release_session(session)
                raise

            session.current_chunk_index += 1
            session.status = LingBotWorldFastSessionStatus.COMMITTED
            logger.info(
                f"Submitted LingBot async VAE chunk {idx + 1}/{session.chunk_count}: "
                f"queue_wait_ms={handle.queue_wait_ms:.3f}"
            )
            return handle
        finally:
            session.transaction_lock.release()

    def wait_async_vae_chunk(
        self,
        session: LingBotWorldFastGenerationSession,
        handle: AsyncVAEChunkHandle,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[Image.Image]:
        """Wait for one async VAE chunk and return frames if it still belongs to the session."""
        manager = getattr(self, "_async_vae_manager", None)
        while not handle.done.wait(timeout=0.1):
            if manager is not None:
                manager.raise_if_failed()
        if handle.exception is not None:
            session.status = LingBotWorldFastSessionStatus.POISONED
            session.poisoned_reason = f"{type(handle.exception).__name__}: {handle.exception}"
            self.release_session(session)
            raise RuntimeError(session.poisoned_reason) from handle.exception
        if handle.canceled or handle.generation_id != session.async_vae_generation_id:
            logger.info(
                "Dropped stale LingBot async VAE output: "
                f"generation_id={handle.generation_id} chunk_id={handle.chunk_id}"
            )
            return []

        frames = handle.frames or []
        session.emitted_frames += len(frames)
        output_ready_ns = handle.output_ready_ns or time.perf_counter_ns()
        chunk_start_ns = int(handle.denoise_profile.get("chunk_start_ns", handle.enqueue_ns))
        chunk_period_ms = (output_ready_ns - chunk_start_ns) / 1_000_000.0
        vae_ms = None
        if handle.vae_start_ns is not None and handle.vae_end_ns is not None:
            vae_ms = (handle.vae_end_ns - handle.vae_start_ns) / 1_000_000.0
        overlap_ratio = None
        if handle.overlap_ms is not None:
            dit_ms = float(handle.denoise_profile.get("dit_total_ms", 0.0) or 0.0)
            overlap_vae_ms = vae_ms
            if overlap_vae_ms is None:
                overlap_vae_ms = (
                    (handle.vae_end_ns or output_ready_ns) - (handle.vae_start_ns or handle.enqueue_ns)
                ) / 1_000_000.0
            denom = min(dit_ms, overlap_vae_ms)
            overlap_ratio = handle.overlap_ms / denom if denom > 0 else None
        self._notify_progress(
            progress_callback,
            "output_ready",
            index=handle.chunk_id,
            frames=len(frames),
            vae_queue_wait_ms=handle.queue_wait_ms,
            vae_ms=vae_ms,
            vae_gpu_ms=handle.vae_gpu_ms,
            chunk_period_ms=chunk_period_ms,
            overlap_ms=handle.overlap_ms,
            overlap_ratio=overlap_ratio,
        )
        logger.info(
            "LingBot async VAE chunk ready: "
            f"chunk_id={handle.chunk_id} frames={len(frames)} chunk_period_ms={chunk_period_ms:.3f} "
            f"queue_wait_ms={handle.queue_wait_ms:.3f} overlap_ms={handle.overlap_ms} "
            f"overlap_ratio={overlap_ratio}"
        )
        return frames
