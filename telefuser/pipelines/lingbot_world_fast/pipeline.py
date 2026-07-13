from __future__ import annotations

import base64
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
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

from .control import LingBotWorldFastControlContext
from .denoising import LingBotWorldFastDenoisingStage
from .session import (
    LingBotWorldFastChunkRequest,
    LingBotWorldFastChunkResult,
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionStatus,
)


@dataclass
class LingBotWorldFastPipelineConfig:
    checkpoint_dir: str = ""
    fast_checkpoint_subdir: str = "lingbot_world_fast"
    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_torch_dtype: torch.dtype = torch.bfloat16
    control_type: str = "cam"
    orig_height: int = 480
    orig_width: int = 832
    max_area: int = 480 * 832
    local_attn_size: int = -1
    sink_size: int = 0
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)


class LingBotWorldFastPipeline(BasePipeline):
    """Pipeline wrapper for LingBot-World-Fast chunked causal generation."""

    def __init__(self, device: str, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16

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

    def init(self, module_manager, config: LingBotWorldFastPipelineConfig) -> None:
        self.config = config
        checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve()
        self._model_info = [{"name": "lingbot_world_fast", "path": str(checkpoint_root)}]
        self.text_device = self._runtime_device(config.text_encoding_config)
        self.vae_device = self._runtime_device(config.vae_config)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.load_state_dict(load_state_dict(str(checkpoint_root / "models_t5_umt5-xxl-enc-bf16.pth")))
        self.text_encoder = self.text_encoder.to(device=self.text_device, dtype=self.torch_dtype).eval()
        tokenizer_path = checkpoint_root / "google" / "umt5-xxl"
        self.tokenizer = HuggingfaceTokenizer(str(tokenizer_path), 512, "whitespace")

        vae_state_dict, vae_cfg = WanVideoVAE.state_dict_converter().from_official(
            load_state_dict(str(checkpoint_root / "Wan2.1_VAE.pth"))
        )
        self.vae = WanVideoVAE(**vae_cfg)
        self.vae.load_state_dict(vae_state_dict, strict=False)
        self.vae = self.vae.to(device=self.vae_device, dtype=self.torch_dtype).eval()

        fast_path = checkpoint_root / config.fast_checkpoint_subdir
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

    @staticmethod
    def _best_output_size(w: int, h: int, expected_area: int, dw: int = 16, dh: int = 16) -> tuple[int, int]:
        ratio = w / h
        ow = math.sqrt(expected_area * ratio)
        oh = expected_area / ow

        ow1 = int(ow // dw * dw)
        oh1 = int(expected_area / max(ow1, 1) // dh * dh)
        oh2 = int(oh // dh * dh)
        ow2 = int(expected_area / max(oh2, 1) // dw * dw)

        if ow1 <= 0 or oh1 <= 0:
            return max(dw, ow2), max(dh, oh2)
        if ow2 <= 0 or oh2 <= 0:
            return max(dw, ow1), max(dh, oh1)

        ratio1 = ow1 / oh1
        ratio2 = ow2 / oh2
        if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
            return ow1, oh1
        return ow2, oh2

    def control_context(self, session_config: LingBotWorldFastSessionConfig) -> LingBotWorldFastControlContext:
        """Return control geometry without allocating a generation runtime."""
        width, height = self._best_output_size(
            session_config.image.width,
            session_config.image.height,
            self.config.max_area,
        )
        height, width = self.check_resize_height_width(height, width)
        frame_num = ((session_config.frame_num - 1) // 4) * 4 + 1
        latent_frames = (frame_num - 1) // 4 + 1
        latent_frames -= latent_frames % session_config.chunk_size
        if latent_frames < session_config.chunk_size:
            raise ValueError("frame_num must contain at least one complete latent chunk")
        return LingBotWorldFastControlContext(
            control_type=self.config.control_type,
            device=self.device,
            torch_dtype=self.torch_dtype,
            orig_height=self.config.orig_height,
            orig_width=self.config.orig_width,
            height=height,
            width=width,
            latent_h=height // 8,
            latent_w=width // 8,
            latent_frames=latent_frames,
            chunk_size=session_config.chunk_size,
        )

    def _prepare_image_tensor(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        image = image.convert("RGB").resize((width, height), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = (
            torch.from_numpy(array).permute(2, 0, 1).sub_(0.5).div_(0.5).to(self.vae_device, dtype=self.torch_dtype)
        )
        return tensor

    def _encode_condition_video(self, image_tensor: torch.Tensor, frame_num: int) -> torch.Tensor:
        h, w = image_tensor.shape[1:]
        video = torch.cat(
            [
                image_tensor.unsqueeze(1),
                torch.zeros(3, frame_num - 1, h, w, device=image_tensor.device, dtype=image_tensor.dtype),
            ],
            dim=1,
        )
        latent = self.vae.encode([video], device=self.vae_device, tiled=False)[0].to(self.device)

        latent_h = latent.shape[2]
        latent_w = latent.shape[3]
        mask = torch.ones(1, frame_num, latent_h, latent_w, device=self.device, dtype=latent.dtype)
        mask[:, 1:] = 0
        mask = torch.cat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, latent_h, latent_w).transpose(1, 2)[0]
        return torch.cat([mask, latent], dim=0)

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
            cache_released = self._release_session_cache(session)
            session.decoder_state.feat_cache = []
            session.decoder_state.feat_idx = [0]
            session.active = False
            if not cache_released:
                session.status = LingBotWorldFastSessionStatus.POISONED
                session.poisoned_reason = f"Failed to release cache handle {session.cache_handle}"
            elif session.status != LingBotWorldFastSessionStatus.POISONED:
                session.status = LingBotWorldFastSessionStatus.RELEASED

    def close(self) -> None:
        """Deterministically close the multi-process denoising worker group."""
        denoise_stage = getattr(self, "denoise_stage", None)
        if isinstance(denoise_stage, ParallelWorker):
            denoise_stage.close()

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
        if not session.active or session.status == LingBotWorldFastSessionStatus.RELEASED:
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
            session.active = False
            self.release_session(session)
            raise

        session.status = LingBotWorldFastSessionStatus.COMMITTED
        if not session.active and not self._release_session_cache(session):
            session.status = LingBotWorldFastSessionStatus.POISONED
            session.poisoned_reason = "Final chunk committed but cache release failed"
            raise RuntimeError(session.poisoned_reason)
        logger.info(
            f"Generated LingBot chunk {request.chunk_index + 1}/{len(session.noise_chunks)}: {len(chunk_frames)} frames"
        )
        return LingBotWorldFastChunkResult(
            chunk_index=request.chunk_index,
            frames=chunk_frames,
            emitted_frames=session.emitted_frames,
            done=not session.active,
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
            if session.active:
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
        self._notify_progress(progress_callback, "encoding_prompt", device=str(self.text_device))
        prompt_emb = self.encode_prompt(session_config.prompt)
        self._notify_progress(progress_callback, "prompt_encoded")

        w0, h0 = session_config.image.size
        width, height = self._best_output_size(w0, h0, self.config.max_area)
        height, width = self.check_resize_height_width(height, width)

        self._notify_progress(progress_callback, "preparing_image", width=width, height=height)
        image_tensor = self._prepare_image_tensor(session_config.image, height, width)

        frame_num = ((session_config.frame_num - 1) // 4) * 4 + 1
        self._notify_progress(
            progress_callback,
            "encoding_condition_video",
            frame_num=frame_num,
            device=str(self.vae_device),
        )
        latent_condition = self._encode_condition_video(image_tensor, frame_num)
        self._notify_progress(progress_callback, "condition_video_encoded")

        lat_h = height // 8
        lat_w = width // 8
        lat_f = (frame_num - 1) // 4 + 1
        lat_f = int(lat_f - (lat_f % session_config.chunk_size))
        frame_num = (lat_f - 1) * 4 + 1
        patch_area = self.dit.patch_size[1] * self.dit.patch_size[2]
        frame_tokens = (lat_h * lat_w) // patch_area
        kv_size = self._resolve_self_kv_size(
            frame_tokens=frame_tokens,
            latent_frames=lat_f,
            config=self.config,
        )
        max_seq_len = session_config.chunk_size * frame_tokens
        max_attention_size = (
            kv_size if session_config.max_attention_size is None else int(session_config.max_attention_size)
        )

        self._notify_progress(
            progress_callback,
            "allocating_runtime",
            latent_frames=lat_f,
            latent_height=lat_h,
            latent_width=lat_w,
            total_chunks=math.ceil(lat_f / session_config.chunk_size),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(session_config.seed))
        noise = torch.randn(
            (1, 16, lat_f, lat_h, lat_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        noise_chunks = list(noise.split(session_config.chunk_size, dim=2))
        condition_chunks = list(latent_condition.unsqueeze(0).split(session_config.chunk_size, dim=2))
        if before_cache is not None:
            before_cache()
        cache_handle = self._next_cache_handle
        self._next_cache_handle += 1
        session = LingBotWorldFastGenerationSession(
            prompt_emb=prompt_emb,
            encoded_image_latent=latent_condition,
            config=session_config,
            noise_chunks=noise_chunks,
            condition_chunks=condition_chunks,
            latent_h=lat_h,
            latent_w=lat_w,
            latent_f=lat_f,
            height=height,
            width=width,
            max_seq_len=max_seq_len,
            frame_tokens=frame_tokens,
            chunk_size=session_config.chunk_size,
            max_attention_size=max_attention_size,
            cache_handle=cache_handle,
            kv_local_attn_size=self.config.local_attn_size,
            kv_sink_size=self.config.sink_size if self.config.local_attn_size != -1 else 0,
        )
        try:
            initialize_cache_kwargs = dict(
                cache_handle=cache_handle,
                batch_size=1,
                kv_size=kv_size,
                max_sequence_length=session_config.max_sequence_length,
                sample_shift=session_config.sample_shift,
                generator_state=generator.get_state().tolist(),
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

    @torch.inference_mode()
    def generate_next_chunk(
        self,
        runtime: LingBotWorldFastGenerationSession,
        control: torch.Tensor,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[Image.Image]:
        if runtime.current_chunk_index >= len(runtime.noise_chunks):
            runtime.active = False
            return []

        with ProfilingContext4Debug("generate_next_chunk"):
            idx = runtime.current_chunk_index
            latent_chunk = runtime.noise_chunks[idx]
            condition_chunk = runtime.condition_chunks[idx]
            control_chunk = control

            self._notify_progress(progress_callback, "denoising_chunk", index=idx)
            current_start = idx * runtime.chunk_size * runtime.frame_tokens
            cached_latent = runtime.world_kv_cached_latents.pop(idx, None) if runtime.world_kv_cached_latents else None
            if cached_latent is not None:
                # On a world_kv fast-forward hit, KV is already seeded and the latent comes
                # from the cached skeleton. Decode it directly and skip clean-KV rewrite.
                self._notify_progress(progress_callback, "decoding_cached_chunk", index=idx)
                denoised = cached_latent.to(device=self.device, dtype=self.torch_dtype)
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
                    )
                    if isinstance(self.denoise_stage, ParallelWorker):
                        denoised = self.denoise_stage.denoise_and_update_cache(**denoise_kwargs, sync=True)
                    else:
                        denoised = self.denoise_stage.denoise_and_update_cache(**denoise_kwargs)

                if runtime.world_kv_binding is not None:
                    try:
                        runtime.world_kv_binding.on_chunk_finalized(runtime, idx, denoised)
                    except Exception as exc:
                        logger.warning(f"world_kv on_chunk_finalized failed at chunk {idx}: {exc}")

            self._notify_progress(progress_callback, "decoding_chunk", index=idx, device=str(self.vae_device))
            with ProfilingContext4Debug("vae_decode"):
                frames = self.decode_video_cached(
                    runtime,
                    denoised,
                    is_first_clip=(idx == 0),
                    is_last_clip=(idx == len(runtime.noise_chunks) - 1),
                )
                images = self.tensor2video(frames)
            self._notify_progress(progress_callback, "chunk_decoded", index=idx, frames=len(images))
            runtime.current_chunk_index += 1
            runtime.emitted_frames += len(images)
            if runtime.current_chunk_index >= len(runtime.noise_chunks):
                runtime.active = False
            return images

    @staticmethod
    def encode_frames_to_b64(frames: list[Image.Image], quality: int = 85) -> list[str]:
        encoded: list[str] = []
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if ok:
                encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        return encoded
