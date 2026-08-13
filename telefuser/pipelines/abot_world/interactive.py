"""Persistent multi-session interaction and batching for ABot-World."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from PIL import Image

from telefuser.core.config import WeightOffloadType
from telefuser.models.wan22_video_vae import Wan22VideoVAEStreamingDecodeState

from .pipeline import ABotWorldPipeline
from .taew_vae import ABotWorldTAEWDecodeState


class ABotWorldSessionLifecycle(str, Enum):
    """Residency and execution lifecycle for one retained ABot session."""

    READY = "ready"
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    MIGRATING = "migrating"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"


@dataclass
class ABotWorldInteractiveSession:
    """State retained across causally generated ABot action blocks."""

    prompt_emb: torch.Tensor
    first_frame_latent: torch.Tensor
    self_cache: list[dict[str, Any]]
    cross_cache: list[dict[str, Any]]
    scheduler: Any
    generator: torch.Generator
    vae_decode_state: Wan22VideoVAEStreamingDecodeState = field(
        default_factory=Wan22VideoVAEStreamingDecodeState
    )
    taew_decode_state: ABotWorldTAEWDecodeState | None = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    next_latent_frame: int = 0
    emitted_frames: int = 0
    lifecycle: ABotWorldSessionLifecycle = ABotWorldSessionLifecycle.READY
    last_activity_at: float = field(default_factory=time.monotonic)
    owner_worker_id: str | None = None
    ownership_epoch: int = 0
    closed: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def is_resident(self) -> bool:
        """Return whether tensors are currently resident on the execution device."""
        return self.lifecycle != ABotWorldSessionLifecycle.SUSPENDED


@dataclass(frozen=True)
class ABotWorldSessionSnapshot:
    """CPU-owned state transferred between ABot workers at a chunk boundary."""

    session_id: str
    prompt_emb: torch.Tensor
    first_frame_latent: torch.Tensor
    self_cache: tuple[dict[str, Any], ...]
    cross_cache: tuple[dict[str, Any], ...]
    vae_feat_cache: tuple[object, ...]
    vae_feat_idx: tuple[int, ...]
    generator_state: torch.Tensor
    next_latent_frame: int
    emitted_frames: int
    ownership_epoch: int


class ABotWorldInteractivePipeline(ABotWorldPipeline):
    """ABot pipeline with shared weights and isolated retained sessions."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._lifecycle_lock = threading.RLock()
        self._execution_lock = threading.RLock()
        self._interactive_sessions: dict[str, ABotWorldInteractiveSession] = {}
        self._models_preloaded = False
        self._last_stage_metrics: dict[str, float | int] = {}

    def preload_models(self) -> None:
        """Place VAE, T5, and DiT on the configured GPU before accepting controls."""
        with self._lifecycle_lock:
            if self._models_preloaded:
                return
            for stage in self._get_stages():
                stage.model_runtime_config.offload_config.offload_type = WeightOffloadType.NO_CPU_OFFLOAD
                stage.onload_models()
                stage.onload_models_flag = True
            self._models_preloaded = True

    @torch.inference_mode()
    def create_interactive_session(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int = 42,
        session_id: str | None = None,
    ) -> ABotWorldInteractiveSession:
        """Encode the start image and allocate session-owned causal caches."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        self.preload_models()
        with self._execution_lock:
            pixels = self.preprocess_image(image.convert("RGB"), self.config.height, self.config.width)
            encode_started_at = time.monotonic()
            start_latent, _ = self.vae_stage.process("encode_image", pixels, None, 1, concat_mask=False)
            encode_seconds = time.monotonic() - encode_started_at
            first_frame_latent = start_latent.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
            text_started_at = time.monotonic()
            prompt_emb = self.text_encoding_stage.process([prompt])[0].to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            text_seconds = time.monotonic() - text_started_at
            self._last_stage_metrics = {
                "batch_size": 1,
                "vae_encode_seconds": encode_seconds,
                "text_encode_seconds": text_seconds,
            }
            self_cache, cross_cache = self.denoise_stage._new_cache(
                first_frame_latent.shape[0],
                first_frame_latent.shape[-2],
                first_frame_latent.shape[-1],
            )
        session = ABotWorldInteractiveSession(
            session_id=session_id or str(uuid.uuid4()),
            prompt_emb=prompt_emb,
            first_frame_latent=first_frame_latent,
            self_cache=self_cache,
            cross_cache=cross_cache,
            scheduler=self.denoise_stage._scheduler(),
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        session.taew_decode_state = self.taew_decode_stage.create_decode_state()
        self.taew_decode_stage.warmup_first_frame(session.taew_decode_state, first_frame_latent)
        with self._lifecycle_lock:
            if session.session_id in self._interactive_sessions:
                raise ValueError(f"ABot interactive session {session.session_id!r} already exists")
            self._interactive_sessions[session.session_id] = session
        return session

    @torch.inference_mode()
    def generate_next_block(
        self,
        session: ABotWorldInteractiveSession,
        actions: Mapping[str, bool] | None = None,
        control_latent_frames: int = 3,
    ) -> list[Image.Image]:
        """Generate one block through the same batch path used by concurrent serving."""
        return self.generate_next_blocks(
            [session],
            [actions],
            control_latent_frames=control_latent_frames,
        )[0]

    @torch.inference_mode()
    def generate_next_blocks(
        self,
        sessions: Sequence[ABotWorldInteractiveSession],
        actions: Sequence[Mapping[str, bool] | None],
        *,
        control_latent_frames: int = 3,
    ) -> list[list[Image.Image]]:
        """Generate one compatible causal block for every session in one model batch."""
        if not sessions or len(sessions) != len(actions):
            raise ValueError("sessions and actions must be non-empty and have equal length")
        if control_latent_frames not in {1, 2, 3}:
            raise ValueError("control_latent_frames must be 1, 2, or 3")
        first_flags = {session.next_latent_frame == 0 for session in sessions}
        if len(first_flags) != 1:
            raise ValueError("ABot first chunks must be batched separately from continuation chunks")
        relative_rope = bool(self.denoise_stage.dit.use_relative_rope)
        if not relative_rope and len({session.next_latent_frame for session in sessions}) != 1:
            raise ValueError("Absolute-RoPE ABot sessions must share next_latent_frame")
        if len({tuple(session.first_frame_latent.shape) for session in sessions}) != 1:
            raise ValueError("ABot batch sessions must have compatible latent shapes")

        with self._lifecycle_lock:
            for session in sessions:
                if self._interactive_sessions.get(session.session_id) is not session or session.closed:
                    raise RuntimeError("ABot interactive session is no longer active")
                if not session.is_resident:
                    raise RuntimeError("ABot interactive session must be restored before generation")

        with self._execution_lock:
            batch_started_at = time.monotonic()
            for session in sessions:
                session.lifecycle = ABotWorldSessionLifecycle.ACTIVE
                session.last_activity_at = time.monotonic()
            frame_count = control_latent_frames
            noises = []
            action_contexts = []
            for session, session_actions in zip(sessions, actions):
                latent_shape = session.first_frame_latent.shape
                noises.append(
                    torch.randn(
                        (1, latent_shape[1], frame_count, latent_shape[3], latent_shape[4]),
                        generator=session.generator,
                        device=self.device,
                        dtype=torch.float32,
                    )
                )
                action_contexts.append(
                    self.build_action_context(
                        session_actions,
                        latent_frames=frame_count,
                        height=self.config.height,
                        width=self.config.width,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                )

            input_prepare_seconds = time.monotonic() - batch_started_at
            cache_collate_started_at = time.monotonic()
            original_global_ends = [
                [int(layer["global_end_index"].item()) for layer in session.self_cache]
                for session in sessions
            ]
            self_cache = self._collate_caches(sessions, "self_cache")
            cross_cache = self._collate_caches(sessions, "cross_cache")
            cache_collate_seconds = time.monotonic() - cache_collate_started_at
            start = sessions[0].next_latent_frame
            # CUDA events provide stage time without treating asynchronous kernel
            # launch latency as DiT runtime.  The final VAE event is synchronized
            # before metrics are read, while the normal stream ordering remains
            # unchanged.
            use_cuda_events = torch.device(self.device).type == "cuda"
            if use_cuda_events:
                denoise_started = torch.cuda.Event(enable_timing=True)
                denoise_finished = torch.cuda.Event(enable_timing=True)
                vae_started = torch.cuda.Event(enable_timing=True)
                vae_finished = torch.cuda.Event(enable_timing=True)
                denoise_started.record()
            else:
                denoise_started_at = time.monotonic()
            latents = self.denoise_stage._denoise_block(
                torch.cat(noises, dim=0).to(dtype=self.torch_dtype),
                torch.cat([session.prompt_emb for session in sessions], dim=0),
                torch.cat(action_contexts, dim=0),
                torch.cat([session.first_frame_latent for session in sessions], dim=0) if start == 0 else None,
                self_cache,
                cross_cache,
                start,
                [session.generator for session in sessions],
                sessions[0].scheduler,
            )
            if use_cuda_events:
                denoise_finished.record()
            else:
                denoise_seconds = time.monotonic() - denoise_started_at
            global_deltas = [
                int(layer["global_end_index"].item()) - original_global_ends[0][layer_index]
                for layer_index, layer in enumerate(self_cache)
            ]
            cache_scatter_started_at = time.monotonic()
            self._scatter_caches(sessions, "self_cache", self_cache)
            for session_index, session in enumerate(sessions):
                for layer_index, delta in enumerate(global_deltas):
                    session.self_cache[layer_index]["global_end_index"].fill_(
                        original_global_ends[session_index][layer_index] + delta
                    )
            self._scatter_caches(sessions, "cross_cache", cross_cache)
            cache_scatter_seconds = time.monotonic() - cache_scatter_started_at
            if use_cuda_events:
                vae_started.record()
            else:
                decode_started_at = time.monotonic()
            if any(session.taew_decode_state is None for session in sessions):
                raise RuntimeError("ABot session is missing its TAeW2.2 decode state")
            decoded = torch.cat(
                [
                    self.taew_decode_stage.decode_chunk(latents[index : index + 1], session.taew_decode_state)
                    for index, session in enumerate(sessions)
                ],
                dim=0,
            )
            if use_cuda_events:
                vae_finished.record()
                vae_finished.synchronize()
                denoise_seconds = denoise_started.elapsed_time(denoise_finished) / 1000.0
                decode_seconds = vae_started.elapsed_time(vae_finished) / 1000.0
            else:
                decode_seconds = time.monotonic() - decode_started_at
            postprocess_started_at = time.monotonic()
            results: list[list[Image.Image]] = []
            for batch_index, session in enumerate(sessions):
                frames = self.tensor2video(decoded[batch_index])
                session.next_latent_frame += frame_count
                session.emitted_frames += len(frames)
                results.append(frames)
            self._last_stage_metrics = {
                "batch_size": len(sessions),
                "input_prepare_seconds": input_prepare_seconds,
                "cache_collate_seconds": cache_collate_seconds,
                "denoise_seconds": denoise_seconds,
                "cache_scatter_seconds": cache_scatter_seconds,
                "vae_decode_seconds": decode_seconds,
                "postprocess_seconds": time.monotonic() - postprocess_started_at,
                "total_seconds": time.monotonic() - batch_started_at,
            }
            return results

    @staticmethod
    def _collate_caches(
        sessions: Sequence[ABotWorldInteractiveSession],
        attribute: str,
    ) -> list[dict[str, Any]]:
        cache_lists = [getattr(session, attribute) for session in sessions]
        if len({len(cache) for cache in cache_lists}) != 1:
            raise ValueError(f"ABot {attribute} layer counts do not match")
        collated: list[dict[str, Any]] = []
        for layer_index in range(len(cache_lists[0])):
            entries = [cache[layer_index] for cache in cache_lists]
            layer: dict[str, Any] = {}
            for key in entries[0]:
                values = [entry[key] for entry in entries]
                if key in {"k", "v"}:
                    layer[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], torch.Tensor):
                    scalar_values = [int(value.item()) for value in values]
                    if key != "global_end_index" and len(set(scalar_values)) != 1:
                        raise ValueError(f"ABot batch cache cursor {key!r} must match")
                    layer[key] = values[0].clone()
                else:
                    if len(set(values)) != 1:
                        raise ValueError(f"ABot batch cache metadata {key!r} must match")
                    layer[key] = values[0]
            collated.append(layer)
        return collated

    @staticmethod
    def _scatter_caches(
        sessions: Sequence[ABotWorldInteractiveSession],
        attribute: str,
        collated: list[dict[str, Any]],
    ) -> None:
        for batch_index, session in enumerate(sessions):
            cache_list = getattr(session, attribute)
            for layer_index, layer in enumerate(collated):
                for key, value in layer.items():
                    if key in {"k", "v"}:
                        cache_list[layer_index][key] = value[batch_index : batch_index + 1].detach().clone()
                    elif isinstance(value, torch.Tensor):
                        cache_list[layer_index][key] = value.detach().clone()
                    else:
                        cache_list[layer_index][key] = value

    def snapshot_interactive_session(
        self,
        session: ABotWorldInteractiveSession,
    ) -> ABotWorldSessionSnapshot:
        """Clone a quiescent session to CPU for suspend or cross-worker migration."""
        with self._execution_lock, session.lock:
            self._require_session(session)
            session.lifecycle = ABotWorldSessionLifecycle.MIGRATING
            return ABotWorldSessionSnapshot(
                session_id=session.session_id,
                prompt_emb=session.prompt_emb.detach().to("cpu").clone(),
                first_frame_latent=session.first_frame_latent.detach().to("cpu").clone(),
                self_cache=tuple(self._clone_cache_to_cpu(session.self_cache)),
                cross_cache=tuple(self._clone_cache_to_cpu(session.cross_cache)),
                vae_feat_cache=tuple(
                    value.detach().to("cpu").clone() if isinstance(value, torch.Tensor) else value
                    for value in session.vae_decode_state.feat_cache
                ),
                vae_feat_idx=tuple(session.vae_decode_state.feat_idx),
                generator_state=session.generator.get_state().to("cpu").clone(),
                next_latent_frame=session.next_latent_frame,
                emitted_frames=session.emitted_frames,
                ownership_epoch=session.ownership_epoch,
            )

    def restore_interactive_snapshot(
        self,
        snapshot: ABotWorldSessionSnapshot,
        *,
        owner_worker_id: str | None = None,
        ownership_epoch: int | None = None,
    ) -> ABotWorldInteractiveSession:
        """Install a transferred CPU snapshot as a new resident session."""
        return self._restore_snapshot(
            snapshot,
            owner_worker_id=owner_worker_id,
            ownership_epoch=ownership_epoch,
            direct_device_tensors=False,
        )

    def restore_interactive_device_snapshot(
        self,
        snapshot: ABotWorldSessionSnapshot,
        *,
        owner_worker_id: str | None = None,
        ownership_epoch: int | None = None,
    ) -> ABotWorldInteractiveSession:
        """Adopt a snapshot already received on this pipeline's CUDA device.

        This is the NCCL migration path. It preserves received target-GPU
        allocations rather than cloning them again after the direct transfer.
        """
        return self._restore_snapshot(
            snapshot,
            owner_worker_id=owner_worker_id,
            ownership_epoch=ownership_epoch,
            direct_device_tensors=True,
        )

    def _restore_snapshot(
        self,
        snapshot: ABotWorldSessionSnapshot,
        *,
        owner_worker_id: str | None,
        ownership_epoch: int | None,
        direct_device_tensors: bool,
    ) -> ABotWorldInteractiveSession:
        with self._execution_lock:
            generator = torch.Generator(device=self.device)
            generator.set_state(snapshot.generator_state)
            if direct_device_tensors:
                expected_device = torch.device(self.device)
                tensors = [snapshot.prompt_emb, snapshot.first_frame_latent]
                tensors.extend(
                    value
                    for cache in (*snapshot.self_cache, *snapshot.cross_cache)
                    for value in cache.values()
                    if isinstance(value, torch.Tensor)
                )
                tensors.extend(value for value in snapshot.vae_feat_cache if isinstance(value, torch.Tensor))
                if any(tensor.device != expected_device for tensor in tensors):
                    raise ValueError("NCCL migration tensors must already reside on the target pipeline device")
                prompt_emb = snapshot.prompt_emb
                first_frame_latent = snapshot.first_frame_latent
                self_cache = [dict(cache) for cache in snapshot.self_cache]
                cross_cache = [dict(cache) for cache in snapshot.cross_cache]
                vae_feat_cache = list(snapshot.vae_feat_cache)
            else:
                prompt_emb = snapshot.prompt_emb.to(self.device, dtype=self.torch_dtype)
                first_frame_latent = snapshot.first_frame_latent.to(self.device, dtype=self.torch_dtype)
                self_cache = self._clone_cache_to_device(snapshot.self_cache, self.device)
                cross_cache = self._clone_cache_to_device(snapshot.cross_cache, self.device)
                vae_feat_cache = [
                    value.to(self.device).clone() if isinstance(value, torch.Tensor) else value
                    for value in snapshot.vae_feat_cache
                ]
            session = ABotWorldInteractiveSession(
                session_id=snapshot.session_id,
                prompt_emb=prompt_emb,
                first_frame_latent=first_frame_latent,
                self_cache=self_cache,
                cross_cache=cross_cache,
                scheduler=self.denoise_stage._scheduler(),
                generator=generator,
                vae_decode_state=Wan22VideoVAEStreamingDecodeState(
                    feat_cache=vae_feat_cache,
                    feat_idx=list(snapshot.vae_feat_idx),
                ),
                next_latent_frame=snapshot.next_latent_frame,
                emitted_frames=snapshot.emitted_frames,
                owner_worker_id=owner_worker_id,
                ownership_epoch=snapshot.ownership_epoch + 1 if ownership_epoch is None else ownership_epoch,
            )
        with self._lifecycle_lock:
            if session.session_id in self._interactive_sessions:
                raise ValueError(f"ABot interactive session {session.session_id!r} already exists")
            self._interactive_sessions[session.session_id] = session
        return session

    @staticmethod
    def _clone_cache_to_cpu(caches: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return ABotWorldInteractivePipeline._clone_cache_to_device(caches, "cpu")

    @staticmethod
    def _clone_cache_to_device(
        caches: Sequence[dict[str, Any]],
        device: str | torch.device,
    ) -> list[dict[str, Any]]:
        return [
            {
                key: value.detach().to(device).clone() if isinstance(value, torch.Tensor) else value
                for key, value in layer.items()
            }
            for layer in caches
        ]

    def last_stage_metrics(self) -> dict[str, float | int]:
        """Return raw timings for the most recently completed model batch."""
        with self._execution_lock:
            return dict(self._last_stage_metrics)

    def suspend_interactive_session(self, session: ABotWorldInteractiveSession) -> None:
        """Move all material session tensors to CPU at a chunk boundary."""
        with self._execution_lock, session.lock:
            self._require_session(session)
            if session.lifecycle == ABotWorldSessionLifecycle.SUSPENDED:
                return
            session.prompt_emb = session.prompt_emb.to("cpu")
            session.first_frame_latent = session.first_frame_latent.to("cpu")
            self._move_cache_tensors(session.self_cache, "cpu")
            self._move_cache_tensors(session.cross_cache, "cpu")
            session.vae_decode_state.feat_cache = [
                value.to("cpu") if isinstance(value, torch.Tensor) else value
                for value in session.vae_decode_state.feat_cache
            ]
            session.lifecycle = ABotWorldSessionLifecycle.SUSPENDED

    def restore_interactive_session(self, session: ABotWorldInteractiveSession) -> None:
        """Restore a suspended session to the pipeline execution device."""
        with self._execution_lock, session.lock:
            self._require_session(session)
            if session.lifecycle != ABotWorldSessionLifecycle.SUSPENDED:
                return
            session.prompt_emb = session.prompt_emb.to(self.device, dtype=self.torch_dtype)
            session.first_frame_latent = session.first_frame_latent.to(self.device, dtype=self.torch_dtype)
            self._move_cache_tensors(session.self_cache, self.device)
            self._move_cache_tensors(session.cross_cache, self.device)
            session.vae_decode_state.feat_cache = [
                value.to(self.device) if isinstance(value, torch.Tensor) else value
                for value in session.vae_decode_state.feat_cache
            ]
            session.lifecycle = ABotWorldSessionLifecycle.READY

    @staticmethod
    def _move_cache_tensors(caches: list[dict[str, Any]], device: str | torch.device) -> None:
        for layer in caches:
            for key, value in tuple(layer.items()):
                if isinstance(value, torch.Tensor):
                    layer[key] = value.to(device)

    def _require_session(self, session: ABotWorldInteractiveSession) -> None:
        with self._lifecycle_lock:
            if self._interactive_sessions.get(session.session_id) is not session or session.closed:
                raise RuntimeError("ABot interactive session is no longer active")

    def close_interactive_session(self, session: ABotWorldInteractiveSession | None = None) -> None:
        """Release only the requested session's retained state."""
        with self._lifecycle_lock:
            targets = list(self._interactive_sessions.values()) if session is None else [session]
            for target in targets:
                if target.closed:
                    continue
                target.lifecycle = ABotWorldSessionLifecycle.CLOSING
                target.closed = True
                target.self_cache.clear()
                target.cross_cache.clear()
                target.vae_decode_state.feat_cache.clear()
                target.vae_decode_state.feat_idx = [0]
                target.lifecycle = ABotWorldSessionLifecycle.CLOSED
                self._interactive_sessions.pop(target.session_id, None)

    def close(self) -> None:
        self.close_interactive_session()
        super().close()
