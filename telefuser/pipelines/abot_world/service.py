"""TurboServe-style LiveKit service for ABot-World-0-5B-LF."""

from __future__ import annotations

import asyncio
import base64
import binascii
import gc
import io
import math
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from telefuser.pipelines.abot_world.interactive import (
    ABotWorldInteractivePipeline,
    ABotWorldInteractiveSession,
    ABotWorldSessionLifecycle,
    ABotWorldSessionSnapshot,
)
from telefuser.service.livekit.nccl_transfer import flatten_tensor_tree, rebuild_tensor_tree
from telefuser.service.livekit.turboserve import TurboServeWorkloadDetector
from telefuser.utils.logging import logger

_CONTROL_ALIASES = {
    "ArrowUp": "W",
    "ArrowDown": "S",
    "ArrowLeft": "A",
    "ArrowRight": "D",
    "KeyW": "W",
    "KeyA": "A",
    "KeyS": "S",
    "KeyD": "D",
    "KeyI": "I",
    "KeyJ": "J",
    "KeyK": "K",
    "KeyL": "L",
    "up": "W",
    "down": "S",
    "left": "A",
    "right": "D",
    "forward": "W",
    "backward": "S",
    "w": "W",
    "a": "A",
    "s": "S",
    "d": "D",
    "i": "I",
    "j": "J",
    "k": "K",
    "l": "L",
}
_VALID_CONTROLS = frozenset("WASDIJKL")
_MAX_INPUT_IMAGE_BYTES = 10 * 1024 * 1024
_DEFAULT_OUTPUT_QUEUE_SIZE = 4
_VIDEO_OUTPUT_TYPES = frozenset({"preview", "chunk"})
_TERMINAL_OUTPUT_TYPES = frozenset({"error", "done"})
_PACING_SAFETY_FACTOR = 1.10
_PACING_MAX_COALESCING_SECONDS = 0.010
_PACING_RENDEZVOUS_WAKE_GUARD_SECONDS = 0.001


@dataclass
class _ABotWorldLiveKitSession:
    session_id: str
    pipeline_session: ABotWorldInteractiveSession
    output_queue: queue.Queue[dict[str, Any]]
    control_event: threading.Event
    config: dict[str, Any]
    control_idle_timeout: float = 10.0
    controls: set[str] = field(default_factory=set)
    last_control_at: float = field(default_factory=time.monotonic)
    next_chunk_index: int = 0
    active: bool = True
    worker: threading.Thread | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    in_flight: bool = False
    ready_since: float | None = None
    next_playout_deadline: float = field(default_factory=time.monotonic)
    created_at: float = field(default_factory=time.monotonic)
    output_queue_high_watermark: int = 0
    dropped_video_payloads: int = 0
    dropped_status_payloads: int = 0
    scheduled_chunks: int = 0
    batch_items: int = 0
    total_queue_wait_seconds: float = 0.0
    total_compute_seconds: float = 0.0
    last_compute_seconds: float = 0.0
    last_chunk_duration_seconds: float = 0.0
    pacing_ready_at: float = field(default_factory=time.monotonic)
    last_error: str | None = None
    migrating: bool = False


@dataclass(frozen=True)
class ABotWorldMigrationBundle:
    """Quiescent service and pipeline state transferred to another ABot worker."""

    snapshot: ABotWorldSessionSnapshot
    config: dict[str, Any]
    controls: frozenset[str]
    control_idle_timeout: float
    last_control_at: float
    next_chunk_index: int
    next_playout_deadline: float


class ABotWorldLiveKitService:
    """One-GPU retained-session owner with coalesced causal-block scheduling.

    By default, compatible ready sessions are coalesced into one DiT invocation
    and their independent KV/RNG state is scattered back afterwards. This is
    the execution model described by TurboServe's paper and is the production
    ABot serving baseline. ``round_robin`` remains available as an ablation: it
    advances exactly one runnable session per scheduler turn.
    """

    def __init__(
        self,
        pipeline: ABotWorldInteractivePipeline,
        *,
        default_fps: int = 12,
        default_session_config: Mapping[str, object] | None = None,
        output_queue_size: int = _DEFAULT_OUTPUT_QUEUE_SIZE,
        control_idle_timeout: float = 10.0,
        close_timeout: float = 300.0,
        max_batch_size: int = 8,
        batching_window_ms: float = 2.0,
        idle_suspension_seconds: float = 5.0,
        scheduler_mode: str = "batched",
    ) -> None:
        if default_fps < 1:
            raise ValueError(f"default_fps must be positive, got {default_fps}")
        if output_queue_size < 1:
            raise ValueError(f"output_queue_size must be positive, got {output_queue_size}")
        if control_idle_timeout <= 0 or close_timeout <= 0:
            raise ValueError("control_idle_timeout and close_timeout must be positive")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if batching_window_ms < 0 or idle_suspension_seconds <= 0:
            raise ValueError("batching_window_ms must be non-negative and idle_suspension_seconds positive")
        if scheduler_mode not in {"round_robin", "batched"}:
            raise ValueError("scheduler_mode must be 'round_robin' or 'batched'")
        self.pipeline = pipeline
        self.default_fps = int(default_fps)
        self.default_session_config = dict(default_session_config or {})
        self.output_queue_size = int(output_queue_size)
        self.control_idle_timeout = float(control_idle_timeout)
        self.close_timeout = float(close_timeout)
        self.max_batch_size = int(max_batch_size)
        self.batching_window_seconds = float(batching_window_ms) / 1000.0
        self.idle_suspension_seconds = float(idle_suspension_seconds)
        self.scheduler_mode = scheduler_mode
        self._sessions: dict[str, _ABotWorldLiveKitSession] = {}
        self._round_robin_order: deque[str] = deque()
        self._sessions_lock = threading.RLock()
        self._scheduler_condition = threading.Condition(self._sessions_lock)
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_stopping = False
        self._capacity_profile: dict[str, object] | None = None
        self._scheduler_paused = False
        self._batch_count = 0
        self._batch_item_count = 0
        self._maximum_batch_size = 0
        self._last_stage_metrics: dict[str, float | int] = {}
        self._pacing_eligible_sessions = 0
        self._pacing_throttled_sessions = 0
        self._pacing_buffered_sessions = 0
        # Observed wall-clock runtimes make deadline rendezvous conservative.
        self._batch_compute_estimates: dict[int, float] = {}
        self._workload_detector = TurboServeWorkloadDetector()

    def start(self) -> None:
        """Preload weights and start the sole GPU scheduling thread."""
        self.pipeline.preload_models()
        self._ensure_scheduler_started()
        logger.info("ABotWorldLiveKitService TurboServe scheduler started")

    def stop(self, *, close_pipeline: bool = True) -> None:
        """Stop admission and release retained sessions.

        Offline experiment suites may keep a preloaded pipeline alive across
        independent scheduler instances by passing ``close_pipeline=False``.
        Production callers retain the original full teardown by default.
        """
        with self._scheduler_condition:
            self._scheduler_stopping = True
            self._scheduler_condition.notify_all()
        scheduler = self._scheduler_thread
        if scheduler is not None and scheduler.is_alive() and scheduler is not threading.current_thread():
            scheduler.join(timeout=self.close_timeout)
        with self._sessions_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.close_session(session_id)
        if close_pipeline:
            self.pipeline.close()

    def configure_session_capacity(self, max_sessions: int | None = None) -> dict[str, object]:
        """Estimate retained-session capacity from free memory and ABot cache geometry."""
        if max_sessions is not None and max_sessions < 1:
            raise ValueError(f"max_sessions must be positive when provided, got {max_sessions}")
        with self._sessions_lock:
            if self._sessions:
                raise RuntimeError("cannot configure retained-session capacity while sessions are active")
            if self._capacity_profile is not None:
                if self._capacity_profile["configured_limit"] != max_sessions:
                    raise RuntimeError("ABot retained-session capacity is already configured with another limit")
                return dict(self._capacity_profile)

        per_session_bytes = self._estimate_session_bytes()
        workspace_peak_bytes = 0
        profiled_session_bytes = 0
        free_bytes = 0
        pipeline_device = torch.device(getattr(self.pipeline, "device", "cpu"))
        if torch.cuda.is_available() and pipeline_device.type == "cuda":
            profile = self._profile_session_memory()
            profiled_session_bytes = int(profile["profiled_session_bytes"])
            workspace_peak_bytes = int(profile["workspace_peak_bytes"])
            per_session_bytes = max(per_session_bytes, profiled_session_bytes)
            free_bytes, _ = torch.cuda.mem_get_info(pipeline_device)
        memory_budget = max(0, int(free_bytes * 0.90))
        if free_bytes:
            computed_capacity = 0
            for candidate in range(1, 65):
                active_batch = min(candidate, self.max_batch_size) if self.scheduler_mode == "batched" else 1
                required_bytes = candidate * per_session_bytes + active_batch * workspace_peak_bytes
                if required_bytes > memory_budget:
                    break
                computed_capacity = candidate
            # A successful real-session warmup proves that one session can run even when the
            # conservative 10% allocator reserve makes the arithmetic round below one.
            computed_capacity = max(1, computed_capacity)
        else:
            computed_capacity = max_sessions or 1
        effective_capacity = min(computed_capacity, max_sessions) if max_sessions is not None else computed_capacity
        effective_batch_size = min(self.max_batch_size, effective_capacity) if self.scheduler_mode == "batched" else 1
        profile: dict[str, object] = {
            "configured_limit": max_sessions,
            "effective_capacity": effective_capacity,
            "computed_capacity": computed_capacity,
            "model": "ABot-World-0-5B-LF",
            "free_device_bytes": int(free_bytes),
            "memory_budget_bytes": memory_budget,
            "estimated_session_bytes": per_session_bytes,
            "profiled_session_bytes": profiled_session_bytes,
            "workspace_peak_bytes": workspace_peak_bytes,
            "estimated_batch_workspace_bytes": effective_batch_size * workspace_peak_bytes,
            "max_batch_size": self.max_batch_size,
            "effective_max_batch_size": effective_batch_size,
            "scheduler_mode": self.scheduler_mode,
        }
        self._capacity_profile = profile
        return dict(profile)

    def _estimate_session_bytes(self) -> int:
        dit = self.pipeline.denoise_stage.dit
        latent_height = self.pipeline.config.height // 16
        latent_width = self.pipeline.config.width // 16
        frame_tokens = (latent_height // dit.patch_size[1]) * (latent_width // dit.patch_size[2])
        head_dim = dit.dim // dit.num_heads
        element_size = torch.empty((), dtype=self.pipeline.torch_dtype).element_size()
        self_cache = 2 * dit.num_layers * dit.local_attn_size * frame_tokens * dit.num_heads * head_dim * element_size
        cross_cache = 2 * dit.num_layers * dit.text_len * dit.num_heads * head_dim * element_size
        prompt = dit.text_len * dit.text_dim * element_size if hasattr(dit, "text_dim") else 0
        # Reserve another 35% for VAE temporal state, latents, allocator fragmentation, and runtime workspaces.
        return max(1, math.ceil((self_cache + cross_cache + prompt) * 1.35))

    def _profile_session_memory(self) -> dict[str, int]:
        """Warm one real chunk and measure retained state separately from workspace peaks."""
        device = torch.device(self.pipeline.device)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        session: ABotWorldInteractiveSession | None = None
        try:
            image = self._load_image(self.default_session_config)
            prompt = str(self.default_session_config.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("ABot capacity warmup requires the default prompt")
            session = self.pipeline.create_interactive_session(
                image,
                prompt,
                seed=int(self.default_session_config.get("seed", 42)),
                session_id="__abot_capacity_warmup__",
            )
            self.pipeline.generate_next_block(
                session,
                {"W": True},
                control_latent_frames=int(self.default_session_config.get("control_latent_frames", 3)),
            )
            torch.cuda.synchronize(device)
            retained = max(1, torch.cuda.memory_allocated(device) - baseline)
            peak = max(retained, torch.cuda.max_memory_allocated(device) - baseline)
            return {
                "profiled_session_bytes": int(retained),
                "workspace_peak_bytes": int(max(0, peak - retained)),
            }
        except Exception:
            logger.exception("ABot session capacity warmup failed; using analytical memory estimate")
            return {"profiled_session_bytes": 0, "workspace_peak_bytes": 0}
        finally:
            if session is not None:
                self.pipeline.close_interactive_session(session)
            gc.collect()
            torch.cuda.empty_cache()

    def session_capacity_profile(self) -> dict[str, object] | None:
        return dict(self._capacity_profile) if self._capacity_profile is not None else None

    def has_session(self, session_id: str) -> bool:
        with self._sessions_lock:
            return session_id in self._sessions

    def create_session(self, config: dict) -> str:
        """Create a preview-only retained session; controls make it scheduler-ready."""
        self._ensure_scheduler_started()
        with self._sessions_lock:
            capacity = int(self._capacity_profile["effective_capacity"]) if self._capacity_profile else 1
            if len(self._sessions) >= capacity:
                raise RuntimeError(f"ABot retained-session capacity is exhausted (capacity={capacity})")

        session_id = str(config.get("session_id") or uuid.uuid4())
        image = self._load_image(config)
        prompt = str(config.get("prompt", self.default_session_config.get("prompt", ""))).strip()
        if not prompt:
            raise ValueError("ABot-World requires a non-empty prompt")
        fps = int(config.get("fps", self.default_session_config.get("fps", self.default_fps)))
        if fps < 1:
            raise ValueError(f"fps must be positive, got {fps}")
        session_idle_timeout = float(config.get("control_idle_timeout", self.control_idle_timeout))
        if session_idle_timeout <= 0:
            raise ValueError(f"control_idle_timeout must be positive, got {session_idle_timeout}")
        control_latent_frames = int(
            config.get("control_latent_frames", self.default_session_config.get("control_latent_frames", 3))
        )
        if control_latent_frames not in {1, 2, 3}:
            raise ValueError("control_latent_frames must be 1, 2, or 3")
        delivery_mode = str(config.get("delivery_mode", self.default_session_config.get("delivery_mode", "latest")))
        if delivery_mode not in {"latest", "lossless"}:
            raise ValueError("delivery_mode must be 'latest' or 'lossless'")
        seed = int(config.get("seed", self.default_session_config.get("seed", 42)))
        pipeline_session = self.pipeline.create_interactive_session(image, prompt, seed=seed, session_id=session_id)
        state = _ABotWorldLiveKitSession(
            session_id=session_id,
            pipeline_session=pipeline_session,
            output_queue=queue.Queue(maxsize=self.output_queue_size),
            control_event=threading.Event(),
            config={
                "fps": fps,
                "control_latent_frames": control_latent_frames,
                "delivery_mode": delivery_mode,
            },
            control_idle_timeout=session_idle_timeout,
        )
        with self._scheduler_condition:
            if session_id in self._sessions:
                self.pipeline.close_interactive_session(pipeline_session)
                raise ValueError(f"ABot session {session_id!r} already exists")
            capacity = int(self._capacity_profile["effective_capacity"]) if self._capacity_profile else 1
            if len(self._sessions) >= capacity:
                self.pipeline.close_interactive_session(pipeline_session)
                raise RuntimeError(f"ABot retained-session capacity is exhausted (capacity={capacity})")
            self._sessions[session_id] = state
            self._round_robin_order.append(session_id)
        self._workload_detector.record_arrival(session_id)

        preview = image.convert("RGB").resize(
            (self.pipeline.config.width, self.pipeline.config.height),
            Image.Resampling.BICUBIC,
        )
        self._put_output(
            state,
            {"type": "preview", "index": -1, "fps": fps, "timestamp": time.time(), "frames": [preview]},
        )
        return session_id

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        """Apply a control message and wake the event-driven scheduler."""
        state = self._session(session_id)
        if state is None:
            return
        message_type = str(chunk.get("type", ""))
        with self._scheduler_condition, state.lock:
            if not state.active:
                return
            was_controlled = bool(state.controls)
            if message_type == "stop":
                state.active = False
                state.controls.clear()
                state.pipeline_session.lifecycle = ABotWorldSessionLifecycle.CLOSING
                self._scheduler_condition.notify_all()
                return
            if message_type == "control_state":
                raw_controls = chunk.get("controls", [])
                if not isinstance(raw_controls, list):
                    raise ValueError("control_state controls must be a list")
                state.controls = self._canonical_controls(raw_controls)
            elif message_type == "control":
                control = self._canonical_control(chunk.get("control", chunk.get("key")))
                event = str(chunk.get("event") or chunk.get("action") or "press").lower()
                if event in {"reset", "reset_pose"}:
                    state.controls.clear()
                elif event == "press":
                    state.controls.add(control)
                else:
                    state.controls.discard(control)
            elif message_type in {"reset", "prompt"}:
                state.controls.clear()
            else:
                raise ValueError(f"Unsupported ABot control message type: {message_type}")
            now = time.monotonic()
            state.last_control_at = now
            state.ready_since = now if state.controls else None
            # A newly reactivated session should not wait behind an old playout
            # prediction. Existing active sessions keep their pacing deadline so
            # frequent control updates cannot force the scheduler to free-run.
            if state.controls and not was_controlled:
                state.pacing_ready_at = now
            state.control_event.set()
            if state.controls:
                self._workload_detector.record_active(session_id, now)
            else:
                self._workload_detector.record_idle(session_id, now)
            self._scheduler_condition.notify_all()

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        """Yield preview and generated frames in per-session sequence order."""
        state = self._session(session_id)
        if state is None:
            return
        while True:
            try:
                payload = await asyncio.to_thread(state.output_queue.get, True, 0.25)
            except queue.Empty:
                with state.lock:
                    if not state.active:
                        return
                continue
            with self._scheduler_condition:
                self._scheduler_condition.notify_all()
            yield payload

    def close_session(self, session_id: str, timeout: float | None = None) -> None:
        """Wait for a chunk boundary, then release only this session's state."""
        effective_timeout = self.close_timeout if timeout is None else timeout
        deadline = time.monotonic() + effective_timeout
        with self._scheduler_condition:
            state = self._sessions.get(session_id)
            if state is None:
                return
            with state.lock:
                state.active = False
                state.controls.clear()
                state.ready_since = None
            self._scheduler_condition.notify_all()
            while state.in_flight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("ABot session did not reach a chunk boundary before close timeout: %s", session_id)
                    return
                self._scheduler_condition.wait(remaining)
            self._sessions.pop(session_id, None)
            self._discard_from_round_robin(session_id)
        self._workload_detector.record_departure(session_id)
        self.pipeline.close_interactive_session(state.pipeline_session)

    def prepare_migration(
        self,
        session_id: str,
        timeout: float | None = None,
    ) -> ABotWorldMigrationBundle:
        """Quiesce one session at a chunk boundary and return a CPU snapshot."""
        state = self._quiesce_migration(session_id, timeout)
        with self._scheduler_condition:
            snapshot = self.pipeline.snapshot_interactive_session(state.pipeline_session)
            return ABotWorldMigrationBundle(
                snapshot=snapshot,
                config=dict(state.config),
                controls=frozenset(state.controls),
                control_idle_timeout=state.control_idle_timeout,
                last_control_at=state.last_control_at,
                next_chunk_index=state.next_chunk_index,
                next_playout_deadline=state.next_playout_deadline,
            )

    def prepare_migration_nccl_metadata(self, session_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Quiesce a session and describe its resident tensors for direct NCCL transfer."""
        state = self._quiesce_migration(session_id, timeout)
        session = state.pipeline_session
        if session.taew_decode_state is None:
            raise RuntimeError("ABot session is missing its TAeW2.2 decode state")
        payload = {
            "prompt_emb": session.prompt_emb,
            "first_frame_latent": session.first_frame_latent,
            "self_cache": session.self_cache,
            "cross_cache": session.cross_cache,
            "vae_feat_cache": session.vae_decode_state.feat_cache,
            "taew_decode_state": self.pipeline.taew_decode_stage.export_decode_state_for_nccl(
                session.taew_decode_state
            ),
        }
        skeleton, manifest, leaves = flatten_tensor_tree(payload)
        return {
            "session_id": session_id,
            "tensor_skeleton": skeleton,
            "tensor_manifest": manifest,
            "generator_state": session.generator.get_state().detach().cpu(),
            "vae_feat_idx": list(session.vae_decode_state.feat_idx),
            "next_latent_frame": session.next_latent_frame,
            "emitted_frames": session.emitted_frames,
            "ownership_epoch": session.ownership_epoch,
            "config": dict(state.config),
            "controls": sorted(state.controls),
            "control_idle_timeout": state.control_idle_timeout,
            "last_control_at": state.last_control_at,
            "next_chunk_index": state.next_chunk_index,
            "next_playout_deadline": state.next_playout_deadline,
            "state_bytes": sum(value.numel() * value.element_size() for value in leaves.values()),
            "_nccl_tensor_leaves": leaves,
        }

    def import_migration_nccl(
        self,
        metadata: Mapping[str, Any],
        tensor_leaves: Mapping[tuple[Any, ...], torch.Tensor],
        *,
        owner_worker_id: str | None = None,
        ownership_epoch: int | None = None,
    ) -> str:
        """Install target-GPU tensors received by NCCL without a CPU snapshot copy."""
        payload = rebuild_tensor_tree(metadata["tensor_skeleton"], dict(tensor_leaves))
        snapshot = ABotWorldSessionSnapshot(
            session_id=str(metadata["session_id"]),
            prompt_emb=payload["prompt_emb"],
            first_frame_latent=payload["first_frame_latent"],
            self_cache=tuple(payload["self_cache"]),
            cross_cache=tuple(payload["cross_cache"]),
            vae_feat_cache=tuple(payload["vae_feat_cache"]),
            vae_feat_idx=tuple(int(value) for value in metadata["vae_feat_idx"]),
            taew_decode_state=dict(payload["taew_decode_state"]),
            generator_state=metadata["generator_state"],
            next_latent_frame=int(metadata["next_latent_frame"]),
            emitted_frames=int(metadata["emitted_frames"]),
            ownership_epoch=int(metadata["ownership_epoch"]),
        )
        bundle = ABotWorldMigrationBundle(
            snapshot=snapshot,
            config=dict(metadata["config"]),
            controls=frozenset(metadata["controls"]),
            control_idle_timeout=float(metadata["control_idle_timeout"]),
            last_control_at=float(metadata["last_control_at"]),
            next_chunk_index=int(metadata["next_chunk_index"]),
            next_playout_deadline=float(metadata["next_playout_deadline"]),
        )
        self._ensure_scheduler_started()
        with self._scheduler_condition:
            capacity = int(self._capacity_profile["effective_capacity"]) if self._capacity_profile else 1
            if len(self._sessions) >= capacity:
                raise RuntimeError(f"ABot retained-session capacity is exhausted (capacity={capacity})")
            if snapshot.session_id in self._sessions:
                raise ValueError(f"ABot session {snapshot.session_id!r} already exists")
        pipeline_session = self.pipeline.restore_interactive_device_snapshot(
            bundle.snapshot,
            owner_worker_id=owner_worker_id,
            ownership_epoch=ownership_epoch,
        )
        state = _ABotWorldLiveKitSession(
            session_id=snapshot.session_id,
            pipeline_session=pipeline_session,
            output_queue=queue.Queue(maxsize=self.output_queue_size),
            control_event=threading.Event(),
            config=dict(bundle.config),
            control_idle_timeout=bundle.control_idle_timeout,
            controls=set(bundle.controls),
            last_control_at=bundle.last_control_at,
            next_chunk_index=bundle.next_chunk_index,
            next_playout_deadline=bundle.next_playout_deadline,
            ready_since=time.monotonic() if bundle.controls else None,
        )
        with self._scheduler_condition:
            if snapshot.session_id in self._sessions:
                self.pipeline.close_interactive_session(pipeline_session)
                raise ValueError(f"ABot session {snapshot.session_id!r} already exists")
            self._sessions[state.session_id] = state
            self._round_robin_order.append(state.session_id)
            self._scheduler_condition.notify_all()
        return state.session_id

    def _quiesce_migration(self, session_id: str, timeout: float | None) -> _ABotWorldLiveKitSession:
        effective_timeout = self.close_timeout if timeout is None else timeout
        deadline = time.monotonic() + effective_timeout
        with self._scheduler_condition:
            state = self._sessions.get(session_id)
            if state is None:
                raise KeyError(f"Unknown ABot session {session_id!r}")
            state.migrating = True
            self._scheduler_condition.notify_all()
            while state.in_flight or not state.output_queue.empty():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    state.migrating = False
                    raise TimeoutError("Timed out waiting for ABot migration chunk boundary and output drain")
                self._scheduler_condition.wait(remaining)
            return state

    def import_migration(
        self,
        bundle: ABotWorldMigrationBundle,
        *,
        owner_worker_id: str | None = None,
        ownership_epoch: int | None = None,
    ) -> str:
        """Install a prepared migration bundle without emitting another preview."""
        self._ensure_scheduler_started()
        with self._scheduler_condition:
            capacity = int(self._capacity_profile["effective_capacity"]) if self._capacity_profile else 1
            if len(self._sessions) >= capacity:
                raise RuntimeError(f"ABot retained-session capacity is exhausted (capacity={capacity})")
            if bundle.snapshot.session_id in self._sessions:
                raise ValueError(f"ABot session {bundle.snapshot.session_id!r} already exists")
        pipeline_session = self.pipeline.restore_interactive_snapshot(
            bundle.snapshot,
            owner_worker_id=owner_worker_id,
            ownership_epoch=ownership_epoch,
        )
        state = _ABotWorldLiveKitSession(
            session_id=bundle.snapshot.session_id,
            pipeline_session=pipeline_session,
            output_queue=queue.Queue(maxsize=self.output_queue_size),
            control_event=threading.Event(),
            config=dict(bundle.config),
            control_idle_timeout=bundle.control_idle_timeout,
            controls=set(bundle.controls),
            last_control_at=bundle.last_control_at,
            next_chunk_index=bundle.next_chunk_index,
            next_playout_deadline=bundle.next_playout_deadline,
            ready_since=time.monotonic() if bundle.controls else None,
        )
        with self._scheduler_condition:
            if bundle.snapshot.session_id in self._sessions:
                self.pipeline.close_interactive_session(pipeline_session)
                raise ValueError(f"ABot session {bundle.snapshot.session_id!r} already exists")
            self._sessions[state.session_id] = state
            self._round_robin_order.append(state.session_id)
            self._scheduler_condition.notify_all()
        return state.session_id

    def commit_migration(self, session_id: str) -> None:
        """Release a source session after target installation and ownership commit."""
        with self._scheduler_condition:
            state = self._sessions.get(session_id)
            if state is None:
                return
            if not state.migrating or state.in_flight:
                raise RuntimeError("ABot source session is not quiescent for migration commit")
            self._sessions.pop(session_id)
            self._discard_from_round_robin(session_id)
        self.pipeline.close_interactive_session(state.pipeline_session)

    def abort_migration(self, session_id: str) -> None:
        """Resume source scheduling when target installation or ownership commit fails."""
        with self._scheduler_condition:
            state = self._sessions.get(session_id)
            if state is None:
                return
            state.migrating = False
            if state.pipeline_session.lifecycle == ABotWorldSessionLifecycle.MIGRATING:
                state.pipeline_session.lifecycle = ABotWorldSessionLifecycle.READY
            state.ready_since = time.monotonic() if state.controls else None
            self._scheduler_condition.notify_all()

    def pause_scheduler(self, timeout: float | None = None) -> None:
        """Stop selecting new batches and wait for current work to reach a boundary."""
        deadline = time.monotonic() + (self.close_timeout if timeout is None else timeout)
        with self._scheduler_condition:
            self._scheduler_paused = True
            self._scheduler_condition.notify_all()
            while any(state.in_flight for state in self._sessions.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._scheduler_paused = False
                    self._scheduler_condition.notify_all()
                    raise TimeoutError("Timed out pausing ABot scheduler at a chunk boundary")
                self._scheduler_condition.wait(remaining)

    def resume_scheduler(self) -> None:
        """Resume admission of ready ABot session batches after a control transaction."""
        with self._scheduler_condition:
            self._scheduler_paused = False
            self._scheduler_condition.notify_all()

    def runtime_metrics(self, session_id: str | None = None) -> dict[str, float | int]:
        """Return raw scheduler facts for service metadata and benchmarks."""
        with self._sessions_lock:
            if session_id is None:
                workload = self._workload_detector.snapshot()
                return {
                    "scheduler_mode": self.scheduler_mode,
                    "sessions": len(self._sessions),
                    "batches": self._batch_count,
                    "batch_items": self._batch_item_count,
                    "maximum_batch_size": self._maximum_batch_size,
                    "pacing_eligible_sessions": self._pacing_eligible_sessions,
                    "pacing_throttled_sessions": self._pacing_throttled_sessions,
                    "pacing_buffered_sessions": self._pacing_buffered_sessions,
                    "active_sessions": workload.active_sessions,
                    "arrivals_per_second": round(workload.arrivals_per_second, 6),
                    "activation_volatility": round(workload.activation_volatility, 6),
                    "mean_chunk_seconds": round(workload.mean_chunk_seconds, 6),
                    "p95_chunk_seconds": round(workload.p95_chunk_seconds, 6),
                    **self._last_stage_metrics,
                }
            state = self._sessions[session_id]
            with state.lock:
                return {
                    "scheduler_mode": self.scheduler_mode,
                    "scheduled_chunks": state.scheduled_chunks,
                    "batch_items": state.batch_items,
                    "output_queue_high_watermark": state.output_queue_high_watermark,
                    "dropped_video_payloads": state.dropped_video_payloads,
                    "dropped_status_payloads": state.dropped_status_payloads,
                    "active": int(bool(state.controls) and state.active),
                    "in_flight": int(state.in_flight),
                    "resident": int(state.pipeline_session.is_resident),
                    "emitted_frames": int(getattr(state.pipeline_session, "emitted_frames", 0)),
                    "total_queue_wait_seconds": round(state.total_queue_wait_seconds, 6),
                    "total_compute_seconds": round(state.total_compute_seconds, 6),
                    "pacing_ready_in_seconds": round(max(0.0, state.pacing_ready_at - time.monotonic()), 6),
                    "pacing_buffered_video_payloads": self._queued_video_payloads(state),
                }

    def _ensure_scheduler_started(self) -> None:
        with self._scheduler_condition:
            if self._scheduler_thread is not None and self._scheduler_thread.is_alive():
                return
            if self._scheduler_stopping:
                raise RuntimeError("ABot scheduler is stopping")
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="abot-world-turboserve",
            )
            self._scheduler_thread.start()

    def _scheduler_loop(self) -> None:
        while True:
            suspend_candidate: _ABotWorldLiveKitSession | None = None
            with self._scheduler_condition:
                if self._scheduler_stopping:
                    return
                now = time.monotonic()
                if self._scheduler_paused:
                    self._scheduler_condition.wait(timeout=0.05)
                    continue
                for state in self._sessions.values():
                    with state.lock:
                        if state.controls and now - state.last_control_at >= state.control_idle_timeout:
                            state.controls.clear()
                            state.ready_since = None
                            state.pipeline_session.lifecycle = ABotWorldSessionLifecycle.IDLE
                        if (
                            not state.controls
                            and not state.in_flight
                            and state.pipeline_session.is_resident
                            and now - state.last_control_at >= self.idle_suspension_seconds
                        ):
                            suspend_candidate = state
                            break
                ready = self._ready_sessions(now)
                if not ready and suspend_candidate is None:
                    self._scheduler_condition.wait(timeout=self._next_scheduler_wake_seconds(now))
                    continue
                if self.scheduler_mode == "batched" and ready and len(ready) < self.max_batch_size:
                    wait_seconds = self._batch_formation_wait_seconds(ready, now)
                    if wait_seconds > 0:
                        self._scheduler_condition.wait(timeout=wait_seconds)
                    now = time.monotonic()
                    ready = self._ready_sessions(now)
                batch = self._select_batch(ready, now=now)
                controls: list[dict[str, bool]] = []
                if batch:
                    for state in batch:
                        with state.lock:
                            state.in_flight = True
                            controls.append({key: True for key in state.controls})

            if not batch:
                if suspend_candidate is not None:
                    try:
                        self.pipeline.suspend_interactive_session(suspend_candidate.pipeline_session)
                    except Exception:
                        logger.exception("Failed to suspend ABot session %s", suspend_candidate.session_id)
                continue
            self._execute_batch(batch, controls)

    def _ready_sessions(self, now: float) -> list[_ABotWorldLiveKitSession]:
        ready: list[_ABotWorldLiveKitSession] = []
        pacing_eligible = 0
        pacing_throttled = 0
        pacing_buffered = 0
        for state in self._sessions.values():
            with state.lock:
                lossless_blocked = (
                    state.config["delivery_mode"] == "lossless" and state.output_queue.full()
                )
                if (
                    state.active
                    and state.controls
                    and not state.in_flight
                    and not state.migrating
                    and not lossless_blocked
                ):
                    if state.ready_since is None:
                        state.ready_since = now
                    if state.config["delivery_mode"] == "latest":
                        buffered_video_payloads = self._queued_video_payloads(state)
                        pacing_ready = now + self._pacing_coalescing_slack_seconds(state) >= state.pacing_ready_at
                        if buffered_video_payloads or not pacing_ready:
                            pacing_throttled += 1
                            pacing_buffered += int(bool(buffered_video_payloads))
                            continue
                        pacing_eligible += 1
                    ready.append(state)
        self._pacing_eligible_sessions = pacing_eligible
        self._pacing_throttled_sessions = pacing_throttled
        self._pacing_buffered_sessions = pacing_buffered
        ready.sort(key=lambda state: (state.next_playout_deadline, state.ready_since or now, state.session_id))
        return ready

    def _next_scheduler_wake_seconds(self, now: float) -> float:
        """Wake promptly for a pacing deadline while retaining event-driven idling."""
        next_wake_at: float | None = None
        for state in self._sessions.values():
            with state.lock:
                if state.active and state.controls:
                    control_expiry = state.last_control_at + state.control_idle_timeout
                    next_wake_at = control_expiry if next_wake_at is None else min(next_wake_at, control_expiry)
                    if (
                        state.config["delivery_mode"] == "latest"
                        and not state.in_flight
                        and not state.migrating
                        and not self._queued_video_payloads(state)
                    ):
                        pacing_wake = state.pacing_ready_at - self._pacing_coalescing_slack_seconds(state)
                        next_wake_at = pacing_wake if next_wake_at is None else min(next_wake_at, pacing_wake)
                elif not state.in_flight and state.pipeline_session.is_resident and not state.controls:
                    suspension_at = state.last_control_at + self.idle_suspension_seconds
                    next_wake_at = suspension_at if next_wake_at is None else min(next_wake_at, suspension_at)
        if next_wake_at is None:
            return 0.05
        return max(0.001, next_wake_at - now)

    def _pacing_coalescing_slack_seconds(self, state: _ABotWorldLiveKitSession) -> float:
        """Allow a small, bounded early window so near-aligned sessions still batch."""
        if state.last_chunk_duration_seconds <= 0:
            return self.batching_window_seconds
        return max(
            self.batching_window_seconds,
            min(_PACING_MAX_COALESCING_SECONDS, state.last_chunk_duration_seconds * 0.05),
        )

    def _batch_formation_wait_seconds(
        self,
        ready: Sequence[_ABotWorldLiveKitSession],
        now: float,
    ) -> float:
        """Wait for a compatible continuation only while all playout deadlines are safe."""
        batch = self._select_batch(ready, now=now)
        if not batch or len(batch) >= self.max_batch_size:
            return 0.0

        # First generated chunks stay latency-critical. They may coalesce when
        # peers are ready in the same turn, but never wait for a future peer.
        if any(state.scheduled_chunks == 0 for state in batch):
            return 0.0
        # Lossless queues are governed by consumer backpressure rather than a
        # playout deadline, so preserve their original micro-batching behavior.
        if any(state.config["delivery_mode"] != "latest" for state in batch):
            return self.batching_window_seconds
        legacy_wait = min(
            self.batching_window_seconds,
            max(0.0, self._latest_safe_batch_start(batch) - now),
        )

        pivot_key = self._batch_key(batch[0])
        selected_ids = {state.session_id for state in batch}
        rendezvous_waits: list[float] = []
        for candidate in self._sessions.values():
            if candidate.session_id in selected_ids:
                continue
            with candidate.lock:
                if (
                    not candidate.active
                    or not candidate.controls
                    or candidate.in_flight
                    or candidate.migrating
                    or candidate.config["delivery_mode"] != "latest"
                    or candidate.scheduled_chunks == 0
                    or self._batch_key(candidate) != pivot_key
                    # A generated chunk owned by the publisher has an external
                    # dequeue time, so it cannot be a rendezvous promise.
                    or self._queued_video_payloads(candidate)
                ):
                    continue

                release_at = candidate.pacing_ready_at - self._pacing_coalescing_slack_seconds(candidate)
                if release_at <= now:
                    # An already-eligible session should be in ready. Avoid
                    # turning a state race into an extra scheduler delay.
                    continue
                proposed_batch = [*batch, candidate]
                latest_safe_start = self._latest_safe_batch_start(proposed_batch)
                if release_at + _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS <= latest_safe_start:
                    rendezvous_waits.append(release_at - now + _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS)

        if rendezvous_waits:
            # Earliest compatible release minimizes queueing; the condition
            # wait is followed by a full readiness/deadline revalidation.
            return min(rendezvous_waits)
        return legacy_wait

    def _latest_safe_batch_start(self, batch: Sequence[_ABotWorldLiveKitSession]) -> float:
        """Return the latest launch time that still meets every playout deadline."""
        if not batch:
            return float("-inf")
        predicted_compute_seconds = self._estimated_batch_compute_seconds(batch)
        return min(state.next_playout_deadline for state in batch) - predicted_compute_seconds

    def _estimated_batch_compute_seconds(self, batch: Sequence[_ABotWorldLiveKitSession]) -> float:
        """Conservatively estimate batch wall time from observed service work."""
        batch_size = len(batch)
        observed = self._batch_compute_estimates.get(batch_size)
        if observed is None:
            singleton_seconds = max((state.last_compute_seconds for state in batch), default=0.0)
            observed = singleton_seconds * batch_size
        return observed * _PACING_SAFETY_FACTOR

    @staticmethod
    def _queued_video_payloads(state: _ABotWorldLiveKitSession) -> int:
        """Return generated chunks not yet taken by the real-time publisher.

        ``latest`` output intentionally evicts stale payloads, so Queue.full()
        cannot express a prefetch bound. The preview is not generated video and
        must not delay the latency-critical first chunk. Keeping at most one
        queued chunk makes it the only chunk ahead of the one being played.
        """
        with state.output_queue.mutex:
            return sum(item.get("type") == "chunk" for item in state.output_queue.queue)

    def _select_batch(
        self,
        ready: Sequence[_ABotWorldLiveKitSession],
        *,
        now: float | None = None,
    ) -> list[_ABotWorldLiveKitSession]:
        if not ready:
            return []
        if self.scheduler_mode == "round_robin":
            return self._select_round_robin_session(ready)
        pivot_key = self._batch_key(ready[0])
        batch = [state for state in ready if self._batch_key(state) == pivot_key][: self.max_batch_size]
        if len(batch) <= 1:
            return batch

        # First chunks are latency-critical but intentionally retain their
        # same-turn coalescing behavior. Lossless sessions are paced by their
        # consumer queues rather than a playout deadline. Only a batch made
        # entirely of already-playing latest-mode sessions has a deadline that
        # makes a larger batch potentially worse than two singleton turns.
        if any(
            state.scheduled_chunks == 0 or state.config["delivery_mode"] != "latest"
            for state in batch
        ):
            return batch

        selected_at = time.monotonic() if now is None else now
        if selected_at <= self._latest_safe_batch_start(batch):
            return batch

        # The earliest state owns the earliest playout deadline because ready
        # is deadline-sorted. Do not let an observed slow B>1 execution turn
        # an otherwise feasible pair of B=1 continuations into an avoidable
        # playout miss.
        return [batch[0]]

    def _select_round_robin_session(
        self,
        ready: Sequence[_ABotWorldLiveKitSession],
    ) -> list[_ABotWorldLiveKitSession]:
        """Pick one runnable session, rotating ownership after every block.

        This follows TurboServe's ``LocalRoundRobinStepScheduler`` rather than
        sorting by a deadline or waiting to form a micro-batch.
        """
        ready_by_id = {state.session_id: state for state in ready}
        for _ in range(len(self._round_robin_order)):
            session_id = self._round_robin_order.popleft()
            state = self._sessions.get(session_id)
            if state is None:
                continue
            self._round_robin_order.append(session_id)
            if session_id in ready_by_id:
                return [ready_by_id[session_id]]
        return []

    def _discard_from_round_robin(self, session_id: str) -> None:
        self._round_robin_order = deque(value for value in self._round_robin_order if value != session_id)

    @staticmethod
    def _batch_key(state: _ABotWorldLiveKitSession) -> tuple[object, ...]:
        session = state.pipeline_session
        local_end = 0
        if session.self_cache:
            value = session.self_cache[0]["local_end_index"]
            local_end = int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        return (
            int(state.config["control_latent_frames"]),
            session.next_latent_frame == 0,
            local_end,
            tuple(session.first_frame_latent.shape),
            session.lifecycle == ABotWorldSessionLifecycle.SUSPENDED,
        )

    def _execute_batch(
        self,
        batch: Sequence[_ABotWorldLiveKitSession],
        controls: Sequence[dict[str, bool]],
    ) -> None:
        started_at = time.monotonic()
        try:
            for state in batch:
                if not state.pipeline_session.is_resident:
                    self.pipeline.restore_interactive_session(state.pipeline_session)
            frame_counts = {int(state.config["control_latent_frames"]) for state in batch}
            if len(frame_counts) != 1:
                raise RuntimeError("ABot scheduler selected an incompatible latent-frame batch")
            if len(batch) == 1:
                results = [
                    self.pipeline.generate_next_block(
                        batch[0].pipeline_session,
                        controls[0],
                        control_latent_frames=frame_counts.pop(),
                    )
                ]
            else:
                results = self.pipeline.generate_next_blocks(
                    [state.pipeline_session for state in batch],
                    list(controls),
                    control_latent_frames=frame_counts.pop(),
                )
        except Exception as exc:
            logger.exception(
                "ABot TurboServe batch generation failed: sessions=%s",
                [item.session_id for item in batch],
            )
            for state in batch:
                with state.lock:
                    state.last_error = str(exc)
                    state.active = False
                    state.pipeline_session.lifecycle = ABotWorldSessionLifecycle.FAILED
                self._put_output(state, {"type": "error", "error": str(exc), "timestamp": time.time()})
        else:
            completed_at = time.monotonic()
            stage_metrics_callback = getattr(self.pipeline, "last_stage_metrics", None)
            self._last_stage_metrics = dict(stage_metrics_callback()) if callable(stage_metrics_callback) else {}
            observed_compute_seconds = completed_at - started_at
            previous_estimate = self._batch_compute_estimates.get(len(batch), 0.0)
            # Keep a service-run high-water mark so a transient fast batch
            # cannot make a later rendezvous overrun a playout deadline.
            self._batch_compute_estimates[len(batch)] = max(
                observed_compute_seconds,
                previous_estimate,
            )
            self._workload_detector.record_chunk(observed_compute_seconds, completed_at)
            self._batch_count += 1
            self._batch_item_count += len(batch)
            self._maximum_batch_size = max(self._maximum_batch_size, len(batch))
            for state, frames, applied_controls in zip(batch, results, controls):
                with state.lock:
                    queue_wait = max(0.0, started_at - (state.ready_since or started_at))
                    state.total_queue_wait_seconds += queue_wait
                    state.total_compute_seconds += completed_at - started_at
                    state.scheduled_chunks += 1
                    state.batch_items += len(batch)
                    payload = {
                        "type": "chunk",
                        "index": state.next_chunk_index,
                        "fps": int(state.config["fps"]),
                        "timestamp": time.time(),
                        "controls": sorted(applied_controls),
                        "frames": frames,
                        "scheduler": {
                            "batch_size": len(batch),
                            "queue_wait_seconds": round(queue_wait, 6),
                            "compute_seconds": round(completed_at - started_at, 6),
                            **self._last_stage_metrics,
                        },
                    }
                    chunk_duration_seconds = max(1, len(frames)) / int(state.config["fps"])
                    previous_compute_seconds = state.last_compute_seconds
                    state.next_chunk_index += 1
                    state.next_playout_deadline = (
                        max(state.next_playout_deadline, completed_at) + chunk_duration_seconds
                    )
                    state.last_chunk_duration_seconds = chunk_duration_seconds
                    state.last_compute_seconds = observed_compute_seconds
                    if state.config["delivery_mode"] == "latest":
                        if state.scheduled_chunks <= 1:
                            # The first chunk is latency-critical; immediately allow one
                            # successor once the publisher has taken this chunk.
                            state.pacing_ready_at = completed_at
                        else:
                            # Start the next block early enough to meet the start of the
                            # sole prefetched block, but never run before this block ends.
                            predicted_compute_seconds = max(
                                observed_compute_seconds, previous_compute_seconds
                            ) * _PACING_SAFETY_FACTOR
                            next_chunk_playout_start = state.next_playout_deadline - chunk_duration_seconds
                            state.pacing_ready_at = max(
                                completed_at, next_chunk_playout_start - predicted_compute_seconds
                            )
                    else:
                        state.pacing_ready_at = completed_at
                    state.ready_since = completed_at if state.controls else None
                self._put_output(state, payload)
        finally:
            with self._scheduler_condition:
                for state in batch:
                    with state.lock:
                        state.in_flight = False
                self._scheduler_condition.notify_all()

    def _put_output(self, state: _ABotWorldLiveKitSession, payload: dict[str, Any]) -> bool:
        """Enqueue one payload without letting a slow latest-mode client retain stale video."""
        payload_type = str(payload.get("type", ""))
        with state.lock:
            if not state.active and payload_type not in _TERMINAL_OUTPUT_TYPES:
                return False
            if not state.output_queue.full():
                state.output_queue.put_nowait(payload)
                state.output_queue_high_watermark = max(
                    state.output_queue_high_watermark,
                    state.output_queue.qsize(),
                )
                return True
            if state.config.get("delivery_mode", "latest") == "lossless":
                return False

            discarded = False
            # Queue consumers run on a different thread; manipulate the backing
            # deque only while Queue's mutex is held so latest-mode eviction
            # cannot race with get()/put().
            with state.output_queue.mutex:
                queued = state.output_queue.queue
                if payload_type in _VIDEO_OUTPUT_TYPES:
                    for item in tuple(queued):
                        if item.get("type") in _VIDEO_OUTPUT_TYPES:
                            queued.remove(item)
                            state.output_queue.unfinished_tasks = max(0, state.output_queue.unfinished_tasks - 1)
                            state.output_queue.not_full.notify()
                            state.dropped_video_payloads += 1
                            discarded = True
                            break
                    if not discarded:
                        state.dropped_video_payloads += 1
                        return False
                elif payload_type in _TERMINAL_OUTPUT_TYPES:
                    for item in tuple(queued):
                        if item.get("type") in _VIDEO_OUTPUT_TYPES:
                            queued.remove(item)
                            state.output_queue.unfinished_tasks = max(0, state.output_queue.unfinished_tasks - 1)
                            state.output_queue.not_full.notify()
                            state.dropped_video_payloads += 1
                            discarded = True
                            break
                    if not discarded and queued:
                        queued.popleft()
                        state.output_queue.unfinished_tasks = max(0, state.output_queue.unfinished_tasks - 1)
                        state.output_queue.not_full.notify()
                        state.dropped_status_payloads += 1
                else:
                    state.dropped_status_payloads += 1
                    return False
            state.output_queue.put_nowait(payload)
            state.output_queue_high_watermark = max(state.output_queue_high_watermark, state.output_queue.qsize())
            return True

    def _session(self, session_id: str) -> _ABotWorldLiveKitSession | None:
        with self._sessions_lock:
            return self._sessions.get(session_id)

    def _load_image(self, config: Mapping[str, object]) -> Image.Image:
        image_value = config.get("image", config.get("image_path", self.default_session_config.get("image_path")))
        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")
        if not isinstance(image_value, str) or not image_value:
            raise ValueError("ABot-World requires an input image, image_path, or data URL")
        if image_value.startswith("data:"):
            try:
                encoded = image_value.split(",", 1)[1]
            except IndexError as exc:
                raise ValueError("Input image data URL is missing its payload") from exc
            if len(encoded) > (_MAX_INPUT_IMAGE_BYTES * 4 // 3) + 4:
                raise ValueError("Input image exceeds the 10 MiB decoded size limit")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Input image data URL is not valid base64") from exc
            if len(raw) > _MAX_INPUT_IMAGE_BYTES:
                raise ValueError("Input image exceeds the 10 MiB decoded size limit")
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB")
        path = Path(image_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ABot input image does not exist: {path}")
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _canonical_control(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError(f"Unsupported ABot control: {value!r}")
        control = _CONTROL_ALIASES.get(value, _CONTROL_ALIASES.get(value.lower()))
        if control is None or control not in _VALID_CONTROLS:
            raise ValueError(f"Unsupported ABot control: {value!r}")
        return control

    @classmethod
    def _canonical_controls(cls, values: list[object]) -> set[str]:
        return {cls._canonical_control(value) for value in values}
