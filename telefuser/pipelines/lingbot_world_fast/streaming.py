"""Pipeline-owned three-stage streaming runtime for LingBot-World-Fast."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from PIL import Image

from telefuser.orchestrator import (
    LocalStageActor,
    ParallelWorkerStageActor,
    StreamingEdgeSpec,
    StreamingPipelineOrchestrator,
    StreamingPipelineSpec,
    StreamingSessionCloseReason,
    StreamingSessionContext,
    StreamingSessionMetrics,
    StreamingSessionStatus,
    StreamingStageIdleInterval,
    StreamingStageInvocation,
    StreamingStageSpec,
)
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker

from .session import LingBotWorldFastGenerationSession, LingBotWorldFastSessionStatus

if TYPE_CHECKING:
    from .pipeline import LingBotWorldFastPipeline


_CONDITION_PREFETCH_DEPTH = 2


@dataclass(frozen=True)
class LingBotWorldFastStreamingSession:
    """Lightweight identity for one session in the shared streaming runtime."""

    session_id: str
    epoch: int
    cache_handle: int


@dataclass
class _LingBotStreamingSessionEntry:
    runtime: LingBotWorldFastGenerationSession
    epoch: int
    progress_callback: Callable[..., None] | None
    next_condition_index: int = 0
    next_control_index: int = 0
    chunk_profiles: dict[int, dict[str, object]] = field(default_factory=dict)


class LingBotWorldFastStreamingRuntime:
    """Own the single actor graph shared by all sessions of one pipeline."""

    def __init__(self, pipeline: LingBotWorldFastPipeline) -> None:
        self.pipeline = pipeline
        self._lock = threading.RLock()
        self._sessions: dict[str, _LingBotStreamingSessionEntry] = {}
        self._closed = False
        self._serialize_dit_decode = self._dit_decode_devices_overlap()
        self._dit_decode_lock = threading.Lock()
        actors = {
            "encode": ParallelWorkerStageActor(
                pipeline.vae_encode_worker,
                "encode_condition_chunk",
                self._encode_inputs,
                self._encode_outputs,
                close_worker=False,
                session_closer=self._release_encode_session,
            ),
            "denoise": self._denoise_actor(),
            "decode": LocalStageActor(
                self._decode,
                name="lingbot-decode-actor",
                session_closer=self._release_decode_session,
            ),
        }
        spec = StreamingPipelineSpec(
            stages=(
                StreamingStageSpec("encode", frozenset({"encode_request"}), frozenset({"condition"})),
                StreamingStageSpec("denoise", frozenset({"condition", "control"}), frozenset({"latent"})),
                StreamingStageSpec("decode", frozenset({"latent"}), frozenset({"frames"})),
            ),
            edges=(
                StreamingEdgeSpec("encode_request", "encode", capacity_per_session=2),
                StreamingEdgeSpec("condition", "denoise", source_stage="encode", capacity_per_session=2),
                StreamingEdgeSpec("control", "denoise", capacity_per_session=2),
                StreamingEdgeSpec("latent", "decode", source_stage="denoise", capacity_per_session=2),
            ),
            output_artifacts=frozenset({"frames"}),
            output_capacity_per_session=2,
            latency_anchor_artifact="control",
        )
        self.orchestrator = StreamingPipelineOrchestrator(spec, actors)

    def create_session(
        self,
        runtime: LingBotWorldFastGenerationSession,
        progress_callback: Callable[..., None] | None = None,
    ) -> LingBotWorldFastStreamingSession:
        """Register an initialized model runtime with the shared scheduler."""
        if runtime.cache_handle is None:
            raise RuntimeError("LingBot streaming requires an initialized cache handle")
        session_id = f"lingbot-{runtime.cache_handle}"
        with self._lock:
            if self._closed:
                raise RuntimeError("LingBot streaming runtime is closed")
            if session_id in self._sessions:
                raise ValueError(f"LingBot streaming session {session_id!r} already exists")
            epoch = self.orchestrator.create_session(session_id, final_sequence_id=runtime.chunk_count - 1)
            entry = _LingBotStreamingSessionEntry(runtime, epoch, progress_callback)
            self._sessions[session_id] = entry
        session = LingBotWorldFastStreamingSession(session_id, epoch, runtime.cache_handle)
        try:
            self._prefetch_conditions(session_id, entry)
        except BaseException:
            self.close_session(session)
            raise
        return session

    def can_submit_chunk(self, session: LingBotWorldFastStreamingSession) -> bool:
        """Return whether the next control can be admitted without mutating ingress."""
        entry = self._require_session(session)
        try:
            with self._lock:
                if self.orchestrator.status(session.session_id) != StreamingSessionStatus.RUNNING:
                    return False
                if entry.next_control_index >= entry.runtime.chunk_count:
                    return False
                artifacts = ["control"]
                if entry.next_condition_index <= entry.next_control_index:
                    artifacts.append("encode_request")
                return self.orchestrator.can_push_inputs(session.session_id, artifacts)
        except RuntimeError:
            if self.orchestrator.error(session.session_id) is not None:
                return False
            raise

    def submit_chunk(
        self,
        session: LingBotWorldFastStreamingSession,
        chunk_index: int,
        control: torch.Tensor,
    ) -> None:
        """Submit one chunk or raise when bounded ingress is unavailable."""
        if not self.try_submit_chunk(session, chunk_index, control):
            raise RuntimeError("LingBot streaming ingress is full")

    def try_submit_chunk(
        self,
        session: LingBotWorldFastStreamingSession,
        chunk_index: int,
        control: torch.Tensor,
    ) -> bool:
        """Submit the next control while condition encoding runs ahead independently."""
        entry = self._require_session(session)
        if chunk_index < 0 or chunk_index >= entry.runtime.chunk_count:
            raise ValueError("chunk_index exceeds the LingBot session length")
        try:
            with self._lock:
                if chunk_index != entry.next_control_index:
                    raise ValueError(f"Expected LingBot control chunk {entry.next_control_index}, got {chunk_index}")
                if entry.next_condition_index < entry.next_control_index:
                    raise RuntimeError("LingBot condition cursor fell behind the control cursor")
                needs_condition = entry.next_condition_index == entry.next_control_index
                inputs: dict[str, object] = {"control": control}
                if needs_condition:
                    inputs["encode_request"] = None
                accepted = self.orchestrator.try_push_inputs(
                    session.session_id,
                    chunk_index,
                    inputs,
                )
                if accepted:
                    if needs_condition:
                        entry.next_condition_index += 1
                    entry.next_control_index += 1
                return accepted
        except RuntimeError:
            error = self.orchestrator.error(session.session_id)
            if error is not None:
                raise RuntimeError("LingBot streaming scheduler failed") from error
            raise

    def _prefetch_conditions(
        self,
        session_id: str,
        entry: _LingBotStreamingSessionEntry,
    ) -> None:
        with self._lock:
            self._prefetch_conditions_locked(session_id, entry)

    def _prefetch_conditions_locked(
        self,
        session_id: str,
        entry: _LingBotStreamingSessionEntry,
    ) -> None:
        """Keep at most two condition chunks ahead of accepted controls."""
        while (
            entry.next_condition_index < entry.runtime.chunk_count
            and entry.next_condition_index - entry.next_control_index < _CONDITION_PREFETCH_DEPTH
        ):
            if not self.orchestrator.try_push_inputs(
                session_id,
                entry.next_condition_index,
                {"encode_request": None},
            ):
                return
            entry.next_condition_index += 1

    def _refill_conditions_after_denoise(
        self,
        session_id: str,
        entry: _LingBotStreamingSessionEntry,
    ) -> None:
        """Overlap future condition encoding with decode instead of current-chunk DiT work."""
        with self._lock:
            if self._sessions.get(session_id) is not entry:
                return
            if self.orchestrator.status(session_id) != StreamingSessionStatus.RUNNING:
                return
            self._prefetch_conditions_locked(session_id, entry)

    def poll_frames(self, session: LingBotWorldFastStreamingSession) -> list[tuple[int, list[Image.Image]]]:
        """Return decoded frame batches in chunk order."""
        self._require_session(session)
        return [(index, frames) for index, _, frames in self.orchestrator.poll_outputs(session.session_id)]

    def error(self, session: LingBotWorldFastStreamingSession) -> BaseException | None:
        """Return the scheduler error for one session."""
        self._require_session(session)
        return self.orchestrator.error(session.session_id)

    def stage_idle_intervals(
        self,
        session: LingBotWorldFastStreamingSession,
        stage_id: str,
    ) -> tuple[StreamingStageIdleInterval, ...]:
        """Return scheduler-observed idle intervals for one stage."""
        self._require_session(session)
        return self.orchestrator.stage_idle_intervals(session.session_id, stage_id)

    def session_metrics(self, session: LingBotWorldFastStreamingSession) -> StreamingSessionMetrics:
        """Return scheduler-observed end-to-end latency metrics for one session."""
        self._require_session(session)
        return self.orchestrator.session_metrics(session.session_id)

    def _pop_chunk_profile(self, session: LingBotWorldFastStreamingSession, index: int) -> dict[str, object]:
        entry = self._require_session(session)
        with self._lock:
            profile = entry.chunk_profiles.pop(index, {})
        for stage_id in ("encode", "denoise", "decode"):
            timing = next(
                (
                    item
                    for item in self.orchestrator.stage_timings(session.session_id, stage_id)
                    if item.sequence_id == index
                ),
                None,
            )
            if timing is None:
                continue
            if timing.admitted_at is not None and timing.completed_at is not None:
                profile[f"{stage_id}_actor_seconds"] = timing.completed_at - timing.admitted_at
            if timing.inputs_ready_at is not None and timing.admitted_at is not None:
                profile[f"{stage_id}_queue_seconds"] = timing.admitted_at - timing.inputs_ready_at
        return profile

    def wait_until_idle(self, session: LingBotWorldFastStreamingSession, timeout: float = 5.0) -> bool:
        """Wait until the session has no admitted or immediately admissible work."""
        self._require_session(session)
        return self.orchestrator.wait_until_idle(session.session_id, timeout=timeout)

    def close_session(self, session: LingBotWorldFastStreamingSession, timeout: float = 300.0) -> None:
        """Drain one session, remove scheduler state, and release all model caches."""
        with self._lock:
            entry = self._sessions.get(session.session_id)
            if entry is None:
                return
            self._validate_handle(session, entry)
        try:
            self.orchestrator.close_session(session.session_id, timeout=timeout)
        except BaseException as exc:
            with entry.runtime.lifecycle_lock:
                entry.runtime.status = LingBotWorldFastSessionStatus.POISONED
                entry.runtime.poisoned_reason = f"Streaming session cleanup failed: {exc}"
            raise
        with self._lock:
            self._sessions.pop(session.session_id, None)
        self._finalize_session_release(entry)

    def close(self) -> None:
        """Drain the shared actor graph and release every remaining session."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        close_error: BaseException | None = None
        try:
            self.orchestrator.close()
        except BaseException as exc:
            close_error = exc
        with self._lock:
            entries = tuple(self._sessions.values())
            self._sessions.clear()
        for entry in entries:
            self._finalize_session_release(entry, close_error)
        if close_error is not None:
            raise RuntimeError("Failed to close LingBot streaming runtime cleanly") from close_error

    def _require_session(self, session: LingBotWorldFastStreamingSession) -> _LingBotStreamingSessionEntry:
        with self._lock:
            try:
                entry = self._sessions[session.session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown LingBot streaming session {session.session_id!r}") from exc
            self._validate_handle(session, entry)
            return entry

    @staticmethod
    def _validate_handle(
        session: LingBotWorldFastStreamingSession,
        entry: _LingBotStreamingSessionEntry,
    ) -> None:
        if session.epoch != entry.epoch or session.cache_handle != entry.runtime.cache_handle:
            raise RuntimeError(f"Stale LingBot streaming session handle {session.session_id!r}")

    def _entry_for_invocation(self, invocation: StreamingStageInvocation) -> _LingBotStreamingSessionEntry:
        with self._lock:
            try:
                entry = self._sessions[invocation.key.session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown LingBot streaming session {invocation.key.session_id!r}") from exc
            if entry.epoch != invocation.key.session_epoch:
                raise RuntimeError(f"Stale LingBot streaming invocation for {invocation.key.session_id!r}")
            return entry

    def _denoise_actor(self) -> LocalStageActor:
        return LocalStageActor(
            self._denoise,
            name="lingbot-denoise-actor",
            session_closer=self._release_denoise_session,
        )

    @staticmethod
    def _runtime_device_ids(runtime_config: object) -> set[int]:
        parallel_config = runtime_config.parallel_config
        if parallel_config.device_ids is not None:
            return set(parallel_config.device_ids)
        return {runtime_config.device_id}

    def _dit_decode_devices_overlap(self) -> bool:
        config = getattr(self.pipeline, "config", None)
        if config is None:
            return False
        dit_devices = self._runtime_device_ids(config.dit_config)
        decode_devices = self._runtime_device_ids(config.vae_decode_config)
        return not dit_devices.isdisjoint(decode_devices)

    def _entry_for_context(self, context: StreamingSessionContext) -> _LingBotStreamingSessionEntry:
        with self._lock:
            try:
                entry = self._sessions[context.session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown LingBot streaming session {context.session_id!r}") from exc
            if entry.epoch != context.session_epoch:
                raise RuntimeError(f"Stale LingBot streaming session cleanup for {context.session_id!r}")
            return entry

    def _release_encode_session(
        self,
        context: StreamingSessionContext,
        reason: StreamingSessionCloseReason,
    ) -> None:
        del reason
        entry = self._entry_for_context(context)
        cache_handle = entry.runtime.cache_handle
        if cache_handle is None:
            return
        result = self.pipeline.vae_encode_worker.release_cache(cache_handle, sync=True)
        if callable(result):
            result()

    def _release_decode_session(
        self,
        context: StreamingSessionContext,
        reason: StreamingSessionCloseReason,
    ) -> None:
        del reason
        entry = self._entry_for_context(context)
        cache_handle = entry.runtime.cache_handle
        if cache_handle is None:
            return
        result = self.pipeline.vae_decode_worker.release_cache(cache_handle, sync=True)
        if callable(result):
            result()

    def _release_denoise_session(
        self,
        context: StreamingSessionContext,
        reason: StreamingSessionCloseReason,
    ) -> None:
        del reason
        entry = self._entry_for_context(context)
        cache_handle = entry.runtime.cache_handle
        if cache_handle is None:
            return
        if isinstance(self.pipeline.denoise_stage, ParallelWorker):
            result = self.pipeline.denoise_stage.release_cache(cache_handle, sync=True)
        else:
            result = self.pipeline.denoise_stage.release_cache(cache_handle)
        if callable(result):
            result()

    @staticmethod
    def _finalize_session_release(
        entry: _LingBotStreamingSessionEntry,
        error: BaseException | None = None,
    ) -> None:
        runtime = entry.runtime
        with runtime.lifecycle_lock:
            runtime.prompt_emb = None
            runtime.condition_image = None
            runtime.world_kv_cached_latents.clear()
            if error is None:
                runtime.cache_handle = None
                runtime.status = LingBotWorldFastSessionStatus.RELEASED
                runtime.poisoned_reason = None
            else:
                runtime.status = LingBotWorldFastSessionStatus.POISONED
                runtime.poisoned_reason = f"Streaming session cleanup failed: {error}"

    def _encode_inputs(self, invocation: StreamingStageInvocation) -> tuple[tuple[object, ...], dict[str, object]]:
        entry = self._entry_for_invocation(invocation)
        runtime = entry.runtime
        index = invocation.key.sequence_id
        self.pipeline._notify_progress(entry.progress_callback, "encoding_condition_chunk", index=index)
        return (), {
            "cache_handle": runtime.cache_handle,
            "chunk_index": index,
            "chunk_count": runtime.chunk_count,
            "chunk_size": runtime.chunk_size,
            "height": runtime.height,
            "width": runtime.width,
            "output_dtype": self.pipeline.torch_dtype,
        }

    def _encode_outputs(self, value: dict[str, object], invocation: StreamingStageInvocation) -> dict[str, object]:
        entry = self._entry_for_invocation(invocation)
        index = invocation.key.sequence_id
        self.pipeline._notify_progress(entry.progress_callback, "condition_chunk_encoded", index=index)
        if index == 0:
            entry.runtime.condition_image = None
        return {"condition": value}

    def _denoise_kwargs(self, invocation: StreamingStageInvocation) -> dict[str, object]:
        runtime = self._entry_for_invocation(invocation).runtime
        index = invocation.key.sequence_id
        kwargs = {
            "cache_handle": runtime.cache_handle,
            "condition_chunk": invocation.inputs["condition"],
            "prompt_emb": None,
            "control_chunk": invocation.inputs["control"],
            "current_start": index * runtime.chunk_size * runtime.frame_tokens,
            "max_attention_size": runtime.max_attention_size,
            "_local_vae_handoff": bool(
                runtime.world_kv_binding is None
                and getattr(self.pipeline.vae_decode_worker, "uses_local_latent_handoff", False)
            ),
            "_benchmark_profile": runtime.config.benchmark_metrics,
        }
        if getattr(self.pipeline, "uses_direct_vae_handoff", False):
            kwargs["_tensor_transport"] = runtime.world_kv_binding is None
        return kwargs

    @torch.inference_mode()
    def _denoise(self, invocation: StreamingStageInvocation) -> dict[str, object]:
        entry = self._entry_for_invocation(invocation)
        runtime = entry.runtime
        index = invocation.key.sequence_id
        cached_latent = runtime.world_kv_cached_latents.pop(index, None) if runtime.world_kv_cached_latents else None
        if cached_latent is not None:
            self.pipeline._notify_progress(entry.progress_callback, "world_kv_cache_hit", index=index)
            advance = self.pipeline.denoise_stage.advance_noise(cache_handle=runtime.cache_handle)
            if callable(advance):
                advance()
            latent = cached_latent.to(device=self.pipeline.device, dtype=self.pipeline.torch_dtype)
        else:
            self.pipeline._notify_progress(entry.progress_callback, "denoising_chunk", index=index)
            kwargs = self._denoise_kwargs(invocation)
            lock = self._dit_decode_lock if self._serialize_dit_decode else nullcontext()
            lock_started_at = time.perf_counter()
            with lock:
                worker_started_at = time.perf_counter()
                result = self.pipeline.denoise_stage.denoise_and_update_cache(**kwargs)
                submit_finished_at = time.perf_counter()
                if callable(result):
                    result = result()
                worker_finished_at = time.perf_counter()
                if isinstance(result, tuple):
                    latent, profile = result
                    profile["denoise_lock_wait_seconds"] = worker_started_at - lock_started_at
                    profile["denoise_submit_seconds"] = submit_finished_at - worker_started_at
                    profile["denoise_result_wait_seconds"] = worker_finished_at - submit_finished_at
                    profile["denoise_worker_seconds"] = worker_finished_at - worker_started_at
                    with self._lock:
                        entry.chunk_profiles.setdefault(index, {}).update(profile)
                else:
                    latent = result
            self.pipeline._notify_progress(entry.progress_callback, "chunk_denoised", index=index)
        if runtime.world_kv_binding is not None:
            try:
                runtime.world_kv_binding.on_chunk_finalized(runtime, index, latent)
            except Exception as exc:
                logger.warning(f"world_kv on_chunk_finalized failed at chunk {index}: {exc}")
        self._refill_conditions_after_denoise(invocation.key.session_id, entry)
        return {"latent": latent}

    def _decode_inputs(self, invocation: StreamingStageInvocation) -> tuple[tuple[object, ...], dict[str, object]]:
        entry = self._entry_for_invocation(invocation)
        runtime = entry.runtime
        index = invocation.key.sequence_id
        self.pipeline._notify_progress(
            entry.progress_callback,
            "decoding_chunk",
            index=index,
            device=str(self.pipeline.vae_device),
        )
        return (), {
            "cache_handle": runtime.cache_handle,
            "latents": invocation.inputs["latent"],
            "is_first_clip": index == 0,
            "is_last_clip": index == runtime.chunk_count - 1,
            "_benchmark_profile": runtime.config.benchmark_metrics,
        }

    def _decode_outputs(self, value: torch.Tensor, invocation: StreamingStageInvocation) -> dict[str, object]:
        entry = self._entry_for_invocation(invocation)
        if value.dtype == torch.uint8:
            arrays = value.permute(1, 2, 3, 0).contiguous().numpy()
            frames = [Image.fromarray(array) for array in arrays]
        else:
            frames = self.pipeline.tensor2video(value)
        self.pipeline._notify_progress(
            entry.progress_callback,
            "chunk_decoded",
            index=invocation.key.sequence_id,
            frames=len(frames),
        )
        return {"frames": frames}

    @torch.inference_mode()
    def _decode(self, invocation: StreamingStageInvocation) -> dict[str, object]:
        args, kwargs = self._decode_inputs(invocation)
        lock = self._dit_decode_lock if self._serialize_dit_decode else nullcontext()
        lock_started_at = time.perf_counter()
        with lock:
            worker_started_at = time.perf_counter()
            result = self.pipeline.vae_decode_worker.decode_chunk(*args, **kwargs)
            submit_finished_at = time.perf_counter()
            if callable(result):
                result = result()
            worker_finished_at = time.perf_counter()
        if isinstance(result, tuple):
            value, profile = result
            profile["decode_lock_wait_seconds"] = worker_started_at - lock_started_at
            profile["decode_submit_seconds"] = submit_finished_at - worker_started_at
            profile["decode_result_wait_seconds"] = worker_finished_at - submit_finished_at
            profile["decode_worker_seconds"] = worker_finished_at - worker_started_at
        else:
            value, profile = result, {}
        convert_start = time.perf_counter()
        output = self._decode_outputs(value, invocation)
        profile["tensor_to_frames_seconds"] = time.perf_counter() - convert_start
        entry = self._entry_for_invocation(invocation)
        with self._lock:
            entry.chunk_profiles.setdefault(invocation.key.sequence_id, {}).update(profile)
        return output
