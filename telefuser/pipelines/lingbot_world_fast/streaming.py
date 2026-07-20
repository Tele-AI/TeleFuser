"""Pipeline-owned three-stage streaming runtime for LingBot-World-Fast."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from PIL import Image

from telefuser.orchestrator import (
    LocalStageActor,
    ParallelWorkerStageActor,
    StreamingEdgeSpec,
    StreamingPipelineOrchestrator,
    StreamingPipelineSpec,
    StreamingSessionStatus,
    StreamingStageIdleInterval,
    StreamingStageInvocation,
    StreamingStageSpec,
)
from telefuser.worker.parallel_worker import ParallelWorker

from .session import LingBotWorldFastGenerationSession

if TYPE_CHECKING:
    from .pipeline import LingBotWorldFastPipeline


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


class LingBotWorldFastStreamingRuntime:
    """Own the single actor graph shared by all sessions of one pipeline."""

    def __init__(self, pipeline: LingBotWorldFastPipeline) -> None:
        self.pipeline = pipeline
        self._lock = threading.RLock()
        self._sessions: dict[str, _LingBotStreamingSessionEntry] = {}
        self._closed = False
        actors = {
            "encode": ParallelWorkerStageActor(
                pipeline.vae_encode_worker,
                "encode_condition_chunk",
                self._encode_inputs,
                self._encode_outputs,
                close_worker=False,
            ),
            "denoise": self._denoise_actor(),
            "decode": ParallelWorkerStageActor(
                pipeline.vae_decode_worker,
                "decode_chunk",
                self._decode_inputs,
                self._decode_outputs,
                close_worker=False,
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
            self._sessions[session_id] = _LingBotStreamingSessionEntry(runtime, epoch, progress_callback)
        return LingBotWorldFastStreamingSession(session_id, epoch, runtime.cache_handle)

    def can_submit_chunk(self, session: LingBotWorldFastStreamingSession) -> bool:
        """Return whether the session can atomically admit another chunk."""
        self._require_session(session)
        try:
            if self.orchestrator.status(session.session_id) != StreamingSessionStatus.RUNNING:
                return False
            return self.orchestrator.can_push_inputs(session.session_id, ("encode_request", "control"))
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
        """Atomically submit one chunk to the shared actor graph."""
        entry = self._require_session(session)
        if chunk_index < 0 or chunk_index >= entry.runtime.chunk_count:
            raise ValueError("chunk_index exceeds the LingBot session length")
        try:
            return self.orchestrator.try_push_inputs(
                session.session_id,
                chunk_index,
                {
                    "encode_request": None,
                    "control": control,
                },
            )
        except RuntimeError:
            error = self.orchestrator.error(session.session_id)
            if error is not None:
                raise RuntimeError("LingBot streaming scheduler failed") from error
            raise

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
        self.orchestrator.close_session(session.session_id, timeout=timeout)
        with self._lock:
            self._sessions.pop(session.session_id, None)
        self.pipeline.release_session(entry.runtime)

    def close(self) -> None:
        """Drain the shared actor graph and release every remaining session."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.orchestrator.close()
        with self._lock:
            entries = tuple(self._sessions.values())
            self._sessions.clear()
        for entry in entries:
            self.pipeline.release_session(entry.runtime)

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

    def _denoise_actor(self):
        if isinstance(self.pipeline.denoise_stage, ParallelWorker):
            return ParallelWorkerStageActor(
                self.pipeline.denoise_stage,
                "denoise_and_update_cache",
                self._denoise_inputs,
                lambda value, _: {"latent": value},
                close_worker=False,
            )
        return LocalStageActor(self._denoise_local, name="lingbot-denoise-actor")

    def _encode_inputs(self, invocation: StreamingStageInvocation) -> tuple[tuple[object, ...], dict[str, object]]:
        runtime = self._entry_for_invocation(invocation).runtime
        index = invocation.key.sequence_id
        return (), {
            "cache_handle": runtime.cache_handle,
            "chunk_index": index,
            "chunk_count": runtime.chunk_count,
            "chunk_size": runtime.chunk_size,
            "height": runtime.height,
            "width": runtime.width,
        }

    def _encode_outputs(self, value: torch.Tensor, invocation: StreamingStageInvocation) -> dict[str, object]:
        entry = self._entry_for_invocation(invocation)
        if invocation.key.sequence_id == 0:
            entry.runtime.condition_image = None
        return {"condition": value.to(device=self.pipeline.device, dtype=self.pipeline.torch_dtype)}

    def _denoise_inputs(self, invocation: StreamingStageInvocation) -> tuple[tuple[object, ...], dict[str, object]]:
        runtime = self._entry_for_invocation(invocation).runtime
        index = invocation.key.sequence_id
        return (), {
            "cache_handle": runtime.cache_handle,
            "condition_chunk": invocation.inputs["condition"],
            "prompt_emb": runtime.prompt_emb,
            "control_chunk": invocation.inputs["control"],
            "current_start": index * runtime.chunk_size * runtime.frame_tokens,
            "max_attention_size": runtime.max_attention_size,
        }

    def _denoise_local(self, invocation: StreamingStageInvocation) -> dict[str, object]:
        _, kwargs = self._denoise_inputs(invocation)
        return {"latent": self.pipeline.denoise_stage.denoise_and_update_cache(**kwargs)}

    def _decode_inputs(self, invocation: StreamingStageInvocation) -> tuple[tuple[object, ...], dict[str, object]]:
        runtime = self._entry_for_invocation(invocation).runtime
        index = invocation.key.sequence_id
        return (), {
            "cache_handle": runtime.cache_handle,
            "latents": invocation.inputs["latent"],
            "is_first_clip": index == 0,
            "is_last_clip": index == runtime.chunk_count - 1,
        }

    def _decode_outputs(self, value: torch.Tensor, invocation: StreamingStageInvocation) -> dict[str, object]:
        entry = self._entry_for_invocation(invocation)
        frames = self.pipeline.tensor2video(value)
        self.pipeline._notify_progress(
            entry.progress_callback,
            "chunk_decoded",
            index=invocation.key.sequence_id,
            frames=len(frames),
        )
        return {"frames": frames}
