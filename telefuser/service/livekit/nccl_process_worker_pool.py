"""Process-isolated ABot model workers with NCCL state migration.

The parent retains LiveKit transport ownership. Child processes retain model
state, so a committed migration changes only the model route, not the room.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from telefuser.service.core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL
from telefuser.service.security.security_validator import SecurityLevel
from telefuser.utils.logging import logger

from .nccl_transfer import allocate_tensor_tree_leaves, transfer_tensor_leaves_nccl
from .pipeline_adapter import LiveKitPipelineAdapter
from .process_worker_pool import (
    ProcessLiveKitWorkerPool,
    ProcessWorkerSpec,
    _close_queue,
    _process_dispatch_trace_gpu_metadata,
)
from .session_registry import SessionRecord
from .token_service import LiveKitTokenService
from .turboserve import TurboServeOwnership, TurboServeOwnershipTable
from .worker import LiveKitWorker

# The parent retains at most one decoded payload waiting for the LiveKit
# transport. The transport acknowledges a payload as soon as it dequeues it,
# which permits exactly one next payload to be prefetched while the current
# one is paced onto WebRTC. Consequently, at most two fully-materialized
# payloads per session live outside ABot's own bounded/latest output queue:
# one being published and one in this parent queue.
_MODEL_OUTPUT_PARENT_QUEUE_SIZE = 1
_VIDEO_OUTPUT_TYPES = frozenset({"preview", "chunk"})
_TERMINAL_OUTPUT_TYPES = frozenset({"error", "done"})
# Process-group creation happens only after every worker has loaded the model,
# so it needs a dedicated budget rather than the normal 15-second IPC timeout.
# Keep the parent request longer than the child process-group timeout so a
# child can return a useful failure instead of being torn down mid-initialization.
_NCCL_INIT_GROUP_TIMEOUT_SECONDS = 180.0
_NCCL_INIT_PARENT_TIMEOUT_SECONDS = 210.0


@dataclass(frozen=True)
class _ModelOutput:
    """One child-model payload plus the worker that owns its output credit."""

    worker_id: str
    payload: dict[str, Any]


class _DispatchTraceWriter:
    """Bounded, parent-owned JSONL writer for experiment audit records."""

    def __init__(self, path: str, *, max_events: int, workers: dict[str, list[str]]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.max_events = int(max_events)
        self.received_events = 0
        self.written_events = 0
        self.dropped_events = 0
        self.write_errors = 0
        self._write_error_logged = False
        self._handle: Any | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"dispatch trace path already exists; choose a fresh run-scoped path: {self.path}")
        self._handle = self.path.open("x", encoding="utf-8")
        self._write_line(
            {
                "schema_version": 1,
                "event_type": "trace_metadata",
                "trace_started_monotonic_seconds": time.monotonic(),
                "trace_started_unix_seconds": time.time(),
                "trace_started_utc": datetime.now(timezone.utc).isoformat(),
                "max_dispatch_events": self.max_events,
                "configured_workers": workers,
            }
        )

    def _write_line(self, record: dict[str, Any]) -> bool:
        handle = self._handle
        if handle is None:
            return False
        try:
            handle.write(json.dumps(record, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            return True
        except (OSError, TypeError, ValueError) as exc:
            self.write_errors += 1
            if not self._write_error_logged:
                self._write_error_logged = True
                logger.warning("Failed to write ABot dispatch trace %s: %s", self.path, exc)
            return False

    def append(self, record: dict[str, Any]) -> None:
        self.received_events += 1
        if self.received_events > self.max_events:
            self.dropped_events += 1
            return
        enriched = dict(record)
        enriched["parent_sequence"] = self.received_events
        enriched["parent_received_monotonic_seconds"] = time.monotonic()
        enriched["parent_received_unix_seconds"] = time.time()
        if self._write_line(enriched):
            self.written_events += 1
        else:
            self.dropped_events += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": True,
            "path": str(self.path),
            "max_events": self.max_events,
            "received_events": self.received_events,
            "written_events": self.written_events,
            "dropped_events": self.dropped_events,
            "write_errors": self.write_errors,
        }

    def close(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()


async def _pump_model_outputs(
    adapter: Any,
    service: Any,
    *,
    worker_id: str,
    session_id: str,
    credits: asyncio.BoundedSemaphore,
    events: Any,
) -> None:
    """Pull only after a parent credit, so IPC cannot outrun WebRTC playback."""
    chunks = adapter.pull_chunks(session_id)
    iterator = chunks.__aiter__()
    credit_held = False
    try:
        while True:
            await credits.acquire()
            credit_held = True
            try:
                payload = await iterator.__anext__()
            except StopAsyncIteration:
                credits.release()
                credit_held = False
                # An inactive model session naturally ends its generator. The
                # parent transport must see this EOF and release its route;
                # otherwise it waits forever in ``pull_model_chunks``.
                events.put(
                    {
                        "type": "model_output_eos",
                        "worker_id": worker_id,
                        "session_id": session_id,
                    }
                )
                return
            except asyncio.CancelledError:
                credits.release()
                credit_held = False
                raise
            except Exception:
                credits.release()
                credit_held = False
                raise
            # No await separates dequeue from IPC submission. If migration
            # pauses this task, a payload is either sent exactly once or has
            # not been removed from the ABot generator.
            events.put(
                {
                    "type": "model_output",
                    "worker_id": worker_id,
                    "session_id": session_id,
                    "payload": payload,
                    "runtime_metrics": adapter.runtime_metrics() or {},
                    "session_runtime_metrics": service.runtime_metrics(session_id),
                }
            )
            # The credit now belongs to the parent queue and transport path.
            credit_held = False
    finally:
        if credit_held:
            with contextlib.suppress(ValueError):
                credits.release()
        aclose = getattr(chunks, "aclose", None)
        if callable(aclose):
            with contextlib.suppress(Exception):
                await aclose()


class _ProcessPipelineAdapter:
    stream_mode = STREAM_MODE_BIDIRECTIONAL

    def __init__(self, pool: "NCCLProcessLiveKitWorkerPool", initial_worker_id: str) -> None:
        self._pool = pool
        self._initial_worker_id = initial_worker_id

    def create_session(self, config: dict) -> str:
        session_id = str(config["session_id"])
        self._pool.create_model_session(self._initial_worker_id, session_id, config)
        return session_id

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self._pool.push_model_chunk(session_id, chunk)

    async def pull_chunks(self, session_id: str):
        async for chunk in self._pool.pull_model_chunks(session_id):
            yield chunk

    def enable_publisher_frame_tracking(self, session_id: str) -> bool:
        return self._pool.enable_publisher_frame_tracking(session_id)

    def report_publisher_frame_progress(
        self, session_id: str, *, event: str, frames_delta: int, sequence: int, observed_monotonic_seconds: float
    ) -> bool:
        del sequence
        return self._pool.report_publisher_frame_progress(
            session_id,
            event=event,
            frames_delta=frames_delta,
            observed_monotonic_seconds=observed_monotonic_seconds,
        )

    def close_session(self, session_id: str) -> None:
        self._pool.close_model_session(session_id)


class _ParentTransportSink:
    def __init__(self, pool: "NCCLProcessLiveKitWorkerPool") -> None:
        self.pool = pool

    def on_worker_status(self, worker_id: str, status: str) -> None:
        del worker_id, status

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        self.pool._event_sink.on_worker_capacity(worker_id, capacity, profile)

    def on_session_status(self, session_id: str, status: str, error: str | None = None) -> None:
        self.pool._event_sink.on_session_status(session_id, status, error)

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        self.pool._event_sink.on_pipeline_session(session_id, pipeline_session_id)

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        self.pool._transport_finished(session_id)
        self.pool._event_sink.on_session_finished(worker_id, session_id, error)

    def on_control_received(self, worker_id: str, session_id: str) -> None:
        callback = getattr(self.pool._event_sink, "on_control_received", None)
        if callable(callback):
            callback(worker_id, session_id)

    def on_chunk_published(
        self, worker_id: str, session_id: str, frames: int, first_frame_at: float | None = None
    ) -> None:
        callback = getattr(self.pool._event_sink, "on_chunk_published", None)
        if callable(callback):
            callback(worker_id, session_id, frames, first_frame_at)


class NCCLProcessLiveKitWorkerPool(ProcessLiveKitWorkerPool):
    """TurboServe-compatible parent transport / GPU model-process pool."""

    def __init__(self, specs: list[ProcessWorkerSpec], **kwargs: Any) -> None:
        super().__init__(specs, **kwargs)
        self._worker_target = _nccl_model_worker_main
        self._ownership = TurboServeOwnershipTable()
        self._model_outputs: dict[str, asyncio.Queue[_ModelOutput | None]] = {}
        self._model_output_inflight: set[str] = set()
        self._model_output_drained: dict[str, asyncio.Event] = {}
        self._model_output_inflight_owner: dict[str, str] = {}
        self._publisher_progress_sequences: dict[str, int] = {}
        self._publisher_frame_tracking: dict[str, bool] = {}
        self._model_output_dropped: dict[str, int] = {}
        self._transport_workers: dict[str, LiveKitWorker] = {}
        self._transport_tasks: dict[str, asyncio.Task[None]] = {}
        self._migrating_controls: dict[str, list[dict]] = {}
        # Worker snapshots include scalar timings/counters plus the bounded
        # scheduler mode string (``batched`` or ``round_robin``).
        self._worker_runtime_metrics: dict[str, dict[str, float | int | str]] = {}
        self._session_runtime_metrics: dict[str, dict[str, float | int]] = {}
        self._migration_total_ms: list[float] = []
        self._nccl_ranks: dict[str, int] = {}
        self._migration_lock = asyncio.Lock()
        self._initializing_workers = False
        # ``ProcessLiveKitWorkerPool`` owns the parent JSONL writer for both
        # isolated-worker modes. Recreating it here would reject the path it
        # has just created, preventing a traced process-NCCL run from starting.

    async def start(self, *, skip_validation: bool = False) -> None:
        # ``ProcessLiveKitWorkerPool.start`` calls this class's ``scale_to``
        # once per replica. Defer communicator construction until all initial
        # workers have completed their sequential checkpoint load.
        self._initializing_workers = True
        try:
            await super().start(skip_validation=skip_validation)
        finally:
            self._initializing_workers = False
        if len(self._active_workers) > 1 and not self._nccl_ranks:
            await self._init_nccl()

    async def scale_to(self, target_workers: int) -> int:
        """Rebuild the static NCCL communicator around a new replica set."""
        async with self._migration_lock:
            if len(self._active_workers) == target_workers:
                return target_workers
            if self._nccl_ranks:
                await asyncio.gather(
                    *(self._request(worker_id, "nccl_destroy") for worker_id in self._nccl_ranks),
                    return_exceptions=True,
                )
                self._nccl_ranks.clear()
            actual = await super().scale_to(target_workers)
            if actual > 1 and not self._initializing_workers:
                await self._init_nccl()
            return actual

    def start_session(self, record: SessionRecord) -> None:
        if record.worker_id is None or record.worker_id not in self._active_workers:
            raise RuntimeError("Model worker is not active")
        runner = LiveKitWorker(
            worker_id=record.worker_id,
            config=self._config,
            pipeline_file=self._pipeline_file,
            token_service=LiveKitTokenService(
                api_key=self._config.livekit_api_key,
                api_secret=self._config.livekit_api_secret,
                token_ttl=self._config.token_ttl,
            ),
            event_sink=_ParentTransportSink(self),
            pipeline_adapter=_ProcessPipelineAdapter(self, record.worker_id),
        )
        task = asyncio.create_task(runner.run_session(record), name=f"livekit-transport-{record.session_id}")
        self._transport_workers[record.session_id] = runner
        self._transport_tasks[record.session_id] = task
        task.add_done_callback(lambda done, sid=record.session_id: self._transport_task_done(sid, done))

    async def stop_session(self, session_id: str) -> None:
        runner = self._transport_workers.get(session_id)
        task = self._transport_tasks.get(session_id)
        if runner is not None:
            await runner.stop_session(session_id)
        if task is not None:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)

    def create_model_session(self, worker_id: str, session_id: str, config: dict) -> None:
        output: asyncio.Queue[_ModelOutput | None] = asyncio.Queue(maxsize=_MODEL_OUTPUT_PARENT_QUEUE_SIZE)
        drained = asyncio.Event()
        drained.set()
        self._model_outputs[session_id] = output
        self._model_output_drained[session_id] = drained
        self._model_output_dropped[session_id] = 0
        self._pipeline_routes[session_id] = worker_id
        self._publisher_progress_sequences[session_id] = -1
        self._publisher_frame_tracking[session_id] = True
        self._session_workers[session_id] = worker_id
        self._ownership.register(session_id, worker_id)
        self._send(
            worker_id,
            {
                "type": "model_create",
                "session_id": session_id,
                "config": dict(config),
                "model_output_credit_window": _MODEL_OUTPUT_PARENT_QUEUE_SIZE,
            },
        )

    def push_model_chunk(self, session_id: str, chunk: dict) -> None:
        if session_id in self._migrating_controls:
            self._migrating_controls[session_id].append(dict(chunk))
            return
        self._send(
            self._pipeline_routes[session_id],
            {"type": "model_push", "session_id": session_id, "chunk": dict(chunk)},
        )

    def enable_publisher_frame_tracking(self, session_id: str) -> bool:
        return bool(self._publisher_frame_tracking.get(session_id, False))

    def report_publisher_frame_progress(
        self, session_id: str, *, event: str, frames_delta: int, observed_monotonic_seconds: float
    ) -> bool:
        worker_id = self._model_output_inflight_owner.get(session_id) or self._pipeline_routes.get(session_id)
        return self._send_publisher_frame_progress(
            session_id,
            worker_id=worker_id,
            event=event,
            frames_delta=frames_delta,
            observed_monotonic_seconds=observed_monotonic_seconds,
        )

    def _send_publisher_frame_progress(
        self,
        session_id: str,
        *,
        worker_id: str | None,
        event: str,
        frames_delta: int,
        observed_monotonic_seconds: float,
    ) -> bool:
        if (
            worker_id is None
            or worker_id not in self._active_workers
            or not self._publisher_frame_tracking.get(session_id, False)
        ):
            return False
        sequence = int(self._publisher_progress_sequences.get(session_id, -1)) + 1
        self._publisher_progress_sequences[session_id] = sequence
        try:
            self._send(
                worker_id,
                {
                    "type": "model_publisher_frame_progress",
                    "session_id": session_id,
                    "event": event,
                    "frames_delta": int(frames_delta),
                    "sequence": sequence,
                    "observed_monotonic_seconds": float(observed_monotonic_seconds),
                },
            )
        except Exception:
            return False
        return True

    def close_model_session(self, session_id: str) -> None:
        worker_id = self._pipeline_routes.pop(session_id, None)
        self._session_workers.pop(session_id, None)
        self._ownership.release(session_id)
        self._migrating_controls.pop(session_id, None)
        self._session_runtime_metrics.pop(session_id, None)
        self._publisher_progress_sequences.pop(session_id, None)
        self._publisher_frame_tracking.pop(session_id, None)
        if worker_id in self._active_workers:
            self._send(worker_id, {"type": "model_close", "session_id": session_id})
        self._close_model_output(session_id)

    async def pull_model_chunks(self, session_id: str):
        output = self._model_outputs.get(session_id)
        if output is None:
            return
        while True:
            item = await output.get()
            if item is None:
                return
            # This is deliberately before ``yield``: it allows one queued
            # prefetch while the transport paces the just-dequeued payload.
            self._model_output_inflight.add(session_id)
            self._model_output_inflight_owner[session_id] = item.worker_id
            self._update_model_output_drained(session_id)
            self._ack_model_output(session_id, item)
            try:
                yield item.payload
            finally:
                self._model_output_inflight.discard(session_id)
                self._model_output_inflight_owner.pop(session_id, None)
                self._update_model_output_drained(session_id)

    def _close_model_output(self, session_id: str) -> None:
        output = self._model_outputs.pop(session_id, None)
        self._model_output_inflight.discard(session_id)
        self._model_output_inflight_owner.pop(session_id, None)
        drained = self._model_output_drained.pop(session_id, None)
        self._model_output_dropped.pop(session_id, None)
        if isinstance(output, asyncio.Queue):
            while True:
                try:
                    output.get_nowait()
                except asyncio.QueueEmpty:
                    break
            with contextlib.suppress(asyncio.QueueFull):
                output.put_nowait(None)
        elif output is not None:
            # Compatibility for simple queue doubles used by isolated tests.
            output.put_nowait(None)
        if drained is not None:
            drained.set()

    def _enqueue_model_output(self, session_id: str, item: _ModelOutput) -> None:
        output = self._model_outputs.get(session_id)
        if output is None:
            return
        if not isinstance(output, asyncio.Queue):
            # Keep legacy light-weight test doubles usable while production
            # always takes the bounded branch below.
            output.put_nowait(item.payload)
            return
        try:
            output.put_nowait(item)
        except asyncio.QueueFull:
            queued = output.get_nowait()
            if queued is None:
                output.put_nowait(None)
                self._record_dropped_model_output(session_id, item)
            elif self._should_replace_queued_output(queued, item):
                output.put_nowait(item)
                self._record_dropped_model_output(session_id, queued)
            else:
                output.put_nowait(queued)
                self._record_dropped_model_output(session_id, item)
        self._update_model_output_drained(session_id)

    @staticmethod
    def _should_replace_queued_output(queued: _ModelOutput, incoming: _ModelOutput) -> bool:
        queued_type = str(queued.payload.get("type", ""))
        incoming_type = str(incoming.payload.get("type", ""))
        if queued_type in _TERMINAL_OUTPUT_TYPES:
            return False
        if incoming_type in _TERMINAL_OUTPUT_TYPES:
            return True
        # Preserve an initial preview until it reaches the transport. Later
        # generated chunks are latest-wins, matching ABot's own queue.
        if queued_type == "preview":
            return False
        if incoming_type == "preview":
            return queued_type in _VIDEO_OUTPUT_TYPES
        return queued_type == "chunk" and incoming_type == "chunk"

    def _record_dropped_model_output(self, session_id: str, item: _ModelOutput, *, acknowledge: bool = True) -> None:
        dropped = getattr(self, "_model_output_dropped", None)
        if isinstance(dropped, dict):
            dropped[session_id] = int(dropped.get(session_id, 0)) + 1
        payload = item.payload
        frames = payload.get("frames")
        if payload.get("type") == "chunk" and isinstance(frames, list) and frames:
            self._send_publisher_frame_progress(
                session_id,
                worker_id=item.worker_id,
                event="dropped",
                frames_delta=-len(frames),
                observed_monotonic_seconds=time.monotonic(),
            )
        if acknowledge:
            self._ack_model_output(session_id, item)

    def _ack_model_output(self, session_id: str, item: _ModelOutput) -> None:
        # Never wait for a child response from the parent event loop. The
        # command only releases that child session's bounded semaphore.
        try:
            if item.worker_id in self._active_workers:
                self._send(
                    item.worker_id,
                    {"type": "model_output_credit", "session_id": session_id},
                )
        except Exception:
            # The owning child can disappear during close/migration; there is
            # then no useful credit to return.
            return

    def _update_model_output_drained(self, session_id: str) -> None:
        drained = self._model_output_drained.get(session_id)
        output = self._model_outputs.get(session_id)
        if drained is None or output is None or not isinstance(output, asyncio.Queue):
            return
        if session_id not in self._model_output_inflight and output.empty():
            drained.set()
        else:
            drained.clear()

    async def _wait_for_model_output_drain(self, session_id: str, *, timeout: float) -> None:
        self._update_model_output_drained(session_id)
        drained = self._model_output_drained.get(session_id)
        if drained is not None:
            await asyncio.wait_for(drained.wait(), timeout=timeout)

    @staticmethod
    def _source_model_output_drain_complete(status: object) -> bool:
        if not isinstance(status, dict):
            return False
        return bool(
            not status.get("in_flight", True)
            and status.get("output_queue_empty", False)
            and int(status.get("publisher_unsubmitted_frames", 1)) == 0
        )

    async def _drain_model_outputs_for_migration(
        self,
        session_id: str,
        *,
        source_worker_id: str,
        timeout: float,
    ) -> None:
        """Drain child Fq and parent publisher Fp before copying model state.

        The child pump intentionally remains active until the source service
        reports no queued model output and no publisher-owned frames. Each
        status request is a command-queue barrier, so it follows all earlier
        publisher-progress commands for this child. A parent drain after the
        barrier accounts for model-output events emitted before that barrier.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out draining model output before NCCL migration")
            await self._wait_for_model_output_drain(session_id, timeout=remaining)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for source drain status before NCCL migration")
            status_event = await self._request(
                source_worker_id,
                "model_output_drain_status",
                session_id=session_id,
                timeout=remaining,
            )
            status = status_event.get("result")
            if not self._source_model_output_drain_complete(status):
                await asyncio.sleep(0.001)
                continue

            # The source barrier follows all of its model-output events; drain
            # those parent transport payloads before accepting the snapshot.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out draining parent transport before NCCL migration")
            await self._wait_for_model_output_drain(session_id, timeout=remaining)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out confirming source drain before NCCL migration")
            final_event = await self._request(
                source_worker_id,
                "model_output_drain_status",
                session_id=session_id,
                timeout=remaining,
            )
            if self._source_model_output_drain_complete(final_event.get("result")):
                return
            await asyncio.sleep(0.001)

    async def migrate_session(self, pipeline_session_id: str, target_worker_id: str) -> TurboServeOwnership:
        async with self._migration_lock:
            source_worker_id = self._pipeline_routes[pipeline_session_id]
            if source_worker_id == target_worker_id:
                return self._ownership.owner(pipeline_session_id)
            if source_worker_id not in self._nccl_ranks or target_worker_id not in self._nccl_ranks:
                raise RuntimeError("NCCL migration requires initialized source and target workers")
            token = self._ownership.prepare_migration(pipeline_session_id, source_worker_id, target_worker_id)
            self._migrating_controls[pipeline_session_id] = []
            started = time.monotonic()
            source_output_paused = False
            try:
                await asyncio.gather(
                    self._request(source_worker_id, "scheduler_pause", timeout=300.0),
                    self._request(target_worker_id, "scheduler_pause", timeout=300.0),
                )
                await self._drain_model_outputs_for_migration(
                    pipeline_session_id,
                    source_worker_id=source_worker_id,
                    timeout=300.0,
                )
                await self._request(
                    source_worker_id,
                    "model_output_pause",
                    session_id=pipeline_session_id,
                    timeout=300.0,
                )
                source_output_paused = True
                paused_status_event = await self._request(
                    source_worker_id,
                    "model_output_drain_status",
                    session_id=pipeline_session_id,
                    timeout=300.0,
                )
                if not self._source_model_output_drain_complete(paused_status_event.get("result")):
                    raise RuntimeError("Source output changed while preparing NCCL migration")
                exported = await self._request(
                    source_worker_id,
                    "nccl_export",
                    session_id=pipeline_session_id,
                    transfer_id=token.token_id,
                    timeout=300.0,
                )
                metadata = dict(exported["result"])
                await self._request(
                    target_worker_id,
                    "nccl_prepare_recv",
                    transfer_id=token.token_id,
                    metadata=metadata,
                    source_rank=self._nccl_ranks[source_worker_id],
                    owner_worker_id=target_worker_id,
                    ownership_epoch=token.source_epoch + 1,
                    model_output_credit_window=_MODEL_OUTPUT_PARENT_QUEUE_SIZE,
                    timeout=300.0,
                )
                await asyncio.gather(
                    self._request(
                        source_worker_id,
                        "nccl_send",
                        transfer_id=token.token_id,
                        target_rank=self._nccl_ranks[target_worker_id],
                        timeout=300.0,
                    ),
                    self._request(
                        target_worker_id,
                        "nccl_recv",
                        transfer_id=token.token_id,
                        source_rank=self._nccl_ranks[source_worker_id],
                        timeout=300.0,
                    ),
                )
                await self._request(
                    source_worker_id, "nccl_commit_source", session_id=pipeline_session_id, timeout=300.0
                )
                ownership = self._ownership.commit_migration(token)
                self._pipeline_routes[pipeline_session_id] = target_worker_id
                self._session_workers[pipeline_session_id] = target_worker_id
                for chunk in self._migrating_controls.pop(pipeline_session_id, []):
                    self._send(
                        target_worker_id, {"type": "model_push", "session_id": pipeline_session_id, "chunk": chunk}
                    )
                self._migration_total_ms.append((time.monotonic() - started) * 1000.0)
                return ownership
            except Exception:
                with contextlib.suppress(Exception):
                    await self._request(
                        target_worker_id,
                        "nccl_discard",
                        transfer_id=token.token_id,
                        session_id=pipeline_session_id,
                    )
                with contextlib.suppress(Exception):
                    await self._request(
                        source_worker_id,
                        "nccl_abort_source",
                        session_id=pipeline_session_id,
                        transfer_id=token.token_id,
                    )
                if source_output_paused:
                    with contextlib.suppress(Exception):
                        await self._request(
                            source_worker_id,
                            "model_output_resume",
                            session_id=pipeline_session_id,
                        )
                self._ownership.abort_migration(token)
                for chunk in self._migrating_controls.pop(pipeline_session_id, []):
                    self._send(
                        source_worker_id, {"type": "model_push", "session_id": pipeline_session_id, "chunk": chunk}
                    )
                raise
            finally:
                # A failed state copy must not leave either GPU permanently
                # paused.  Resume is deliberately best-effort: the original
                # migration exception is the meaningful caller-visible error.
                await asyncio.gather(
                    self._request(source_worker_id, "scheduler_resume"),
                    self._request(target_worker_id, "scheduler_resume"),
                    return_exceptions=True,
                )

    def turboserve_snapshot(self) -> dict[str, object]:
        snapshot = super().turboserve_snapshot()
        snapshot.update(
            {
                "migration_supported": bool(self._nccl_ranks),
                "migration_backend": "process_nccl" if self._nccl_ranks else None,
                "nccl_ranks": dict(self._nccl_ranks),
                "worker_runtime_metrics": {
                    worker_id: dict(self._worker_runtime_metrics.get(worker_id, {})) for worker_id in self._specs
                },
                "session_runtime_metrics": dict(self._session_runtime_metrics),
                "migration_calibration": {
                    "average_total_ms": sum(self._migration_total_ms) / len(self._migration_total_ms)
                    if self._migration_total_ms
                    else 0.0
                },
                "model_output_flow_control": self._model_output_flow_snapshot(),
                "dispatch_trace": (
                    self._dispatch_trace.snapshot()
                    if getattr(self, "_dispatch_trace", None) is not None
                    else {"enabled": False}
                ),
            }
        )
        return snapshot

    def _model_output_flow_snapshot(self) -> dict[str, object]:
        outputs = getattr(self, "_model_outputs", {})
        inflight = getattr(self, "_model_output_inflight", set())
        backlog: dict[str, int] = {}
        for session_id, output in outputs.items():
            qsize = getattr(output, "qsize", None)
            if callable(qsize):
                queued = int(qsize())
            else:
                queued = len(getattr(output, "items", ()))
            backlog[session_id] = queued + int(session_id in inflight)
        return {
            "parent_queue_capacity": _MODEL_OUTPUT_PARENT_QUEUE_SIZE,
            "ack_on_dequeue": True,
            "max_materialized_payloads_per_session": _MODEL_OUTPUT_PARENT_QUEUE_SIZE + 1,
            "backlog": backlog,
            "dropped_payloads": dict(getattr(self, "_model_output_dropped", {})),
        }

    async def aclose(self) -> None:
        try:
            for session_id in tuple(self._transport_workers):
                with contextlib.suppress(Exception):
                    await self.stop_session(session_id)
            if self._nccl_ranks:
                await asyncio.gather(
                    *(self._request(worker_id, "nccl_destroy") for worker_id in self._nccl_ranks),
                    return_exceptions=True,
                )
                self._nccl_ranks.clear()
            await super().aclose()
        finally:
            trace = getattr(self, "_dispatch_trace", None)
            if trace is not None:
                trace.close()

    async def _init_nccl(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        workers = sorted(self._active_workers)
        results = await asyncio.gather(
            *(
                self._request(
                    worker_id,
                    "nccl_init",
                    rank=rank,
                    world_size=len(workers),
                    init_method=f"tcp://127.0.0.1:{port}",
                    timeout=_NCCL_INIT_PARENT_TIMEOUT_SECONDS,
                )
                for rank, worker_id in enumerate(workers)
            ),
            return_exceptions=True,
        )
        failures = [
            f"{worker_id}: {result}"
            for worker_id, result in zip(workers, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise RuntimeError(f"NCCL process-group initialization failed: {'; '.join(failures)}")
        self._nccl_ranks = {worker_id: rank for rank, worker_id in enumerate(workers)}

    def _transport_finished(self, session_id: str) -> None:
        self.close_model_session(session_id)

    def _transport_task_done(self, session_id: str, task: asyncio.Task[None]) -> None:
        self._transport_tasks.pop(session_id, None)
        self._transport_workers.pop(session_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("LiveKit transport failed: session=%s error=%s", session_id, task.exception())

    def _record_dispatch_trace(self, event: dict[str, Any]) -> None:
        """Persist one child-reported model invocation in the parent process."""
        trace = getattr(self, "_dispatch_trace", None)
        raw = event.get("trace")
        if trace is None or not isinstance(raw, dict):
            return
        gpu = event.get("gpu")
        record = dict(raw)
        record["worker_id"] = str(event.get("worker_id", "unknown"))
        record["gpu"] = dict(gpu) if isinstance(gpu, dict) else {}
        trace.append(record)

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "model_publisher_frame_tracking":
            session_id = str(event.get("session_id", ""))
            if session_id in self._publisher_frame_tracking:
                self._publisher_frame_tracking[session_id] = bool(event.get("enabled", False))
            return
        if event.get("type") == "model_dispatch_trace":
            self._record_dispatch_trace(event)
            return
        if event.get("type") == "model_output_eos":
            # ``close_model_session`` installs the parent queue sentinel and
            # asks the child to release retained state. It is deliberately
            # idempotent because the transport finally block also calls back.
            self.close_model_session(str(event["session_id"]))
            return
        if event.get("type") == "model_output":
            metrics = event.get("runtime_metrics")
            if isinstance(metrics, dict):
                self._worker_runtime_metrics[event["worker_id"]] = {
                    key: value for key, value in metrics.items() if isinstance(value, int | float | str)
                }
            session_metrics = event.get("session_runtime_metrics")
            if isinstance(session_metrics, dict):
                self._session_runtime_metrics[event["session_id"]] = {
                    key: value for key, value in session_metrics.items() if isinstance(value, int | float)
                }
            output_callback = getattr(self._event_sink, "on_model_output", None)
            if callable(output_callback):
                output_callback(
                    event["worker_id"],
                    event["session_id"],
                    event["payload"],
                    runtime_metrics=metrics if isinstance(metrics, dict) else None,
                    session_runtime_metrics=session_metrics if isinstance(session_metrics, dict) else None,
                )
            self._enqueue_model_output(
                event["session_id"],
                _ModelOutput(worker_id=event["worker_id"], payload=event["payload"]),
            )
            return
        super()._dispatch_event(event)


def _nccl_model_worker_main(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    try:
        asyncio.run(
            _run_nccl_model_worker(spec, config_values, pipeline_file, skip_validation, security_name, commands, events)
        )
    finally:
        _close_queue(commands, join=False)
        _close_queue(events)


async def _run_nccl_model_worker(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    if not spec.gpu_ids:
        raise RuntimeError("process-nccl requires one CUDA GPU per worker")
    torch.cuda.set_device(int(spec.gpu_ids[0]))
    adapter = LiveKitPipelineAdapter(security_level=SecurityLevel[security_name] if security_name else None)
    adapter.start(pipeline_file, skip_validation=skip_validation, gpu_num=1, gpu_ids=spec.gpu_ids)
    if adapter.stream_mode != STREAM_MODE_BIDIRECTIONAL:
        raise RuntimeError("process-nccl requires a bidirectional pipeline")
    # Honour the operator ceiling in process-NCCL too. Previously this path
    # always auto-sized and then overwrote --max-sessions-per-worker at the
    # parent scheduler, which can violate a measured per-session FPS SLO.
    from .config import LiveKitServeConfig

    config = LiveKitServeConfig(**config_values)
    profile = adapter.configure_session_capacity(config.session_capacity_limit())
    events.put(
        {
            "type": "worker_capacity",
            "worker_id": spec.worker_id,
            "capacity": int((profile or {}).get("effective_capacity", 1)),
            "profile": profile,
        }
    )
    events.put({"type": "worker_status", "worker_id": spec.worker_id, "status": "idle"})
    events.put({"type": "worker_ready", "worker_id": spec.worker_id})
    service = adapter.stream_service.service
    set_dispatch_trace_callback = getattr(service, "set_dispatch_trace_callback", None)
    trace_path_value = config.dispatch_trace_path
    if isinstance(trace_path_value, str) and trace_path_value.strip() and callable(set_dispatch_trace_callback):
        try:
            logical_cuda_device: int | None = int(torch.cuda.current_device())
        except Exception:
            logical_cuda_device = None
        trace_gpu = _process_dispatch_trace_gpu_metadata(
            spec,
            logical_cuda_device=logical_cuda_device,
        )

        def forward_dispatch_trace(record: dict[str, Any]) -> None:
            events.put(
                {
                    "type": "model_dispatch_trace",
                    "worker_id": spec.worker_id,
                    "gpu": trace_gpu,
                    "trace": record,
                }
            )

        set_dispatch_trace_callback(forward_dispatch_trace)
    outputs: dict[str, asyncio.Task[None]] = {}
    output_credits: dict[str, asyncio.BoundedSemaphore] = {}
    outgoing: dict[str, dict[tuple[Any, ...], torch.Tensor]] = {}
    incoming: dict[str, tuple[dict[str, Any], dict[tuple[Any, ...], torch.Tensor], str, int, int]] = {}

    def start_pump(session_id: str, *, credit_window: int = _MODEL_OUTPUT_PARENT_QUEUE_SIZE) -> None:
        credits = output_credits.get(session_id)
        if credits is None:
            credits = asyncio.BoundedSemaphore(max(1, int(credit_window)))
            output_credits[session_id] = credits
        if session_id not in outputs:
            outputs[session_id] = asyncio.create_task(
                _pump_model_outputs(
                    adapter,
                    service,
                    worker_id=spec.worker_id,
                    session_id=session_id,
                    credits=credits,
                    events=events,
                ),
                name=f"model-output-{session_id}",
            )

    async def stop_pump(session_id: str, *, drop_credit_state: bool = False) -> None:
        task = outputs.pop(session_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if drop_credit_state:
            output_credits.pop(session_id, None)

    async def result(request_id: str | None, value: Any = True, error: Exception | None = None) -> None:
        if request_id is not None:
            events.put(
                {
                    "type": "command_result",
                    "worker_id": spec.worker_id,
                    "request_id": request_id,
                    "result": value,
                    "error": repr(error) if error else None,
                }
            )

    try:
        while True:
            command = await asyncio.to_thread(commands.get)
            request_id, kind = command.get("request_id"), command["type"]
            try:
                if kind == "model_create":
                    session_id = adapter.create_session(command["config"])
                    try:
                        publisher_tracking_enabled = bool(adapter.enable_publisher_frame_tracking(session_id))
                    except Exception:
                        publisher_tracking_enabled = False
                    events.put(
                        {
                            "type": "model_publisher_frame_tracking",
                            "worker_id": spec.worker_id,
                            "session_id": session_id,
                            "enabled": publisher_tracking_enabled,
                        }
                    )
                    start_pump(
                        session_id,
                        credit_window=int(command.get("model_output_credit_window", _MODEL_OUTPUT_PARENT_QUEUE_SIZE)),
                    )
                elif kind == "model_push":
                    adapter.push_chunk(command["session_id"], command["chunk"])
                elif kind == "model_publisher_frame_progress":
                    adapter.report_publisher_frame_progress(
                        command["session_id"],
                        event=str(command["event"]),
                        frames_delta=int(command["frames_delta"]),
                        sequence=int(command["sequence"]),
                        observed_monotonic_seconds=float(command["observed_monotonic_seconds"]),
                    )
                elif kind == "model_close":
                    await stop_pump(command["session_id"], drop_credit_state=True)
                    adapter.close_session(command["session_id"])
                elif kind == "model_output_credit":
                    credits = output_credits.get(command["session_id"])
                    if credits is not None:
                        with contextlib.suppress(ValueError):
                            credits.release()
                elif kind == "model_output_pause":
                    await stop_pump(command["session_id"])
                elif kind == "model_output_drain_status":
                    migration_drain_status = getattr(service, "migration_drain_status", None)
                    if not callable(migration_drain_status):
                        raise RuntimeError("Pipeline service does not expose migration drain status")
                    status = await asyncio.to_thread(migration_drain_status, command["session_id"])
                    await result(request_id, status)
                    continue

                elif kind == "model_output_resume":
                    has_session = getattr(service, "has_session", None)
                    if not callable(has_session) or has_session(command["session_id"]):
                        start_pump(command["session_id"])
                elif kind == "nccl_init":
                    device_id = torch.device("cuda", torch.cuda.current_device())
                    await asyncio.to_thread(
                        dist.init_process_group,
                        "nccl",
                        init_method=command["init_method"],
                        rank=command["rank"],
                        world_size=command["world_size"],
                        timeout=timedelta(seconds=_NCCL_INIT_GROUP_TIMEOUT_SECONDS),
                        device_id=device_id,
                    )
                elif kind == "scheduler_pause":
                    await asyncio.to_thread(service.pause_scheduler)
                elif kind == "scheduler_resume":
                    service.resume_scheduler()
                elif kind == "nccl_export":
                    metadata = await asyncio.to_thread(service.prepare_migration_nccl_metadata, command["session_id"])
                    outgoing[command["transfer_id"]] = metadata.pop("_nccl_tensor_leaves")
                    await result(request_id, metadata)
                    continue
                elif kind == "nccl_prepare_recv":
                    metadata = command["metadata"]
                    leaves = allocate_tensor_tree_leaves(
                        metadata["tensor_manifest"], torch.device(f"cuda:{spec.gpu_ids[0]}")
                    )
                    incoming[command["transfer_id"]] = (
                        metadata,
                        leaves,
                        command["owner_worker_id"],
                        command["ownership_epoch"],
                        int(command.get("model_output_credit_window", _MODEL_OUTPUT_PARENT_QUEUE_SIZE)),
                    )
                elif kind == "nccl_send":
                    transfer_tensor_leaves_nccl(
                        outgoing.pop(command["transfer_id"]), peer_rank=command["target_rank"], send=True
                    )
                elif kind == "nccl_recv":
                    metadata, leaves, owner, epoch, credit_window = incoming.pop(command["transfer_id"])
                    transfer_tensor_leaves_nccl(leaves, peer_rank=command["source_rank"], send=False)
                    session_id = service.import_migration_nccl(
                        metadata, leaves, owner_worker_id=owner, ownership_epoch=epoch
                    )
                    try:
                        publisher_tracking_enabled = bool(adapter.enable_publisher_frame_tracking(session_id))
                    except Exception:
                        publisher_tracking_enabled = False
                    events.put(
                        {
                            "type": "model_publisher_frame_tracking",
                            "worker_id": spec.worker_id,
                            "session_id": session_id,
                            "enabled": publisher_tracking_enabled,
                        }
                    )
                    start_pump(session_id, credit_window=credit_window)
                elif kind == "nccl_commit_source":
                    await stop_pump(command["session_id"], drop_credit_state=True)
                    service.commit_migration(command["session_id"])
                elif kind == "nccl_abort_source":
                    service.abort_migration(command["session_id"])
                    outgoing.pop(command.get("transfer_id", ""), None)
                elif kind == "nccl_discard":
                    incoming.pop(command["transfer_id"], None)
                    await stop_pump(command["session_id"], drop_credit_state=True)
                    if service.has_session(command["session_id"]):
                        service.close_session(command["session_id"])
                elif kind == "nccl_destroy":
                    if dist.is_initialized():
                        dist.destroy_process_group()
                elif kind == "shutdown":
                    break
                else:
                    raise ValueError(f"Unknown process-nccl command {kind!r}")
                await result(request_id)
            except Exception as exc:
                await result(request_id, error=exc)
    finally:
        for session_id in tuple(outputs):
            await stop_pump(session_id, drop_credit_state=True)
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
        await adapter.aclose()
