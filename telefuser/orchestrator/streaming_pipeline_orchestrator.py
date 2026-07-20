"""Actor-style orchestration for stateful chunked generation pipelines."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class StageOrdering(str, Enum):
    """Ordering contract for a streaming stage."""

    NONE = "none"
    PER_SESSION_STRICT = "per_session_strict"


class StreamingActorState(str, Enum):
    """Lifecycle and health state of one stage actor."""

    RUNNING = "running"
    FAILED = "failed"
    CLOSED = "closed"


class StreamingSessionStatus(str, Enum):
    """Lifecycle state for one streaming session."""

    RUNNING = "running"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CLOSED = "closed"


class StreamingSessionCloseReason(str, Enum):
    """Reason passed to stage-owned session cleanup."""

    CANCELLED = "cancelled"
    FAILED = "failed"
    CLOSED = "closed"
    SHUTDOWN = "shutdown"


class StreamingActorBusyError(RuntimeError):
    """Raised when a stage actor cannot accept another message yet."""


class StreamingActorFailedError(RuntimeError):
    """Raised when a failed actor poisons every session assigned to it."""


@dataclass(frozen=True)
class StreamingTaskKey:
    """Identity of one actor invocation."""

    session_id: str
    session_epoch: int
    sequence_id: int
    stage_id: str
    request_id: str


@dataclass(frozen=True)
class StreamingSessionContext:
    """Stable identity passed to stage session-lifecycle hooks."""

    session_id: str
    session_epoch: int


@dataclass(frozen=True)
class StreamingStageSpec:
    """Input, output, and ordering contract for a stage actor."""

    stage_id: str
    consumes: frozenset[str]
    produces: frozenset[str]
    ordering: StageOrdering = StageOrdering.PER_SESSION_STRICT
    max_in_flight_per_session: int = 1
    max_in_flight_global: int = 1
    resource_group: str | None = None

    def __post_init__(self) -> None:
        if not self.stage_id or not self.consumes or not self.produces:
            raise ValueError("A stage requires a non-empty ID, at least one input, and at least one output")
        if self.resource_group == "":
            raise ValueError("resource_group must be non-empty when provided")
        if self.max_in_flight_per_session < 1:
            raise ValueError("max_in_flight_per_session must be at least one")
        if self.max_in_flight_global < 1:
            raise ValueError("max_in_flight_global must be at least one")
        if self.max_in_flight_per_session > self.max_in_flight_global:
            raise ValueError("max_in_flight_per_session cannot exceed max_in_flight_global")
        if self.ordering == StageOrdering.PER_SESSION_STRICT and self.max_in_flight_per_session != 1:
            raise ValueError("PER_SESSION_STRICT stages require max_in_flight_per_session=1")


@dataclass(frozen=True)
class StreamingResourceGroupSpec:
    """Concurrency limit shared by one or more logical stages."""

    name: str
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A resource group requires a non-empty name")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")


@dataclass(frozen=True)
class StreamingEdgeSpec:
    """Bounded artifact path; a None source denotes external ingress."""

    artifact: str
    target_stage: str
    source_stage: str | None = None
    capacity_per_session: int = 1

    def __post_init__(self) -> None:
        if not self.artifact or not self.target_stage:
            raise ValueError("An edge requires artifact and target_stage")
        if self.capacity_per_session < 1:
            raise ValueError("capacity_per_session must be at least one")


@dataclass(frozen=True)
class StreamingPipelineSpec:
    """Typed dataflow declaration for StreamingPipelineOrchestrator."""

    stages: tuple[StreamingStageSpec, ...]
    edges: tuple[StreamingEdgeSpec, ...]
    output_artifacts: frozenset[str] = frozenset()
    output_capacity_per_session: int = 1
    resource_groups: tuple[StreamingResourceGroupSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.output_capacity_per_session < 1:
            raise ValueError("output_capacity_per_session must be at least one")


@dataclass(frozen=True)
class StreamingStageInvocation:
    """Message from the scheduler loop to a stage actor."""

    key: StreamingTaskKey
    inputs: Mapping[str, object]
    is_first: bool
    is_last: bool


@dataclass(frozen=True)
class StreamingStageTiming:
    """Scheduler timestamps for one sequence item admitted to a stage."""

    stage_id: str
    sequence_id: int
    inputs_ready_at: float | None
    admitted_at: float | None
    completed_at: float | None


@dataclass(frozen=True)
class StreamingLatencySummary:
    """Distribution summary for scheduler-observed latency samples."""

    count: int
    minimum_seconds: float | None
    p50_seconds: float | None
    p95_seconds: float | None
    maximum_seconds: float | None
    mean_seconds: float | None


@dataclass(frozen=True)
class StreamingSessionMetrics:
    """Immutable end-to-end timing snapshot for one streaming session."""

    ingress_accepted_at: tuple[tuple[int, float], ...]
    output_emitted_at: tuple[tuple[int, float], ...]
    first_output_latency_seconds: float | None
    control_to_output_latency: StreamingLatencySummary
    chunk_period: StreamingLatencySummary


@dataclass(frozen=True)
class StreamingActorHealth:
    """Immutable health snapshot for one stage actor."""

    state: StreamingActorState
    pending_invocations: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class StreamingArtifactStats:
    """Live scheduler-owned artifact slots and consumer references."""

    slot_count: int
    reference_count: int


@dataclass(frozen=True)
class StreamingSchedulerDiagnostics:
    """Counts of anomalous actor results and cleanup failures."""

    stale_epoch_results: int
    orphaned_results: int
    duplicate_results: int
    cleanup_failures: int
    slot_leaks: int


@dataclass(frozen=True)
class StreamingStageIdleInterval:
    """Gap between two causally ordered stage admissions."""

    stage_id: str
    previous_sequence_id: int
    sequence_id: int
    idle_seconds: float
    reason: str
    missing_inputs: tuple[str, ...]


class StreamingStageActor(Protocol):
    """Backend that executes one logical state-owning stage actor."""

    def submit(self, invocation: StreamingStageInvocation) -> Future[Mapping[str, object]]:
        """Submit a request-correlated invocation."""

    def health(self) -> StreamingActorHealth:
        """Return a non-blocking actor health snapshot."""

    def barrier(self, timeout: float = 5.0) -> None:
        """Wait for every accepted invocation to resolve."""

    def close_session(
        self,
        context: StreamingSessionContext,
        reason: StreamingSessionCloseReason,
        timeout: float = 5.0,
    ) -> None:
        """Release actor-owned state for one drained session."""

    def close(self) -> None:
        """Stop admission and release execution resources."""


@dataclass
class _ActorMessage:
    invocation: StreamingStageInvocation
    future: Future[Mapping[str, object]]


@dataclass
class _ActorSessionCloseMessage:
    context: StreamingSessionContext
    reason: StreamingSessionCloseReason
    future: Future[None]


class LocalStageActor:
    """Single-threaded actor with a bounded mailbox for local tests and stages."""

    def __init__(
        self,
        handler: Callable[[StreamingStageInvocation], Mapping[str, object]],
        mailbox_capacity: int = 1,
        name: str = "local-stage-actor",
        session_closer: Callable[[StreamingSessionContext, StreamingSessionCloseReason], None] | None = None,
    ) -> None:
        if mailbox_capacity < 1:
            raise ValueError("mailbox_capacity must be at least one")
        self._handler = handler
        self._session_closer = session_closer
        self._mailbox: queue.Queue[_ActorMessage | _ActorSessionCloseMessage | None] = queue.Queue(
            maxsize=mailbox_capacity
        )
        self._closed = False
        self._failure_reason: str | None = None
        self._pending_invocations = 0
        self._pending_session_closes = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._thread.start()

    def submit(self, invocation: StreamingStageInvocation) -> Future[Mapping[str, object]]:
        """Queue one invocation without waiting for a full mailbox."""
        with self._idle:
            if self._closed:
                raise RuntimeError("Stage actor is closed")
            if self._failure_reason is not None:
                raise RuntimeError(f"Stage actor has failed: {self._failure_reason}")
            future: Future[Mapping[str, object]] = Future()
            try:
                self._mailbox.put_nowait(_ActorMessage(invocation, future))
            except queue.Full as exc:
                raise StreamingActorBusyError("Stage actor mailbox is full") from exc
            self._pending_invocations += 1
            return future

    def health(self) -> StreamingActorHealth:
        """Return actor state without blocking model execution."""
        with self._idle:
            if self._failure_reason is not None:
                state = StreamingActorState.FAILED
            elif self._closed:
                state = StreamingActorState.CLOSED
            else:
                state = StreamingActorState.RUNNING
            return StreamingActorHealth(state, self._pending_invocations, self._failure_reason)

    def barrier(self, timeout: float = 5.0) -> None:
        """Wait until all accepted invocations have reached a terminal future state."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._pending_invocations or self._pending_session_closes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for stage actor quiescence")
                self._idle.wait(remaining)

    def close_session(
        self,
        context: StreamingSessionContext,
        reason: StreamingSessionCloseReason,
        timeout: float = 5.0,
    ) -> None:
        """Run session cleanup in actor order on the actor thread."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        future: Future[None] = Future()
        with self._idle:
            if self._closed:
                raise RuntimeError("Stage actor is closed")
            if self._failure_reason is not None:
                raise RuntimeError(f"Stage actor has failed: {self._failure_reason}")
            self._pending_session_closes += 1
        try:
            self._mailbox.put(_ActorSessionCloseMessage(context, reason, future), timeout=timeout)
        except queue.Full as exc:
            with self._idle:
                self._pending_session_closes -= 1
                self._idle.notify_all()
            raise TimeoutError("Timed out submitting stage session cleanup") from exc
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeoutError as exc:
            raise TimeoutError("Timed out waiting for stage session cleanup") from exc

    def close(self) -> None:
        """Stop after already queued work has completed."""
        with self._idle:
            if self._closed:
                return
            self._closed = True
        self._mailbox.put(None)
        self._thread.join()

    def _run(self) -> None:
        try:
            while True:
                message = self._mailbox.get()
                if message is None:
                    return
                if isinstance(message, _ActorSessionCloseMessage):
                    try:
                        if self._session_closer is not None:
                            self._session_closer(message.context, message.reason)
                    except BaseException as exc:
                        message.future.set_exception(exc)
                    else:
                        message.future.set_result(None)
                    finally:
                        del message
                        self._finish_session_close()
                    continue
                try:
                    if not message.future.set_running_or_notify_cancel():
                        continue
                    try:
                        result = self._handler(message.invocation)
                    except BaseException as exc:
                        message.future.set_exception(exc)
                    else:
                        message.future.set_result(result)
                finally:
                    del message
                    self._finish_invocation()
        except BaseException as exc:
            self._fail(exc)

    def _finish_invocation(self) -> None:
        with self._idle:
            self._pending_invocations -= 1
            self._idle.notify_all()

    def _finish_session_close(self) -> None:
        with self._idle:
            self._pending_session_closes -= 1
            self._idle.notify_all()

    def _fail(self, exc: BaseException) -> None:
        failure_reason = f"{type(exc).__name__}: {exc}"
        with self._idle:
            self._failure_reason = failure_reason
            while True:
                try:
                    message = self._mailbox.get_nowait()
                except queue.Empty:
                    break
                if message is None:
                    continue
                if not message.future.done():
                    try:
                        message.future.set_exception(RuntimeError(f"Stage actor failed: {failure_reason}"))
                    except InvalidStateError:
                        pass
                if isinstance(message, _ActorSessionCloseMessage):
                    self._pending_session_closes -= 1
                else:
                    self._pending_invocations -= 1
            self._idle.notify_all()


@dataclass
class _ArtifactSlot:
    value: object
    remaining_consumers: set[str]

    @property
    def reference_count(self) -> int:
        """Return the number of stages that still own this artifact."""
        return len(self.remaining_consumers)


@dataclass
class _SessionRuntime:
    session_id: str
    epoch: int
    final_sequence_id: int | None
    status: StreamingSessionStatus = StreamingSessionStatus.RUNNING
    frontiers: dict[str, int] = field(default_factory=dict)
    artifacts: dict[tuple[int, str], _ArtifactSlot] = field(default_factory=dict)
    in_flight: dict[tuple[str, int], StreamingTaskKey] = field(default_factory=dict)
    outputs: deque[tuple[int, str, object]] = field(default_factory=deque)
    pending_outputs: dict[tuple[int, str], object] = field(default_factory=dict)
    next_emit_sequence: int = 0
    timings: dict[tuple[str, int], StreamingStageTiming] = field(default_factory=dict)
    ingress_accepted_at: dict[int, float] = field(default_factory=dict)
    output_emitted_at: dict[int, float] = field(default_factory=dict)
    missing_inputs_after_completion: dict[tuple[str, int], tuple[str, ...]] = field(default_factory=dict)
    admission_blocking_reasons: dict[tuple[str, int], str] = field(default_factory=dict)
    cleaned_stages: set[str] = field(default_factory=set)
    cleanup_scheduled: bool = False
    cleanup_complete: bool = False
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    error: BaseException | None = None
    cleanup_error: BaseException | None = None


class StreamingPipelineOrchestrator:
    """Coordinate bounded artifacts between independent stage actors.

    It is intentionally separate from FlexiblePipelineOrchestrator: this class
    schedules sequence-indexed stream items, whereas the existing orchestrator
    retains request-level stage-group semantics.
    """

    def __init__(self, spec: StreamingPipelineSpec, actors: Mapping[str, StreamingStageActor]) -> None:
        self.spec = spec
        self._stages = {stage.stage_id: stage for stage in spec.stages}
        self._resource_groups = {group.name: group for group in spec.resource_groups}
        self._actors = dict(actors)
        self._edges_by_artifact: dict[str, tuple[StreamingEdgeSpec, ...]] = {}
        self._sessions: dict[str, _SessionRuntime] = {}
        self._session_order: deque[str] = deque()
        self._topological_stage_ids: tuple[str, ...] = ()
        self._epoch = 0
        self._closed = False
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._recent_terminal_keys: deque[StreamingTaskKey] = deque()
        self._recent_terminal_key_set: set[StreamingTaskKey] = set()
        self._stale_epoch_results = 0
        self._orphaned_results = 0
        self._duplicate_results = 0
        self._cleanup_failures = 0
        self._slot_leaks = 0
        self._validate()
        self._failed_cleanup_queue: queue.Queue[_SessionRuntime | None] = queue.Queue()
        self._failed_cleanup_thread = threading.Thread(
            target=self._run_failed_cleanups,
            daemon=True,
            name="streaming-failed-session-cleanup",
        )
        self._failed_cleanup_thread.start()

    def create_session(self, session_id: str, final_sequence_id: int | None = None) -> int:
        """Create a session and return its epoch."""
        if final_sequence_id is not None and final_sequence_id < 0:
            raise ValueError("final_sequence_id must be non-negative")
        with self._lock:
            if self._closed:
                raise RuntimeError("Streaming pipeline orchestrator is closed")
            current = self._sessions.get(session_id)
            if current is not None and (current.status == StreamingSessionStatus.RUNNING or current.in_flight):
                raise ValueError(f"Session {session_id!r} is still active")
            self._epoch += 1
            runtime = _SessionRuntime(
                session_id=session_id,
                epoch=self._epoch,
                final_sequence_id=final_sequence_id,
                frontiers={stage_id: 0 for stage_id in self._stages},
            )
            self._sessions[session_id] = runtime
            try:
                self._session_order.remove(session_id)
            except ValueError:
                pass
            self._session_order.append(session_id)
            return runtime.epoch

    def push_input(self, session_id: str, sequence_id: int, artifact: str, value: object) -> None:
        """Provide an external artifact for one sequence item."""
        if not self.try_push_inputs(session_id, sequence_id, {artifact: value}):
            raise RuntimeError(f"Input edge for {artifact!r} is full")

    def try_push_inputs(self, session_id: str, sequence_id: int, inputs: Mapping[str, object]) -> bool:
        """Atomically admit named ingress artifacts, or return False under backpressure."""
        if sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if not inputs:
            raise ValueError("At least one input artifact is required")
        with self._lock:
            runtime = self._active_runtime(session_id)
            if runtime.final_sequence_id is not None and sequence_id > runtime.final_sequence_id:
                raise ValueError(f"Sequence {sequence_id} exceeds final sequence {runtime.final_sequence_id}")
            edges_by_artifact: dict[str, tuple[StreamingEdgeSpec, ...]] = {}
            for artifact in inputs:
                edges = self._external_input_edges(artifact)
                if (sequence_id, artifact) in runtime.artifacts:
                    raise ValueError(f"Artifact {artifact!r} already exists for sequence {sequence_id}")
                edges_by_artifact[artifact] = edges
            if any(
                self._edge_occupancy(runtime, edge) >= edge.capacity_per_session
                for edges in edges_by_artifact.values()
                for edge in edges
            ):
                return False
            runtime.ingress_accepted_at.setdefault(sequence_id, time.monotonic())
            for artifact, value in inputs.items():
                runtime.artifacts[(sequence_id, artifact)] = _ArtifactSlot(
                    value, {edge.target_stage for edge in edges_by_artifact[artifact]}
                )
            self._observe_input_readiness(runtime)
            self._pump_all()
            return True

    def can_push_input(self, session_id: str, artifact: str) -> bool:
        """Return whether an external artifact can be admitted without backpressure."""
        return self.can_push_inputs(session_id, (artifact,))

    def can_push_inputs(self, session_id: str, artifacts: Collection[str]) -> bool:
        """Return whether every named ingress artifact has capacity."""
        if not artifacts:
            raise ValueError("At least one input artifact is required")
        with self._lock:
            runtime = self._active_runtime(session_id)
            return all(
                self._edge_occupancy(runtime, edge) < edge.capacity_per_session
                for artifact in artifacts
                for edge in self._external_input_edges(artifact)
            )

    def finish_inputs(self, session_id: str, final_sequence_id: int) -> None:
        """Mark the final sequence for an open-ended session."""
        if final_sequence_id < 0:
            raise ValueError("final_sequence_id must be non-negative")
        with self._lock:
            runtime = self._active_runtime(session_id)
            if runtime.final_sequence_id not in {None, final_sequence_id}:
                raise ValueError("final_sequence_id cannot be changed")
            if any(
                sequence_id == final_sequence_id and timing.admitted_at is not None
                for (_, sequence_id), timing in runtime.timings.items()
            ):
                raise RuntimeError("final_sequence_id must be declared before its first stage is admitted")
            runtime.final_sequence_id = final_sequence_id
            self._pump_all()

    def poll_outputs(self, session_id: str) -> list[tuple[int, str, object]]:
        """Return and clear emitted artifacts in sequence order."""
        with self._lock:
            runtime = self._runtime(session_id)
            outputs = list(runtime.outputs)
            runtime.outputs.clear()
            self._pump_all()
            return outputs

    def status(self, session_id: str) -> StreamingSessionStatus:
        """Return the lifecycle state of a session."""
        with self._lock:
            return self._runtime(session_id).status

    def error(self, session_id: str) -> BaseException | None:
        """Return the error that failed a session, if any."""
        with self._lock:
            return self._runtime(session_id).error

    def cleanup_error(self, session_id: str) -> BaseException | None:
        """Return the stage cleanup error for a terminal session, if any."""
        with self._lock:
            return self._runtime(session_id).cleanup_error

    def artifact_stats(self, session_id: str) -> StreamingArtifactStats:
        """Return live scheduler-owned artifact slot and reference counts."""
        with self._lock:
            slots = self._runtime(session_id).artifacts.values()
            return StreamingArtifactStats(
                slot_count=len(slots),
                reference_count=sum(slot.reference_count for slot in slots),
            )

    def resource_group_usage(self, name: str) -> int:
        """Return the number of in-flight tasks holding a resource-group permit."""
        with self._lock:
            if name not in self._resource_groups:
                raise KeyError(f"Unknown streaming resource group {name!r}")
            return self._resource_group_in_flight_count(name)

    def diagnostics(self) -> StreamingSchedulerDiagnostics:
        """Return scheduler anomaly counters for lifecycle validation."""
        with self._lock:
            return StreamingSchedulerDiagnostics(
                stale_epoch_results=self._stale_epoch_results,
                orphaned_results=self._orphaned_results,
                duplicate_results=self._duplicate_results,
                cleanup_failures=self._cleanup_failures,
                slot_leaks=self._slot_leaks,
            )

    def stage_timings(self, session_id: str, stage_id: str) -> tuple[StreamingStageTiming, ...]:
        """Return scheduler admission timings for one stage in sequence order."""
        with self._lock:
            if stage_id not in self._stages:
                raise KeyError(f"Unknown streaming stage {stage_id!r}")
            return tuple(
                timing
                for (_, _), timing in sorted(self._runtime(session_id).timings.items(), key=lambda item: item[0][1])
                if timing.stage_id == stage_id
            )

    def session_metrics(self, session_id: str) -> StreamingSessionMetrics:
        """Return end-to-end latency distributions without consuming session state."""
        with self._lock:
            runtime = self._runtime(session_id)
            ingress = tuple(sorted(runtime.ingress_accepted_at.items()))
            emitted = tuple(sorted(runtime.output_emitted_at.items()))
            control_to_output = [
                emitted_at - runtime.ingress_accepted_at[sequence_id]
                for sequence_id, emitted_at in emitted
                if sequence_id in runtime.ingress_accepted_at
            ]
            chunk_periods = [current[1] - previous[1] for previous, current in zip(emitted, emitted[1:])]
            first_output_latency = None
            if ingress and emitted:
                first_output_latency = emitted[0][1] - ingress[0][1]
            return StreamingSessionMetrics(
                ingress_accepted_at=ingress,
                output_emitted_at=emitted,
                first_output_latency_seconds=first_output_latency,
                control_to_output_latency=self._summarize_latencies(control_to_output),
                chunk_period=self._summarize_latencies(chunk_periods),
            )

    def stage_idle_intervals(self, session_id: str, stage_id: str) -> tuple[StreamingStageIdleInterval, ...]:
        """Classify causal stage-admission gaps from scheduler-observed inputs."""
        with self._lock:
            runtime = self._runtime(session_id)
            timings = [
                timing
                for (_, _), timing in sorted(runtime.timings.items(), key=lambda item: item[0][1])
                if timing.stage_id == stage_id and timing.admitted_at is not None and timing.completed_at is not None
            ]
            intervals: list[StreamingStageIdleInterval] = []
            for previous, current in zip(timings, timings[1:]):
                idle_seconds = max(0.0, current.admitted_at - previous.completed_at)
                missing_inputs = runtime.missing_inputs_after_completion.get((stage_id, current.sequence_id), ())
                if idle_seconds <= 0.001:
                    reason = "continuous"
                elif current.inputs_ready_at is None or current.inputs_ready_at > previous.completed_at:
                    causal_inputs = tuple(
                        artifact for artifact in ("condition", "control") if artifact in missing_inputs
                    )
                    reason = "+".join(causal_inputs) if causal_inputs else "inputs_not_ready"
                else:
                    reason = runtime.admission_blocking_reasons.get(
                        (stage_id, current.sequence_id),
                        "scheduler_admission",
                    )
                intervals.append(
                    StreamingStageIdleInterval(
                        stage_id=stage_id,
                        previous_sequence_id=previous.sequence_id,
                        sequence_id=current.sequence_id,
                        idle_seconds=idle_seconds,
                        reason=reason,
                        missing_inputs=missing_inputs,
                    )
                )
            return tuple(intervals)

    def actor_health(self, stage_id: str) -> StreamingActorHealth:
        """Return a health snapshot for one configured stage actor."""
        try:
            actor = self._actors[stage_id]
        except KeyError as exc:
            raise KeyError(f"Unknown streaming stage {stage_id!r}") from exc
        return actor.health()

    def barrier(self, timeout: float = 5.0) -> None:
        """Wait until the graph has no admitted or immediately admissible work."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._idle:
            while any(
                runtime.in_flight
                or (runtime.status == StreamingSessionStatus.RUNNING and self._has_admissible_task(runtime))
                or self._cleanup_is_pending(runtime)
                for runtime in self._sessions.values()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for streaming graph quiescence")
                self._idle.wait(remaining)
        for actor in self._actors.values():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for streaming graph quiescence")
            actor.barrier(remaining)

    def wait_until_idle(self, session_id: str, timeout: float = 5.0) -> bool:
        """Wait for all admitted actor invocations for a session."""
        deadline = time.monotonic() + timeout
        with self._idle:
            runtime = self._runtime(session_id)
            while (
                runtime.in_flight
                or (runtime.status == StreamingSessionStatus.RUNNING and self._has_admissible_task(runtime))
                or self._cleanup_is_pending(runtime)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return (
                not runtime.in_flight
                and not (runtime.status == StreamingSessionStatus.RUNNING and self._has_admissible_task(runtime))
                and not self._cleanup_is_pending(runtime)
            )

    def cancel_session(self, session_id: str, timeout: float = 5.0) -> None:
        """Stop admission, drain submitted work, and release stage-owned session state."""
        with self._idle:
            runtime = self._runtime(session_id)
            if runtime.status == StreamingSessionStatus.RUNNING:
                runtime.status = StreamingSessionStatus.CANCELLED
                self._discard_buffered_state(runtime)
                self._idle.notify_all()
            elif runtime.status != StreamingSessionStatus.CANCELLED:
                return
        self._cleanup_session(runtime, StreamingSessionCloseReason.CANCELLED, timeout)

    def close_session(self, session_id: str, timeout: float = 5.0) -> None:
        """Drain and clean one session before discarding its scheduler state."""
        with self._idle:
            runtime = self._runtime(session_id)
            if runtime.status == StreamingSessionStatus.RUNNING:
                runtime.status = StreamingSessionStatus.CLOSED
            self._discard_buffered_state(runtime)
            reason = self._session_close_reason(runtime)
            self._idle.notify_all()
        self._cleanup_session(runtime, reason, timeout)
        with self._idle:
            if self._sessions.get(session_id) is runtime:
                del self._sessions[session_id]
                try:
                    self._session_order.remove(session_id)
                except ValueError:
                    pass
            self._idle.notify_all()

    def close(self, timeout: float = 300.0) -> None:
        """Drain sessions, release stage state, and close every actor."""
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._idle:
            if self._closed:
                return
            self._closed = True
            runtimes = tuple(self._sessions.values())
            for runtime in runtimes:
                if runtime.status == StreamingSessionStatus.RUNNING:
                    runtime.status = StreamingSessionStatus.CLOSED
                self._discard_buffered_state(runtime)
            self._idle.notify_all()

        first_error: BaseException | None = None
        for runtime in runtimes:
            try:
                self._cleanup_session(
                    runtime,
                    self._session_close_reason(runtime, shutdown=True),
                    max(0.0, deadline - time.monotonic()),
                )
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        self._failed_cleanup_queue.put(None)
        self._failed_cleanup_thread.join()
        for stage_id in reversed(self._topological_stage_ids):
            try:
                self._actors[stage_id].close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        with self._idle:
            self._sessions.clear()
            self._session_order.clear()
            self._idle.notify_all()
        if first_error is not None:
            raise RuntimeError("Failed to close one or more streaming stage actors or sessions") from first_error

    @staticmethod
    def _session_close_reason(
        runtime: _SessionRuntime,
        *,
        shutdown: bool = False,
    ) -> StreamingSessionCloseReason:
        if runtime.status == StreamingSessionStatus.CANCELLED:
            return StreamingSessionCloseReason.CANCELLED
        if runtime.status == StreamingSessionStatus.FAILED:
            return StreamingSessionCloseReason.FAILED
        if shutdown:
            return StreamingSessionCloseReason.SHUTDOWN
        return StreamingSessionCloseReason.CLOSED

    @staticmethod
    def _discard_buffered_state(runtime: _SessionRuntime) -> None:
        runtime.artifacts.clear()
        runtime.pending_outputs.clear()
        runtime.outputs.clear()

    @staticmethod
    def _cleanup_is_pending(runtime: _SessionRuntime) -> bool:
        return runtime.cleanup_scheduled and not runtime.cleanup_complete and runtime.cleanup_error is None

    def _cleanup_session(
        self,
        runtime: _SessionRuntime,
        reason: StreamingSessionCloseReason,
        timeout: float,
    ) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        if not runtime.lifecycle_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            error = TimeoutError(f"Timed out waiting to clean streaming session {runtime.session_id!r}")
            with self._idle:
                self._cleanup_failures += 1
                runtime.cleanup_error = error
                self._idle.notify_all()
            raise error
        try:
            if runtime.cleanup_complete:
                return
            with self._idle:
                runtime.cleanup_error = None
                while runtime.in_flight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        error = TimeoutError(f"Timed out draining streaming session {runtime.session_id!r}")
                        self._cleanup_failures += 1
                        runtime.cleanup_error = error
                        self._idle.notify_all()
                        raise error
                    self._idle.wait(remaining)

            context = StreamingSessionContext(runtime.session_id, runtime.epoch)
            first_error: BaseException | None = None
            for stage_id in reversed(self._topological_stage_ids):
                if stage_id in runtime.cleaned_stages:
                    continue
                closer = getattr(self._actors[stage_id], "close_session", None)
                if closer is None:
                    runtime.cleaned_stages.add(stage_id)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    exc = TimeoutError(f"Timed out cleaning stage {stage_id!r} for session {runtime.session_id!r}")
                else:
                    try:
                        closer(context, reason, remaining)
                    except BaseException as caught:
                        exc = caught
                    else:
                        runtime.cleaned_stages.add(stage_id)
                        continue
                with self._lock:
                    self._cleanup_failures += 1
                if first_error is None:
                    first_error = exc

            with self._idle:
                leaked_slots = len(runtime.artifacts)
                if leaked_slots:
                    self._slot_leaks += leaked_slots
                    self._cleanup_failures += 1
                    self._discard_buffered_state(runtime)
                    if first_error is None:
                        first_error = RuntimeError(
                            f"Session {runtime.session_id!r} retained {leaked_slots} artifact slots after cleanup"
                        )
                runtime.cleanup_complete = first_error is None and len(runtime.cleaned_stages) == len(
                    self._topological_stage_ids
                )
                runtime.cleanup_error = first_error
                self._idle.notify_all()
            if first_error is not None:
                raise RuntimeError(f"Failed to clean streaming session {runtime.session_id!r}") from first_error
        finally:
            runtime.lifecycle_lock.release()

    def _schedule_failed_cleanup(self, runtime: _SessionRuntime) -> None:
        if runtime.status != StreamingSessionStatus.FAILED or runtime.in_flight or runtime.cleanup_scheduled:
            return
        runtime.cleanup_scheduled = True
        self._failed_cleanup_queue.put_nowait(runtime)

    def _run_failed_cleanups(self) -> None:
        while True:
            runtime = self._failed_cleanup_queue.get()
            if runtime is None:
                return
            try:
                self._cleanup_session(runtime, StreamingSessionCloseReason.FAILED, timeout=300.0)
            except BaseException as exc:
                with self._idle:
                    if runtime.cleanup_error is None:
                        self._cleanup_failures += 1
                        runtime.cleanup_error = exc
                    self._idle.notify_all()

    def _validate(self) -> None:
        if not self._stages:
            raise ValueError("A streaming pipeline requires at least one stage")
        if len(self._resource_groups) != len(self.spec.resource_groups):
            raise ValueError("Streaming resource group names must be unique")
        unknown_resource_groups = {
            stage.resource_group
            for stage in self.spec.stages
            if stage.resource_group is not None and stage.resource_group not in self._resource_groups
        }
        if unknown_resource_groups:
            raise ValueError(f"Unknown streaming resource groups: {sorted(unknown_resource_groups)}")
        if len(self._stages) != len(self.spec.stages) or set(self._actors) != set(self._stages):
            raise ValueError("Stage IDs and actor IDs must match exactly")

        declared_producers: dict[str, str] = {}
        for stage in self.spec.stages:
            for artifact in stage.produces:
                previous = declared_producers.get(artifact)
                if previous is not None and previous != stage.stage_id:
                    raise ValueError(f"Artifact {artifact!r} has multiple producers")
                declared_producers[artifact] = stage.stage_id

        grouped: dict[str, list[StreamingEdgeSpec]] = {}
        external_artifacts: set[str] = set()
        seen_edges: set[tuple[str, str | None, str]] = set()
        adjacency: dict[str, set[str]] = {stage_id: set() for stage_id in self._stages}
        indegree: dict[str, int] = {stage_id: 0 for stage_id in self._stages}
        for edge in self.spec.edges:
            edge_key = (edge.artifact, edge.source_stage, edge.target_stage)
            if edge_key in seen_edges:
                raise ValueError(f"Duplicate edge for artifact {edge.artifact!r}")
            seen_edges.add(edge_key)
            target = self._stages.get(edge.target_stage)
            if target is None or edge.artifact not in target.consumes:
                raise ValueError(f"Invalid target for artifact {edge.artifact!r}")
            if edge.source_stage is None:
                if edge.artifact in declared_producers:
                    raise ValueError(f"Produced artifact {edge.artifact!r} cannot also be external input")
                external_artifacts.add(edge.artifact)
            else:
                source = self._stages.get(edge.source_stage)
                if source is None or declared_producers.get(edge.artifact) != edge.source_stage:
                    raise ValueError(f"Invalid source for artifact {edge.artifact!r}")
                if edge.target_stage not in adjacency[edge.source_stage]:
                    adjacency[edge.source_stage].add(edge.target_stage)
                    indegree[edge.target_stage] += 1
            grouped.setdefault(edge.artifact, []).append(edge)

        for stage in self.spec.stages:
            if any(
                not any(edge.target_stage == stage.stage_id for edge in grouped.get(artifact, ()))
                for artifact in stage.consumes
            ):
                raise ValueError(f"Stage {stage.stage_id!r} consumes an unconnected artifact")
            if any(
                not any(edge.source_stage == stage.stage_id for edge in grouped.get(artifact, ()))
                and artifact not in self.spec.output_artifacts
                for artifact in stage.produces
            ):
                raise ValueError(f"Stage {stage.stage_id!r} produces an unconnected artifact")

        for artifact in self.spec.output_artifacts:
            if artifact not in declared_producers:
                raise ValueError(f"Output artifact {artifact!r} has no producer")

        stage_positions = {stage.stage_id: index for index, stage in enumerate(self.spec.stages)}
        roots = deque(stage.stage_id for stage in self.spec.stages if indegree[stage.stage_id] == 0)
        topological_stage_ids: list[str] = []
        while roots:
            stage_id = roots.popleft()
            topological_stage_ids.append(stage_id)
            for target_stage in sorted(adjacency[stage_id], key=stage_positions.__getitem__):
                indegree[target_stage] -= 1
                if indegree[target_stage] == 0:
                    roots.append(target_stage)
        if len(topological_stage_ids) != len(self._stages):
            raise ValueError("Streaming pipeline graph contains a cycle")
        self._topological_stage_ids = tuple(topological_stage_ids)

        reachable_artifacts = set(external_artifacts)
        reachable_stages: set[str] = set()
        made_progress = True
        while made_progress:
            made_progress = False
            for stage in self.spec.stages:
                if stage.stage_id in reachable_stages or not stage.consumes <= reachable_artifacts:
                    continue
                reachable_stages.add(stage.stage_id)
                reachable_artifacts.update(stage.produces)
                made_progress = True
        unreachable_outputs = self.spec.output_artifacts - reachable_artifacts
        if unreachable_outputs:
            raise ValueError(f"Unreachable output artifacts: {sorted(unreachable_outputs)}")

        self._edges_by_artifact = {artifact: tuple(edges) for artifact, edges in grouped.items()}

    def _external_input_edges(self, artifact: str) -> tuple[StreamingEdgeSpec, ...]:
        edges = self._edges_by_artifact.get(artifact, ())
        if not edges or any(edge.source_stage is not None for edge in edges):
            raise ValueError(f"Artifact {artifact!r} is not an external input")
        return edges

    def _pump_all(self) -> None:
        """Admit ready work round-robin across all active sessions."""
        made_progress = True
        while made_progress:
            made_progress = False
            for _ in range(len(self._session_order)):
                session_id = self._session_order.popleft()
                self._session_order.append(session_id)
                runtime = self._sessions.get(session_id)
                if runtime is None or runtime.status != StreamingSessionStatus.RUNNING:
                    continue
                made_progress = self._pump_once(runtime) or made_progress

    def _pump_once(self, runtime: _SessionRuntime) -> bool:
        for stage in self.spec.stages:
            actor_failure = self._actor_failure(stage.stage_id)
            if actor_failure is not None:
                self._fail_all_sessions_for_actor(actor_failure)
                return False
            sequence_id = self._runnable_sequence(runtime, stage)
            if sequence_id is None:
                continue
            if not self._has_execution_capacity(stage):
                reason = "resource_group" if self._resource_group_is_full(stage) else "stage_capacity"
                runtime.admission_blocking_reasons.setdefault((stage.stage_id, sequence_id), reason)
                continue
            if not self._has_output_capacity(runtime, stage):
                runtime.admission_blocking_reasons.setdefault((stage.stage_id, sequence_id), "edge_capacity")
                continue
            inputs = self._take_inputs(runtime, stage, sequence_id)
            key = StreamingTaskKey(runtime.session_id, runtime.epoch, sequence_id, stage.stage_id, uuid.uuid4().hex)
            invocation = StreamingStageInvocation(
                key=key,
                inputs=inputs,
                is_first=sequence_id == 0,
                is_last=runtime.final_sequence_id == sequence_id,
            )
            try:
                future = self._actors[stage.stage_id].submit(invocation)
            except StreamingActorBusyError:
                self._restore_inputs(runtime, stage, sequence_id, inputs)
                continue
            except BaseException as exc:
                self._restore_inputs(runtime, stage, sequence_id, inputs)
                actor_failure = self._actor_failure(stage.stage_id)
                if actor_failure is None:
                    self._fail_runtime(runtime, exc)
                else:
                    self._fail_all_sessions_for_actor(actor_failure, runtime, exc)
                self._idle.notify_all()
                return False
            self._set_timing(runtime, stage.stage_id, sequence_id, admitted_at=time.monotonic())
            runtime.in_flight[(stage.stage_id, sequence_id)] = key
            future.add_done_callback(
                lambda completed, sid=runtime.session_id, task_key=key: self._complete(sid, task_key, completed)
            )
            return True
        return False

    def _has_admissible_task(self, runtime: _SessionRuntime) -> bool:
        return any(
            self._runnable_sequence(runtime, stage) is not None
            and self._has_execution_capacity(stage)
            and self._has_output_capacity(runtime, stage)
            for stage in self.spec.stages
        )

    def _runnable_sequence(self, runtime: _SessionRuntime, stage: StreamingStageSpec) -> int | None:
        count = sum(1 for stage_id, _ in runtime.in_flight if stage_id == stage.stage_id)
        if count >= stage.max_in_flight_per_session:
            return None
        candidates = (
            (runtime.frontiers[stage.stage_id],)
            if stage.ordering == StageOrdering.PER_SESSION_STRICT
            else sorted({sequence_id for sequence_id, _ in runtime.artifacts})
        )
        for sequence_id in candidates:
            if (stage.stage_id, sequence_id) in runtime.in_flight:
                continue
            if all(
                (slot := runtime.artifacts.get((sequence_id, artifact))) is not None
                and stage.stage_id in slot.remaining_consumers
                for artifact in stage.consumes
            ):
                return sequence_id
        return None

    def _has_execution_capacity(self, stage: StreamingStageSpec) -> bool:
        if self._stage_in_flight_count(stage.stage_id) >= stage.max_in_flight_global:
            return False
        if stage.resource_group is None:
            return True
        group = self._resource_groups[stage.resource_group]
        return self._resource_group_in_flight_count(group.name) < group.max_concurrency

    def _resource_group_is_full(self, stage: StreamingStageSpec) -> bool:
        if stage.resource_group is None:
            return False
        group = self._resource_groups[stage.resource_group]
        return self._resource_group_in_flight_count(group.name) >= group.max_concurrency

    def _stage_in_flight_count(self, stage_id: str) -> int:
        return sum(
            1
            for runtime in self._sessions.values()
            for in_flight_stage_id, _ in runtime.in_flight
            if in_flight_stage_id == stage_id
        )

    def _resource_group_in_flight_count(self, name: str) -> int:
        return sum(
            1
            for runtime in self._sessions.values()
            for stage_id, _ in runtime.in_flight
            if self._stages[stage_id].resource_group == name
        )

    def _has_output_capacity(self, runtime: _SessionRuntime, stage: StreamingStageSpec) -> bool:
        internal_capacity = all(
            self._edge_occupancy(runtime, edge) + self._reserved_artifact_count(runtime, edge.artifact)
            < edge.capacity_per_session
            for artifact in stage.produces
            for edge in self._edges_by_artifact.get(artifact, ())
            if edge.source_stage == stage.stage_id
        )
        output_capacity = all(
            self._output_occupancy(runtime, artifact) + self._reserved_artifact_count(runtime, artifact)
            < self.spec.output_capacity_per_session
            for artifact in stage.produces & self.spec.output_artifacts
        )
        return internal_capacity and output_capacity

    def _take_inputs(self, runtime: _SessionRuntime, stage: StreamingStageSpec, sequence_id: int) -> dict[str, object]:
        inputs: dict[str, object] = {}
        for artifact in stage.consumes:
            slot = runtime.artifacts[(sequence_id, artifact)]
            slot.remaining_consumers.remove(stage.stage_id)
            inputs[artifact] = slot.value
            if not slot.remaining_consumers:
                del runtime.artifacts[(sequence_id, artifact)]
        return inputs

    def _restore_inputs(
        self, runtime: _SessionRuntime, stage: StreamingStageSpec, sequence_id: int, inputs: Mapping[str, object]
    ) -> None:
        for artifact, value in inputs.items():
            slot = runtime.artifacts.get((sequence_id, artifact))
            if slot is None:
                runtime.artifacts[(sequence_id, artifact)] = _ArtifactSlot(value, {stage.stage_id})
            else:
                slot.remaining_consumers.add(stage.stage_id)

    def _complete(self, session_id: str, key: StreamingTaskKey, future: Future[Mapping[str, object]]) -> None:
        with self._idle:
            runtime = self._sessions.get(session_id)
            if runtime is None:
                if key in self._recent_terminal_key_set:
                    self._duplicate_results += 1
                else:
                    self._orphaned_results += 1
                self._pump_all()
                self._idle.notify_all()
                return
            if runtime.epoch != key.session_epoch:
                self._stale_epoch_results += 1
                self._pump_all()
                self._idle.notify_all()
                return
            active_key = runtime.in_flight.get((key.stage_id, key.sequence_id))
            if active_key != key:
                if key in self._recent_terminal_key_set:
                    self._duplicate_results += 1
                else:
                    self._orphaned_results += 1
                self._pump_all()
                self._idle.notify_all()
                return

            del runtime.in_flight[(key.stage_id, key.sequence_id)]
            self._record_terminal_key(key)
            self._set_timing(runtime, key.stage_id, key.sequence_id, completed_at=time.monotonic())
            try:
                result = future.result()
            except BaseException as exc:
                result = None
                execution_error: BaseException | None = exc
            else:
                execution_error = None

            actor_failure = self._actor_failure(key.stage_id)
            if actor_failure is not None:
                self._fail_all_sessions_for_actor(actor_failure, runtime, execution_error)
            elif runtime.status == StreamingSessionStatus.RUNNING and execution_error is not None:
                self._fail_runtime(runtime, execution_error)
            elif runtime.status == StreamingSessionStatus.RUNNING:
                try:
                    assert result is not None
                    stage = self._stages[key.stage_id]
                    missing, unexpected = stage.produces - set(result), set(result) - stage.produces
                    if missing or unexpected:
                        raise ValueError(
                            f"Invalid output from {key.stage_id!r}: missing={missing}, unexpected={unexpected}"
                        )
                    for artifact, value in result.items():
                        edges = tuple(
                            edge
                            for edge in self._edges_by_artifact.get(artifact, ())
                            if edge.source_stage == key.stage_id
                        )
                        if artifact in self.spec.output_artifacts:
                            runtime.pending_outputs[(key.sequence_id, artifact)] = value
                        if edges:
                            runtime.artifacts[(key.sequence_id, artifact)] = _ArtifactSlot(
                                value, {edge.target_stage for edge in edges}
                            )
                    self._emit_ready_outputs(runtime)
                    if stage.ordering == StageOrdering.PER_SESSION_STRICT:
                        runtime.frontiers[key.stage_id] = key.sequence_id + 1
                    self._record_successor_missing_inputs(runtime, stage, key.sequence_id)
                    self._observe_input_readiness(runtime)
                except BaseException as exc:
                    self._fail_runtime(runtime, exc)
            self._schedule_failed_cleanup(runtime)
            self._pump_all()
            self._idle.notify_all()

    def _actor_failure(self, stage_id: str) -> StreamingActorFailedError | None:
        health_method = getattr(self._actors[stage_id], "health", None)
        if health_method is None:
            return None
        try:
            health = health_method()
        except BaseException as exc:
            return StreamingActorFailedError(f"Streaming stage actor {stage_id!r} health check failed: {exc}")
        if health.state == StreamingActorState.RUNNING:
            return None
        detail = f": {health.failure_reason}" if health.failure_reason else ""
        return StreamingActorFailedError(f"Streaming stage actor {stage_id!r} is {health.state.value}{detail}")

    def _fail_all_sessions_for_actor(
        self,
        actor_error: StreamingActorFailedError,
        originating_runtime: _SessionRuntime | None = None,
        originating_error: BaseException | None = None,
    ) -> None:
        for runtime in self._sessions.values():
            if runtime.status != StreamingSessionStatus.RUNNING:
                continue
            error = (
                originating_error if runtime is originating_runtime and originating_error is not None else actor_error
            )
            self._fail_runtime(runtime, error)

    def _fail_runtime(self, runtime: _SessionRuntime, error: BaseException) -> None:
        if runtime.status != StreamingSessionStatus.RUNNING:
            return
        runtime.status = StreamingSessionStatus.FAILED
        runtime.error = error
        self._discard_buffered_state(runtime)
        self._schedule_failed_cleanup(runtime)

    def _record_terminal_key(self, key: StreamingTaskKey) -> None:
        if key in self._recent_terminal_key_set:
            return
        if len(self._recent_terminal_keys) >= 4096:
            expired = self._recent_terminal_keys.popleft()
            self._recent_terminal_key_set.remove(expired)
        self._recent_terminal_keys.append(key)
        self._recent_terminal_key_set.add(key)

    @staticmethod
    def _edge_occupancy(runtime: _SessionRuntime, edge: StreamingEdgeSpec) -> int:
        return sum(
            1
            for (_, artifact), slot in runtime.artifacts.items()
            if artifact == edge.artifact and edge.target_stage in slot.remaining_consumers
        )

    def _reserved_artifact_count(self, runtime: _SessionRuntime, artifact: str) -> int:
        return sum(1 for stage_id, _ in runtime.in_flight if artifact in self._stages[stage_id].produces)

    @staticmethod
    def _output_occupancy(runtime: _SessionRuntime, artifact: str) -> int:
        return sum(1 for _, queued_artifact, _ in runtime.outputs if queued_artifact == artifact) + sum(
            1 for _, pending_artifact in runtime.pending_outputs if pending_artifact == artifact
        )

    def _emit_ready_outputs(self, runtime: _SessionRuntime) -> None:
        if not self.spec.output_artifacts:
            return
        while all(
            (runtime.next_emit_sequence, artifact) in runtime.pending_outputs for artifact in self.spec.output_artifacts
        ):
            sequence_id = runtime.next_emit_sequence
            for artifact in sorted(self.spec.output_artifacts):
                value = runtime.pending_outputs.pop((sequence_id, artifact))
                runtime.outputs.append((sequence_id, artifact, value))
            runtime.output_emitted_at[sequence_id] = time.monotonic()
            runtime.next_emit_sequence += 1

    @staticmethod
    def _summarize_latencies(values: Collection[float]) -> StreamingLatencySummary:
        ordered = sorted(values)
        if not ordered:
            return StreamingLatencySummary(0, None, None, None, None, None)

        def percentile(fraction: float) -> float:
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        return StreamingLatencySummary(
            count=len(ordered),
            minimum_seconds=ordered[0],
            p50_seconds=percentile(0.50),
            p95_seconds=percentile(0.95),
            maximum_seconds=ordered[-1],
            mean_seconds=sum(ordered) / len(ordered),
        )

    def _observe_input_readiness(self, runtime: _SessionRuntime) -> None:
        sequence_ids = {sequence_id for sequence_id, _ in runtime.artifacts}
        for stage in self.spec.stages:
            for sequence_id in sequence_ids:
                if all(
                    (slot := runtime.artifacts.get((sequence_id, artifact))) is not None
                    and stage.stage_id in slot.remaining_consumers
                    for artifact in stage.consumes
                ):
                    self._set_timing(runtime, stage.stage_id, sequence_id, inputs_ready_at=time.monotonic())

    def _record_successor_missing_inputs(
        self,
        runtime: _SessionRuntime,
        stage: StreamingStageSpec,
        sequence_id: int,
    ) -> None:
        successor = sequence_id + 1
        missing_inputs = tuple(
            sorted(
                artifact
                for artifact in stage.consumes
                if (slot := runtime.artifacts.get((successor, artifact))) is None
                or stage.stage_id not in slot.remaining_consumers
            )
        )
        runtime.missing_inputs_after_completion[(stage.stage_id, successor)] = missing_inputs

    @staticmethod
    def _set_timing(
        runtime: _SessionRuntime,
        stage_id: str,
        sequence_id: int,
        *,
        inputs_ready_at: float | None = None,
        admitted_at: float | None = None,
        completed_at: float | None = None,
    ) -> None:
        current = runtime.timings.get(
            (stage_id, sequence_id), StreamingStageTiming(stage_id, sequence_id, None, None, None)
        )
        runtime.timings[(stage_id, sequence_id)] = StreamingStageTiming(
            stage_id=stage_id,
            sequence_id=sequence_id,
            inputs_ready_at=current.inputs_ready_at if current.inputs_ready_at is not None else inputs_ready_at,
            admitted_at=admitted_at if admitted_at is not None else current.admitted_at,
            completed_at=completed_at if completed_at is not None else current.completed_at,
        )

    def _runtime(self, session_id: str) -> _SessionRuntime:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown streaming session {session_id!r}") from exc

    def _active_runtime(self, session_id: str) -> _SessionRuntime:
        runtime = self._runtime(session_id)
        if runtime.status != StreamingSessionStatus.RUNNING:
            raise RuntimeError(f"Session {session_id!r} is not accepting input")
        return runtime
