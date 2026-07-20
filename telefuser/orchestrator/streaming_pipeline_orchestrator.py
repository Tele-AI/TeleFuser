"""Actor-style orchestration for stateful chunked generation pipelines."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class StageOrdering(str, Enum):
    """Ordering contract for a streaming stage."""

    NONE = "none"
    PER_SESSION_STRICT = "per_session_strict"


class StreamingSessionStatus(str, Enum):
    """Lifecycle state for one streaming session."""

    RUNNING = "running"
    CANCELLED = "cancelled"
    FAILED = "failed"
    CLOSED = "closed"


class StreamingActorBusyError(RuntimeError):
    """Raised when a stage actor cannot accept another message yet."""


@dataclass(frozen=True)
class StreamingTaskKey:
    """Identity of one actor invocation."""

    session_id: str
    session_epoch: int
    sequence_id: int
    stage_id: str
    request_id: str


@dataclass(frozen=True)
class StreamingStageSpec:
    """Input, output, and ordering contract for a stage actor."""

    stage_id: str
    consumes: frozenset[str]
    produces: frozenset[str]
    ordering: StageOrdering = StageOrdering.PER_SESSION_STRICT
    max_in_flight_per_session: int = 1

    def __post_init__(self) -> None:
        if not self.stage_id or not self.consumes or not self.produces:
            raise ValueError("A stage requires a non-empty ID, at least one input, and at least one output")
        if self.max_in_flight_per_session < 1:
            raise ValueError("max_in_flight_per_session must be at least one")
        if self.ordering == StageOrdering.PER_SESSION_STRICT and self.max_in_flight_per_session != 1:
            raise ValueError("PER_SESSION_STRICT stages require max_in_flight_per_session=1")


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

    def close(self) -> None:
        """Stop admission and release execution resources."""


@dataclass
class _ActorMessage:
    invocation: StreamingStageInvocation
    future: Future[Mapping[str, object]]


class LocalStageActor:
    """Single-threaded actor with a bounded mailbox for local tests and stages."""

    def __init__(
        self,
        handler: Callable[[StreamingStageInvocation], Mapping[str, object]],
        mailbox_capacity: int = 1,
        name: str = "local-stage-actor",
    ) -> None:
        if mailbox_capacity < 1:
            raise ValueError("mailbox_capacity must be at least one")
        self._handler = handler
        self._mailbox: queue.Queue[_ActorMessage | None] = queue.Queue(maxsize=mailbox_capacity)
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._thread.start()

    def submit(self, invocation: StreamingStageInvocation) -> Future[Mapping[str, object]]:
        """Queue one invocation without waiting for a full mailbox."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Stage actor is closed")
            future: Future[Mapping[str, object]] = Future()
            try:
                self._mailbox.put_nowait(_ActorMessage(invocation, future))
            except queue.Full as exc:
                raise StreamingActorBusyError("Stage actor mailbox is full") from exc
            return future

    def close(self) -> None:
        """Stop after already queued work has completed."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._mailbox.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            message = self._mailbox.get()
            if message is None:
                return
            if not message.future.set_running_or_notify_cancel():
                continue
            try:
                message.future.set_result(self._handler(message.invocation))
            except BaseException as exc:
                message.future.set_exception(exc)


@dataclass
class _ArtifactSlot:
    value: object
    consumers: set[str]


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
    missing_inputs_after_completion: dict[tuple[str, int], tuple[str, ...]] = field(default_factory=dict)
    error: BaseException | None = None


class StreamingPipelineOrchestrator:
    """Coordinate bounded artifacts between independent stage actors.

    It is intentionally separate from FlexiblePipelineOrchestrator: this class
    schedules sequence-indexed stream items, whereas the existing orchestrator
    retains request-level stage-group semantics.
    """

    def __init__(self, spec: StreamingPipelineSpec, actors: Mapping[str, StreamingStageActor]) -> None:
        self.spec = spec
        self._stages = {stage.stage_id: stage for stage in spec.stages}
        self._actors = dict(actors)
        self._edges_by_artifact: dict[str, tuple[StreamingEdgeSpec, ...]] = {}
        self._sessions: dict[str, _SessionRuntime] = {}
        self._session_order: deque[str] = deque()
        self._epoch = 0
        self._lock = threading.RLock()
        self._idle = threading.Condition(self._lock)
        self._validate()

    def create_session(self, session_id: str, final_sequence_id: int | None = None) -> int:
        """Create a session and return its epoch."""
        if final_sequence_id is not None and final_sequence_id < 0:
            raise ValueError("final_sequence_id must be non-negative")
        with self._lock:
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
                    reason = "inputs_not_ready"
                else:
                    reason = "scheduler_admission"
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

    def wait_until_idle(self, session_id: str, timeout: float = 5.0) -> bool:
        """Wait for all admitted actor invocations for a session."""
        deadline = time.monotonic() + timeout
        with self._idle:
            runtime = self._runtime(session_id)
            while runtime.in_flight or (
                runtime.status == StreamingSessionStatus.RUNNING and self._has_admissible_task(runtime)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return not runtime.in_flight and not (
                runtime.status == StreamingSessionStatus.RUNNING and self._has_admissible_task(runtime)
            )

    def cancel_session(self, session_id: str) -> None:
        """Reject new work and discard artifacts; in-flight work is drained."""
        with self._idle:
            runtime = self._runtime(session_id)
            if runtime.status != StreamingSessionStatus.RUNNING:
                return
            runtime.status = StreamingSessionStatus.CANCELLED
            runtime.artifacts.clear()
            runtime.pending_outputs.clear()
            runtime.outputs.clear()
            self._idle.notify_all()

    def close_session(self, session_id: str, timeout: float = 5.0) -> None:
        """Stop one session, drain submitted work, and discard scheduler-owned state."""
        with self._idle:
            runtime = self._runtime(session_id)
            if runtime.status == StreamingSessionStatus.RUNNING:
                runtime.status = StreamingSessionStatus.CLOSED
            runtime.artifacts.clear()
            runtime.pending_outputs.clear()
            runtime.outputs.clear()
            self._idle.notify_all()
        if not self.wait_until_idle(session_id, timeout=timeout):
            raise TimeoutError(f"Timed out draining streaming session {session_id!r}")
        with self._idle:
            if self._sessions.get(session_id) is runtime:
                del self._sessions[session_id]
                try:
                    self._session_order.remove(session_id)
                except ValueError:
                    pass
            self._idle.notify_all()

    def close(self) -> None:
        """Close the independent actor graph without changing other orchestrators."""
        with self._idle:
            for runtime in self._sessions.values():
                if runtime.status == StreamingSessionStatus.RUNNING:
                    runtime.status = StreamingSessionStatus.CLOSED
                runtime.artifacts.clear()
                runtime.pending_outputs.clear()
                runtime.outputs.clear()
            self._idle.notify_all()
        for actor in self._actors.values():
            actor.close()
        with self._idle:
            self._sessions.clear()
            self._session_order.clear()

    def _validate(self) -> None:
        if not self._stages:
            raise ValueError("A streaming pipeline requires at least one stage")
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

        roots = deque(stage_id for stage_id, degree in indegree.items() if degree == 0)
        visited = 0
        while roots:
            stage_id = roots.popleft()
            visited += 1
            for target_stage in adjacency[stage_id]:
                indegree[target_stage] -= 1
                if indegree[target_stage] == 0:
                    roots.append(target_stage)
        if visited != len(self._stages):
            raise ValueError("Streaming pipeline graph contains a cycle")

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
            sequence_id = self._runnable_sequence(runtime, stage)
            if sequence_id is None or not self._has_output_capacity(runtime, stage):
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
                runtime.status = StreamingSessionStatus.FAILED
                runtime.error = exc
                runtime.artifacts.clear()
                runtime.pending_outputs.clear()
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
            self._runnable_sequence(runtime, stage) is not None and self._has_output_capacity(runtime, stage)
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
                and stage.stage_id in slot.consumers
                for artifact in stage.consumes
            ):
                return sequence_id
        return None

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
            slot.consumers.remove(stage.stage_id)
            inputs[artifact] = slot.value
            if not slot.consumers and artifact not in self.spec.output_artifacts:
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
                slot.consumers.add(stage.stage_id)

    def _complete(self, session_id: str, key: StreamingTaskKey, future: Future[Mapping[str, object]]) -> None:
        with self._idle:
            runtime = self._sessions.get(session_id)
            if runtime is None or runtime.epoch != key.session_epoch:
                self._pump_all()
                return
            active_key = runtime.in_flight.pop((key.stage_id, key.sequence_id), None)
            if active_key != key:
                self._pump_all()
                self._idle.notify_all()
                return
            self._set_timing(runtime, key.stage_id, key.sequence_id, completed_at=time.monotonic())
            if runtime.status != StreamingSessionStatus.RUNNING:
                self._pump_all()
                self._idle.notify_all()
                return
            try:
                result = future.result()
                stage = self._stages[key.stage_id]
                missing, unexpected = stage.produces - set(result), set(result) - stage.produces
                if missing or unexpected:
                    raise ValueError(
                        f"Invalid output from {key.stage_id!r}: missing={missing}, unexpected={unexpected}"
                    )
                for artifact, value in result.items():
                    edges = tuple(
                        edge for edge in self._edges_by_artifact.get(artifact, ()) if edge.source_stage == key.stage_id
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
                runtime.status = StreamingSessionStatus.FAILED
                runtime.error = exc
                runtime.artifacts.clear()
                runtime.pending_outputs.clear()
            self._pump_all()
            self._idle.notify_all()

    @staticmethod
    def _edge_occupancy(runtime: _SessionRuntime, edge: StreamingEdgeSpec) -> int:
        return sum(
            1
            for (_, artifact), slot in runtime.artifacts.items()
            if artifact == edge.artifact and edge.target_stage in slot.consumers
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
            runtime.next_emit_sequence += 1

    def _observe_input_readiness(self, runtime: _SessionRuntime) -> None:
        sequence_ids = {sequence_id for sequence_id, _ in runtime.artifacts}
        for stage in self.spec.stages:
            for sequence_id in sequence_ids:
                if all(
                    (slot := runtime.artifacts.get((sequence_id, artifact))) is not None
                    and stage.stage_id in slot.consumers
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
                or stage.stage_id not in slot.consumers
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
