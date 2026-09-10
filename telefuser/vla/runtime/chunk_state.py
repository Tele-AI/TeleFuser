"""Deterministic lifecycle for inferred action chunks."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from threading import RLock

from ..contracts import RobotActionChunk


class ChunkStatus(str, Enum):
    """Lifecycle state of one submitted action chunk request."""

    PENDING = "pending"
    READY = "ready"
    EXECUTING = "executing"
    EXECUTED = "executed"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    REJECTED = "rejected"


class RuntimeState(str, Enum):
    """Externally visible state of the chunk runtime."""

    EMPTY = "empty"
    PENDING = "pending"
    READY = "ready"
    EXECUTING = "executing"
    HOLDING = "holding"
    STOPPED = "stopped"


class RemainderPolicy(str, Enum):
    """Treatment of actions beyond one execution horizon."""

    DISCARD = "discard"
    RETAIN = "retain"


class DisconnectPolicy(str, Enum):
    """Fail-closed state selected when the transport disconnects."""

    HOLD = "hold"
    STOP = "stop"


@dataclass(frozen=True)
class ChunkTicket:
    """Identity and independent deadlines for one prediction request."""

    generation: int
    sequence_id: int
    observation_timestamp_ns: int
    received_at: float
    deadline_at: float | None


@dataclass
class _ChunkRecord:
    ticket: ChunkTicket
    status: ChunkStatus
    inference_started: bool = False
    inference_completed: bool = False
    recovery_applied: bool = False
    reason: str | None = None


class ActionChunkStateMachine:
    """Track pending, ready, and executing chunks independently of transport."""

    def __init__(
        self,
        episode_id: str,
        *,
        execute_horizon: int,
        max_observation_age_ns: int | None = None,
        remainder_policy: RemainderPolicy = RemainderPolicy.DISCARD,
        disconnect_policy: DisconnectPolicy = DisconnectPolicy.HOLD,
        stateful_policy: bool = False,
        recover_stateful_policy: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        terminal_history: int = 128,
    ) -> None:
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(execute_horizon, int) or isinstance(execute_horizon, bool) or execute_horizon < 1:
            raise ValueError("execute_horizon must be a positive integer")
        if max_observation_age_ns is not None and (
            not isinstance(max_observation_age_ns, int)
            or isinstance(max_observation_age_ns, bool)
            or max_observation_age_ns < 1
        ):
            raise ValueError("max_observation_age_ns must be a positive integer")
        if stateful_policy and recover_stateful_policy is None:
            raise ValueError("stateful policies require a discard recovery callback")
        if not isinstance(terminal_history, int) or isinstance(terminal_history, bool) or terminal_history < 1:
            raise ValueError("terminal_history must be a positive integer")
        self.episode_id = episode_id
        self.execute_horizon = execute_horizon
        self.max_observation_age_ns = max_observation_age_ns
        self.remainder_policy = RemainderPolicy(remainder_policy)
        self.disconnect_policy = DisconnectPolicy(disconnect_policy)
        self.stateful_policy = stateful_policy
        self._recover_stateful_policy = recover_stateful_policy
        self._clock = clock
        self._terminal_history = terminal_history
        self._generation = 0
        self._latest_sequence_id: int | None = None
        self._latest_observation_timestamp_ns: int | None = None
        self._pending: ChunkTicket | None = None
        self._ready: tuple[ChunkTicket, RobotActionChunk] | None = None
        self._executing: tuple[ChunkTicket, RobotActionChunk] | None = None
        self._remainder: RobotActionChunk | None = None
        self._disconnected_state: RuntimeState | None = None
        self._records: OrderedDict[int, _ChunkRecord] = OrderedDict()
        self._lock = RLock()

    @property
    def state(self) -> RuntimeState:
        """Return the current externally visible runtime state."""
        with self._lock:
            if self._disconnected_state is not None:
                return self._disconnected_state
            if self._executing is not None:
                return RuntimeState.EXECUTING
            if self._ready is not None:
                return RuntimeState.READY
            if self._pending is not None:
                return RuntimeState.PENDING
            return RuntimeState.EMPTY

    @property
    def latest_sequence_id(self) -> int | None:
        """Return the newest accepted request sequence."""
        with self._lock:
            return self._latest_sequence_id

    def submit(
        self,
        sequence_id: int,
        observation_timestamp_ns: int,
        *,
        request_ttl_ms: float | None = None,
        observation_clock_now_ns: int | None = None,
    ) -> ChunkTicket:
        """Admit a prediction request and supersede older pending or ready work."""
        self._validate_nonnegative_int(sequence_id, "sequence_id")
        self._validate_nonnegative_int(observation_timestamp_ns, "observation_timestamp_ns")
        if observation_clock_now_ns is not None:
            self._validate_nonnegative_int(observation_clock_now_ns, "observation_clock_now_ns")
        ttl_ms = self._validate_ttl(request_ttl_ms)
        with self._lock:
            now = self._now()
            self._generation += 1
            ticket = ChunkTicket(
                generation=self._generation,
                sequence_id=sequence_id,
                observation_timestamp_ns=observation_timestamp_ns,
                received_at=now,
                deadline_at=None if ttl_ms is None else now + ttl_ms / 1000.0,
            )
            record = _ChunkRecord(ticket=ticket, status=ChunkStatus.PENDING)
            self._records[ticket.generation] = record
            if self._disconnected_state is not None:
                self._set_terminal(record, ChunkStatus.REJECTED, "runtime is disconnected")
                return ticket
            if self._latest_sequence_id is not None and sequence_id <= self._latest_sequence_id:
                self._set_terminal(record, ChunkStatus.REJECTED, "sequence_id is not newer than the latest request")
                return ticket
            if self._observation_is_stale(observation_timestamp_ns, observation_clock_now_ns):
                self._set_terminal(record, ChunkStatus.EXPIRED, "observation timestamp is stale")
                return ticket
            if (
                self._latest_observation_timestamp_ns is not None
                and observation_timestamp_ns < self._latest_observation_timestamp_ns
            ):
                self._set_terminal(record, ChunkStatus.REJECTED, "observation timestamp moved backwards")
                return ticket
            if self._pending is not None:
                self._set_terminal(
                    self._record(self._pending),
                    ChunkStatus.SUPERSEDED,
                    "a newer observation superseded the pending request",
                )
            if self._ready is not None:
                self._set_terminal(
                    self._record(self._ready[0]),
                    ChunkStatus.SUPERSEDED,
                    "a newer observation superseded the ready chunk",
                )
                self._ready = None
            self._pending = ticket
            self._latest_sequence_id = sequence_id
            if (
                self._latest_observation_timestamp_ns is None
                or observation_timestamp_ns > self._latest_observation_timestamp_ns
            ):
                self._latest_observation_timestamp_ns = observation_timestamp_ns
            self._trim_history()
            return ticket

    def mark_inference_started(self, ticket: ChunkTicket) -> ChunkStatus:
        """Mark that a policy call may mutate state for this request."""
        with self._lock:
            record = self._record(ticket)
            if record.status is ChunkStatus.PENDING:
                record.inference_started = True
            return record.status

    def complete(
        self,
        ticket: ChunkTicket,
        chunk: RobotActionChunk,
        *,
        observation_clock_now_ns: int | None = None,
    ) -> ChunkStatus:
        """Publish a completed chunk or deterministically discard it."""
        if observation_clock_now_ns is not None:
            self._validate_nonnegative_int(observation_clock_now_ns, "observation_clock_now_ns")
        with self._lock:
            record = self._record(ticket)
            record.inference_completed = True
            if record.status is not ChunkStatus.PENDING or self._pending != ticket:
                self._recover_discarded_state(record)
                self._trim_history()
                return record.status
            try:
                self._validate_chunk(ticket, chunk)
            except Exception:
                self._pending = None
                self._set_terminal(record, ChunkStatus.REJECTED, "completed chunk failed validation")
                self._recover_discarded_state(record)
                raise
            now = self._now()
            if ticket.deadline_at is not None and now >= ticket.deadline_at:
                self._pending = None
                self._set_terminal(record, ChunkStatus.EXPIRED, "request_ttl_ms elapsed before completion")
                self._recover_discarded_state(record)
                return record.status
            if self._observation_is_stale(ticket.observation_timestamp_ns, observation_clock_now_ns):
                self._pending = None
                self._set_terminal(record, ChunkStatus.EXPIRED, "observation became stale before completion")
                self._recover_discarded_state(record)
                return record.status
            self._pending = None
            self._ready = (ticket, chunk)
            record.status = ChunkStatus.READY
            record.reason = None
            return record.status

    def begin_execution(self) -> RobotActionChunk | None:
        """Move the newest ready chunk into execution and apply execute_horizon."""
        with self._lock:
            if self._disconnected_state is not None:
                return None
            if self._executing is not None:
                raise RuntimeError("an action chunk is already executing")
            if self._ready is None:
                return None
            ticket, chunk = self._ready
            self._ready = None
            execution_length = min(self.execute_horizon, chunk.valid_length)
            execution_chunk = chunk.trim(execution_length)
            if self.remainder_policy is RemainderPolicy.RETAIN and execution_length < chunk.valid_length:
                remaining = chunk.actions[execution_length : chunk.valid_length]
                self._remainder = replace(chunk, actions=remaining, valid_length=int(remaining.shape[0]))
            else:
                self._remainder = None
            self._executing = (ticket, execution_chunk)
            self._record(ticket).status = ChunkStatus.EXECUTING
            return execution_chunk

    def finish_execution(self, *, success: bool = True) -> RuntimeState:
        """Finish the active horizon and optionally expose retained remainder."""
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")
        with self._lock:
            if self._executing is None:
                raise RuntimeError("no action chunk is executing")
            ticket, _ = self._executing
            record = self._record(ticket)
            self._executing = None
            if not success:
                self._remainder = None
                self._set_terminal(record, ChunkStatus.REJECTED, "simulator execution failed")
            elif self._remainder is not None:
                self._ready = (ticket, self._remainder)
                self._remainder = None
                record.status = ChunkStatus.READY
                record.reason = None
            else:
                record.status = ChunkStatus.EXECUTED
                record.reason = None
                self._trim_history()
            return self.state

    def disconnect(self) -> RuntimeState:
        """Discard buffered work and enter the configured fail-closed state."""
        with self._lock:
            for ticket in self._active_tickets():
                self._set_terminal(self._record(ticket), ChunkStatus.REJECTED, "transport disconnected")
            self._pending = None
            self._ready = None
            self._executing = None
            self._remainder = None
            self._disconnected_state = (
                RuntimeState.HOLDING if self.disconnect_policy is DisconnectPolicy.HOLD else RuntimeState.STOPPED
            )
            self._trim_history()
            return self._disconnected_state

    def reset(self, episode_id: str | None = None) -> None:
        """Clear every chunk slot and restart sequence ordering."""
        if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
            raise ValueError("episode_id must be a non-empty string")
        with self._lock:
            for ticket in self._active_tickets():
                record = self._record(ticket)
                self._set_terminal(record, ChunkStatus.REJECTED, "runtime reset")
                self._recover_discarded_state(record)
            if episode_id is not None:
                self.episode_id = episode_id
            self._pending = None
            self._ready = None
            self._executing = None
            self._remainder = None
            self._latest_sequence_id = None
            self._latest_observation_timestamp_ns = None
            self._disconnected_state = None
            self._trim_history()

    def status(self, ticket: ChunkTicket) -> ChunkStatus:
        """Return the current status of a known ticket."""
        with self._lock:
            return self._record(ticket).status

    def reason(self, ticket: ChunkTicket) -> str | None:
        """Return the terminal reason for a known ticket."""
        with self._lock:
            return self._record(ticket).reason

    def _observation_is_stale(self, timestamp_ns: int, observation_clock_now_ns: int | None) -> bool:
        if self.max_observation_age_ns is None or observation_clock_now_ns is None:
            return False
        return observation_clock_now_ns - timestamp_ns > self.max_observation_age_ns

    def _validate_chunk(self, ticket: ChunkTicket, chunk: RobotActionChunk) -> None:
        if not isinstance(chunk, RobotActionChunk):
            raise TypeError("completed action must be a RobotActionChunk")
        if chunk.episode_id != self.episode_id:
            raise ValueError("completed chunk episode_id does not match the runtime")
        if chunk.sequence_id != ticket.sequence_id:
            raise ValueError("completed chunk sequence_id does not match its ticket")
        if chunk.observation_timestamp_ns != ticket.observation_timestamp_ns:
            raise ValueError("completed chunk observation timestamp does not match its ticket")

    def _recover_discarded_state(self, record: _ChunkRecord) -> None:
        if not self.stateful_policy or not record.inference_started or record.recovery_applied:
            return
        assert self._recover_stateful_policy is not None
        self._recover_stateful_policy(self.episode_id)
        record.recovery_applied = True

    def _set_terminal(self, record: _ChunkRecord, status: ChunkStatus, reason: str) -> None:
        record.status = status
        record.reason = reason
        self._trim_history()

    def _record(self, ticket: ChunkTicket) -> _ChunkRecord:
        try:
            return self._records[ticket.generation]
        except KeyError as error:
            raise KeyError(f"unknown or expired chunk ticket: {ticket.generation}") from error

    def _active_tickets(self) -> tuple[ChunkTicket, ...]:
        tickets: list[ChunkTicket] = []
        if self._pending is not None:
            tickets.append(self._pending)
        if self._ready is not None:
            tickets.append(self._ready[0])
        if self._executing is not None:
            tickets.append(self._executing[0])
        return tuple({ticket.generation: ticket for ticket in tickets}.values())

    def _trim_history(self) -> None:
        terminal = {
            ChunkStatus.EXECUTED,
            ChunkStatus.SUPERSEDED,
            ChunkStatus.EXPIRED,
            ChunkStatus.REJECTED,
        }
        protected = {ticket.generation for ticket in self._active_tickets()}
        protected.update(
            generation
            for generation, record in self._records.items()
            if record.inference_started and not record.inference_completed
        )
        removable = [
            generation
            for generation, record in self._records.items()
            if record.status in terminal and generation not in protected
        ]
        for generation in removable[: -self._terminal_history]:
            self._records.pop(generation)

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("chunk runtime clock must return a finite number")
        return float(value)

    @staticmethod
    def _validate_nonnegative_int(value: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _validate_ttl(value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("request_ttl_ms must be a positive finite number")
        ttl_ms = float(value)
        if not math.isfinite(ttl_ms) or ttl_ms <= 0:
            raise ValueError("request_ttl_ms must be a positive finite number")
        return ttl_ms
