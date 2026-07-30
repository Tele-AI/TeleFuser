"""Exclusive execution leases for time-sliced LingBot sessions."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

ExecutionLeaseStatus = Literal["waiting", "queued", "active", "parked", "closing"]


@dataclass(frozen=True)
class ExecutionLeaseTransition:
    """One externally observable lease state transition."""

    session_id: str
    status: ExecutionLeaseStatus


@dataclass(frozen=True)
class ExecutionLeaseSnapshot:
    """Immutable state for one registered session."""

    session_id: str
    status: ExecutionLeaseStatus
    last_activity_at: float | None
    busy: bool
    queue_position: int | None


@dataclass
class _ExecutionLeaseEntry:
    idle_timeout: float
    status: ExecutionLeaseStatus = "waiting"
    last_activity_at: float | None = None
    busy: bool = False
    has_run_since_grant: bool = False
    protect_next_grant: bool = False


class ExecutionLeaseManager:
    """Grant one model-execution lease and yield it at chunk boundaries."""

    def __init__(self) -> None:
        self._entries: dict[str, _ExecutionLeaseEntry] = {}
        self._waiting: deque[str] = deque()
        self._active_session_id: str | None = None
        self._closed = False
        self._condition = threading.Condition(threading.RLock())

    def register(self, session_id: str, *, idle_timeout: float) -> None:
        """Register a session that will request execution on its first control."""
        if idle_timeout <= 0:
            raise ValueError("idle_timeout must be positive")
        with self._condition:
            if self._closed:
                raise RuntimeError("Execution lease manager is closed")
            if session_id in self._entries:
                raise ValueError(f"Execution lease session {session_id!r} already exists")
            self._entries[session_id] = _ExecutionLeaseEntry(idle_timeout=float(idle_timeout))

    def record_activity(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> tuple[ExecutionLeaseTransition, ...]:
        """Renew an active lease or queue a parked session for execution."""
        observed_at = time.monotonic() if now is None else now
        with self._condition:
            entry = self._require_entry(session_id)
            if entry.status == "closing":
                return ()
            entry.last_activity_at = observed_at
            transitions: list[ExecutionLeaseTransition] = []
            if entry.status in {"waiting", "parked"}:
                entry.protect_next_grant = self._active_session_id is not None
                entry.status = "queued"
                if session_id not in self._waiting:
                    self._waiting.append(session_id)
                transitions.append(ExecutionLeaseTransition(session_id, "queued"))

            active_id = self._active_session_id
            if active_id is not None and active_id != session_id:
                active = self._entries[active_id]
                if active.has_run_since_grant and not active.busy and self._is_expired(active, observed_at):
                    transitions.extend(self._park_active_locked())
            transitions.extend(self._grant_next_locked())
            self._condition.notify_all()
            return tuple(transitions)

    def wait_for_turn(self, session_id: str) -> bool:
        """Block until ``session_id`` owns the lease or begins closing."""
        with self._condition:
            while True:
                entry = self._entries.get(session_id)
                if entry is None or entry.status == "closing" or self._closed:
                    return False
                if self._active_session_id == session_id and entry.status == "active":
                    return True
                self._condition.wait()

    def begin_chunk(self, session_id: str) -> bool:
        """Reserve the active lease until the submitted chunk completes."""
        with self._condition:
            entry = self._entries.get(session_id)
            if entry is None or entry.status != "active" or self._active_session_id != session_id or entry.busy:
                return False
            entry.busy = True
            entry.has_run_since_grant = True
            return True

    def finish_chunk(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> tuple[ExecutionLeaseTransition, ...]:
        """Release a chunk boundary and yield an expired lease when demand exists."""
        observed_at = time.monotonic() if now is None else now
        with self._condition:
            entry = self._require_entry(session_id)
            if not entry.busy:
                raise RuntimeError(f"Execution lease session {session_id!r} has no chunk in flight")
            entry.busy = False
            transitions: list[ExecutionLeaseTransition] = []
            if entry.status == "active" and self._waiting and self._is_expired(entry, observed_at):
                transitions.extend(self._park_active_locked())
                transitions.extend(self._grant_next_locked())
            self._condition.notify_all()
            return tuple(transitions)

    def yield_if_idle(
        self,
        session_id: str,
        *,
        now: float | None = None,
    ) -> tuple[ExecutionLeaseTransition, ...]:
        """Yield an idle lease when demand arrived before the timeout elapsed."""
        observed_at = time.monotonic() if now is None else now
        with self._condition:
            entry = self._require_entry(session_id)
            transitions: list[ExecutionLeaseTransition] = []
            if (
                self._active_session_id == session_id
                and entry.status == "active"
                and entry.has_run_since_grant
                and not entry.busy
                and self._waiting
                and self._is_expired(entry, observed_at)
            ):
                transitions.extend(self._park_active_locked())
                transitions.extend(self._grant_next_locked())
                self._condition.notify_all()
            return tuple(transitions)

    def abort_chunk(self, session_id: str) -> None:
        """Clear an in-flight reservation after submission fails."""
        with self._condition:
            entry = self._entries.get(session_id)
            if entry is not None:
                entry.busy = False
            self._condition.notify_all()

    def deactivate(self, session_id: str) -> None:
        """Stop a session from acquiring more work while cleanup drains."""
        with self._condition:
            entry = self._entries.get(session_id)
            if entry is None:
                return
            entry.status = "closing"
            self._remove_waiter_locked(session_id)
            self._condition.notify_all()

    def release(self, session_id: str) -> tuple[ExecutionLeaseTransition, ...]:
        """Forget a cleaned session and grant any newly available lease."""
        with self._condition:
            entry = self._entries.pop(session_id, None)
            if entry is None:
                return ()
            self._remove_waiter_locked(session_id)
            if self._active_session_id == session_id:
                self._active_session_id = None
            transitions = tuple(self._grant_next_locked())
            self._condition.notify_all()
            return transitions

    def snapshot(self, session_id: str) -> ExecutionLeaseSnapshot:
        """Return the current lease state for diagnostics and tests."""
        with self._condition:
            entry = self._require_entry(session_id)
            queue_position = None
            if session_id in self._waiting:
                queue_position = tuple(self._waiting).index(session_id) + 1
            return ExecutionLeaseSnapshot(
                session_id=session_id,
                status=entry.status,
                last_activity_at=entry.last_activity_at,
                busy=entry.busy,
                queue_position=queue_position,
            )

    def close(self) -> None:
        """Wake every waiter and reject subsequent registrations."""
        with self._condition:
            self._closed = True
            self._waiting.clear()
            for entry in self._entries.values():
                entry.status = "closing"
            self._condition.notify_all()

    def _grant_next_locked(self) -> list[ExecutionLeaseTransition]:
        if self._active_session_id is not None:
            return []
        while self._waiting:
            session_id = self._waiting.popleft()
            entry = self._entries.get(session_id)
            if entry is None or entry.status != "queued":
                continue
            entry.status = "active"
            entry.has_run_since_grant = not entry.protect_next_grant
            entry.protect_next_grant = False
            self._active_session_id = session_id
            return [ExecutionLeaseTransition(session_id, "active")]
        return []

    def _park_active_locked(self) -> list[ExecutionLeaseTransition]:
        session_id = self._active_session_id
        if session_id is None:
            return []
        entry = self._entries[session_id]
        if entry.busy:
            return []
        entry.status = "parked"
        self._active_session_id = None
        return [ExecutionLeaseTransition(session_id, "parked")]

    @staticmethod
    def _is_expired(entry: _ExecutionLeaseEntry, now: float) -> bool:
        return entry.last_activity_at is not None and now - entry.last_activity_at >= entry.idle_timeout

    def _remove_waiter_locked(self, session_id: str) -> None:
        try:
            self._waiting.remove(session_id)
        except ValueError:
            pass

    def _require_entry(self, session_id: str) -> _ExecutionLeaseEntry:
        try:
            return self._entries[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown execution lease session {session_id!r}") from exc
