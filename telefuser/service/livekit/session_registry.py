"""In-memory session registry for LiveKit serving."""

from __future__ import annotations

import threading

from pydantic import BaseModel, Field

from .schemas import SessionStatus, new_session_id, utc_timestamp

TERMINAL_SESSION_STATUSES: frozenset[str] = frozenset({"closed", "failed", "expired"})


class SessionRecord(BaseModel):
    """Authoritative API-side state for one LiveKit-backed TeleFuser session."""

    session_id: str
    room_name: str
    controller_identity: str
    status: SessionStatus
    worker_id: str | None = None
    pipeline_session_id: str | None = None
    config: dict = Field(default_factory=dict)
    error: str | None = None
    created_at: float
    updated_at: float
    expires_at: float | None = None
    participant_count: int = 0


class SessionRegistry:
    """Thread-safe in-memory registry for session records."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}
        self._lock = threading.RLock()

    def create(
        self,
        *,
        controller_identity: str,
        config: dict,
        session_id: str | None = None,
        room_name: str | None = None,
        timeout_s: int | None = None,
    ) -> SessionRecord:
        """Create and store a new pending session."""
        now = utc_timestamp()
        sid = session_id or new_session_id()
        record = SessionRecord(
            session_id=sid,
            room_name=room_name or f"tf-world-{sid}",
            controller_identity=controller_identity,
            status="pending",
            config=dict(config),
            created_at=now,
            updated_at=now,
            expires_at=now + timeout_s if timeout_s else None,
        )
        with self._lock:
            if sid in self._records:
                raise ValueError(f"Session {sid} already exists")
            self._records[sid] = record
            return record.model_copy(deep=True)

    def get(self, session_id: str) -> SessionRecord | None:
        """Return a session record copy if present."""
        with self._lock:
            record = self._records.get(session_id)
            return record.model_copy(deep=True) if record is not None else None

    def require(self, session_id: str) -> SessionRecord:
        """Return a session record or raise ``KeyError``."""
        record = self.get(session_id)
        if record is None:
            raise KeyError(f"Session {session_id} not found")
        return record

    def assign_worker(self, session_id: str, worker_id: str) -> SessionRecord:
        """Mark a session as assigned to a worker."""
        return self.update(session_id, status="assigned", worker_id=worker_id)

    def set_pipeline_session(self, session_id: str, pipeline_session_id: str) -> SessionRecord:
        """Record the pipeline-owned session id returned by ``create_session``."""
        return self.update(session_id, pipeline_session_id=pipeline_session_id)

    def update_status(self, session_id: str, status: SessionStatus, *, error: str | None = None) -> SessionRecord:
        """Update only the public session status and optional error."""
        return self.update(session_id, status=status, error=error)

    def set_participant_count(self, session_id: str, participant_count: int) -> SessionRecord:
        """Update the current room participant count."""
        return self.update(session_id, participant_count=participant_count)

    def close(self, session_id: str) -> SessionRecord:
        """Mark a session closed."""
        return self.update_status(session_id, "closed")

    def fail(self, session_id: str, error: str) -> SessionRecord:
        """Mark a session failed."""
        return self.update_status(session_id, "failed", error=error)

    def expire(self, session_id: str) -> SessionRecord:
        """Mark a session expired."""
        return self.update_status(session_id, "expired")

    def update(self, session_id: str, **updates: object) -> SessionRecord:
        """Apply field updates and return a copy of the new record."""
        with self._lock:
            record = self._records.get(session_id)
            if record is None:
                raise KeyError(f"Session {session_id} not found")
            for key, value in updates.items():
                setattr(record, key, value)
            record.updated_at = utc_timestamp()
            return record.model_copy(deep=True)

    def delete(self, session_id: str) -> bool:
        """Delete a session record."""
        with self._lock:
            return self._records.pop(session_id, None) is not None

    def list_records(self) -> list[SessionRecord]:
        """Return copies of all known sessions."""
        with self._lock:
            return [record.model_copy(deep=True) for record in self._records.values()]
