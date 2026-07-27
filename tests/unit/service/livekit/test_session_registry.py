from __future__ import annotations

from telefuser.service.livekit.session_registry import SessionRegistry


def test_session_registry_lifecycle() -> None:
    registry = SessionRegistry()

    record = registry.create(
        session_id="session-1",
        room_name="room-1",
        controller_identity="user-1",
        config={"fps": 16},
        timeout_s=60,
    )

    assert record.status == "pending"
    assert record.expires_at is not None

    assigned = registry.assign_worker("session-1", "worker-0")
    assert assigned.status == "assigned"
    assert assigned.worker_id == "worker-0"

    pipeline_record = registry.set_pipeline_session("session-1", "pipeline-1")
    assert pipeline_record.pipeline_session_id == "pipeline-1"

    closed = registry.close("session-1")
    assert closed.status == "closed"


def test_session_registry_returns_copies() -> None:
    registry = SessionRegistry()
    record = registry.create(controller_identity="user-1", config={}, session_id="session-1")
    record.status = "failed"

    stored = registry.require("session-1")

    assert stored.status == "pending"
