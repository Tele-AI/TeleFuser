from __future__ import annotations

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.runtime import LiveKitServeRuntime
from telefuser.service.livekit.scheduler import LiveKitScheduler
from telefuser.service.livekit.schemas import SessionCreateRequest


class _TokenService:
    def create_token(self, *, identity: str, room_name: str, role: str, **kwargs: object) -> str:
        return f"{role}:{identity}:{room_name}"


class _WorkerPool:
    def __init__(self) -> None:
        self.started: list[str] = []

    async def start(self, *, skip_validation: bool = False) -> None:
        return None

    def start_session(self, record) -> None:
        self.started.append(record.session_id)

    async def stop_session(self, session_id: str) -> None:
        return None

    async def aclose(self) -> None:
        return None


def test_scheduler_uses_all_retained_slots_before_queueing() -> None:
    scheduler = LiveKitScheduler(num_workers=1, max_sessions_per_worker=2, queue_size=1)

    first = scheduler.assign(session_id="session-1", room_name="room-1")
    second = scheduler.assign(session_id="session-2", room_name="room-2")
    third = scheduler.assign(session_id="session-3", room_name="room-3")

    assert first.status == second.status == "assigned"
    assert third.status == "queued"
    assert scheduler.workers()[0].session_ids == ["session-1", "session-2"]


def test_scheduler_releases_one_slot_without_idling_other_sessions() -> None:
    scheduler = LiveKitScheduler(num_workers=1, max_sessions_per_worker=2, queue_size=1)
    scheduler.assign(session_id="session-1", room_name="room-1")
    scheduler.assign(session_id="session-2", room_name="room-2")
    scheduler.assign(session_id="session-3", room_name="room-3")

    admission = scheduler.release_session("session-1")

    assert admission is not None
    assert admission.session_id == "session-3"
    assert scheduler.workers()[0].session_ids == ["session-2", "session-3"]
    assert scheduler.health_snapshot()["workers_busy"] == 1


def test_scheduler_preserves_the_compatibility_room_for_a_remaining_session() -> None:
    scheduler = LiveKitScheduler(num_workers=1, max_sessions_per_worker=2)
    scheduler.assign(session_id="session-1", room_name="room-1")
    scheduler.assign(session_id="session-2", room_name="room-2")

    scheduler.release_session("session-1")

    worker = scheduler.workers()[0]
    assert worker.session_id == "session-2"
    assert worker.room_name == "room-2"


def test_scheduler_does_not_resurrect_a_failed_worker_when_a_session_finishes() -> None:
    scheduler = LiveKitScheduler(num_workers=1, max_sessions_per_worker=2, queue_size=1)
    scheduler.assign(session_id="session-1", room_name="room-1")
    scheduler.assign(session_id="session-2", room_name="room-2")
    scheduler.assign(session_id="session-3", room_name="room-3")
    scheduler.fail_worker("worker-0", "worker failed")

    admission = scheduler.release_session("session-1")

    assert admission is None
    worker = scheduler.workers()[0]
    assert worker.status == "failed"
    assert worker.error == "worker failed"
    assert worker.session_ids == ["session-2"]
    assert scheduler.health_snapshot()["queued_sessions"] == 1


def test_runtime_starts_two_sessions_on_one_model_worker() -> None:
    worker_pool = _WorkerPool()
    runtime = LiveKitServeRuntime(
        config=LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            max_sessions_per_worker=2,
            control_idle_timeout=8.0,
        ),
        pipeline_file="pipeline.py",
        token_service=_TokenService(),
        worker_pool=worker_pool,
    )

    first = runtime.create_session(SessionCreateRequest(identity="controller-1"))
    second = runtime.create_session(SessionCreateRequest(identity="controller-2"))

    assert first.admission.status == second.admission.status == "assigned"
    assert worker_pool.started == [first.record.session_id, second.record.session_id]
    assert runtime.registry.require(first.record.session_id).config["control_idle_timeout"] == 8.0
    assert runtime.registry.require(second.record.session_id).worker_id == "worker-0"
