from __future__ import annotations

from telefuser.service.livekit.scheduler import LiveKitScheduler


def test_scheduler_assigns_first_idle_worker() -> None:
    scheduler = LiveKitScheduler(num_workers=1, gpu_groups=[["0"]])

    admission = scheduler.assign(session_id="session-1", room_name="room-1")

    assert admission.status == "assigned"
    assert admission.worker_id == "worker-0"
    worker = scheduler.workers()[0]
    assert worker.session_id == "session-1"
    assert worker.gpu_ids == ["0"]


def test_scheduler_rejects_when_busy_and_queue_disabled() -> None:
    scheduler = LiveKitScheduler(num_workers=1, queue_size=0)
    scheduler.assign(session_id="session-1", room_name="room-1")

    admission = scheduler.assign(session_id="session-2", room_name="room-2")

    assert admission.status == "rejected"
    assert admission.reason == "no_idle_worker"


def test_scheduler_queues_and_assigns_on_release() -> None:
    scheduler = LiveKitScheduler(num_workers=1, queue_size=1)
    scheduler.assign(session_id="session-1", room_name="room-1")

    queued = scheduler.assign(session_id="session-2", room_name="room-2")
    next_admission = scheduler.release_session("session-1")

    assert queued.status == "queued"
    assert queued.queue_position == 1
    assert next_admission is not None
    assert next_admission.status == "assigned"
    worker = scheduler.workers()[0]
    assert worker.session_id == "session-2"


def test_scheduler_health_counts_failed_workers() -> None:
    scheduler = LiveKitScheduler(num_workers=2)
    scheduler.assign(session_id="session-1", room_name="room-1")
    scheduler.fail_worker("worker-1", "boom")

    assert scheduler.health_snapshot() == {
        "workers_total": 2,
        "workers_idle": 0,
        "workers_busy": 1,
        "workers_failed": 1,
        "queued_sessions": 0,
    }
