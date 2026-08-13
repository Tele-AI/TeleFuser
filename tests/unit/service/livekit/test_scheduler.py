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


def test_scheduler_balances_by_normalized_retained_load() -> None:
    scheduler = LiveKitScheduler(num_workers=2, max_sessions_per_worker=2)
    first = scheduler.assign(session_id="session-1", room_name="room-1")
    second = scheduler.assign(session_id="session-2", room_name="room-2")

    assert first.worker_id == "worker-0"
    assert second.worker_id == "worker-1"


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


def test_scheduler_drains_queue_across_new_capacity() -> None:
    scheduler = LiveKitScheduler(num_workers=2, queue_size=2)
    scheduler.update_worker_status("worker-1", "stopped")
    scheduler.assign(session_id="session-1", room_name="room-1")
    assert scheduler.assign(session_id="session-2", room_name="room-2").status == "queued"
    scheduler.update_worker_status("worker-1", "idle")

    admissions = scheduler.drain_queue()

    assert [(item.session_id, item.worker_id) for item in admissions] == [("session-2", "worker-1")]


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


def test_scheduler_does_not_count_stopped_workers_as_idle() -> None:
    scheduler = LiveKitScheduler(num_workers=2)
    scheduler.update_worker_status("worker-1", "stopped")

    assert scheduler.health_snapshot() == {
        "workers_total": 2,
        "workers_idle": 1,
        "workers_busy": 0,
        "workers_failed": 0,
        "queued_sessions": 0,
    }


def test_scheduler_updates_worker_capacity_before_admission() -> None:
    scheduler = LiveKitScheduler(num_workers=1, max_sessions_per_worker=1)

    worker = scheduler.update_worker_capacity("worker-0", 3)

    assert worker.session_capacity == 3
    assert scheduler.workers()[0].session_capacity == 3


def test_scheduler_reassigns_an_admitted_session_after_migration() -> None:
    scheduler = LiveKitScheduler(num_workers=2, max_sessions_per_worker=2)
    scheduler.assign(session_id="session-1", room_name="room-1")

    admission = scheduler.reassign_session("session-1", "worker-1")

    assert admission.worker_id == "worker-1"
    workers = {worker.worker_id: worker for worker in scheduler.workers()}
    assert workers["worker-0"].session_ids == []
    assert workers["worker-1"].session_ids == ["session-1"]
