from __future__ import annotations

from telefuser.pipelines.lingbot_world_fast.lease import ExecutionLeaseManager


def test_execution_lease_yields_expired_busy_session_at_chunk_boundary() -> None:
    manager = ExecutionLeaseManager()
    manager.register("session-a", idle_timeout=5.0)
    manager.register("session-b", idle_timeout=5.0)

    assert [transition.status for transition in manager.record_activity("session-a", now=0.0)] == [
        "queued",
        "active",
    ]
    assert manager.begin_chunk("session-a") is True

    assert [transition.status for transition in manager.record_activity("session-b", now=6.0)] == ["queued"]
    assert manager.snapshot("session-a").status == "active"

    transitions = manager.finish_chunk("session-a", now=6.0)

    assert [(transition.session_id, transition.status) for transition in transitions] == [
        ("session-a", "parked"),
        ("session-b", "active"),
    ]
    assert manager.snapshot("session-b").status == "active"


def test_execution_lease_yields_after_queued_waiter_reaches_idle_deadline() -> None:
    manager = ExecutionLeaseManager()
    manager.register("session-a", idle_timeout=5.0)
    manager.register("session-b", idle_timeout=5.0)
    manager.record_activity("session-a", now=0.0)
    manager.record_activity("session-b", now=1.0)

    assert manager.yield_if_idle("session-a", now=4.99) == ()

    transitions = manager.yield_if_idle("session-a", now=5.0)

    assert [(transition.session_id, transition.status) for transition in transitions] == [
        ("session-a", "parked"),
        ("session-b", "active"),
    ]
    assert manager.snapshot("session-a").status == "parked"
    assert manager.snapshot("session-b").status == "active"


def test_execution_lease_switches_idle_session_immediately_when_no_chunk_is_running() -> None:
    manager = ExecutionLeaseManager()
    manager.register("session-a", idle_timeout=5.0)
    manager.register("session-b", idle_timeout=5.0)
    manager.record_activity("session-a", now=0.0)

    transitions = manager.record_activity("session-b", now=5.0)

    assert [(transition.session_id, transition.status) for transition in transitions] == [
        ("session-b", "queued"),
        ("session-a", "parked"),
        ("session-b", "active"),
    ]


def test_execution_lease_requeues_parked_session_at_fifo_tail() -> None:
    manager = ExecutionLeaseManager()
    manager.register("session-a", idle_timeout=1.0)
    manager.register("session-b", idle_timeout=1.0)
    manager.record_activity("session-a", now=0.0)
    manager.begin_chunk("session-a")
    manager.record_activity("session-b", now=2.0)
    manager.finish_chunk("session-a", now=2.0)

    transitions = manager.record_activity("session-a", now=3.0)

    assert [(transition.session_id, transition.status) for transition in transitions] == [("session-a", "queued")]
    assert manager.snapshot("session-b").status == "active"

    assert manager.begin_chunk("session-b") is True
    transitions = manager.finish_chunk("session-b", now=3.0)

    assert [(transition.session_id, transition.status) for transition in transitions] == [
        ("session-b", "parked"),
        ("session-a", "active"),
    ]


def test_execution_lease_guarantees_one_chunk_after_a_stale_waiter_is_granted() -> None:
    manager = ExecutionLeaseManager()
    manager.register("session-a", idle_timeout=5.0)
    manager.register("session-b", idle_timeout=5.0)
    manager.record_activity("session-a", now=0.0)
    manager.begin_chunk("session-a")
    manager.record_activity("session-b", now=1.0)

    manager.finish_chunk("session-a", now=10.0)
    transitions = manager.record_activity("session-a", now=10.1)

    assert [(transition.session_id, transition.status) for transition in transitions] == [("session-a", "queued")]
    assert manager.snapshot("session-b").status == "active"

    assert manager.begin_chunk("session-b") is True
    transitions = manager.finish_chunk("session-b", now=10.2)
    assert [(transition.session_id, transition.status) for transition in transitions] == [
        ("session-b", "parked"),
        ("session-a", "active"),
    ]


def test_execution_lease_deactivation_waits_for_explicit_cleanup_release() -> None:
    manager = ExecutionLeaseManager()
    manager.register("session-a", idle_timeout=1.0)
    manager.register("session-b", idle_timeout=1.0)
    manager.record_activity("session-a", now=0.0)
    manager.record_activity("session-b", now=2.0)
    assert manager.snapshot("session-b").status == "active"

    manager.deactivate("session-b")
    assert manager.snapshot("session-b").status == "closing"
    assert manager.snapshot("session-a").status == "parked"

    assert manager.release("session-b") == ()
