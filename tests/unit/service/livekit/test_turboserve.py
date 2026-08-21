from __future__ import annotations

import pytest

from telefuser.service.livekit.turboserve import (
    TurboServeAutoscalingController,
    TurboServeOwnershipTable,
    TurboServePlacementController,
    TurboServeSessionDemand,
    TurboServeWorkerLoad,
    TurboServeWorkloadDetector,
)


def test_workload_detector_reports_activity_volatility_and_chunk_latency() -> None:
    detector = TurboServeWorkloadDetector(window_seconds=10.0, volatility_bins=5)
    detector.record_arrival("a", now=1.0)
    detector.record_active("a", now=2.0)
    detector.record_chunk(0.2, now=2.0)
    detector.record_chunk(0.4, now=3.0)
    detector.record_idle("a", now=4.0)

    snapshot = detector.snapshot(now=5.0)

    assert snapshot.active_sessions == 0
    assert snapshot.arrivals_per_second == pytest.approx(0.1)
    assert snapshot.mean_chunk_seconds == pytest.approx(0.3)
    assert snapshot.p95_chunk_seconds == pytest.approx(0.4)
    assert snapshot.activation_volatility > 0


def test_placement_retains_owner_and_rebalance_accounts_for_migration_cost() -> None:
    controller = TurboServePlacementController(
        migration_bandwidth_bytes_per_second=1_000_000_000,
        migration_penalty=0.1,
    )
    workers = [
        TurboServeWorkerLoad("gpu-0", 8, 4, 4, 4.0),
        TurboServeWorkerLoad("gpu-1", 8, 1, 1, 1.0),
    ]
    retained = controller.place(
        TurboServeSessionDemand("session-a", True, 100, owner_worker_id="gpu-0"),
        workers,
    )
    assert retained.worker_id == "gpu-0"

    plans = controller.plan_rebalance(
        [TurboServeSessionDemand("session-a", True, 100, owner_worker_id="gpu-0")],
        workers,
    )
    assert len(plans) == 1
    assert plans[0].source_worker_id == "gpu-0"
    assert plans[0].target_worker_id == "gpu-1"
    assert plans[0].gain_seconds > 0


def test_autoscaler_applies_capacity_hysteresis_and_cooldown() -> None:
    controller = TurboServeAutoscalingController(
        sessions_per_worker=4,
        target_utilization=0.75,
        hysteresis=0.05,
        cooldown_seconds=10.0,
        max_workers=8,
    )
    scale_out = controller.decide(8, 1, now=20.0)
    assert scale_out.action == "scale_out"
    assert scale_out.target_workers == 3
    assert controller.decide(8, 1, now=21.0).reason == "cooldown"

    scale_in = controller.decide(1, 4, now=31.0)
    assert scale_in.action == "scale_in"
    assert scale_in.target_workers == 1


def test_ownership_migration_commit_is_atomic_and_epoch_guarded() -> None:
    table = TurboServeOwnershipTable()
    assert table.register("session-a", "gpu-0").epoch == 1
    token = table.prepare_migration("session-a", "gpu-0", "gpu-1")

    committed = table.commit_migration(token)

    assert committed.worker_id == "gpu-1"
    assert committed.epoch == 2
    with pytest.raises(RuntimeError, match="stale"):
        table.commit_migration(token)
