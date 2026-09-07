from __future__ import annotations

import asyncio

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.runtime import LiveKitServeRuntime
from telefuser.service.livekit.schemas import SessionCreateRequest
from telefuser.service.livekit.turboserve import TurboServeOwnership


class FakeTokenService:
    def create_token(self, *, identity: str, room_name: str, role: str, **kwargs: object) -> str:
        return f"{role}:{identity}:{room_name}"


class FakeWorkerPool:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.closed = False
        self.start_options: list[bool] = []
        self.close_calls = 0
        self.active_workers = 1
        self.scale_targets: list[int] = []
        self.on_scale = None

    async def start(self, *, skip_validation: bool = False) -> None:
        self.start_options.append(skip_validation)

    def start_session(self, record) -> None:
        self.started.append(record.session_id)

    async def stop_session(self, session_id: str) -> None:
        self.stopped.append(session_id)

    def active_worker_count(self) -> int:
        return self.active_workers

    async def scale_to(self, target_workers: int) -> int:
        self.scale_targets.append(target_workers)
        self.active_workers = target_workers
        if self.on_scale is not None:
            self.on_scale()
        return target_workers

    async def aclose(self) -> None:
        self.closed = True
        self.close_calls += 1


class MigratingWorkerPool(FakeWorkerPool):
    def __init__(self) -> None:
        super().__init__()
        self.migrations: list[tuple[str, str]] = []

    async def migrate_session(self, pipeline_session_id: str, target_worker_id: str) -> TurboServeOwnership:
        self.migrations.append((pipeline_session_id, target_worker_id))
        return TurboServeOwnership(pipeline_session_id, target_worker_id, len(self.migrations) + 1)

    def turboserve_snapshot(self) -> dict[str, object]:
        return {
            "migration_supported": True,
            "worker_runtime_metrics": {
                "worker-0": {"active_sessions": 2, "mean_chunk_seconds": 2.0, "p95_chunk_seconds": 4.0},
                "worker-1": {"active_sessions": 0, "mean_chunk_seconds": 0.5, "p95_chunk_seconds": 1.0},
            },
        }


def test_runtime_autoscaling_scales_out_and_drains_capacity() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            num_workers=2,
            worker_gpu_map="0;1",
            queue_size=2,
            autoscaling_enabled=True,
            autoscaling_min_workers=1,
            autoscaling_cooldown_seconds=0,
        )
        pool = FakeWorkerPool()
        runtime = LiveKitServeRuntime(
            config=config, pipeline_file="pipeline.py", token_service=FakeTokenService(), worker_pool=pool
        )
        runtime.scheduler.update_worker_status("worker-1", "stopped")
        pool.on_scale = lambda: runtime.scheduler.update_worker_status("worker-1", "idle")
        first = runtime.create_session(SessionCreateRequest(identity="controller-1"))
        second = runtime.create_session(SessionCreateRequest(identity="controller-2"))
        assert first.admission.status == "assigned"
        assert second.admission.status == "queued"

        decision = await runtime._autoscale_once()

        assert decision.target_workers == 2
        assert pool.scale_targets == [2]
        assert runtime.registry.require(second.record.session_id).status == "assigned"

    asyncio.run(_run())


def test_runtime_allows_multiple_in_process_gpu_workers() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            num_workers=2,
            worker_gpu_map="0;1",
        )
        pool = FakeWorkerPool()
        runtime = LiveKitServeRuntime(
            config=config, pipeline_file="pipeline.py", token_service=FakeTokenService(), worker_pool=pool
        )
        await runtime.start()
        assert runtime.is_ready is True
        await runtime.aclose()

    asyncio.run(_run())


def test_runtime_starts_queued_session_when_worker_is_released() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            queue_size=1,
        )
        worker_pool = FakeWorkerPool()
        runtime = LiveKitServeRuntime(
            config=config,
            pipeline_file="pipeline.py",
            token_service=FakeTokenService(),
            worker_pool=worker_pool,
        )

        first = runtime.create_session(SessionCreateRequest(identity="controller-1"))
        second = runtime.create_session(SessionCreateRequest(identity="controller-2"))
        await runtime.delete_session(first.record.session_id)

        assert first.record.status == "assigned"
        assert second.record.status == "queued"
        assert worker_pool.stopped == [first.record.session_id]
        assert worker_pool.started == [first.record.session_id, second.record.session_id]
        assert runtime.registry.require(first.record.session_id).status == "closed"
        second_record = runtime.registry.require(second.record.session_id)
        assert second_record.status == "assigned"
        assert second_record.worker_id == "worker-0"

    asyncio.run(_run())


def test_runtime_worker_callbacks_release_capacity() -> None:
    config = LiveKitServeConfig(livekit_url="wss://livekit.example", livekit_api_key="key", livekit_api_secret="secret")
    worker_pool = FakeWorkerPool()
    runtime = LiveKitServeRuntime(
        config=config,
        pipeline_file="pipeline.py",
        token_service=FakeTokenService(),
        worker_pool=worker_pool,
    )
    result = runtime.create_session(SessionCreateRequest(identity="controller-1"))

    runtime.on_pipeline_session(result.record.session_id, "pipeline-1")
    runtime.on_session_status(result.record.session_id, "running")
    runtime.on_session_finished("worker-0", result.record.session_id)

    record = runtime.registry.require(result.record.session_id)
    assert record.status == "closed"
    assert record.pipeline_session_id == "pipeline-1"
    assert runtime.scheduler.health_snapshot()["workers_idle"] == 1


def test_runtime_migration_updates_registry_and_admission_owner() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            num_workers=2,
            worker_gpu_map="0;1",
            max_sessions_per_worker=2,
        )
        pool = MigratingWorkerPool()
        runtime = LiveKitServeRuntime(
            config=config, pipeline_file="pipeline.py", token_service=FakeTokenService(), worker_pool=pool
        )
        created = runtime.create_session(SessionCreateRequest(identity="controller-1"))
        runtime.on_pipeline_session(created.record.session_id, "pipeline-1")

        ownership = await runtime.migrate_session(created.record.session_id, "worker-1")

        assert ownership.worker_id == "worker-1"
        assert pool.migrations == [("pipeline-1", "worker-1")]
        assert runtime.registry.require(created.record.session_id).worker_id == "worker-1"
        workers = {worker.worker_id: worker for worker in runtime.scheduler.workers()}
        assert workers["worker-0"].session_ids == []
        assert workers["worker-1"].session_ids == [created.record.session_id]

    asyncio.run(_run())


def test_runtime_rebalances_one_profitable_migration_from_measured_load() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            num_workers=2,
            worker_gpu_map="0;1",
            max_sessions_per_worker=3,
            turboserve_migration_bandwidth_gbps=1000,
        )
        pool = MigratingWorkerPool()
        runtime = LiveKitServeRuntime(
            config=config, pipeline_file="pipeline.py", token_service=FakeTokenService(), worker_pool=pool
        )
        first = runtime.create_session(SessionCreateRequest(identity="controller-1"))
        second = runtime.create_session(SessionCreateRequest(identity="controller-2"))
        runtime.on_pipeline_session(first.record.session_id, "pipeline-1")
        runtime.on_pipeline_session(second.record.session_id, "pipeline-2")
        runtime.scheduler.reassign_session(second.record.session_id, "worker-0")
        runtime.registry.assign_worker(second.record.session_id, "worker-0")
        runtime.on_worker_capacity("worker-0", 3, {"estimated_session_bytes": 1})
        runtime.on_worker_capacity("worker-1", 3, {"estimated_session_bytes": 1})

        await runtime._rebalance_once()

        assert pool.migrations == [("pipeline-1", "worker-1")]
        assert runtime.registry.require(first.record.session_id).worker_id == "worker-1"
        assert runtime.metadata()["turboserve_rebalance"]["last_plan"]["source_worker_id"] == "worker-0"

    asyncio.run(_run())


def test_runtime_reports_livekit_connected_only_after_room_connection() -> None:
    config = LiveKitServeConfig(livekit_url="wss://livekit.example", livekit_api_key="key", livekit_api_secret="secret")
    runtime = LiveKitServeRuntime(
        config=config,
        pipeline_file="pipeline.py",
        token_service=FakeTokenService(),
        worker_pool=FakeWorkerPool(),
    )
    runtime.create_session(SessionCreateRequest(identity="controller-1"))

    assert runtime.health().livekit_connected is False

    runtime.on_worker_status("worker-0", "starting_pipeline")

    assert runtime.health().livekit_connected is True


def test_runtime_exposes_worker_calculated_capacity() -> None:
    config = LiveKitServeConfig(livekit_url="wss://livekit.example", livekit_api_key="key", livekit_api_secret="secret")
    runtime = LiveKitServeRuntime(
        config=config,
        pipeline_file="pipeline.py",
        token_service=FakeTokenService(),
        worker_pool=FakeWorkerPool(),
    )

    runtime.on_worker_capacity("worker-0", 3, {"effective_capacity": 3, "limiting_device": 1})

    metadata = runtime.metadata()
    assert metadata["max_sessions_per_worker"] == 3
    assert metadata["configured_max_sessions_per_worker"] == "auto"
    assert metadata["session_capacity"] == {"worker-0": {"effective_capacity": 3, "limiting_device": 1}}


def test_runtime_uses_only_reported_capacities_for_autoscaling() -> None:
    config = LiveKitServeConfig(
        livekit_url="wss://livekit.example",
        livekit_api_key="key",
        livekit_api_secret="secret",
        num_workers=2,
        worker_gpu_map="0;1",
        autoscaling_enabled=True,
        queue_size=1,
        autoscaling_min_workers=1,
    )
    runtime = LiveKitServeRuntime(
        config=config,
        pipeline_file="pipeline.py",
        token_service=FakeTokenService(),
        worker_pool=FakeWorkerPool(),
    )
    runtime.scheduler.update_worker_status("worker-1", "stopped")

    runtime.on_worker_capacity("worker-0", 3, {"effective_capacity": 3})

    assert runtime._autoscaling_controller.sessions_per_worker == 3
    assert runtime.metadata()["max_sessions_per_worker"] == 3


def test_runtime_is_unhealthy_when_all_configured_workers_are_failed_or_stopped() -> None:
    config = LiveKitServeConfig(
        livekit_url="wss://livekit.example",
        livekit_api_key="key",
        livekit_api_secret="secret",
        num_workers=2,
    )
    runtime = LiveKitServeRuntime(
        config=config,
        pipeline_file="pipeline.py",
        token_service=FakeTokenService(),
        worker_pool=FakeWorkerPool(),
    )
    runtime.scheduler.update_worker_status("worker-0", "failed")
    runtime.scheduler.update_worker_status("worker-1", "stopped")

    health = runtime.health()

    assert health.status == "unhealthy"
    assert health.workers_idle == 0


def test_runtime_start_and_close_are_idempotent() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
        )
        worker_pool = FakeWorkerPool()
        runtime = LiveKitServeRuntime(
            config=config,
            pipeline_file="pipeline.py",
            token_service=FakeTokenService(),
            worker_pool=worker_pool,
            skip_validation=True,
        )

        await runtime.start()
        await runtime.start()
        assert runtime.is_ready is True
        assert worker_pool.start_options == [True]

        await runtime.aclose()
        await runtime.aclose()
        assert runtime.is_ready is False
        assert worker_pool.close_calls == 1

    asyncio.run(_run())


def test_runtime_releases_capacity_after_worker_reports_failure() -> None:
    config = LiveKitServeConfig(livekit_url="wss://livekit.example", livekit_api_key="key", livekit_api_secret="secret")
    runtime = LiveKitServeRuntime(
        config=config,
        pipeline_file="pipeline.py",
        token_service=FakeTokenService(),
        worker_pool=FakeWorkerPool(),
    )
    result = runtime.create_session(SessionCreateRequest(identity="controller-1"))

    runtime.on_session_status(result.record.session_id, "failed", error="room connect failed")
    runtime.on_session_finished("worker-0", result.record.session_id, error="room connect failed")

    record = runtime.registry.require(result.record.session_id)
    assert record.status == "failed"
    assert record.error == "room connect failed"
    assert runtime.scheduler.health_snapshot()["workers_idle"] == 1
