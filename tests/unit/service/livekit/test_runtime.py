from __future__ import annotations

import asyncio

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.runtime import LiveKitServeRuntime
from telefuser.service.livekit.schemas import SessionCreateRequest


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

    async def start(self, *, skip_validation: bool = False) -> None:
        self.start_options.append(skip_validation)

    def start_session(self, record) -> None:
        self.started.append(record.session_id)

    async def stop_session(self, session_id: str) -> None:
        self.stopped.append(session_id)

    async def aclose(self) -> None:
        self.closed = True
        self.close_calls += 1


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
