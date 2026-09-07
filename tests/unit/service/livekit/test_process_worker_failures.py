from __future__ import annotations

import asyncio
import multiprocessing
from typing import Any

import pytest

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.process_worker_pool import ProcessLiveKitWorkerPool, ProcessWorkerSpec
from telefuser.service.livekit.runtime import LiveKitServeRuntime
from telefuser.service.livekit.session_registry import SessionRecord


def _exit_during_start(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    del spec, config_values, pipeline_file, skip_validation, security_name, commands, events
    raise SystemExit(9)


def _exit_during_stop(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    del config_values, pipeline_file, skip_validation, security_name
    events.put({"type": "worker_ready", "worker_id": spec.worker_id})
    while True:
        command = commands.get()
        if command["type"] == "stop_session":
            raise SystemExit(11)


def _exit_before_shutdown_ack(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    del config_values, pipeline_file, skip_validation, security_name
    events.put({"type": "worker_ready", "worker_id": spec.worker_id})
    while commands.get()["type"] != "shutdown":
        pass


class _Sink:
    def __init__(self) -> None:
        self.worker_statuses: list[tuple[str, str]] = []

    def on_worker_status(self, worker_id: str, status: str) -> None:
        self.worker_statuses.append((worker_id, status))

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        return None

    def on_session_status(self, session_id: str, status: str, error: str | None = None) -> None:
        return None

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        return None

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        return None


def _config(**updates: object) -> LiveKitServeConfig:
    values: dict[str, object] = {
        "livekit_url": "wss://livekit.example",
        "livekit_api_key": "key",
        "livekit_api_secret": "secret",
        "worker_mode": "process",
        **updates,
    }
    return LiveKitServeConfig(**values)


def test_process_worker_start_fails_immediately_when_child_exits() -> None:
    async def _run() -> None:
        pool = ProcessLiveKitWorkerPool(
            [ProcessWorkerSpec("worker-0", ["0"])],
            config=_config(),
            pipeline_file="pipeline.py",
            event_sink=_Sink(),
            context=multiprocessing.get_context("spawn"),
            worker_target=_exit_during_start,
        )

        with pytest.raises(RuntimeError, match="exited during startup with code 9"):
            await asyncio.wait_for(pool.start(), timeout=10)

    asyncio.run(_run())


def test_process_runtime_requires_explicit_gpu_map_for_multiple_workers() -> None:
    async def _run() -> None:
        runtime = LiveKitServeRuntime(
            config=_config(num_workers=2),
            pipeline_file="pipeline.py",
        )
        with pytest.raises(ValueError, match="worker_gpu_map is required"):
            await runtime.start()
        await runtime.aclose()

    asyncio.run(_run())


def test_pending_command_fails_when_child_exits() -> None:
    async def _run() -> None:
        pool = ProcessLiveKitWorkerPool(
            [ProcessWorkerSpec("worker-0", ["0"])],
            config=_config(),
            pipeline_file="pipeline.py",
            event_sink=_Sink(),
            context=multiprocessing.get_context("spawn"),
            worker_target=_exit_during_stop,
        )
        await pool.start()
        pool.start_session(
            SessionRecord(
                session_id="session-1",
                room_name="room-1",
                controller_identity="controller-1",
                status="assigned",
                worker_id="worker-0",
                config={},
                created_at=0,
                updated_at=0,
            )
        )

        with pytest.raises(RuntimeError, match="exited unexpectedly with code 11"):
            await asyncio.wait_for(pool.stop_session("session-1"), timeout=3)
        await pool.aclose()

    asyncio.run(_run())


def test_clean_child_exit_completes_shutdown_without_ack() -> None:
    async def _run() -> None:
        sink = _Sink()
        pool = ProcessLiveKitWorkerPool(
            [ProcessWorkerSpec("worker-0", ["0"])],
            config=_config(),
            pipeline_file="pipeline.py",
            event_sink=sink,
            context=multiprocessing.get_context("spawn"),
            worker_target=_exit_before_shutdown_ack,
        )
        await pool.start()

        await asyncio.wait_for(pool.aclose(), timeout=3)

        assert ("worker-0", "failed") not in sink.worker_statuses

    asyncio.run(_run())
