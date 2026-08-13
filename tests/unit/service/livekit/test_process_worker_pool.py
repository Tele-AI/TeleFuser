from __future__ import annotations

import asyncio
import multiprocessing
import time
from typing import Any
from unittest.mock import patch

import pytest

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.process_worker_pool import ProcessLiveKitWorkerPool, ProcessWorkerSpec
from telefuser.service.livekit.runtime import LiveKitServeRuntime
from telefuser.service.livekit.session_registry import SessionRecord


def _fake_process_worker(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    del config_values, pipeline_file, skip_validation, security_name
    events.put(
        {
            "type": "worker_capacity",
            "worker_id": spec.worker_id,
            "capacity": 2,
            "profile": {"effective_capacity": 2},
        }
    )
    events.put({"type": "worker_status", "worker_id": spec.worker_id, "status": "idle"})
    events.put({"type": "worker_ready", "worker_id": spec.worker_id})
    pipeline_sessions: dict[str, str] = {}
    while True:
        command = commands.get()
        command_type = command["type"]
        request_id = command.get("request_id")
        if command_type == "start_session":
            record = command["record"]
            session_id = record["session_id"]
            pipeline_session_id = f"pipeline-{session_id}"
            pipeline_sessions[session_id] = pipeline_session_id
            events.put(
                {
                    "type": "pipeline_session",
                    "worker_id": spec.worker_id,
                    "session_id": session_id,
                    "pipeline_session_id": pipeline_session_id,
                }
            )
            events.put(
                {
                    "type": "session_status",
                    "worker_id": spec.worker_id,
                    "session_id": session_id,
                    "status": "running",
                    "error": None,
                }
            )
            continue
        if command_type == "stop_session":
            session_id = command["session_id"]
            events.put(
                {
                    "type": "session_finished",
                    "worker_id": spec.worker_id,
                    "session_id": session_id,
                    "pipeline_session_id": pipeline_sessions.pop(session_id, None),
                    "error": None,
                }
            )
        if request_id is not None:
            events.put(
                {
                    "type": "command_result",
                    "worker_id": spec.worker_id,
                    "request_id": request_id,
                }
            )
        if command_type == "shutdown":
            return


class _RecordingSink:
    def __init__(self) -> None:
        self.worker_statuses: list[tuple[str, str]] = []
        self.capacities: list[tuple[str, int, dict[str, object] | None]] = []
        self.session_statuses: list[tuple[str, str, str | None]] = []
        self.pipeline_sessions: list[tuple[str, str]] = []
        self.finished: list[tuple[str, str, str | None]] = []

    def on_worker_status(self, worker_id: str, status: str) -> None:
        self.worker_statuses.append((worker_id, status))

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        self.capacities.append((worker_id, capacity, profile))

    def on_session_status(self, session_id: str, status: str, error: str | None = None) -> None:
        self.session_statuses.append((session_id, status, error))

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        self.pipeline_sessions.append((session_id, pipeline_session_id))

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        self.finished.append((worker_id, session_id, error))


def _record() -> SessionRecord:
    return SessionRecord(
        session_id="session-1",
        room_name="room-1",
        controller_identity="controller-1",
        status="assigned",
        worker_id="worker-0",
        config={},
        created_at=0,
        updated_at=0,
    )


async def _wait_until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for process-worker event")
        await asyncio.sleep(0.01)


def test_process_worker_pool_forwards_lifecycle_over_spawn_ipc() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            worker_mode="process",
        )
        sink = _RecordingSink()
        pool = ProcessLiveKitWorkerPool(
            [ProcessWorkerSpec("worker-0", ["0"])],
            config=config,
            pipeline_file="pipeline.py",
            event_sink=sink,
            context=multiprocessing.get_context("spawn"),
            worker_target=_fake_process_worker,
        )
        await pool.start(skip_validation=True)
        assert pool.active_worker_count() == 1

        pool.start_session(_record())
        assert sink.capacities == [("worker-0", 2, {"effective_capacity": 2})]
        await _wait_until(lambda: bool(sink.pipeline_sessions))
        assert sink.pipeline_sessions == [("session-1", "pipeline-session-1")]
        assert pool.turboserve_snapshot()["retained_sessions_by_worker"] == {"worker-0": 1}

        await pool.stop_session("session-1")
        await _wait_until(lambda: bool(sink.finished))
        assert sink.finished == [("worker-0", "session-1", None)]
        assert pool.turboserve_snapshot()["retained_sessions_by_worker"] == {"worker-0": 0}
        await pool.aclose()

    asyncio.run(_run())


def test_process_worker_pool_scales_spawned_replicas() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            worker_mode="process",
        )
        sink = _RecordingSink()
        pool = ProcessLiveKitWorkerPool(
            [ProcessWorkerSpec("worker-0", ["0"]), ProcessWorkerSpec("worker-1", ["1"])],
            config=config,
            pipeline_file="pipeline.py",
            event_sink=sink,
            initial_workers=1,
            context=multiprocessing.get_context("spawn"),
            worker_target=_fake_process_worker,
        )
        await pool.start(skip_validation=True)
        assert pool.active_worker_count() == 1

        assert await pool.scale_to(2) == 2
        assert await pool.scale_to(1) == 1
        assert pool.turboserve_snapshot()["active_workers"] == ["worker-0"]
        await pool.aclose()

    asyncio.run(_run())


def test_process_worker_pool_rolls_back_route_when_session_command_fails() -> None:
    async def _run() -> None:
        config = LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            worker_mode="process",
        )
        pool = ProcessLiveKitWorkerPool(
            [ProcessWorkerSpec("worker-0", ["0"])],
            config=config,
            pipeline_file="pipeline.py",
            event_sink=_RecordingSink(),
            context=multiprocessing.get_context("spawn"),
            worker_target=_fake_process_worker,
        )
        await pool.start(skip_validation=True)

        with (
            patch.object(pool, "_send", side_effect=RuntimeError("command send failed")),
            pytest.raises(RuntimeError, match="command send failed"),
        ):
            pool.start_session(_record())

        assert pool.turboserve_snapshot()["retained_sessions_by_worker"] == {"worker-0": 0}
        await pool.aclose()

    asyncio.run(_run())


def test_runtime_selects_process_worker_pool() -> None:
    config = LiveKitServeConfig(
        livekit_url="wss://livekit.example",
        livekit_api_key="key",
        livekit_api_secret="secret",
        num_workers=2,
        worker_gpu_map="0;1",
        worker_mode="process",
    )
    runtime = LiveKitServeRuntime(config=config, pipeline_file="pipeline.py")

    assert isinstance(runtime.worker_pool, ProcessLiveKitWorkerPool)
    assert runtime.worker_pool.turboserve_snapshot()["configured_workers"] == 2
