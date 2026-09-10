from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Mapping

import pytest

from telefuser.pipelines.lingbot_vla_v2.action_scheduler import ActionChunkScheduler
from telefuser.vla.runtime import ActionChunkScheduler as CommonActionChunkScheduler


def test_lingbot_scheduler_import_is_a_compatibility_alias() -> None:
    assert ActionChunkScheduler is CommonActionChunkScheduler


def test_scheduler_discards_inflight_and_pending_work_when_newer_observation_arrives() -> None:
    started = threading.Event()
    release = threading.Event()

    def infer(request: Mapping[str, Any]) -> dict[str, Any]:
        if request["sequence_id"] == 0:
            started.set()
            assert release.wait(timeout=2)
        return {"action": request["sequence_id"]}

    async def scenario() -> None:
        scheduler = ActionChunkScheduler(infer)
        await scheduler.start()
        first = scheduler.submit({"request_id": "first", "sequence_id": 0}, session_key="session")
        assert await asyncio.to_thread(started.wait, 2)
        second = scheduler.submit({"request_id": "second", "sequence_id": 1}, session_key="session")
        third = scheduler.submit({"request_id": "third", "sequence_id": 2}, session_key="session")

        assert (await second)["scheduler_status"] == "superseded"
        release.set()
        assert (await first)["scheduler_status"] == "superseded"
        completed = await third
        assert completed["action"] == 2
        assert completed["sequence_id"] == 2
        assert completed["scheduler_status"] == "completed"
        assert completed["server_timing"]["queue_wait_ms"] >= 0
        await scheduler.close()

    asyncio.run(scenario())


def test_scheduler_expires_result_after_non_cancellable_inference() -> None:
    def infer(_request: Mapping[str, Any]) -> dict[str, Any]:
        time.sleep(0.02)
        return {"action": "too-late"}

    async def scenario() -> None:
        scheduler = ActionChunkScheduler(infer)
        await scheduler.start()
        response = await scheduler.submit(
            {"request_id": "expired", "request_ttl_ms": 1},
            session_key="session",
        )
        assert response["action"] is None
        assert response["scheduler_status"] == "expired"
        assert response["error"]["code"] == "expired"
        await scheduler.close()

    asyncio.run(scenario())


def test_scheduler_rejects_stale_sequence_without_running_inference() -> None:
    calls: list[int] = []

    def infer(request: Mapping[str, Any]) -> dict[str, Any]:
        calls.append(request["sequence_id"])
        return {"action": request["sequence_id"]}

    async def scenario() -> None:
        scheduler = ActionChunkScheduler(infer)
        await scheduler.start()
        assert (await scheduler.submit({"sequence_id": 4}, session_key="session"))["action"] == 4
        stale = await scheduler.submit({"sequence_id": 4}, session_key="session")
        assert stale["scheduler_status"] == "stale_sequence"
        assert calls == [4]
        await scheduler.close()

    asyncio.run(scenario())


def test_scheduler_bounds_pending_sessions() -> None:
    started = threading.Event()
    release = threading.Event()

    def infer(request: Mapping[str, Any]) -> dict[str, Any]:
        if request["request_id"] == "active":
            started.set()
            assert release.wait(timeout=2)
        return {"action": request["request_id"]}

    async def scenario() -> None:
        scheduler = ActionChunkScheduler(infer, max_pending_sessions=1)
        await scheduler.start()
        active = scheduler.submit({"request_id": "active"}, session_key="active")
        assert await asyncio.to_thread(started.wait, 2)
        pending = scheduler.submit({"request_id": "pending"}, session_key="pending")
        overloaded = await scheduler.submit({"request_id": "overloaded"}, session_key="overloaded")
        assert overloaded["scheduler_status"] == "overloaded"
        release.set()
        assert (await active)["scheduler_status"] == "completed"
        assert (await pending)["scheduler_status"] == "completed"
        await scheduler.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"sequence_id": -1}, "sequence_id"),
        ({"sequence_id": True}, "sequence_id"),
        ({"request_ttl_ms": 0}, "request_ttl_ms"),
        ({"request_ttl_ms": float("inf")}, "request_ttl_ms"),
    ],
)
def test_scheduler_rejects_invalid_controls(payload: dict[str, Any], match: str) -> None:
    async def scenario() -> None:
        scheduler = ActionChunkScheduler(lambda _request: {"action": 1})
        await scheduler.start()
        with pytest.raises(ValueError, match=match):
            scheduler.submit(payload, session_key="session")
        await scheduler.close()

    asyncio.run(scenario())
