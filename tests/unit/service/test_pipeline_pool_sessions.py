"""CPU-only tests for session-affine PipelinePool leases."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from telefuser.service.core.pipeline_pool import PipelinePool
from telefuser.service.core.replica_worker import ReplicaDeadError


def _pool(num_replicas: int = 2) -> tuple[PipelinePool, list[MagicMock]]:
    pool = PipelinePool(
        num_replicas=num_replicas,
        replica_device_ids=[[str(index)] for index in range(num_replicas)],
        security_level_name="NONE",
    )
    handles = []
    for replica_id in range(num_replicas):
        handle = MagicMock()
        handle.replica_id = replica_id
        handle._dead = False
        handle.process.is_alive.return_value = True
        handles.append(handle)
        pool._available.put_nowait(replica_id)
    pool._handles = list(handles)
    pool._instance_status = ["idle"] * num_replicas
    return pool, handles


def test_sessions_are_pinned_to_distinct_replicas_and_close_returns_capacity() -> None:
    pool, handles = _pool()

    async def scenario() -> None:
        assert await pool.open_session("one") == 0
        assert await pool.open_session("two") == 1
        assert await pool.session_bindings() == {"one": 0, "two": 1}

        async with pool.acquire_session("one") as first:
            assert first is handles[0]
        async with pool.acquire_session("one") as again:
            assert again is handles[0]

        assert await pool.close_session("one") == 0
        async with pool.acquire() as released:
            assert released is handles[0]

    asyncio.run(scenario())


def test_same_session_requests_are_serialized() -> None:
    pool, _ = _pool(num_replicas=1)

    async def scenario() -> None:
        await pool.open_session("one")
        first_entered = asyncio.Event()
        release_first = asyncio.Event()
        entered: list[str] = []

        async def use_session(name: str) -> None:
            async with pool.acquire_session("one"):
                entered.append(name)
                if name == "first":
                    first_entered.set()
                    await release_first.wait()

        first = asyncio.create_task(use_session("first"))
        await first_entered.wait()
        second = asyncio.create_task(use_session("second"))
        await asyncio.sleep(0)
        assert entered == ["first"]
        release_first.set()
        await asyncio.gather(first, second)
        assert entered == ["first", "second"]

    asyncio.run(scenario())


def test_dead_session_replica_invalidates_binding_deterministically() -> None:
    pool, handles = _pool()

    async def scenario() -> None:
        await pool.open_session("one")
        handles[0].process.is_alive.return_value = False

        with pytest.raises(ReplicaDeadError, match="not alive"):
            async with pool.acquire_session("one"):
                pass

        assert await pool.session_bindings() == {}
        assert pool._instance_status == ["dead", "idle"]
        assert pool._live_count == 1
        with pytest.raises(KeyError, match="unknown pipeline session"):
            await pool.session_replica_id("one")

    asyncio.run(scenario())


def test_replica_failure_during_session_execution_evicts_once() -> None:
    pool, handles = _pool(num_replicas=1)

    async def scenario() -> None:
        await pool.open_session("one")
        with pytest.raises(ReplicaDeadError, match="failed"):
            async with pool.acquire_session("one"):
                raise ReplicaDeadError("failed")
        assert await pool.session_bindings() == {}
        assert pool._live_count == 0
        handles[0].shutdown.assert_called_once()

    asyncio.run(scenario())


def test_pool_shutdown_clears_all_session_bindings() -> None:
    pool, handles = _pool()

    async def scenario() -> None:
        await pool.open_session("one")
        await pool.open_session("two")
        await pool.aclose()
        assert await pool.session_bindings() == {}
        with pytest.raises(RuntimeError, match="shutting down"):
            await pool.open_session("three", timeout_s=0.01)

    asyncio.run(scenario())
    for handle in handles:
        handle.shutdown.assert_called_once()


def test_pool_shutdown_waits_for_active_session_request() -> None:
    pool, handles = _pool(num_replicas=1)

    async def scenario() -> None:
        await pool.open_session("one")
        entered = asyncio.Event()
        release = asyncio.Event()

        async def use_session() -> None:
            async with pool.acquire_session("one"):
                entered.set()
                await release.wait()

        request = asyncio.create_task(use_session())
        await entered.wait()
        shutdown = asyncio.create_task(pool.aclose())
        await asyncio.sleep(0)
        assert not shutdown.done()
        release.set()
        await asyncio.gather(request, shutdown)
        assert await pool.session_bindings() == {}

    asyncio.run(scenario())
    handles[0].shutdown.assert_called_once()


def test_pool_shutdown_wakes_session_waiting_for_capacity() -> None:
    pool, _ = _pool(num_replicas=1)

    async def scenario() -> None:
        await pool.open_session("one")
        waiting = asyncio.create_task(pool.open_session("two"))
        await asyncio.sleep(0)
        await pool.aclose()
        with pytest.raises(RuntimeError, match="shutting down"):
            await waiting

    asyncio.run(scenario())
