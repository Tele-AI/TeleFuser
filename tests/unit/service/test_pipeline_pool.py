"""Tests for PipelinePool, ReplicaWorker, and multi-slot TaskManager.

Covers test plan items 1-17 from pipeline_pool_development_plan.md.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from telefuser.platforms import current_platform
from telefuser.service.api.schema import TaskRequest
from telefuser.service.core.config import ServerConfig
from telefuser.service.core.task_manager import TaskManager, TaskStatus

_DEVICE_ENV_VAR = current_platform.device_control_env_var


# ============================================================================
# Test 1: TaskManager atomic claim — concurrent claim, single winner
# ============================================================================


def test_multi_slot_concurrent_claim_single_winner() -> None:
    """With max_concurrent_processing=2, only 2 tasks can be claimed concurrently."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=2)

    ids = [tm.create_task(TaskRequest(task="t2i")) for _ in range(4)]

    first = tm.claim_next_pending_task()
    second = tm.claim_next_pending_task()
    third = tm.claim_next_pending_task()

    assert first == ids[0]
    assert second == ids[1]
    assert third is None  # slots full

    assert tm.get_task(ids[0]).status == TaskStatus.PROCESSING
    assert tm.get_task(ids[1]).status == TaskStatus.PROCESSING
    assert tm.get_task(ids[2]).status == TaskStatus.PENDING


# ============================================================================
# Test 2: TaskManager capacity — max_concurrent=N, N+1 returns None
# ============================================================================


def test_capacity_limit_respected() -> None:
    """claim returns None when all N slots are occupied."""
    for n in (1, 3, 5):
        tm = TaskManager(max_queue_size=20, max_concurrent_processing=n)
        for _ in range(n + 2):
            tm.create_task(TaskRequest(task="t2i"))

        claimed = []
        for _ in range(n + 1):
            c = tm.claim_next_pending_task()
            if c is not None:
                claimed.append(c)

        assert len(claimed) == n


# ============================================================================
# Test 3: TaskManager backward compat — max_concurrent=1 identical to current
# ============================================================================


def test_backward_compat_single_slot() -> None:
    """max_concurrent_processing=1 behaves like the original single-slot implementation."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=1)

    first = tm.create_task(TaskRequest(task="t2i"))
    second = tm.create_task(TaskRequest(task="t2i"))

    assert tm.claim_next_pending_task() == first
    assert tm.claim_next_pending_task() is None

    tm.release_processing_slot(first)
    assert tm.claim_next_pending_task() == second


# ============================================================================
# Test 4: TaskManager current_task stability — OrderedDict first-inserted key
# ============================================================================


def test_current_task_points_to_first_processing() -> None:
    """current_task always points to the first (earliest) processing task."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=3)

    ids = [tm.create_task(TaskRequest(task="t2i")) for _ in range(3)]

    for i in range(3):
        tm.claim_next_pending_task()

    status = tm.get_service_status()
    assert status["current_task"] == ids[0]
    assert len(status["current_tasks"]) == 3

    # Release first; current_task should shift to second
    tm.release_processing_slot(ids[0])
    status = tm.get_service_status()
    assert status["current_task"] == ids[1]
    assert len(status["current_tasks"]) == 2


# ============================================================================
# Test: Multi-slot release cycle
# ============================================================================


def test_multi_slot_release_cycle() -> None:
    """Releasing a slot allows the next pending task to be claimed."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=2)
    ids = [tm.create_task(TaskRequest(task="t2i")) for _ in range(5)]

    c1 = tm.claim_next_pending_task()
    c2 = tm.claim_next_pending_task()
    assert tm.claim_next_pending_task() is None

    tm.release_processing_slot(c1)
    c3 = tm.claim_next_pending_task()
    assert c3 == ids[2]

    tm.release_processing_slot(c2)
    c4 = tm.claim_next_pending_task()
    assert c4 == ids[3]


# ============================================================================
# Test: Concurrent multi-slot claims across threads
# ============================================================================


def test_concurrent_multi_slot_claims() -> None:
    """Concurrent claims under multi-slot never hand the same task to two callers."""
    tm = TaskManager(max_queue_size=100, max_concurrent_processing=4)
    for _ in range(40):
        tm.create_task(TaskRequest(task="t2i"))

    claimed: list[str] = []
    claimed_lock = threading.Lock()

    def worker() -> None:
        for _ in range(40):
            task_id = tm.claim_next_pending_task()
            if task_id is not None:
                with claimed_lock:
                    claimed.append(task_id)
                tm.release_processing_slot(task_id)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(claimed) == len(set(claimed))
    assert len(claimed) == 40


# ============================================================================
# Test: Zero capacity — fail-fast
# ============================================================================


def test_zero_capacity_fails_pending_tasks() -> None:
    """set_max_concurrent_processing(0) fails all pending tasks immediately."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=2)
    ids = [tm.create_task(TaskRequest(task="t2i")) for _ in range(3)]

    tm.claim_next_pending_task()  # first goes to PROCESSING

    tm.set_max_concurrent_processing(0)

    # Remaining pending tasks should be FAILED
    for task_id in ids[1:]:
        task = tm.get_task(task_id)
        assert task.status == TaskStatus.FAILED
        assert "all pipeline replicas are dead" in task.error.lower()


def test_zero_capacity_rejects_new_tasks() -> None:
    """create_task raises RuntimeError when zero capacity."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=2)

    tm.set_max_concurrent_processing(0)

    with pytest.raises(RuntimeError, match="No inference capacity"):
        tm.create_task(TaskRequest(task="t2i"))


def test_zero_capacity_claim_drains_stragglers() -> None:
    """claim_next_pending_task drains remaining pending tasks in zero capacity mode."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=2)
    ids = [tm.create_task(TaskRequest(task="t2i")) for _ in range(3)]

    tm.set_max_concurrent_processing(0)

    result = tm.claim_next_pending_task()
    assert result is None

    for task_id in ids:
        assert tm.get_task(task_id).status == TaskStatus.FAILED


# ============================================================================
# Test: get_service_status with multi-slot info
# ============================================================================


def test_service_status_multi_slot_fields() -> None:
    """get_service_status includes multi-slot fields."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=3)
    for _ in range(4):
        tm.create_task(TaskRequest(task="t2i"))

    tm.claim_next_pending_task()
    tm.claim_next_pending_task()

    status = tm.get_service_status()
    assert status["service_status"] == "busy"
    assert status["processing_count"] == 2
    assert status["max_concurrent_processing"] == 3
    assert len(status["current_tasks"]) == 2
    assert status["pending_tasks"] == 2


def test_service_status_idle_when_no_processing() -> None:
    """service_status is 'idle' when no tasks are processing."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=3)

    status = tm.get_service_status()
    assert status["service_status"] == "idle"
    assert status["processing_count"] == 0
    assert status["current_tasks"] == []


# ============================================================================
# Test: max_concurrent_processing property
# ============================================================================


def test_max_concurrent_processing_property() -> None:
    """max_concurrent_processing property reflects dynamic changes."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=4)
    assert tm.max_concurrent_processing == 4

    tm.set_max_concurrent_processing(2)
    assert tm.max_concurrent_processing == 2


# ============================================================================
# Test 14: Config resolve_replica_device_ids
# ============================================================================


def test_resolve_replica_device_ids_basic() -> None:
    """4 GPUs, num_replicas=2 → 2 groups of 2."""
    config = ServerConfig(num_replicas=2)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_DEVICE_ENV_VAR, None)
        groups = config.resolve_replica_device_ids(4)

    assert groups == [["0", "1"], ["2", "3"]]


def test_resolve_replica_device_ids_with_cvd() -> None:
    """Device env var=4,5,6,7, parallelism=4, num_replicas=2 → groups of 4,5 and 6,7."""
    config = ServerConfig(num_replicas=2)
    with patch.dict(os.environ, {_DEVICE_ENV_VAR: "4,5,6,7"}):
        groups = config.resolve_replica_device_ids(4)

    assert groups == [["4", "5"], ["6", "7"]]


def test_resolve_replica_device_ids_mig() -> None:
    """MIG device tokens are treated as opaque strings."""
    config = ServerConfig(num_replicas=2)
    mig = "MIG-GPU-xxx/1/0,MIG-GPU-xxx/2/0"
    with patch.dict(os.environ, {_DEVICE_ENV_VAR: mig}):
        groups = config.resolve_replica_device_ids(2)

    assert groups == [["MIG-GPU-xxx/1/0"], ["MIG-GPU-xxx/2/0"]]


def test_resolve_replica_device_ids_not_divisible() -> None:
    """parallelism not divisible by num_replicas raises ValueError."""
    config = ServerConfig(num_replicas=3)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_DEVICE_ENV_VAR, None)
        with pytest.raises(ValueError, match="must be divisible"):
            config.resolve_replica_device_ids(5)


def test_resolve_replica_device_ids_fewer_gpus_than_parallelism() -> None:
    """Device env var has fewer GPUs than parallelism raises ValueError."""
    config = ServerConfig(num_replicas=2)
    with patch.dict(os.environ, {_DEVICE_ENV_VAR: "0,1"}):
        with pytest.raises(ValueError, match="only 2 GPUs visible"):
            config.resolve_replica_device_ids(4)


def test_resolve_replica_device_ids_cvd_selects_first_n() -> None:
    """Device env var has 8 GPUs but parallelism=4 → only first 4 used."""
    config = ServerConfig(num_replicas=2)
    with patch.dict(os.environ, {_DEVICE_ENV_VAR: "0,1,2,3,4,5,6,7"}):
        groups = config.resolve_replica_device_ids(4)

    assert groups == [["0", "1"], ["2", "3"]]


def test_resolve_replica_device_ids_single_gpu_per_replica() -> None:
    """num_replicas=4, parallelism=4 → 4 replicas with 1 GPU each."""
    config = ServerConfig(num_replicas=4)
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(_DEVICE_ENV_VAR, None)
        groups = config.resolve_replica_device_ids(4)

    assert groups == [["0"], ["1"], ["2"], ["3"]]


# ============================================================================
# Test: is_processing with multi-slot
# ============================================================================


def test_is_processing_multi_slot() -> None:
    """is_processing is True when any slot is occupied, False when all empty."""
    tm = TaskManager(max_queue_size=10, max_concurrent_processing=2)

    assert tm.is_processing() is False

    tm.create_task(TaskRequest(task="t2i"))
    c1 = tm.claim_next_pending_task()
    assert tm.is_processing() is True

    tm.create_task(TaskRequest(task="t2i"))
    c2 = tm.claim_next_pending_task()
    assert tm.is_processing() is True

    tm.release_processing_slot(c1)
    assert tm.is_processing() is True

    tm.release_processing_slot(c2)
    assert tm.is_processing() is False
