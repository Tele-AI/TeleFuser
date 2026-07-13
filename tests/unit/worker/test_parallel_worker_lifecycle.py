import threading
from queue import Empty
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from telefuser.worker.parallel_worker import ParallelWorker


def _worker() -> ParallelWorker:
    worker = ParallelWorker.__new__(ParallelWorker)
    worker.name = "Parallel Worker TestStage"
    worker.timeout = 1
    worker._lifecycle_lock = threading.Lock()
    worker._failed = False
    worker._closed = False
    worker._failure_reason = None
    worker.queue_out = MagicMock()
    worker.queue_in = [MagicMock(), MagicMock()]
    worker.ctx = SimpleNamespace(processes=[MagicMock(), MagicMock()])
    for process in worker.ctx.processes:
        process.is_alive.return_value = True
    return worker


def test_timeout_marks_group_failed_terminates_all_ranks_and_rejects_reuse() -> None:
    worker = _worker()
    worker.queue_out.get.side_effect = Empty()

    with pytest.raises(RuntimeError, match="timeout after 1 seconds"):
        worker._wait_result("denoise")

    assert worker.failed is True
    assert worker.closed is False
    assert worker.failure_reason == "denoise timeout after 1 seconds"
    for process in worker.ctx.processes:
        process.kill.assert_called_once_with()
        process.join.assert_called_once_with(timeout=2)

    with pytest.raises(RuntimeError, match="has failed"):
        worker._ensure_usable()


def test_remote_rank_error_marks_entire_group_failed() -> None:
    worker = _worker()
    worker.queue_out.get.return_value = RuntimeError("Parallel worker rank 2 failed")

    with pytest.raises(RuntimeError, match="rank 2 failed"):
        worker._wait_result("initialize_cache")

    assert worker.failed is True
    assert "initialize_cache failed" in worker.failure_reason
    for process in worker.ctx.processes:
        process.kill.assert_called_once_with()


def test_close_is_idempotent_and_rejects_new_work() -> None:
    worker = _worker()
    for process in worker.ctx.processes:
        process.is_alive.return_value = False

    worker.close()
    worker.close()

    assert worker.closed is True
    assert worker.failed is False
    for queue in worker.queue_in:
        queue.put.assert_called_once_with(["exit", None, None])
        queue.close.assert_called_once_with()
    worker.queue_out.close.assert_called_once_with()

    with pytest.raises(RuntimeError, match="is closed"):
        worker._ensure_usable()


def test_close_after_failure_does_not_submit_more_worker_commands() -> None:
    worker = _worker()
    worker._mark_failed("rank 1 failed")

    worker.close()

    assert worker.failed is True
    assert worker.closed is True
    for queue in worker.queue_in:
        queue.put.assert_not_called()
        queue.close.assert_called_once_with()
