from __future__ import annotations

import asyncio
from typing import Any

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.nccl_process_worker_pool import (
    _MODEL_OUTPUT_PARENT_QUEUE_SIZE,
    NCCLProcessLiveKitWorkerPool,
    _pump_model_outputs,
)
from telefuser.service.livekit.process_worker_pool import (
    ProcessLiveKitWorkerPool,
    ProcessWorkerSpec,
)
from telefuser.service.livekit.worker import NullWorkerEventSink


class _EventCollector:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.updated = asyncio.Event()

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)
        self.updated.set()


class _PumpAdapter:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.pull_count = 0

    async def pull_chunks(self, session_id: str):
        del session_id
        for payload in self.payloads:
            self.pull_count += 1
            yield payload

    def runtime_metrics(self) -> dict[str, int]:
        return {"active_sessions": 1}


class _PumpService:
    def runtime_metrics(self, session_id: str) -> dict[str, int]:
        del session_id
        return {"active": 1}


def _pool() -> NCCLProcessLiveKitWorkerPool:
    config = LiveKitServeConfig(
        livekit_url="wss://livekit.example",
        livekit_api_key="key",
        livekit_api_secret="secret",
        worker_mode="process-nccl",
        num_workers=2,
        worker_gpu_map="0;1",
    )
    pool = NCCLProcessLiveKitWorkerPool(
        [ProcessWorkerSpec("worker-0", ["0"]), ProcessWorkerSpec("worker-1", ["1"])],
        config=config,
        pipeline_file="pipeline.py",
        event_sink=NullWorkerEventSink(),
    )
    pool._active_workers = {"worker-0"}
    return pool


def _model_output(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "model_output",
        "worker_id": "worker-0",
        "session_id": session_id,
        "payload": payload,
    }


async def _wait_for_count(events: _EventCollector, count: int) -> None:
    while len(events.items) < count:
        events.updated.clear()
        await asyncio.wait_for(events.updated.wait(), timeout=1.0)


def test_child_pump_waits_for_credit_before_reading_next_abot_payload() -> None:
    async def run() -> None:
        events = _EventCollector()
        adapter = _PumpAdapter([{"type": "chunk", "index": 0}, {"type": "chunk", "index": 1}])
        credits = asyncio.BoundedSemaphore(_MODEL_OUTPUT_PARENT_QUEUE_SIZE)
        task = asyncio.create_task(
            _pump_model_outputs(
                adapter,
                _PumpService(),
                worker_id="worker-0",
                session_id="pipeline-1",
                credits=credits,
                events=events,
            )
        )
        await _wait_for_count(events, 1)
        await asyncio.sleep(0)
        assert adapter.pull_count == 1
        assert [item["payload"]["index"] for item in events.items] == [0]

        credits.release()
        await _wait_for_count(events, 2)
        assert adapter.pull_count == 2
        assert [item["payload"]["index"] for item in events.items] == [0, 1]

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_parent_queue_preserves_preview_then_replaces_stale_video_and_returns_credit() -> None:
    async def run() -> None:
        pool = _pool()
        sent: list[tuple[str, dict[str, Any]]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        pool.create_model_session("worker-0", "pipeline-1", {})
        sent.clear()

        pool._dispatch_event(_model_output("pipeline-1", {"type": "preview", "index": -1}))
        pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 0}))
        assert pool._model_outputs["pipeline-1"].qsize() == 1
        assert pool._model_output_dropped["pipeline-1"] == 1

        chunks = pool.pull_model_chunks("pipeline-1")
        assert (await chunks.__anext__())["type"] == "preview"
        pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 1}))
        pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 2}))
        assert (await chunks.__anext__())["index"] == 2
        snapshot = pool.turboserve_snapshot()["model_output_flow_control"]
        assert snapshot["parent_queue_capacity"] == 1
        assert snapshot["max_materialized_payloads_per_session"] == 2
        assert snapshot["dropped_payloads"] == {"pipeline-1": 2}
        credits = [command for _, command in sent if command["type"] == "model_output_credit"]
        assert [command["session_id"] for command in credits] == ["pipeline-1"] * 4
        await chunks.aclose()

    asyncio.run(run())


def test_parent_queue_prioritizes_terminal_payload_over_queued_video() -> None:
    async def run() -> None:
        pool = _pool()
        pool._send = lambda worker_id, command: None
        pool.create_model_session("worker-0", "pipeline-1", {})
        pool._dispatch_event(_model_output("pipeline-1", {"type": "preview"}))
        pool._dispatch_event(_model_output("pipeline-1", {"type": "error", "error": "model failed"}))

        chunks = pool.pull_model_chunks("pipeline-1")
        payload = await chunks.__anext__()
        assert payload == {"type": "error", "error": "model failed"}
        assert pool._model_output_dropped["pipeline-1"] == 1
        await chunks.aclose()

    asyncio.run(run())


def test_initial_start_builds_one_nccl_group_after_all_workers(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = set()
        init_sizes: list[int] = []

        async def fake_parent_scale_to(self, target_workers: int) -> int:
            self._active_workers = set(list(self._specs)[:target_workers])
            return len(self._active_workers)

        async def fake_parent_start(self, *, skip_validation: bool = False) -> None:
            assert skip_validation
            # This mirrors ProcessLiveKitWorkerPool.start: its virtual
            # scale_to calls occur once for each sequential worker startup.
            await self.scale_to(1)
            await self.scale_to(2)

        async def fake_init_nccl() -> None:
            init_sizes.append(len(pool._active_workers))
            pool._nccl_ranks = {worker_id: index for index, worker_id in enumerate(sorted(pool._active_workers))}

        monkeypatch.setattr(ProcessLiveKitWorkerPool, "scale_to", fake_parent_scale_to)
        monkeypatch.setattr(ProcessLiveKitWorkerPool, "start", fake_parent_start)
        pool._init_nccl = fake_init_nccl

        await pool.start(skip_validation=True)

        assert init_sizes == [2]
        assert not pool._initializing_workers
        assert pool._nccl_ranks == {"worker-0": 0, "worker-1": 1}

    asyncio.run(run())
