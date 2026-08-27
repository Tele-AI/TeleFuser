from __future__ import annotations

import asyncio
from typing import Any

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.nccl_process_worker_pool import (
    _MODEL_OUTPUT_PARENT_QUEUE_SIZE,
    _NCCL_INIT_PARENT_TIMEOUT_SECONDS,
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


def _model_output_eos(session_id: str) -> dict[str, Any]:
    return {
        "type": "model_output_eos",
        "worker_id": "worker-0",
        "session_id": session_id,
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


def test_child_pump_emits_eos_when_abot_output_iterator_ends() -> None:
    async def run() -> None:
        events = _EventCollector()
        await _pump_model_outputs(
            _PumpAdapter([]),
            _PumpService(),
            worker_id="worker-0",
            session_id="pipeline-1",
            credits=asyncio.BoundedSemaphore(_MODEL_OUTPUT_PARENT_QUEUE_SIZE),
            events=events,
        )

        assert [(item["type"], item["worker_id"], item["session_id"]) for item in events.items] == [
            ("model_output_eos", "worker-0", "pipeline-1")
        ]

    asyncio.run(run())


def test_parent_eos_finishes_pull_and_releases_route() -> None:
    async def run() -> None:
        pool = _pool()
        sent: list[tuple[str, dict[str, Any]]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        pool.create_model_session("worker-0", "pipeline-1", {})
        sent.clear()

        chunks = pool.pull_model_chunks("pipeline-1")
        pool._dispatch_event(_model_output_eos("pipeline-1"))
        try:
            await asyncio.wait_for(chunks.__anext__(), timeout=0.2)
        except StopAsyncIteration:
            pass
        else:
            raise AssertionError("parent pull did not finish after child model-output EOF")

        assert "pipeline-1" not in pool._pipeline_routes
        assert "pipeline-1" not in pool._session_workers
        assert "pipeline-1" not in pool._model_outputs
        assert sent == [("worker-0", {"type": "model_close", "session_id": "pipeline-1"})]
        await chunks.aclose()

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


def test_init_nccl_uses_dedicated_parent_timeout() -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        requests: list[tuple[str, str, dict[str, object]]] = []

        async def fake_request(worker_id: str, command_type: str, **kwargs: object) -> dict[str, object]:
            requests.append((worker_id, command_type, kwargs))
            return {"result": True}

        pool._request = fake_request
        await pool._init_nccl()

        assert [(worker_id, command_type) for worker_id, command_type, _ in requests] == [
            ("worker-0", "nccl_init"),
            ("worker-1", "nccl_init"),
        ]
        assert [kwargs["rank"] for _, _, kwargs in requests] == [0, 1]
        assert all(kwargs["world_size"] == 2 for _, _, kwargs in requests)
        assert all(kwargs["timeout"] == _NCCL_INIT_PARENT_TIMEOUT_SECONDS for _, _, kwargs in requests)
        assert len({kwargs["init_method"] for _, _, kwargs in requests}) == 1
        assert pool._nccl_ranks == {"worker-0": 0, "worker-1": 1}

    asyncio.run(run())


def test_parent_progress_follows_dequeued_payload_owner_after_route_change() -> None:
    async def run() -> None:
        pool = _pool()
        pool._active_workers = {"worker-0", "worker-1"}
        sent: list[tuple[str, dict[str, Any]]] = []
        pool._send = lambda worker_id, command: sent.append((worker_id, command))
        pool.create_model_session("worker-0", "pipeline-1", {})
        sent.clear()
        pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "frames": [object(), object()]}))
        chunks = pool.pull_model_chunks("pipeline-1")
        assert (await chunks.__anext__())["type"] == "chunk"
        pool._pipeline_routes["pipeline-1"] = "worker-1"
        assert pool.report_publisher_frame_progress(
            "pipeline-1", event="submitted", frames_delta=-1, observed_monotonic_seconds=12.0
        )
        progress = [
            (worker_id, command) for worker_id, command in sent if command["type"] == "model_publisher_frame_progress"
        ]
        assert progress == [
            (
                "worker-0",
                {
                    "type": "model_publisher_frame_progress",
                    "session_id": "pipeline-1",
                    "event": "submitted",
                    "frames_delta": -1,
                    "sequence": 0,
                    "observed_monotonic_seconds": 12.0,
                },
            )
        ]
        await chunks.aclose()
        assert "pipeline-1" not in pool._model_output_inflight_owner

    asyncio.run(run())


def test_latest_parent_queue_replacement_rebates_source_publisher_credit() -> None:
    pool = _pool()
    sent: list[tuple[str, dict[str, Any]]] = []
    pool._send = lambda worker_id, command: sent.append((worker_id, command))
    pool.create_model_session("worker-0", "pipeline-1", {})
    sent.clear()
    pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 0, "frames": [object()] * 12}))
    pool._dispatch_event(_model_output("pipeline-1", {"type": "chunk", "index": 1, "frames": [object()] * 12}))
    progress = [
        (worker_id, command) for worker_id, command in sent if command["type"] == "model_publisher_frame_progress"
    ]
    assert progress == [
        (
            "worker-0",
            {
                "type": "model_publisher_frame_progress",
                "session_id": "pipeline-1",
                "event": "dropped",
                "frames_delta": -12,
                "sequence": 0,
                "observed_monotonic_seconds": progress[0][1]["observed_monotonic_seconds"],
            },
        )
    ]


def test_migration_drain_waits_for_child_queue_and_publisher_frames(monkeypatch) -> None:
    async def run() -> None:
        pool = _pool()
        calls: list[tuple[str, object]] = []
        statuses = iter(
            [
                # Child Fq remains: the parent cannot accept this snapshot.
                {"in_flight": False, "output_queue_empty": False, "publisher_unsubmitted_frames": 0},
                # Fq has crossed to the publisher, but child Fp remains.
                {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 12},
                # First zero snapshot is followed by a parent-drain barrier.
                {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 0},
                # Only a second zero status after that barrier is safe.
                {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 0},
            ]
        )

        async def fake_wait_for_model_output_drain(session_id: str, *, timeout: float) -> None:
            assert session_id == "pipeline-1"
            assert timeout > 0
            calls.append(("parent_drain", None))

        async def fake_request(worker_id: str, request_type: str, **kwargs: object) -> dict[str, object]:
            assert worker_id == "worker-0"
            assert request_type == "model_output_drain_status"
            assert kwargs["session_id"] == "pipeline-1"
            assert float(kwargs["timeout"]) > 0
            status = next(statuses)
            calls.append(("child_status", status))
            return {"result": status}

        monkeypatch.setattr(pool, "_wait_for_model_output_drain", fake_wait_for_model_output_drain)
        monkeypatch.setattr(pool, "_request", fake_request)

        await pool._drain_model_outputs_for_migration("pipeline-1", source_worker_id="worker-0", timeout=1.0)

        assert [kind for kind, _ in calls] == [
            "parent_drain",
            "child_status",
            "parent_drain",
            "child_status",
            "parent_drain",
            "child_status",
            "parent_drain",
            "child_status",
        ]
        assert [value for kind, value in calls if kind == "child_status"] == [
            {"in_flight": False, "output_queue_empty": False, "publisher_unsubmitted_frames": 0},
            {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 12},
            {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 0},
            {"in_flight": False, "output_queue_empty": True, "publisher_unsubmitted_frames": 0},
        ]
