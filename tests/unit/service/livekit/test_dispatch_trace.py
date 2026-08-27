from __future__ import annotations

import json
from queue import SimpleQueue

from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService
from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.nccl_process_worker_pool import NCCLProcessLiveKitWorkerPool, _DispatchTraceWriter
from telefuser.service.livekit.process_worker_pool import (
    ProcessLiveKitWorkerPool,
    ProcessWorkerSpec,
    _install_process_dispatch_trace_callback,
)
from telefuser.service.livekit.process_worker_pool import (
    _DispatchTraceWriter as ProcessDispatchTraceWriter,
)


def test_service_dispatch_trace_has_stage_vae_and_session_audit_fields() -> None:
    service = object.__new__(ABotWorldLiveKitService)
    service.scheduler_mode = "batched"
    service._dispatch_trace_sequence = 0
    records: list[dict] = []
    service._dispatch_trace_callback = records.append

    service._emit_dispatch_trace(
        selected_at=100.0,
        selected_wall_time=1_700_000_000.0,
        model_started_at=100.01,
        model_started_wall_time=1_700_000_000.01,
        completed_at=100.51,
        completed_wall_time=1_700_000_000.51,
        control_latent_frames=3,
        session_traces=[
            {
                "session_id": "user-7",
                "chunk_index": 4,
                "next_latent_frame_before": 12,
                "next_latent_frame_after": 15,
                "emitted_frames_before": 48,
                "emitted_frames_after": 60,
                "frames": 12,
                "controls": ["W"],
                "queue_wait_seconds": 0.02,
            }
        ],
        stage_metrics={
            "input_prepare_seconds": 0.01,
            "cache_collate_seconds": 0.02,
            "denoise_seconds": 0.34,
            "cache_scatter_seconds": 0.03,
            "vae_decode_seconds": 0.07,
            "postprocess_seconds": 0.04,
            "total_seconds": 0.51,
            "taew_decode_mode": 1,
            "taew_decode_items": 2,
            "taew_decode_batch_size": 2,
            "taew_decode_invocations": 1,
        },
        outcome="ok",
    )

    assert len(records) == 1
    record = records[0]
    assert record["trace_sequence"] == 1
    assert record["batch_size"] == 1
    assert record["control_latent_frames"] == 3
    assert record["model_started_monotonic_seconds"] == 100.01
    assert record["model_completed_monotonic_seconds"] == 100.51
    assert record["stages_seconds"]["denoise"] == 0.34
    assert record["vae_decode"] == {
        "mode": 1,
        "mode_name": "synchronized_batch",
        "items": 2,
        "effective_batch_size": 2,
        "invocations": 1,
    }
    assert record["sessions"][0]["session_id"] == "user-7"
    assert record["sessions"][0]["chunk_index"] == 4


def test_parent_dispatch_trace_jsonl_is_bounded_and_enriched(tmp_path) -> None:
    path = tmp_path / "dispatch-trace.jsonl"
    writer = _DispatchTraceWriter(str(path), max_events=1, workers={"worker-0": ["0"]})
    pool = object.__new__(NCCLProcessLiveKitWorkerPool)
    pool._dispatch_trace = writer

    pool._dispatch_event(
        {
            "type": "model_dispatch_trace",
            "worker_id": "worker-0",
            "gpu": {"physical_gpu_id": "0", "logical_cuda_device": 0},
            "trace": {
                "schema_version": 1,
                "event_type": "model_dispatch",
                "batch_size": 2,
                "sessions": [{"session_id": "user-a"}, {"session_id": "user-b"}],
            },
        }
    )
    pool._dispatch_event(
        {
            "type": "model_dispatch_trace",
            "worker_id": "worker-0",
            "gpu": {"physical_gpu_id": "0", "logical_cuda_device": 0},
            "trace": {"schema_version": 1, "event_type": "model_dispatch", "batch_size": 1},
        }
    )
    snapshot = writer.snapshot()
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event_type"] for record in records] == ["trace_metadata", "model_dispatch"]
    assert records[1]["parent_sequence"] == 1
    assert records[1]["worker_id"] == "worker-0"
    assert records[1]["gpu"]["physical_gpu_id"] == "0"
    assert records[1]["sessions"] == [{"session_id": "user-a"}, {"session_id": "user-b"}]
    assert snapshot["written_events"] == 1
    assert snapshot["dropped_events"] == 1


class _TraceCallbackService:
    def __init__(self) -> None:
        self.callback = None

    def set_dispatch_trace_callback(self, callback) -> None:
        self.callback = callback


def test_process_mode_trace_forwards_to_parent_and_preserves_physical_gpu(monkeypatch, tmp_path) -> None:
    path = tmp_path / "dispatch-trace.jsonl"
    config = LiveKitServeConfig(
        livekit_url="wss://livekit.example",
        livekit_api_key="key",
        livekit_api_secret="secret",
        worker_mode="process",
        dispatch_trace_path=str(path),
    )
    service = _TraceCallbackService()
    events = SimpleQueue()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    spec = ProcessWorkerSpec("worker-0", ["0"])

    assert _install_process_dispatch_trace_callback(
        service=service,
        config=config,
        spec=spec,
        events=events,
        logical_cuda_device=0,
    )
    assert callable(service.callback)
    service.callback(
        {
            "schema_version": 1,
            "event_type": "model_dispatch",
            "batch_size": 1,
            "sessions": [{"session_id": "user-a"}],
        }
    )
    event = events.get()
    assert event["gpu"] == {
        "physical_gpu_id": "1",
        "configured_gpu_id": "0",
        "logical_cuda_device": 0,
    }

    writer = ProcessDispatchTraceWriter(str(path), max_events=1, workers={"worker-0": ["0"]})
    pool = object.__new__(ProcessLiveKitWorkerPool)
    pool._dispatch_trace = writer
    ProcessLiveKitWorkerPool._dispatch_event(pool, event)
    snapshot = writer.snapshot()
    writer.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[1]["worker_id"] == "worker-0"
    assert records[1]["gpu"] == event["gpu"]
    assert records[1]["parent_sequence"] == 1
    assert snapshot["written_events"] == 1
