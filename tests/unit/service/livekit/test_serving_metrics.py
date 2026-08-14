from __future__ import annotations

from queue import SimpleQueue

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.nccl_process_worker_pool import (
    NCCLProcessLiveKitWorkerPool,
    _ParentTransportSink,
)
from telefuser.service.livekit.process_worker_pool import ProcessLiveKitWorkerPool, _ProcessEventSink
from telefuser.service.livekit.runtime import LiveKitServeRuntime
from telefuser.service.livekit.schemas import SessionCreateRequest


class _TokenService:
    def create_token(self, *, identity: str, room_name: str, role: str, **kwargs: object) -> str:
        del kwargs
        return f"{role}:{identity}:{room_name}"


class _WorkerPool:
    def __init__(self) -> None:
        self.snapshot = {
            "worker_runtime_metrics": {
                "worker-0": {
                    "scheduler_mode": "batched",
                    "active_sessions": 1,
                    "mean_chunk_seconds": 0.5,
                    "p95_chunk_seconds": 0.75,
                    "maximum_batch_size": 2,
                    "denoise_seconds": 0.42,
                    "vae_decode_seconds": 0.06,
                    "taew_decode_items": 2,
                    "taew_decode_batch_size": 1,
                    "taew_decode_invocations": 2,
                    "taew_decode_mode": 2,
                }
            },
            "session_runtime_metrics": {
                "pipeline-session-1": {
                    "active": 1,
                    "emitted_frames": 12,
                }
            },
        }

    async def start(self, *, skip_validation: bool = False) -> None:
        del skip_validation

    def start_session(self, record) -> None:
        del record

    async def stop_session(self, session_id: str) -> None:
        del session_id

    async def aclose(self) -> None:
        return None

    def turboserve_snapshot(self) -> dict:
        return self.snapshot


def _runtime() -> LiveKitServeRuntime:
    return LiveKitServeRuntime(
        config=LiveKitServeConfig(
            livekit_url="wss://livekit.example",
            livekit_api_key="key",
            livekit_api_secret="secret",
            worker_gpu_map="4",
            max_sessions_per_worker=2,
            default_fps=12,
        ),
        pipeline_file="pipeline.py",
        token_service=_TokenService(),
        worker_pool=_WorkerPool(),
    )


def test_serving_metrics_render_scheduler_pipeline_slo_and_no_session_id_labels() -> None:
    runtime = _runtime()
    created = runtime.create_session(SessionCreateRequest(identity="controller", config={"fps": 12}))
    runtime.on_pipeline_session(created.record.session_id, "pipeline-session-1")
    runtime.on_session_status(created.record.session_id, "running")

    runtime.on_control_received("worker-0", created.record.session_id)
    runtime.on_model_output(
        "worker-0",
        "pipeline-session-1",
        {
            "type": "chunk",
            "fps": 12,
            "frame_count": 12,
            "scheduler": {
                "batch_size": 2,
                "queue_wait_seconds": 0.05,
                "compute_seconds": 0.55,
                "denoise_seconds": 0.42,
                "vae_decode_seconds": 0.06,
            },
        },
    )
    runtime.on_chunk_published("worker-0", created.record.session_id, 12)

    rendered = runtime.prometheus_metrics()

    assert 'telefuser_serving_worker_sessions{gpu="4",worker_id="worker-0"} 1' in rendered
    assert 'telefuser_serving_scheduler_mode_info{mode="batched"} 1' in rendered
    assert 'telefuser_serving_pipeline_stage_latency_seconds_bucket{le="0.5",stage="dit"} 1' in rendered
    assert 'telefuser_serving_slo_chunks_total{result="met"} 1' in rendered
    assert "telefuser_serving_action_to_first_frame_seconds_count 1" in rendered
    assert 'telefuser_serving_published_fps{scope="aggregate"}' in rendered
    assert created.record.session_id not in rendered
    assert "pipeline-session-1" not in rendered

    summary = runtime.serving_metrics_snapshot()["summary"]
    assert summary["sessions"] == {"retained": 1, "active": 1, "idle": 0, "waiting": 0}
    assert summary["scheduler_mode"] == "batched"


def test_serving_metrics_records_migration_errors() -> None:
    runtime = _runtime()
    runtime._serving_metrics.record_migration(success=False, error="CUDA out of memory")
    rendered = runtime.prometheus_metrics()

    assert 'telefuser_serving_migrations_total{result="error"} 1' in rendered
    assert 'telefuser_serving_errors_total{kind="oom"} 1' in rendered


class _ForwardedEventSink:
    def __init__(self) -> None:
        self.controls: list[tuple[str, str]] = []
        self.published: list[tuple[str, str, int, float | None]] = []
        self.outputs: list[tuple[str, str, dict, dict | None, dict | None]] = []

    def on_control_received(self, worker_id: str, session_id: str) -> None:
        self.controls.append((worker_id, session_id))

    def on_chunk_published(
        self,
        worker_id: str,
        session_id: str,
        frames: int,
        first_frame_at: float | None = None,
    ) -> None:
        self.published.append((worker_id, session_id, frames, first_frame_at))

    def on_model_output(
        self,
        worker_id: str,
        session_id: str,
        payload: dict,
        runtime_metrics: dict | None = None,
        session_runtime_metrics: dict | None = None,
    ) -> None:
        self.outputs.append((worker_id, session_id, payload, runtime_metrics, session_runtime_metrics))


class _OutputQueue:
    def __init__(self) -> None:
        self.items: list[dict | None] = []

    def put_nowait(self, item: dict | None) -> None:
        self.items.append(item)


class _NCCLTransportPool:
    def __init__(self, event_sink: _ForwardedEventSink) -> None:
        self._event_sink = event_sink


def test_process_worker_ipc_events_reach_runtime_sink() -> None:
    child_events = SimpleQueue()
    child = _ProcessEventSink("worker-0", child_events)
    payload = {"type": "chunk", "frame_count": 12}
    child.on_control_received("worker-0", "http-session-1")
    child.on_chunk_published("worker-0", "http-session-1", 12, 123.0)
    child.on_model_output(
        "worker-0",
        "pipeline-session-1",
        payload,
        runtime_metrics={"scheduler_mode": "batched"},
        session_runtime_metrics={"active": 1},
    )

    forwarded = _ForwardedEventSink()
    parent = object.__new__(ProcessLiveKitWorkerPool)
    parent._event_sink = forwarded
    for _ in range(3):
        ProcessLiveKitWorkerPool._dispatch_event(parent, child_events.get())

    assert forwarded.controls == [("worker-0", "http-session-1")]
    assert forwarded.published == [("worker-0", "http-session-1", 12, 123.0)]
    assert forwarded.outputs == [
        (
            "worker-0",
            "pipeline-session-1",
            payload,
            {"scheduler_mode": "batched"},
            {"active": 1},
        )
    ]


def test_nccl_parent_transport_and_model_event_hooks_preserve_scheduler_mode() -> None:
    forwarded = _ForwardedEventSink()
    output = _OutputQueue()
    parent = object.__new__(NCCLProcessLiveKitWorkerPool)
    parent._event_sink = forwarded
    parent._worker_runtime_metrics = {}
    parent._session_runtime_metrics = {}
    parent._model_outputs = {"pipeline-session-1": output}
    parent._specs = {"worker-0": object()}
    parent._session_workers = {}
    parent._pipeline_routes = {}
    parent._active_workers = set()
    parent._nccl_ranks = {}
    parent._migration_total_ms = []

    payload = {"type": "chunk", "frame_count": 12}
    NCCLProcessLiveKitWorkerPool._dispatch_event(
        parent,
        {
            "type": "model_output",
            "worker_id": "worker-0",
            "session_id": "pipeline-session-1",
            "payload": payload,
            "runtime_metrics": {"scheduler_mode": "batched", "maximum_batch_size": 2},
            "session_runtime_metrics": {"active": 1},
        },
    )

    transport = _ParentTransportSink(_NCCLTransportPool(forwarded))
    transport.on_control_received("worker-0", "http-session-1")
    transport.on_chunk_published("worker-0", "http-session-1", 12, 234.0)

    snapshot = NCCLProcessLiveKitWorkerPool.turboserve_snapshot(parent)
    assert snapshot["worker_runtime_metrics"] == {"worker-0": {"scheduler_mode": "batched", "maximum_batch_size": 2}}
    assert output.items == [payload]
    assert forwarded.controls == [("worker-0", "http-session-1")]
    assert forwarded.published == [("worker-0", "http-session-1", 12, 234.0)]
    assert forwarded.outputs == [
        (
            "worker-0",
            "pipeline-session-1",
            payload,
            {"scheduler_mode": "batched", "maximum_batch_size": 2},
            {"active": 1},
        )
    ]



def test_serving_metrics_distinguish_native_taew_batch_from_dit_batch() -> None:
    runtime = _runtime()
    synchronized_scheduler = {
        "batch_size": 2,
        "taew_decode_items": 2,
        "taew_decode_batch_size": 2,
        "taew_decode_invocations": 1,
        "taew_decode_mode": 1,
    }
    serial_fallback_scheduler = {
        "batch_size": 2,
        "taew_decode_items": 2,
        "taew_decode_batch_size": 1,
        "taew_decode_invocations": 2,
        "taew_decode_mode": 2,
    }
    for scheduler in (
        synchronized_scheduler,
        synchronized_scheduler,
        serial_fallback_scheduler,
        serial_fallback_scheduler,
    ):
        runtime.on_model_output(
            "worker-0",
            "pipeline-session-1",
            {"type": "chunk", "frame_count": 12, "scheduler": scheduler},
        )

    rendered = runtime.prometheus_metrics()

    assert "telefuser_serving_taew_decode_synchronized_items_total 2" in rendered
    assert "telefuser_serving_taew_decode_synchronized_executions_total 1" in rendered
    assert "telefuser_serving_taew_decode_serial_fallback_items_total 2" in rendered
    assert "telefuser_serving_taew_decode_serial_fallback_executions_total 2" in rendered
    assert "telefuser_serving_taew_decode_mean_native_batch_size 1.33333333333" in rendered
    assert "telefuser_serving_worker_taew_decode_mode" in rendered
    assert "telefuser_serving_taew_decode_synchronized_items_total{mode=" not in rendered
    assert "telefuser_serving_taew_decode_serial_fallback_items_total{mode=" not in rendered
