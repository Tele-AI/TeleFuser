from __future__ import annotations

import asyncio
import queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from telefuser.pipelines.abot_world.interactive import ABotWorldSessionLifecycle
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService, _ABotWorldLiveKitSession
from telefuser.service.core.stream_pipeline_service import BidirectionalService


class _FakePipelineSession:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.next_latent_frame = 0
        self.first_frame_latent = torch.zeros(1, 1, 1, 1, 1)
        self.self_cache = [
            {
                "local_end_index": torch.zeros(1, dtype=torch.long),
                "global_end_index": torch.zeros(1, dtype=torch.long),
            }
        ]
        self.lifecycle = ABotWorldSessionLifecycle.READY
        self.closed = False

    @property
    def is_resident(self) -> bool:
        return self.lifecycle != ABotWorldSessionLifecycle.SUSPENDED


class _FakePipeline:
    def __init__(self) -> None:
        self.config = SimpleNamespace(width=8, height=8)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.denoise_stage = SimpleNamespace(
            dit=SimpleNamespace(
                patch_size=(1, 2, 2),
                dim=8,
                num_heads=2,
                num_layers=2,
                local_attn_size=18,
                text_len=8,
            )
        )
        self.generate_calls: list[tuple[str, dict[str, bool]]] = []
        self.batch_sizes: list[int] = []
        self.closed_sessions: list[str] = []
        self.suspended_sessions: list[str] = []
        self.restored_sessions: list[str] = []
        self.closed = False

    def preload_models(self) -> None:
        return None

    def create_interactive_session(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int,
        session_id: str | None = None,
    ) -> _FakePipelineSession:
        del seed
        assert image.mode == "RGB"
        assert prompt
        assert session_id is not None
        return _FakePipelineSession(session_id)

    def generate_next_block(
        self,
        session: _FakePipelineSession,
        controls: dict[str, bool],
        *,
        control_latent_frames: int,
    ) -> list[Image.Image]:
        assert control_latent_frames == 3
        self.generate_calls.append((session.session_id, controls))
        self.batch_sizes.append(1)
        session.next_latent_frame += control_latent_frames
        return [Image.new("RGB", (8, 8), color=(20, len(self.generate_calls) % 255, 40))]

    def generate_next_blocks(
        self,
        sessions: list[_FakePipelineSession],
        controls: list[dict[str, bool]],
        *,
        control_latent_frames: int,
    ) -> list[list[Image.Image]]:
        self.batch_sizes.append(len(sessions))
        results = []
        for session, state in zip(sessions, controls):
            self.generate_calls.append((session.session_id, state))
            session.next_latent_frame += control_latent_frames
            results.append([Image.new("RGB", (8, 8), color=(20, len(self.generate_calls) % 255, 40))])
        return results

    def suspend_interactive_session(self, session: _FakePipelineSession) -> None:
        session.lifecycle = ABotWorldSessionLifecycle.SUSPENDED
        self.suspended_sessions.append(session.session_id)

    def restore_interactive_session(self, session: _FakePipelineSession) -> None:
        session.lifecycle = ABotWorldSessionLifecycle.READY
        self.restored_sessions.append(session.session_id)

    def close_interactive_session(self, session: _FakePipelineSession) -> None:
        session.closed = True
        self.closed_sessions.append(session.session_id)

    def close(self) -> None:
        self.closed = True


def _service(**kwargs: object) -> tuple[ABotWorldLiveKitService, _FakePipeline]:
    pipeline = _FakePipeline()
    service = ABotWorldLiveKitService(
        pipeline,
        default_session_config={"prompt": "test prompt"},
        **kwargs,
    )
    return service, pipeline


def _create(service: ABotWorldLiveKitService, session_id: str, **config: object) -> str:
    return service.create_session(
        {
            "session_id": session_id,
            "image": Image.new("RGB", (8, 8)),
            **config,
        }
    )


def test_service_matches_shared_multi_session_bidirectional_contract() -> None:
    service, _ = _service()
    assert isinstance(service, BidirectionalService)
    profile = service.configure_session_capacity(3)
    assert profile["effective_capacity"] == 3
    assert profile["max_batch_size"] == 8
    service.stop()


def test_capacity_profile_accepts_explicit_cuda_device_string(monkeypatch) -> None:
    service, pipeline = _service()
    pipeline.device = "cuda:3"
    monkeypatch.setattr("telefuser.pipelines.abot_world.service.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(service, "_profile_session_memory", lambda: {
        "profiled_session_bytes": 100,
        "workspace_peak_bytes": 200,
    })
    observed = {}

    def fake_mem_get_info(device):
        observed["device"] = device
        return 10_000, 20_000

    monkeypatch.setattr("telefuser.pipelines.abot_world.service.torch.cuda.mem_get_info", fake_mem_get_info)

    profile = service.configure_session_capacity(2)

    assert observed["device"] == torch.device("cuda:3")
    assert profile["effective_capacity"] == 2
    service.stop()

def test_round_robin_capacity_uses_one_active_workspace(monkeypatch) -> None:
    service, pipeline = _service(max_batch_size=8)
    pipeline.device = "cuda:0"
    monkeypatch.setattr("telefuser.pipelines.abot_world.service.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(service, "_estimate_session_bytes", lambda: 100)
    monkeypatch.setattr(service, "_profile_session_memory", lambda: {
        "profiled_session_bytes": 100,
        "workspace_peak_bytes": 200,
    })
    monkeypatch.setattr(
        "telefuser.pipelines.abot_world.service.torch.cuda.mem_get_info",
        lambda device: (1_000, 2_000),
    )

    profile = service.configure_session_capacity(10)

    assert profile["computed_capacity"] == 7
    assert profile["effective_capacity"] == 7
    assert profile["estimated_batch_workspace_bytes"] == 200
    assert profile["scheduler_mode"] == "round_robin"
    service.stop()



def test_two_ready_sessions_are_generated_in_one_batch_and_keep_order() -> None:
    service, pipeline = _service(output_queue_size=4, batching_window_ms=30, scheduler_mode="batched")
    service.configure_session_capacity(2)
    first = _create(service, "first")
    second = _create(service, "second")
    first_state = service._session(first)
    second_state = service._session(second)
    assert first_state is not None and second_state is not None
    assert first_state.output_queue.get(timeout=1)["type"] == "preview"
    assert second_state.output_queue.get(timeout=1)["type"] == "preview"

    service.push_chunk(first, {"type": "control_state", "controls": ["KeyW"]})
    service.push_chunk(second, {"type": "control_state", "controls": ["KeyD"]})
    first_chunk = first_state.output_queue.get(timeout=2)
    second_chunk = second_state.output_queue.get(timeout=2)

    assert first_chunk["index"] == 0
    assert second_chunk["index"] == 0
    assert first_chunk["scheduler"]["batch_size"] == 2
    assert second_chunk["scheduler"]["batch_size"] == 2
    assert 2 in pipeline.batch_sizes
    service.stop()


def test_service_accepts_two_latent_experimental_chunk() -> None:
    service, _ = _service()
    service.configure_session_capacity(1)
    session_id = _create(service, "two-latent", fps=8, control_latent_frames=2)
    try:
        state = service._session(session_id)
        assert state is not None
        assert state.config["fps"] == 8
        assert state.config["control_latent_frames"] == 2
    finally:
        service.close_session(session_id)
        service.stop()


def test_default_scheduler_round_robins_single_session_steps() -> None:
    service, pipeline = _service(output_queue_size=4)
    service.configure_session_capacity(2)
    first = _create(service, "first")
    second = _create(service, "second")
    first_state = service._session(first)
    second_state = service._session(second)
    assert first_state is not None and second_state is not None
    assert first_state.output_queue.get(timeout=1)["type"] == "preview"
    assert second_state.output_queue.get(timeout=1)["type"] == "preview"

    service.push_chunk(first, {"type": "control_state", "controls": ["KeyW"]})
    service.push_chunk(second, {"type": "control_state", "controls": ["KeyD"]})
    first_chunk = first_state.output_queue.get(timeout=2)
    second_chunk = second_state.output_queue.get(timeout=2)

    assert first_chunk["scheduler"]["batch_size"] == 1
    assert second_chunk["scheduler"]["batch_size"] == 1
    assert pipeline.batch_sizes[:2] == [1, 1]
    assert service.runtime_metrics()["scheduler_mode"] == "round_robin"
    service.stop()


def test_lossless_sessions_each_stream_thirty_chunks_without_drops() -> None:
    service, _ = _service(output_queue_size=2, batching_window_ms=10, control_idle_timeout=30)
    service.configure_session_capacity(2)
    session_ids = [_create(service, value, delivery_mode="lossless") for value in ("a", "b")]

    async def collect(session_id: str) -> list[int]:
        indexes: list[int] = []
        async for payload in service.pull_chunks(session_id):
            if payload["type"] == "preview":
                service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
                continue
            indexes.append(payload["index"])
            if len(indexes) == 30:
                service.push_chunk(session_id, {"type": "control", "control": "KeyW", "event": "release"})
                return indexes
        return indexes

    async def run() -> list[list[int]]:
        return await asyncio.gather(*(collect(session_id) for session_id in session_ids))

    try:
        indexes = asyncio.run(run())
        assert indexes == [list(range(30)), list(range(30))]
        for session_id in session_ids:
            assert service.runtime_metrics(session_id)["dropped_video_payloads"] == 0
    finally:
        service.stop()


def test_latest_queue_discards_oldest_video_and_records_metric() -> None:
    service, _ = _service(output_queue_size=1)
    pipeline_session = _FakePipelineSession("test")
    state = _ABotWorldLiveKitSession(
        session_id="test",
        pipeline_session=pipeline_session,
        output_queue=queue.Queue(maxsize=1),
        control_event=threading.Event(),
        config={"fps": 12, "control_latent_frames": 3, "delivery_mode": "latest"},
    )
    state.output_queue.put({"type": "chunk", "index": 0})

    assert service._put_output(state, {"type": "chunk", "index": 1})
    assert state.output_queue.get_nowait()["index"] == 1
    assert state.dropped_video_payloads == 1
    service.stop()


def test_migration_waits_for_already_generated_output_to_drain() -> None:
    service, pipeline = _service()
    service.configure_session_capacity(1)
    session_id = _create(service, "migrating")
    state = service._session(session_id)
    assert state is not None
    pipeline.snapshot_interactive_session = lambda session: SimpleNamespace(session_id=session.session_id)

    def drain_preview() -> None:
        time.sleep(0.05)
        state.output_queue.get_nowait()
        with service._scheduler_condition:
            service._scheduler_condition.notify_all()

    thread = threading.Thread(target=drain_preview)
    thread.start()
    started_at = time.monotonic()
    bundle = service.prepare_migration(session_id, timeout=1)
    thread.join()

    assert bundle.snapshot.session_id == session_id
    assert time.monotonic() - started_at >= 0.04
    service.abort_migration(session_id)
    service.close_session(session_id)
    service.stop()


def test_idle_session_suspends_and_restores_on_new_control() -> None:
    service, pipeline = _service(
        output_queue_size=2,
        batching_window_ms=0,
        idle_suspension_seconds=0.02,
        control_idle_timeout=30,
    )
    service.configure_session_capacity(1)
    session_id = _create(service, "idle")
    state = service._session(session_id)
    assert state is not None
    state.output_queue.get(timeout=1)

    deadline = time.monotonic() + 1
    while session_id not in pipeline.suspended_sessions and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session_id in pipeline.suspended_sessions

    service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
    assert state.output_queue.get(timeout=1)["type"] == "chunk"
    assert session_id in pipeline.restored_sessions
    service.stop()


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "control_state", "controls": "KeyW"},
        {"type": "control", "control": "unsupported", "event": "press"},
        {"type": "unsupported"},
    ],
)
def test_invalid_livekit_control_payloads_are_rejected(payload: dict) -> None:
    service, _ = _service()
    service.configure_session_capacity(1)
    session_id = _create(service, "invalid")
    try:
        with pytest.raises(ValueError):
            service.push_chunk(session_id, payload)
    finally:
        service.stop()
