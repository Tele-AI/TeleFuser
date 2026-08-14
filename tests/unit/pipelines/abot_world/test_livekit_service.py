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
        self.call_times: list[float] = []
        self.frames_per_chunk = 1
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
        self.call_times.append(time.monotonic())
        self.batch_sizes.append(1)
        session.next_latent_frame += control_latent_frames
        return [
            Image.new("RGB", (8, 8), color=(20, len(self.generate_calls) % 255, 40))
            for _ in range(self.frames_per_chunk)
        ]

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
            self.call_times.append(time.monotonic())
            session.next_latent_frame += control_latent_frames
            results.append(
                [
                    Image.new("RGB", (8, 8), color=(20, len(self.generate_calls) % 255, 40))
                    for _ in range(self.frames_per_chunk)
                ]
            )
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


def _take_and_notify(
    service: ABotWorldLiveKitService,
    state: _ABotWorldLiveKitSession,
    *,
    timeout: float = 1.0,
) -> dict[str, object]:
    payload = state.output_queue.get(timeout=timeout)
    with service._scheduler_condition:
        service._scheduler_condition.notify_all()
    return payload


def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    assert predicate()


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

def test_batched_capacity_accounts_for_active_batch_workspace(monkeypatch) -> None:
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

    assert profile["computed_capacity"] == 3
    assert profile["effective_capacity"] == 3
    assert profile["estimated_batch_workspace_bytes"] == 600
    assert profile["scheduler_mode"] == "batched"
    service.stop()



def test_two_ready_sessions_are_generated_in_one_batch_and_keep_order() -> None:
    service, pipeline = _service(output_queue_size=4, batching_window_ms=30)
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


def test_default_scheduler_coalesces_compatible_sessions() -> None:
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

    assert first_chunk["scheduler"]["batch_size"] == 2
    assert second_chunk["scheduler"]["batch_size"] == 2
    assert pipeline.batch_sizes[:1] == [2]
    assert service.runtime_metrics()["scheduler_mode"] == "batched"
    service.stop()


def test_round_robin_remains_single_session_ablation() -> None:
    service, pipeline = _service(output_queue_size=4, scheduler_mode="round_robin")
    service.configure_session_capacity(2)
    first = _create(service, "first")
    second = _create(service, "second")
    first_state = service._session(first)
    second_state = service._session(second)
    assert first_state is not None and second_state is not None
    try:
        assert _take_and_notify(service, first_state)["type"] == "preview"
        assert _take_and_notify(service, second_state)["type"] == "preview"
        service.push_chunk(first, {"type": "control_state", "controls": ["KeyW"]})
        service.push_chunk(second, {"type": "control_state", "controls": ["KeyD"]})
        assert _take_and_notify(service, first_state, timeout=2)["type"] == "chunk"
        assert _take_and_notify(service, second_state, timeout=2)["type"] == "chunk"
        assert pipeline.batch_sizes[:2] == [1, 1]
    finally:
        service.stop()


def test_latest_mode_bounds_prefetch_and_resumes_at_playout_deadline() -> None:
    service, pipeline = _service(output_queue_size=4, batching_window_ms=0, control_idle_timeout=30)
    pipeline.frames_per_chunk = 12
    service.configure_session_capacity(1)
    session_id = _create(service, "paced", fps=12, control_latent_frames=3)
    state = service._session(session_id)
    assert state is not None
    try:
        assert _take_and_notify(service, state)["type"] == "preview"
        service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
        _wait_for(lambda: len(pipeline.generate_calls) >= 1)
        assert _take_and_notify(service, state)["type"] == "chunk"
        _wait_for(lambda: len(pipeline.generate_calls) >= 2)

        with state.lock:
            expected_resume_at = state.next_playout_deadline - state.last_chunk_duration_seconds
            assert state.pacing_ready_at <= expected_resume_at
        remaining = expected_resume_at - time.monotonic()
        if remaining > 0.05:
            time.sleep(remaining - 0.05)
        # The second chunk is the sole prefetch. It remains queued while the
        # first 12-frame chunk is being played, so there is no free-running c3.
        assert len(pipeline.generate_calls) == 2
        metrics = service.runtime_metrics(session_id)
        assert metrics["pacing_buffered_video_payloads"] == 1
        assert service.runtime_metrics()["pacing_throttled_sessions"] == 1

        remaining = expected_resume_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        assert _take_and_notify(service, state)["type"] == "chunk"
        dequeued_at = time.monotonic()
        _wait_for(lambda: len(pipeline.generate_calls) >= 3)
        third_started_at = pipeline.call_times[2]
        assert third_started_at >= expected_resume_at - 0.05
        assert third_started_at <= dequeued_at + 0.15
    finally:
        service.stop()


def test_latest_mode_batches_mildly_staggered_playout_consumers() -> None:
    service, pipeline = _service(output_queue_size=4, batching_window_ms=20, control_idle_timeout=30)
    pipeline.frames_per_chunk = 12
    service.configure_session_capacity(2)
    first = _create(service, "first", fps=12, control_latent_frames=3)
    second = _create(service, "second", fps=12, control_latent_frames=3)
    first_state = service._session(first)
    second_state = service._session(second)
    assert first_state is not None and second_state is not None
    try:
        assert _take_and_notify(service, first_state)["type"] == "preview"
        assert _take_and_notify(service, second_state)["type"] == "preview"
        service.push_chunk(first, {"type": "control_state", "controls": ["KeyW"]})
        service.push_chunk(second, {"type": "control_state", "controls": ["KeyD"]})
        _wait_for(lambda: len(pipeline.batch_sizes) >= 1)
        assert pipeline.batch_sizes[0] == 2

        assert _take_and_notify(service, first_state)["type"] == "chunk"
        time.sleep(0.003)
        assert _take_and_notify(service, second_state)["type"] == "chunk"
        _wait_for(lambda: len(pipeline.batch_sizes) >= 2)
        assert pipeline.batch_sizes[1] == 2

        with first_state.lock:
            continuation_at = first_state.next_playout_deadline - first_state.last_chunk_duration_seconds
        remaining = continuation_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        assert _take_and_notify(service, first_state)["type"] == "chunk"
        time.sleep(0.003)
        assert _take_and_notify(service, second_state)["type"] == "chunk"
        _wait_for(lambda: len(pipeline.batch_sizes) >= 3)
        assert pipeline.batch_sizes[2] == 2
    finally:
        service.stop()


def _prepare_latest_continuation(
    state: _ABotWorldLiveKitSession,
    *,
    now: float,
    pacing_ready_at: float,
    next_playout_deadline: float,
) -> None:
    with state.lock:
        state.controls = {"W"}
        state.ready_since = now
        state.scheduled_chunks = 2
        state.last_chunk_duration_seconds = 1.0
        state.last_compute_seconds = 0.05
        state.pacing_ready_at = pacing_ready_at
        state.next_playout_deadline = next_playout_deadline
        state.pipeline_session.next_latent_frame = 6


def test_latest_mode_rendezvouses_staggered_continuations_within_deadline_slack() -> None:
    service, pipeline = _service(output_queue_size=4, batching_window_ms=2, control_idle_timeout=30)
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    first = _create(service, "first")
    second = _create(service, "second")
    first_state = service._session(first)
    second_state = service._session(second)
    assert first_state is not None and second_state is not None
    try:
        now = 10_000.0
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )
        _prepare_latest_continuation(
            second_state,
            now=now,
            pacing_ready_at=now + 0.12,
            next_playout_deadline=now + 1.12,
        )

        ready = service._ready_sessions(now)
        assert [state.session_id for state in ready] == [first]
        wait_seconds = service._batch_formation_wait_seconds(ready, now)
        # 10 ms pacing slack makes the second continuation eligible at +110 ms.
        assert wait_seconds == pytest.approx(0.11, abs=0.002)

        ready = service._ready_sessions(now + wait_seconds + 0.001)
        batch = service._select_batch(ready, now=now + wait_seconds + 0.001)
        assert [state.session_id for state in batch] == [first, second]
        service._execute_batch(batch, [{"W": True}, {"D": True}])
        assert pipeline.batch_sizes[-1] == 2
    finally:
        service.stop()


def test_latest_mode_rendezvous_never_waits_past_a_playout_deadline() -> None:
    service, _ = _service(output_queue_size=4, batching_window_ms=2, control_idle_timeout=30)
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    first = _create(service, "first")
    second = _create(service, "second")
    first_state = service._session(first)
    second_state = service._session(second)
    assert first_state is not None and second_state is not None
    try:
        now = 20_000.0
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 0.20,
        )
        _prepare_latest_continuation(
            second_state,
            now=now,
            pacing_ready_at=now + 0.16,
            next_playout_deadline=now + 0.36,
        )

        ready = service._ready_sessions(now)
        assert [state.session_id for state in ready] == [first]
        wait_seconds = service._batch_formation_wait_seconds(ready, now)
        # A B=2 estimate is 2 * 50 ms * 1.1, leaving only 90 ms for first.
        # The peer releases at 150 ms, so use only the legacy 2 ms window.
        assert wait_seconds == pytest.approx(service.batching_window_seconds)
        assert now + wait_seconds < service._latest_safe_batch_start([first_state, second_state])
        ready_after_window = service._ready_sessions(now + wait_seconds)
        assert [state.session_id for state in service._select_batch(ready_after_window, now=now + wait_seconds)] == [first]
    finally:
        service.stop()


def test_latest_mode_falls_back_to_singleton_when_observed_batch_misses_deadline() -> None:
    service, _ = _service(output_queue_size=4, batching_window_ms=2, control_idle_timeout=30)
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    first = _create(service, "first")
    second = _create(service, "second")
    first_state = service._session(first)
    second_state = service._session(second)
    assert first_state is not None and second_state is not None
    try:
        now = 30_000.0
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 0.50,
        )
        _prepare_latest_continuation(
            second_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.00,
        )
        # A measured B=2 takes 1.01 seconds, while each B=1 takes 0.40s.
        # With the 10% safety margin, B=2 misses the first deadline, but B=1
        # followed by B=1 still fits the two respective deadlines.
        service._batch_compute_estimates.update({1: 0.40, 2: 1.01})
        ready = service._ready_sessions(now)
        assert [state.session_id for state in ready] == [first, second]
        assert service._latest_safe_batch_start(ready) < now
        assert now <= service._latest_safe_batch_start([first_state])
        assert now + service._estimated_batch_compute_seconds([first_state]) <= service._latest_safe_batch_start(
            [second_state]
        )

        batch = service._select_batch(ready, now=now)

        assert [state.session_id for state in batch] == [first]
    finally:
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
