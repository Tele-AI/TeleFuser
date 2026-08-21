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
from telefuser.pipelines.abot_world.service import (
    _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS,
    ABotWorldLiveKitService,
    _ABotWorldLiveKitSession,
)
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
    def __init__(self, *, use_relative_rope: bool = True) -> None:
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
                use_relative_rope=use_relative_rope,
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


def _service(
    *,
    use_relative_rope: bool = True,
    **kwargs: object,
) -> tuple[ABotWorldLiveKitService, _FakePipeline]:
    pipeline = _FakePipeline(use_relative_rope=use_relative_rope)
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


def test_offline_batch_compute_prior_seeds_b2_then_keeps_online_high_water() -> None:
    service, _ = _service(
        max_batch_size=2,
        batch_compute_profile_name="h100_lf3_eager_full_pipeline_v1",
        batch_compute_prior_seconds={2: 0.7404982000589371},
        batch_compute_safety_factor=1.05,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    first, second = (_create(service, value) for value in ("first", "second"))
    states = [service._session(session_id) for session_id in (first, second)]
    assert all(state is not None for state in states)
    try:
        assert service._estimated_batch_compute_seconds(states) == pytest.approx(0.7404982000589371 * 1.05)
        service._batch_compute_estimates[2] = 0.80
        assert service._estimated_batch_compute_seconds(states) == pytest.approx(0.84)
        assert service.runtime_metrics()["batch_compute_safety_factor"] == pytest.approx(1.05)
        assert service.runtime_metrics()["batch_compute_profile_name"] == "h100_lf3_eager_full_pipeline_v1"
    finally:
        service.stop()


def test_capacity_profile_accepts_explicit_cuda_device_string(monkeypatch) -> None:
    service, pipeline = _service()
    pipeline.device = "cuda:3"
    monkeypatch.setattr("telefuser.pipelines.abot_world.service.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        service,
        "_profile_session_memory",
        lambda: {
            "profiled_session_bytes": 100,
            "workspace_peak_bytes": 200,
        },
    )
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
    monkeypatch.setattr(
        service,
        "_profile_session_memory",
        lambda: {
            "profiled_session_bytes": 100,
            "workspace_peak_bytes": 200,
        },
    )
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


def test_deadline_batch_wait_uses_one_persistent_timeout_then_falls_back() -> None:
    service, _ = _service(
        output_queue_size=4,
        batching_window_ms=2,
        max_deadline_batch_wait_ms=150,
        control_idle_timeout=30,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    first = _create(service, "first")
    first_state = service._session(first)
    assert first_state is not None
    try:
        now = 70_000.0
        service._batch_compute_estimates.update({1: 0.05, 2: 0.70})
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )

        ready = service._ready_sessions(now)
        first_wait = service._batch_formation_wait_seconds(ready, now)
        assert first_wait == pytest.approx(0.15)
        assert first_state.deadline_batch_wait_until == pytest.approx(now + 0.15)

        # An unrelated condition wake must not restart the full timeout.
        remaining_wait = service._batch_formation_wait_seconds(ready, now + 0.05)
        assert remaining_wait == pytest.approx(0.10)

        timeout_wait = service._batch_formation_wait_seconds(ready, now + 0.151)
        assert timeout_wait == 0.0
        assert first_state.deadline_batch_wait_until is None
        assert service.runtime_metrics()["deadline_batch_wait_timeouts"] == 1
    finally:
        service.stop()


def test_deadline_batch_wait_dispatches_b2_when_peer_arrives_before_timeout() -> None:
    service, pipeline = _service(
        output_queue_size=4,
        batching_window_ms=2,
        max_deadline_batch_wait_ms=200,
        control_idle_timeout=30,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    first, second = (_create(service, value) for value in ("first", "second"))
    first_state = service._session(first)
    second_state = service._session(second)
    assert first_state is not None and second_state is not None
    try:
        now = 71_000.0
        service._batch_compute_estimates.update({1: 0.05, 2: 0.60})
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )
        initial_wait = service._batch_formation_wait_seconds(service._ready_sessions(now), now)
        assert initial_wait == pytest.approx(0.20)

        peer_ready_at = now + 0.05
        _prepare_latest_continuation(
            second_state,
            now=peer_ready_at,
            pacing_ready_at=peer_ready_at,
            next_playout_deadline=peer_ready_at + 1.0,
        )
        ready = service._ready_sessions(peer_ready_at)
        batch = service._select_batch(ready, now=peer_ready_at)
        assert [state.session_id for state in batch] == [first, second]
        assert service._batch_formation_wait_seconds(ready, peer_ready_at) == 0.0
        service._execute_batch(batch, [{"W": True}, {"D": True}])
        assert pipeline.batch_sizes[-1] == 2
    finally:
        service.stop()


def test_deadline_batch_wait_promotes_held_b2_to_b3_for_safe_future_peer() -> None:
    """A persistent A hold may keep A+B together long enough to board safe C."""
    service, pipeline = _service(
        output_queue_size=4,
        batching_window_ms=2,
        max_batch_size=3,
        max_deadline_batch_wait_ms=500,
        batch_compute_safety_factor=1.0,
        control_idle_timeout=30,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(3)
    first, second, third = (_create(service, value) for value in ("first", "second", "third"))
    states = [service._session(session_id) for session_id in (first, second, third)]
    assert all(state is not None for state in states)
    first_state, second_state, third_state = states
    assert first_state is not None and second_state is not None and third_state is not None
    try:
        now = 71_500.0
        service._batch_compute_estimates.update({1: 0.05, 2: 0.60, 3: 0.70})
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.50,
        )
        # A starts one persistent B=2-bounded hold before either peer is ready.
        assert service._batch_formation_wait_seconds(service._ready_sessions(now), now) == pytest.approx(0.50)
        assert first_state.deadline_batch_wait_until == pytest.approx(now + 0.50)

        peer_ready_at = now + 0.05
        _prepare_latest_continuation(
            second_state,
            now=peer_ready_at,
            pacing_ready_at=peer_ready_at,
            next_playout_deadline=peer_ready_at + 1.50,
        )
        # Latest-mode pacing admits a continuation 10 ms ahead of its nominal
        # pacing point. C therefore becomes eligible at ``third_release_at``.
        third_release_at = now + 0.15
        _prepare_latest_continuation(
            third_state,
            now=peer_ready_at,
            pacing_ready_at=third_release_at + 0.01,
            next_playout_deadline=now + 1.70,
        )

        ready = service._ready_sessions(peer_ready_at)
        assert [state.session_id for state in ready] == [first, second]
        expected_wait = third_release_at - peer_ready_at + _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS
        assert service._batch_formation_wait_seconds(ready, peer_ready_at) == pytest.approx(expected_wait)
        # The original A cap remains the B=2 fallback, rather than being reset
        # when B arrives.
        assert first_state.deadline_batch_wait_until == pytest.approx(now + 0.50)

        selected_at = peer_ready_at + expected_wait
        ready = service._ready_sessions(selected_at)
        assert [state.session_id for state in ready] == [first, second, third]
        batch = service._select_batch(ready, now=selected_at)
        assert [state.session_id for state in batch] == [first, second, third]
        assert selected_at <= service._latest_safe_batch_start(batch, now=selected_at)
        service._execute_batch(batch, [{"W": True}, {"D": True}, {"A": True}])
        assert pipeline.batch_sizes[-1] == 3
    finally:
        service.stop()


def test_deadline_batch_wait_dispatches_b2_when_future_third_peer_misses_b3_deadline() -> None:
    """A C that is safe for B=2 but late for B=3 must not extend the A+B hold."""
    service, pipeline = _service(
        output_queue_size=4,
        batching_window_ms=2,
        max_batch_size=3,
        max_deadline_batch_wait_ms=500,
        batch_compute_safety_factor=1.0,
        control_idle_timeout=30,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(3)
    first, second, third = (_create(service, value) for value in ("first", "second", "third"))
    states = [service._session(session_id) for session_id in (first, second, third)]
    assert all(state is not None for state in states)
    first_state, second_state, third_state = states
    assert first_state is not None and second_state is not None and third_state is not None
    try:
        now = 71_600.0
        service._batch_compute_estimates.update({1: 0.05, 2: 0.60, 3: 0.65})
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 0.80,
        )
        # B=2 is the A fallback and fixes the original hold at +200 ms.
        assert service._batch_formation_wait_seconds(service._ready_sessions(now), now) == pytest.approx(0.20)
        assert first_state.deadline_batch_wait_until == pytest.approx(now + 0.20)

        peer_ready_at = now + 0.05
        _prepare_latest_continuation(
            second_state,
            now=peer_ready_at,
            pacing_ready_at=peer_ready_at,
            next_playout_deadline=peer_ready_at + 0.80,
        )
        third_release_at = now + 0.18
        _prepare_latest_continuation(
            third_state,
            now=peer_ready_at,
            pacing_ready_at=third_release_at + 0.01,
            next_playout_deadline=now + 1.00,
        )

        ready = service._ready_sessions(peer_ready_at)
        assert [state.session_id for state in ready] == [first, second]
        b2_latest_safe_start = service._latest_safe_batch_start(ready, now=peer_ready_at)
        assert b2_latest_safe_start == pytest.approx(now + 0.20)
        b3_latest_safe_start = service._latest_safe_batch_start([*ready, third_state], now=peer_ready_at)
        assert b3_latest_safe_start == pytest.approx(now + 0.15)
        assert third_release_at + _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS <= b2_latest_safe_start
        assert third_release_at + _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS > b3_latest_safe_start

        # C could arrive before the retained B=2 fallback, but it would miss
        # the B=3 deadline. Launch B=2 now rather than extending the hold.
        assert service._batch_formation_wait_seconds(ready, peer_ready_at) == 0.0
        batch = service._select_batch(ready, now=peer_ready_at)
        assert [state.session_id for state in batch] == [first, second]
        assert peer_ready_at <= service._latest_safe_batch_start(batch, now=peer_ready_at)
        service._execute_batch(batch, [{"W": True}, {"D": True}])
        assert pipeline.batch_sizes[-1] == 2
    finally:
        service.stop()


def test_deadline_batch_wait_is_clamped_by_predicted_b2_deadline() -> None:
    service, _ = _service(
        output_queue_size=4,
        batching_window_ms=2,
        max_deadline_batch_wait_ms=300,
        control_idle_timeout=30,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    first = _create(service, "first")
    first_state = service._session(first)
    assert first_state is not None
    try:
        now = 72_000.0
        service._batch_compute_estimates.update({1: 0.05, 2: 0.75})
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )

        wait_seconds = service._batch_formation_wait_seconds(service._ready_sessions(now), now)
        # B=2 uses a 10% safety factor: 1.0 - 1.1 * 0.75 = 0.175 s.
        assert wait_seconds == pytest.approx(0.175)
    finally:
        service.stop()


@pytest.mark.parametrize(
    ("first_deadline_offset", "urgent_deadline_offset", "expected_session", "expected_fillers"),
    [
        (1.0, 0.30, "urgent", 1),
        (0.70, 0.50, "first", 0),
    ],
    ids=("runs-safe-earlier-edf-filler", "protects-held-singleton-fallback"),
)
def test_deadline_batch_wait_only_allows_edf_work_that_preserves_held_fallback(
    first_deadline_offset: float,
    urgent_deadline_offset: float,
    expected_session: str,
    expected_fillers: int,
) -> None:
    service, _ = _service(
        output_queue_size=4,
        batching_window_ms=2,
        max_deadline_batch_wait_ms=250,
        control_idle_timeout=30,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    first, urgent = (_create(service, value) for value in ("first", "urgent"))
    first_state = service._session(first)
    urgent_state = service._session(urgent)
    assert first_state is not None and urgent_state is not None
    try:
        now = 73_000.0
        service._batch_compute_estimates.update({1: 0.05, 2: 0.06})
        if expected_session == first:
            service._batch_compute_estimates.update({1: 0.40, 2: 0.50})
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + first_deadline_offset,
        )
        # Start the peer-wait before the incompatible EDF job appears.
        assert service._batch_formation_wait_seconds(service._ready_sessions(now), now) > 0

        _prepare_latest_continuation(
            urgent_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + urgent_deadline_offset,
        )
        urgent_state.pipeline_session.self_cache[0]["local_end_index"].fill_(1)
        ready = service._ready_sessions(now)
        assert [state.session_id for state in ready] == [urgent, first]

        assert service._batch_formation_wait_seconds(ready, now) == 0.0
        batch = service._select_batch(ready, now=now)
        assert [state.session_id for state in batch] == [expected_session]
        assert service.runtime_metrics()["deadline_batch_filler_dispatches"] == expected_fillers
    finally:
        service.stop()


def test_latest_mode_aligns_three_staggered_lf3_continuations_without_frame_shrinking() -> None:
    """A 3-way LF3 batch forms through safe wakeups, not unvalidated LF1/LF2 bridges."""
    service, pipeline = _service(output_queue_size=4, batching_window_ms=2, control_idle_timeout=30)
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(3)
    first, second, third = (_create(service, value) for value in ("first", "second", "third"))
    states = [service._session(session_id) for session_id in (first, second, third)]
    assert all(state is not None for state in states)
    first_state, second_state, third_state = states
    assert first_state is not None and second_state is not None and third_state is not None
    try:
        now = 40_000.0
        service._batch_compute_estimates.update({1: 0.05, 2: 0.06, 3: 0.08})
        _prepare_latest_continuation(
            first_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )
        _prepare_latest_continuation(
            second_state,
            now=now,
            pacing_ready_at=now + 0.06,
            next_playout_deadline=now + 1.06,
        )
        _prepare_latest_continuation(
            third_state,
            now=now,
            pacing_ready_at=now + 0.12,
            next_playout_deadline=now + 1.12,
        )

        ready = service._ready_sessions(now)
        assert [state.session_id for state in ready] == [first]
        first_wait = service._batch_formation_wait_seconds(ready, now)
        assert 0 < first_wait < 0.10

        after_first_wait = now + first_wait + _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS
        ready = service._ready_sessions(after_first_wait)
        assert [state.session_id for state in ready] == [first, second]
        second_wait = service._batch_formation_wait_seconds(ready, after_first_wait)
        assert 0 < second_wait < 0.10

        aligned_at = after_first_wait + second_wait + _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS
        ready = service._ready_sessions(aligned_at)
        batch = service._select_batch(ready, now=aligned_at)
        assert [state.session_id for state in batch] == [first, second, third]
        assert aligned_at <= service._latest_safe_batch_start(batch)

        service._execute_batch(batch, [{"W": True}, {"D": True}, {"A": True}])
        assert pipeline.batch_sizes[-1] == 3
        assert [state.pipeline_session.next_latent_frame for state in batch] == [9, 9, 9]
    finally:
        service.stop()


@pytest.mark.parametrize(
    ("lagging_local_end", "expected_ids", "expected_batch_size"),
    [
        (72, ["ahead", "lagging"], 2),
        (60, ["ahead"], 1),
    ],
    ids=("matching-local-window", "different-local-window"),
)
def test_relative_rope_rendezvouses_mixed_global_positions_with_compatible_local_windows(
    lagging_local_end: int,
    expected_ids: list[str],
    expected_batch_size: int,
) -> None:
    """Only a matching retained KV layout may join a Relative-RoPE micro-batch."""
    service, pipeline = _service(
        use_relative_rope=True,
        output_queue_size=4,
        batching_window_ms=20,
        control_idle_timeout=30,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    ahead, lagging = (_create(service, value) for value in ("ahead", "lagging"))
    ahead_state = service._session(ahead)
    lagging_state = service._session(lagging)
    assert ahead_state is not None and lagging_state is not None
    try:
        now = 45_000.0
        service._batch_compute_estimates.update({1: 0.05, 2: 0.06})
        _prepare_latest_continuation(
            ahead_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )
        _prepare_latest_continuation(
            lagging_state,
            now=now,
            pacing_ready_at=now + 0.025,
            next_playout_deadline=now + 1.025,
        )
        ahead_state.pipeline_session.next_latent_frame = 9
        lagging_state.pipeline_session.next_latent_frame = 12
        ahead_cache = ahead_state.pipeline_session.self_cache[0]
        lagging_cache = lagging_state.pipeline_session.self_cache[0]
        ahead_cache["global_end_index"].fill_(108)
        lagging_cache["global_end_index"].fill_(144)
        ahead_cache["local_end_index"].fill_(72)
        lagging_cache["local_end_index"].fill_(lagging_local_end)

        ready = service._ready_sessions(now)
        assert [state.session_id for state in ready] == [ahead]
        wait_seconds = service._batch_formation_wait_seconds(ready, now)
        if expected_batch_size == 2:
            assert 0 < wait_seconds < service.batching_window_seconds
        else:
            assert wait_seconds == pytest.approx(service.batching_window_seconds)

        selected_at = now + wait_seconds + _PACING_RENDEZVOUS_WAKE_GUARD_SECONDS
        ready = service._ready_sessions(selected_at)
        assert [state.session_id for state in ready] == [ahead, lagging]
        assert (service._batch_key(ahead_state) == service._batch_key(lagging_state)) is (expected_batch_size == 2)
        batch = service._select_batch(ready, now=selected_at)
        assert [state.session_id for state in batch] == expected_ids
        service._execute_batch(batch, [{"W": True} for _ in batch])
        assert pipeline.batch_sizes[-1] == expected_batch_size
    finally:
        service.stop()


def test_absolute_rope_sessions_at_different_positions_do_not_share_a_batch() -> None:
    """A lagging LF3 session must catch up alone before it can phase-lock."""
    service, _ = _service(use_relative_rope=False, output_queue_size=4, batching_window_ms=2, control_idle_timeout=30)
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    ahead, lagging = (_create(service, value) for value in ("ahead", "lagging"))
    ahead_state = service._session(ahead)
    lagging_state = service._session(lagging)
    assert ahead_state is not None and lagging_state is not None
    try:
        now = 50_000.0
        _prepare_latest_continuation(
            ahead_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )
        _prepare_latest_continuation(
            lagging_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )
        ahead_state.pipeline_session.next_latent_frame = 9
        lagging_state.pipeline_session.next_latent_frame = 6

        ready = service._ready_sessions(now)

        assert [state.session_id for state in ready] == [ahead, lagging]
        assert service._batch_key(ahead_state) != service._batch_key(lagging_state)
        assert [state.session_id for state in service._select_batch(ready, now=now)] == [ahead]
    finally:
        service.stop()


def test_lagging_lf3_session_can_catch_up_then_rejoin_a_deadline_safe_batch() -> None:
    """A conservative phase-lock trace needs no LF shrink or model-interface change."""
    service, pipeline = _service(
        use_relative_rope=False, output_queue_size=4, batching_window_ms=2, control_idle_timeout=30
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(2)
    ahead, lagging = (_create(service, value) for value in ("ahead", "lagging"))
    ahead_state = service._session(ahead)
    lagging_state = service._session(lagging)
    assert ahead_state is not None and lagging_state is not None
    assert _take_and_notify(service, ahead_state)["type"] == "preview"
    assert _take_and_notify(service, lagging_state)["type"] == "preview"
    try:
        now = 60_000.0
        _prepare_latest_continuation(
            ahead_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )
        _prepare_latest_continuation(
            lagging_state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 0.5,
        )
        ahead_state.pipeline_session.next_latent_frame = 9
        lagging_state.pipeline_session.next_latent_frame = 6

        catch_up = service._select_batch(service._ready_sessions(now), now=now)
        assert [state.session_id for state in catch_up] == [lagging]
        assert now <= service._latest_safe_batch_start(catch_up)
        service._execute_batch(catch_up, [{"W": True}])
        assert pipeline.batch_sizes[-1] == 1
        assert lagging_state.pipeline_session.next_latent_frame == 9
        assert _take_and_notify(service, lagging_state)["type"] == "chunk"

        # Once the lagging session has completed a full LF3 block, it has the
        # same Absolute-RoPE position as the buffered-ahead peer. The existing
        # deadline-safe selector can now form B=2 without any frame shrink.
        aligned_at = time.monotonic()
        service._batch_compute_estimates.update({1: 0.04, 2: 0.06})
        for state in (ahead_state, lagging_state):
            _prepare_latest_continuation(
                state,
                now=aligned_at,
                pacing_ready_at=aligned_at,
                next_playout_deadline=aligned_at + 1.0,
            )
            state.pipeline_session.next_latent_frame = 9
        phase_locked = service._select_batch(service._ready_sessions(aligned_at), now=aligned_at)
        assert [state.session_id for state in phase_locked] == [ahead, lagging]
        assert aligned_at <= service._latest_safe_batch_start(phase_locked)
        service._execute_batch(phase_locked, [{"W": True}, {"D": True}])
        assert pipeline.batch_sizes[-2:] == [1, 2]
        assert [state.pipeline_session.next_latent_frame for state in phase_locked] == [12, 12]
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
        assert [state.session_id for state in service._select_batch(ready_after_window, now=now + wait_seconds)] == [
            first
        ]
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


def test_publisher_frame_credit_moves_dequeued_chunk_and_ignores_duplicate_progress() -> None:
    service, _ = _service(
        output_queue_size=4,
        publisher_frame_credit_enabled=True,
        publisher_frame_credit_target_seconds=1.5,
        publisher_frame_credit_reserve_frames=4,
        publisher_frame_credit_guard_ms=50,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(1)
    session_id = _create(service, "frame-credit", fps=12, control_latent_frames=3)
    state = service._session(session_id)
    assert state is not None
    try:
        assert state.output_queue.get(timeout=1)["type"] == "preview"
        assert service.enable_publisher_frame_tracking(session_id)
        state.output_queue.put({"type": "chunk", "frames": [Image.new("RGB", (8, 8)) for _ in range(12)]})

        async def pull_one() -> dict:
            iterator = service.pull_chunks(session_id)
            payload = await anext(iterator)
            await iterator.aclose()
            return payload

        assert asyncio.run(pull_one())["type"] == "chunk"
        metrics = service.runtime_metrics(session_id)
        assert metrics["queued_video_frames"] == 0
        assert metrics["publisher_unsubmitted_frames"] == 12
        assert metrics["frame_credit_frames"] == 12
        assert service.report_publisher_frame_progress(session_id, event="submitted", frames_delta=-1, sequence=1)
        assert not service.report_publisher_frame_progress(session_id, event="submitted", frames_delta=-1, sequence=1)
        metrics = service.runtime_metrics(session_id)
        assert metrics["publisher_unsubmitted_frames"] == 11
        assert metrics["publisher_frames_submitted"] == 1
    finally:
        service.stop()


def test_frame_credit_edf_uses_livekit_buffer_not_queue_payload_count() -> None:
    service, _ = _service(
        output_queue_size=4,
        publisher_frame_credit_enabled=True,
        publisher_frame_credit_target_seconds=1.5,
        publisher_frame_credit_reserve_frames=4,
        publisher_frame_credit_guard_ms=50,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(1)
    session_id = _create(service, "edf-credit", fps=12, control_latent_frames=3)
    state = service._session(session_id)
    assert state is not None
    try:
        state.output_queue.get(timeout=1)
        assert service.enable_publisher_frame_tracking(session_id)
        now = 91_000.0
        _prepare_latest_continuation(
            state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 0.01,
        )
        with state.lock:
            state.publisher_unsubmitted_frames = 19
        assert service._ready_sessions(now) == []
        assert service._frame_credit_ready_at(state, now) == pytest.approx(now + 1 / 12)
        with state.lock:
            state.publisher_unsubmitted_frames = 18
        assert service._ready_sessions(now) == [state]
        assert service._session_deadline(state, now) == pytest.approx(now + (14 / 12) - 0.05)
        metrics = service.runtime_metrics(session_id)
        assert metrics["frame_credit_frames"] == 18
        assert metrics["frame_credit_target_frames"] == 18
        assert metrics["pacing_buffered_video_payloads"] == 0
    finally:
        service.stop()


def test_frame_credit_explicit_target_frames_overrides_seconds_but_not_reserve_deadline_floor() -> None:
    service, _ = _service(
        output_queue_size=4,
        publisher_frame_credit_enabled=True,
        publisher_frame_credit_target_seconds=1.5,
        publisher_frame_credit_target_frames=36,
        publisher_frame_credit_reserve_frames=4,
        publisher_frame_credit_guard_ms=50,
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(1)
    session_id = _create(service, "edf-credit-frames", fps=12, control_latent_frames=3)
    state = service._session(session_id)
    assert state is not None
    try:
        state.output_queue.get(timeout=1)
        assert service.enable_publisher_frame_tracking(session_id)
        now = 92_000.0
        _prepare_latest_continuation(
            state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 0.01,
        )
        with state.lock:
            state.publisher_unsubmitted_frames = 37
        assert service._ready_sessions(now) == []
        assert service._frame_credit_ready_at(state, now) == pytest.approx(now + 1 / 12)
        with state.lock:
            state.publisher_unsubmitted_frames = 36
        assert service._ready_sessions(now) == [state]
        assert service._session_deadline(state, now) == pytest.approx(now + (32 / 12) - 0.05)
        metrics = service.runtime_metrics(session_id)
        assert metrics["frame_credit_frames"] == 36
        assert metrics["frame_credit_target_frames"] == 36
    finally:
        service.stop()


def test_offline_batch_compute_prior_expands_b2_deadline_wait_before_first_batch() -> None:
    prior_seconds = 0.7404982000589371
    service, _ = _service(
        max_batch_size=2,
        max_deadline_batch_wait_ms=1000,
        batch_compute_profile_name="h100_lf3_eager_full_pipeline_v1",
        batch_compute_prior_seconds={2: prior_seconds},
    )
    with service._scheduler_condition:
        service._scheduler_paused = True
    service.configure_session_capacity(1)
    session_id = _create(service, "first")
    state = service._session(session_id)
    assert state is not None
    try:
        now = 10_000.0
        _prepare_latest_continuation(
            state,
            now=now,
            pacing_ready_at=now,
            next_playout_deadline=now + 1.0,
        )
        assert service._batch_formation_wait_seconds(service._ready_sessions(now), now) == pytest.approx(
            1.0 - prior_seconds * 1.10
        )
    finally:
        service.stop()
