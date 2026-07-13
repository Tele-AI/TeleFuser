import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline
from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastChunkRequest,
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionStatus,
)
from telefuser.worker.parallel_worker import ParallelWorker


def _session(
    *,
    active: bool = True,
    current_chunk_index: int = 0,
    chunk_count: int = 2,
) -> LingBotWorldFastGenerationSession:
    empty = torch.empty(0)
    return LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        status=LingBotWorldFastSessionStatus.READY,
        prompt_emb=empty,
        noise_chunks=[empty for _ in range(chunk_count)],
        condition_chunks=[empty for _ in range(chunk_count)],
        latent_h=1,
        latent_w=1,
        latent_f=chunk_count,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=1,
        cache_handle=7,
        active=active,
        current_chunk_index=current_chunk_index,
    )


def _control() -> torch.Tensor:
    return torch.zeros(1, 384, 1, 1, 1, dtype=torch.float32)


def _pipeline() -> LingBotWorldFastPipeline:
    pipeline = LingBotWorldFastPipeline(device="cpu")
    pipeline.config = SimpleNamespace(control_type="cam")
    return pipeline


def test_pipeline_call_generates_one_explicitly_indexed_chunk() -> None:
    pipeline = _pipeline()
    pipeline.denoise_stage = MagicMock()
    runtime = _session()
    expected = [Image.new("RGB", (8, 8), "red") for _ in range(9)]

    def generate_next_chunk(runtime_state, control=None, progress_callback=None):
        runtime_state.current_chunk_index += 1
        runtime_state.emitted_frames += len(expected)
        return expected

    pipeline.generate_next_chunk = MagicMock(side_effect=generate_next_chunk)
    progress_callback = MagicMock()
    control = _control()

    result = pipeline(
        runtime,
        LingBotWorldFastChunkRequest(
            chunk_index=0,
            session_id="session-a",
            control=control,
        ),
        progress_callback=progress_callback,
    )

    assert result.frames == expected
    assert result.chunk_index == 0
    assert result.session_id == "session-a"
    assert result.emitted_frames == 9
    assert result.done is False
    assert runtime.status == LingBotWorldFastSessionStatus.COMMITTED
    pipeline.generate_next_chunk.assert_called_once_with(
        runtime,
        control=control,
        progress_callback=progress_callback,
    )


def test_chunk_request_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LingBotWorldFastChunkRequest(chunk_index=-1, control=torch.zeros(1))

    with pytest.raises(TypeError, match="model control tensor"):
        LingBotWorldFastChunkRequest(chunk_index=0, control=object())


def test_pipeline_call_rejects_inactive_runtime() -> None:
    pipeline = _pipeline()
    runtime = _session(active=False)
    pipeline.generate_next_chunk = MagicMock()

    with pytest.raises(RuntimeError, match="inactive"):
        pipeline(runtime, LingBotWorldFastChunkRequest(chunk_index=0, control=torch.zeros(1)))

    pipeline.generate_next_chunk.assert_not_called()


def test_pipeline_call_rejects_out_of_order_chunk() -> None:
    pipeline = _pipeline()
    runtime = _session(current_chunk_index=1)
    pipeline.generate_next_chunk = MagicMock()

    with pytest.raises(ValueError, match="does not match session index"):
        pipeline(runtime, LingBotWorldFastChunkRequest(chunk_index=0, control=torch.zeros(1)))

    pipeline.generate_next_chunk.assert_not_called()


def test_new_session_rejects_out_of_order_chunk_without_initializing() -> None:
    pipeline = _pipeline()
    session = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8)))
    )
    deferred_control = MagicMock(return_value=_control())
    pipeline._create_initialized_session = MagicMock()

    with pytest.raises(ValueError, match="does not match session index"):
        pipeline(session, LingBotWorldFastChunkRequest(chunk_index=1, control=deferred_control))

    deferred_control.assert_not_called()
    pipeline._create_initialized_session.assert_not_called()


def test_first_pipeline_call_initializes_the_external_session() -> None:
    pipeline = _pipeline()
    pipeline.denoise_stage = MagicMock()
    initialized = _session()
    session = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8)))
    )
    control = _control()
    events: list[str] = []

    def initialize(config, progress_callback, before_cache):
        assert config is session.config
        assert progress_callback is None
        events.append("initialization_started")
        before_cache()
        events.append("cache_initialization_ready")
        return initialized

    def deferred_control() -> torch.Tensor:
        events.append("control_materialized")
        return control

    pipeline.generate_next_chunk = MagicMock(return_value=[])
    pipeline._create_initialized_session = MagicMock(side_effect=initialize)

    pipeline(session, LingBotWorldFastChunkRequest(chunk_index=0, control=deferred_control))

    pipeline._create_initialized_session.assert_called_once()
    assert events == ["initialization_started", "control_materialized", "cache_initialization_ready"]
    pipeline.generate_next_chunk.assert_called_once_with(session, control=control, progress_callback=None)


def test_pipeline_call_releases_runtime_when_generation_fails() -> None:
    pipeline = _pipeline()
    pipeline.denoise_stage = MagicMock()
    runtime = _session(chunk_count=1)
    pipeline.generate_next_chunk = MagicMock(side_effect=RuntimeError("generation failed"))

    with pytest.raises(RuntimeError, match="generation failed"):
        pipeline(runtime, LingBotWorldFastChunkRequest(chunk_index=0, control=_control()))

    assert runtime.active is False
    assert runtime.status == LingBotWorldFastSessionStatus.POISONED
    assert runtime.poisoned_reason == "RuntimeError: generation failed"
    pipeline.denoise_stage.release_cache.assert_called_once_with(7)

    with pytest.raises(RuntimeError, match="poisoned"):
        pipeline(runtime, LingBotWorldFastChunkRequest(chunk_index=0, control=_control()))


def test_pipeline_call_rejects_control_with_wrong_shape() -> None:
    pipeline = _pipeline()
    pipeline.denoise_stage = MagicMock()
    runtime = _session()

    with pytest.raises(ValueError, match="Control shape"):
        pipeline(runtime, LingBotWorldFastChunkRequest(chunk_index=0, control=torch.zeros(1, dtype=torch.float32)))

    pipeline.denoise_stage.release_cache.assert_not_called()


def test_final_chunk_releases_decoder_state_and_cache() -> None:
    pipeline = _pipeline()
    pipeline.denoise_stage = MagicMock()
    runtime = _session(chunk_count=1)
    runtime.decoder_state.feat_cache = [torch.ones(1)]
    runtime.decoder_state.feat_idx = [1]

    def generate_next_chunk(session, control, progress_callback=None):
        session.current_chunk_index = 1
        session.active = False
        return []

    pipeline.generate_next_chunk = MagicMock(side_effect=generate_next_chunk)

    result = pipeline(runtime, LingBotWorldFastChunkRequest(chunk_index=0, control=_control()))

    assert result.done is True
    assert runtime.cache_handle is None
    assert runtime.decoder_state.feat_cache == []
    assert runtime.decoder_state.feat_idx == [0]
    assert runtime.status == LingBotWorldFastSessionStatus.RELEASED
    pipeline.denoise_stage.release_cache.assert_called_once_with(7)


def test_release_session_is_idempotent() -> None:
    pipeline = _pipeline()
    pipeline.denoise_stage = MagicMock()
    session = _session()

    pipeline.release_session(session)
    pipeline.release_session(session)

    pipeline.denoise_stage.release_cache.assert_called_once_with(7)
    assert session.cache_handle is None
    assert session.status == LingBotWorldFastSessionStatus.RELEASED


def test_concurrent_chunk_on_same_session_is_rejected() -> None:
    pipeline = _pipeline()
    session = _session()
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_transaction() -> None:
        with session.transaction_lock:
            lock_acquired.set()
            release_lock.wait(timeout=2.0)

    holder = threading.Thread(target=hold_transaction, daemon=True)
    holder.start()
    assert lock_acquired.wait(timeout=1.0)

    try:
        with pytest.raises(RuntimeError, match="already has a chunk in progress"):
            pipeline(
                session,
                LingBotWorldFastChunkRequest(
                    chunk_index=0,
                    control=torch.zeros(1),
                ),
            )
    finally:
        release_lock.set()
        holder.join(timeout=1.0)

    assert not holder.is_alive()
    assert session.status == LingBotWorldFastSessionStatus.READY


def test_generate_video_drains_runtime_and_releases_it() -> None:
    pipeline = LingBotWorldFastPipeline(device="cpu")
    pipeline.release_session = MagicMock(side_effect=lambda state: setattr(state, "active", False))
    frame = Image.new("RGB", (8, 8))

    def generate(runtime_state, request, progress_callback=None):
        runtime_state.current_chunk_index += 1
        runtime_state.emitted_frames += 1
        if runtime_state.current_chunk_index == 2:
            runtime_state.active = False
        return SimpleNamespace(frames=[frame])

    config = LingBotWorldFastSessionConfig(prompt="test", image=frame)

    with patch.object(LingBotWorldFastPipeline, "__call__", side_effect=generate) as generate_chunk:
        frames = pipeline.generate_video(config, controls=[torch.tensor([1]), torch.tensor([2])])

    assert frames == [frame, frame]
    assert generate_chunk.call_count == 2
    pipeline.release_session.assert_called_once()


def test_pipeline_close_delegates_to_parallel_worker() -> None:
    pipeline = LingBotWorldFastPipeline(device="cpu")
    worker = object.__new__(ParallelWorker)
    worker.close = MagicMock()
    pipeline.denoise_stage = worker

    pipeline.close()

    worker.close.assert_called_once_with()
