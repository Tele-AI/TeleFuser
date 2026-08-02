from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline, LingBotWorldFastPipelineConfig
from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionStatus,
)
from telefuser.pipelines.lingbot_world_fast.streaming import LingBotWorldFastStreamingRuntime


class _Worker:
    def __init__(self, name: str, release_order: list[str]):
        self.name = name
        self.release_order = release_order
        self.calls: list[dict[str, object]] = []
        self.release_calls: list[int] = []
        self.release_failures = 0
        self.closed = False

    def encode_condition_chunk(self, **kwargs):
        self.calls.append(kwargs)
        return lambda: torch.tensor([1.0])

    def decode_chunk(self, **kwargs):
        self.calls.append(kwargs)
        return lambda: torch.tensor([2.0])

    def release_cache(self, cache_handle: int, sync: bool = False):
        assert sync
        if self.release_failures:
            self.release_failures -= 1
            raise RuntimeError(f"{self.name} release failed")
        self.release_calls.append(cache_handle)
        self.release_order.append(self.name)
        return True

    def close(self):
        self.closed = True


class _Denoise:
    def __init__(self, release_order: list[str]):
        self.release_order = release_order
        self.advance_calls: list[int] = []
        self.calls: list[dict[str, object]] = []
        self.inference_modes: list[bool] = []
        self.release_calls: list[int] = []
        self.fail_denoise = False
        self.started: threading.Event | None = None
        self.release: threading.Event | None = None

    def denoise_and_update_cache(self, **kwargs):
        self.inference_modes.append(torch.is_inference_mode_enabled())
        if self.fail_denoise:
            raise RuntimeError("injected denoise failure")
        self.calls.append(kwargs)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            self.release.wait(timeout=1)
        return kwargs["condition_chunk"]

    def advance_noise(self, cache_handle: int):
        self.advance_calls.append(cache_handle)

    def release_cache(self, cache_handle: int):
        self.release_calls.append(cache_handle)
        self.release_order.append("denoise")
        return True


class _Pipeline:
    device = "cpu"
    vae_device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self):
        self.release_order: list[str] = []
        self.vae_encode_worker = _Worker("encode", self.release_order)
        self.vae_decode_worker = _Worker("decode", self.release_order)
        self.denoise_stage = _Denoise(self.release_order)
        self.released: list[int | None] = []

    @staticmethod
    def tensor2video(value):
        return [Image.new("RGB", (1, 1), color=(int(value.item()), 0, 0))]

    @staticmethod
    def _notify_progress(progress_callback, stage: str, **data: object) -> None:
        if progress_callback is not None:
            progress_callback(stage, **data)

    def release_session(self, runtime: LingBotWorldFastGenerationSession) -> None:
        self.released.append(runtime.cache_handle)


def test_independent_vae_stage_configs_are_resolved_independently() -> None:
    config = LingBotWorldFastPipelineConfig(
        vae_encode_config=ModelRuntimeConfig(
            device_type="cuda",
            device_id=0,
            torch_dtype=torch.float32,
            parallel_config=ParallelConfig(device_ids=[0]),
        ),
        vae_decode_config=ModelRuntimeConfig(
            device_type="cuda",
            device_id=1,
            torch_dtype=torch.float32,
            parallel_config=ParallelConfig(device_ids=[1]),
        ),
    )
    assert config.vae_encode_config.device_id == 0
    assert config.vae_encode_config.parallel_config.device_ids == [0]
    assert config.vae_decode_config.device_id == 1
    assert config.vae_decode_config.parallel_config.device_ids == [1]
    LingBotWorldFastPipeline._validate_vae_stage_runtime_config(config.vae_encode_config)
    LingBotWorldFastPipeline._validate_vae_stage_runtime_config(config.vae_decode_config)


def test_only_vae_decode_stage_accepts_spatial_parallel_workers() -> None:
    config = ModelRuntimeConfig(
        device_type="cuda",
        device_id=0,
        torch_dtype=torch.float32,
        parallel_config=ParallelConfig(device_ids=[0, 1, 2, 3], sp_ulysses_degree=4),
    )

    LingBotWorldFastPipeline._validate_vae_stage_runtime_config(config, allow_parallel=True)
    with pytest.raises(ValueError, match="VAE encode stage currently requires exactly one GPU"):
        LingBotWorldFastPipeline._validate_vae_stage_runtime_config(config)


def test_streaming_session_routes_one_chunk_through_three_stages() -> None:
    pipeline = _Pipeline()
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        prompt_emb=torch.tensor([0.0]),
        latent_h=1,
        latent_w=1,
        latent_f=1,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=1,
        cache_handle=9,
    )
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    assert streaming_runtime.orchestrator.spec.resource_groups == ()
    assert all(stage.resource_group is None for stage in streaming_runtime.orchestrator.spec.stages)
    session = streaming_runtime.create_session(runtime)
    try:
        streaming_runtime.submit_chunk(session, 0, torch.tensor([4.0]))
        assert streaming_runtime.wait_until_idle(session)
        assert streaming_runtime.error(session) is None
        outputs = streaming_runtime.poll_frames(session)
    finally:
        streaming_runtime.close_session(session)
        streaming_runtime.close()

    assert len(outputs) == 1
    assert outputs[0][0] == 0
    assert len(outputs[0][1]) == 1
    assert pipeline.vae_encode_worker.calls[0]["cache_handle"] == 9
    assert pipeline.vae_decode_worker.calls[0]["is_first_clip"] is True
    assert pipeline.vae_decode_worker.calls[0]["is_last_clip"] is True
    assert pipeline.vae_encode_worker.closed is False
    assert pipeline.vae_decode_worker.closed is False
    assert pipeline.vae_encode_worker.release_calls == [9]
    assert pipeline.vae_decode_worker.release_calls == [9]
    assert pipeline.denoise_stage.release_calls == [9]
    assert pipeline.denoise_stage.inference_modes == [True]
    assert pipeline.release_order == ["decode", "denoise", "encode"]
    assert pipeline.released == []
    assert runtime.cache_handle is None


def test_streaming_session_prefetches_two_conditions_ahead_of_controls() -> None:
    pipeline = _Pipeline()
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        prompt_emb=torch.tensor([0.0]),
        latent_h=1,
        latent_w=1,
        latent_f=4,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=1,
        cache_handle=15,
    )
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    session = streaming_runtime.create_session(runtime)
    denoise_started = threading.Event()
    release_denoise = threading.Event()
    pipeline.denoise_stage.started = denoise_started
    pipeline.denoise_stage.release = release_denoise
    try:
        assert streaming_runtime.wait_until_idle(session)
        assert [call["chunk_index"] for call in pipeline.vae_encode_worker.calls] == [0, 1]
        assert pipeline.denoise_stage.calls == []

        streaming_runtime.submit_chunk(session, 0, torch.tensor([4.0]))
        assert denoise_started.wait(timeout=1)
        assert [call["chunk_index"] for call in pipeline.vae_encode_worker.calls] == [0, 1]
        release_denoise.set()
        assert streaming_runtime.wait_until_idle(session)
        assert [call["chunk_index"] for call in pipeline.vae_encode_worker.calls] == [0, 1, 2]
        assert len(pipeline.denoise_stage.calls) == 1
        assert streaming_runtime.poll_frames(session)[0][0] == 0
        metrics = streaming_runtime.session_metrics(session)
        assert [index for index, _ in metrics.ingress_accepted_at] == [0]
    finally:
        release_denoise.set()
        streaming_runtime.close_session(session)
        streaming_runtime.close()


def test_streaming_session_overlaps_encode_denoise_and_decode() -> None:
    pipeline = _Pipeline()
    encode_started = threading.Event()
    denoise_started = threading.Event()
    decode_started = threading.Event()
    release_stages = threading.Event()

    original_encode = pipeline.vae_encode_worker.encode_condition_chunk
    original_decode = pipeline.vae_decode_worker.decode_chunk
    original_denoise = pipeline.denoise_stage.denoise_and_update_cache

    def encode_condition_chunk(**kwargs):
        result = original_encode(**kwargs)
        if kwargs["chunk_index"] != 2:
            return result

        def wait_for_release():
            encode_started.set()
            release_stages.wait(timeout=1)
            return result()

        return wait_for_release

    def denoise_and_update_cache(**kwargs):
        if kwargs["current_start"] == 1:
            denoise_started.set()
            release_stages.wait(timeout=1)
        return original_denoise(**kwargs)

    def decode_chunk(**kwargs):
        result = original_decode(**kwargs)
        if not kwargs["is_first_clip"]:
            return result

        def wait_for_release():
            decode_started.set()
            release_stages.wait(timeout=1)
            return result()

        return wait_for_release

    pipeline.vae_encode_worker.encode_condition_chunk = encode_condition_chunk
    pipeline.denoise_stage.denoise_and_update_cache = denoise_and_update_cache
    pipeline.vae_decode_worker.decode_chunk = decode_chunk
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        prompt_emb=torch.tensor([0.0]),
        latent_h=1,
        latent_w=1,
        latent_f=3,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=3,
        cache_handle=17,
    )
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    session = streaming_runtime.create_session(runtime)
    try:
        assert streaming_runtime.wait_until_idle(session)
        streaming_runtime.submit_chunk(session, 0, torch.tensor([4.0]))
        streaming_runtime.submit_chunk(session, 1, torch.tensor([5.0]))

        assert encode_started.wait(timeout=1)
        assert denoise_started.wait(timeout=1)
        assert decode_started.wait(timeout=1)
    finally:
        release_stages.set()
        streaming_runtime.close_session(session)
        streaming_runtime.close()


def test_capacity_poll_is_pure_and_control_fallback_is_atomic() -> None:
    pipeline = _Pipeline()
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        prompt_emb=torch.tensor([0.0]),
        latent_h=1,
        latent_w=1,
        latent_f=4,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=1,
        cache_handle=19,
    )
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    session = streaming_runtime.create_session(runtime)
    denoise_started = threading.Event()
    release_denoise = threading.Event()
    pipeline.denoise_stage.started = denoise_started
    pipeline.denoise_stage.release = release_denoise
    try:
        assert streaming_runtime.wait_until_idle(session)
        assert [call["chunk_index"] for call in pipeline.vae_encode_worker.calls] == [0, 1]

        streaming_runtime.submit_chunk(session, 0, torch.tensor([4.0]))
        assert denoise_started.wait(timeout=1)
        streaming_runtime.submit_chunk(session, 1, torch.tensor([5.0]))
        assert [call["chunk_index"] for call in pipeline.vae_encode_worker.calls] == [0, 1]

        assert streaming_runtime.can_submit_chunk(session)
        assert [call["chunk_index"] for call in pipeline.vae_encode_worker.calls] == [0, 1]

        assert streaming_runtime.try_submit_chunk(session, 2, torch.tensor([6.0]))
        deadline = time.monotonic() + 1
        while len(pipeline.vae_encode_worker.calls) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [call["chunk_index"] for call in pipeline.vae_encode_worker.calls] == [0, 1, 2]

        assert not streaming_runtime.try_submit_chunk(session, 3, torch.tensor([7.0]))
        assert [call["chunk_index"] for call in pipeline.vae_encode_worker.calls] == [0, 1, 2]
    finally:
        release_denoise.set()
        streaming_runtime.close_session(session)
        streaming_runtime.close()


def test_streaming_session_rejects_out_of_order_or_duplicate_controls() -> None:
    pipeline = _Pipeline()
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        prompt_emb=torch.tensor([0.0]),
        latent_h=1,
        latent_w=1,
        latent_f=2,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=1,
        cache_handle=16,
    )
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    session = streaming_runtime.create_session(runtime)
    try:
        with pytest.raises(ValueError, match="Expected LingBot control chunk 0, got 1"):
            streaming_runtime.try_submit_chunk(session, 1, torch.tensor([4.0]))
        assert streaming_runtime.try_submit_chunk(session, 0, torch.tensor([4.0]))
        with pytest.raises(ValueError, match="Expected LingBot control chunk 1, got 0"):
            streaming_runtime.try_submit_chunk(session, 0, torch.tensor([4.0]))
    finally:
        streaming_runtime.close_session(session)
        streaming_runtime.close()


def test_streaming_runtime_shares_one_actor_graph_across_sessions() -> None:
    pipeline = _Pipeline()
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    sessions = []
    try:
        for cache_handle in (10, 11):
            runtime = LingBotWorldFastGenerationSession(
                config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
                prompt_emb=torch.tensor([0.0]),
                latent_h=1,
                latent_w=1,
                latent_f=1,
                height=8,
                width=8,
                frame_tokens=1,
                chunk_size=1,
                max_attention_size=1,
                cache_handle=cache_handle,
            )
            session = streaming_runtime.create_session(runtime)
            sessions.append(session)
            streaming_runtime.submit_chunk(session, 0, torch.tensor([float(cache_handle)]))

        outputs = []
        for session in sessions:
            assert streaming_runtime.wait_until_idle(session)
            outputs.extend(streaming_runtime.poll_frames(session))
    finally:
        for session in sessions:
            streaming_runtime.close_session(session)
        streaming_runtime.close()

    assert [index for index, _ in outputs] == [0, 0]
    assert {call["cache_handle"] for call in pipeline.vae_encode_worker.calls} == {10, 11}
    assert {call["cache_handle"] for call in pipeline.vae_decode_worker.calls} == {10, 11}
    assert pipeline.vae_encode_worker.release_calls == [10, 11]
    assert pipeline.vae_decode_worker.release_calls == [10, 11]
    assert pipeline.denoise_stage.release_calls == [10, 11]
    assert pipeline.release_order == ["decode", "denoise", "encode"] * 2
    assert pipeline.released == []


def test_streaming_runtime_preserves_world_kv_decode_only_hits() -> None:
    pipeline = _Pipeline()
    binding = MagicMock()
    cached_latent = torch.tensor([7.0])
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        prompt_emb=torch.tensor([0.0]),
        latent_h=1,
        latent_w=1,
        latent_f=1,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=1,
        cache_handle=12,
        world_kv_binding=binding,
        world_kv_cached_latents={0: cached_latent},
    )
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    session = streaming_runtime.create_session(runtime)
    try:
        streaming_runtime.submit_chunk(session, 0, torch.tensor([4.0]))
        assert streaming_runtime.wait_until_idle(session)
        assert streaming_runtime.error(session) is None
        assert streaming_runtime.poll_frames(session)
    finally:
        streaming_runtime.close_session(session)
        streaming_runtime.close()

    assert pipeline.denoise_stage.advance_calls == [12]
    decode_call = pipeline.vae_decode_worker.calls[0]
    torch.testing.assert_close(decode_call["latents"], cached_latent)
    binding.on_chunk_finalized.assert_called_once()


def test_streaming_session_cleanup_failure_is_poisoned_and_retryable() -> None:
    pipeline = _Pipeline()
    pipeline.vae_decode_worker.release_failures = 1
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        prompt_emb=torch.tensor([0.0]),
        latent_h=1,
        latent_w=1,
        latent_f=1,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=1,
        cache_handle=13,
    )
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    session = streaming_runtime.create_session(runtime)
    try:
        with pytest.raises(RuntimeError, match="Failed to clean streaming session"):
            streaming_runtime.close_session(session, timeout=1)
        assert runtime.status == LingBotWorldFastSessionStatus.POISONED
        assert runtime.cache_handle == 13
        assert pipeline.release_order == ["denoise", "encode"]

        streaming_runtime.close_session(session, timeout=1)
        assert pipeline.release_order == ["denoise", "encode", "decode"]
        assert runtime.status == LingBotWorldFastSessionStatus.RELEASED
        assert runtime.cache_handle is None
    finally:
        streaming_runtime.close()


def test_streaming_stage_failure_automatically_releases_lingbot_caches() -> None:
    pipeline = _Pipeline()
    pipeline.denoise_stage.fail_denoise = True
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        prompt_emb=torch.tensor([0.0]),
        latent_h=1,
        latent_w=1,
        latent_f=1,
        height=8,
        width=8,
        frame_tokens=1,
        chunk_size=1,
        max_attention_size=1,
        cache_handle=14,
    )
    streaming_runtime = LingBotWorldFastStreamingRuntime(pipeline)
    session = streaming_runtime.create_session(runtime)
    try:
        streaming_runtime.submit_chunk(session, 0, torch.tensor([4.0]))
        assert streaming_runtime.wait_until_idle(session, timeout=1)
        assert str(streaming_runtime.error(session)) == "injected denoise failure"
        assert pipeline.release_order == ["decode", "denoise", "encode"]
        assert pipeline.vae_encode_worker.release_calls == [14]
        assert pipeline.vae_decode_worker.release_calls == [14]
        assert pipeline.denoise_stage.release_calls == [14]

        streaming_runtime.close_session(session)
        assert pipeline.release_order == ["decode", "denoise", "encode"]
        assert runtime.status == LingBotWorldFastSessionStatus.RELEASED
        assert runtime.cache_handle is None
    finally:
        streaming_runtime.close()
