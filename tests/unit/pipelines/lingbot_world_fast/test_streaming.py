from __future__ import annotations

import threading
import time

import torch
from PIL import Image

from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
)
from telefuser.pipelines.lingbot_world_fast.streaming import LingBotWorldFastStreamingRuntime


class _Worker:
    def __init__(self):
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def encode_condition_chunk(self, **kwargs):
        self.calls.append(kwargs)
        return lambda: torch.tensor([1.0])

    def decode_chunk(self, **kwargs):
        self.calls.append(kwargs)
        return lambda: torch.tensor([2.0])

    def close(self):
        self.closed = True


class _Denoise:
    def denoise_and_update_cache(self, **kwargs):
        return kwargs["condition_chunk"]


class _Pipeline:
    device = "cpu"
    torch_dtype = torch.float32

    def __init__(self):
        self.vae_encode_worker = _Worker()
        self.vae_decode_worker = _Worker()
        self.denoise_stage = _Denoise()
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
    assert pipeline.released == [9]


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
    assert pipeline.released == [10, 11]
