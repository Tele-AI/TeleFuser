from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch
from PIL import Image

from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService
from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionState,
)


class _ImmediateLoop:
    @staticmethod
    def call_soon_threadsafe(callback, payload) -> None:
        callback(payload)


class _FakePipeline:
    def __init__(self) -> None:
        self.device = "cpu"
        self.vae_device = torch.device("cpu")
        self.text_device = torch.device("cpu")
        self.dit = SimpleNamespace(local_attn_size=18, sink_size=6)

    @staticmethod
    def create_runtime(config, progress_callback=None):
        return SimpleNamespace(
            active=True,
            height=480,
            width=832,
            latent_f=21,
            frame_tokens=1560,
            chunk_size=3,
            max_attention_size=32760,
            noise_chunks=[object()],
            self_kv_cache=[{"k": SimpleNamespace(shape=(1, 32760))}],
        )

    @staticmethod
    def generate_next_chunk(runtime, **kwargs):
        runtime.active = False
        return [Image.new("RGB", (16, 16)) for _ in range(3)]

    @staticmethod
    def encode_frames_to_b64(frames):
        return ["encoded" for _ in frames]

    @staticmethod
    def build_control_override(runtime, incoming):
        return None


def test_lingbot_service_emits_runtime_and_chunk_measurements(monkeypatch) -> None:
    service = LingBotWorldFastService(_FakePipeline())
    state = LingBotWorldFastSessionState(
        config=LingBotWorldFastSessionConfig(
            prompt="test",
            image=Image.new("RGB", (16, 16)),
            benchmark_metrics=True,
            show_control_hud=False,
        ),
        output_queue=asyncio.Queue(),
        loop=_ImmediateLoop(),
    )
    service._sessions["session"] = state
    monkeypatch.setattr(service, "_emit_preview_frame", lambda current: None)
    monkeypatch.setattr(service, "_release_runtime", lambda current: None)

    service._worker_loop("session")

    payloads = []
    while not state.output_queue.empty():
        payloads.append(state.output_queue.get_nowait())
    runtime_ready = next(item for item in payloads if item.get("stage") == "runtime_ready")
    chunk_sent = next(item for item in payloads if item.get("stage") == "chunk_sent")

    assert runtime_ready["measurement"]["name"] == "runtime_creation"
    assert runtime_ready["runtime"]["kv_cache_capacity_tokens"] == 32760
    assert runtime_ready["runtime"]["kv_local_attn_size"] == 18
    assert chunk_sent["measurement"]["index"] == 0
    assert chunk_sent["measurement"]["frames"] == 3
    assert chunk_sent["measurement"]["compute_seconds"] >= 0
    assert chunk_sent["measurement"]["encode_seconds"] >= 0
