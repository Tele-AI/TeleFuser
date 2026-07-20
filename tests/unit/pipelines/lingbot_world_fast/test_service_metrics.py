from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from telefuser.metrics.runtime import start_runtime_measurement
from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService
from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionState,
)


def test_lingbot_actor_service_emits_runtime_and_chunk_measurements() -> None:
    config = LingBotWorldFastSessionConfig(
        prompt="test",
        image=Image.new("RGB", (16, 16)),
        chunk_size=1,
        frame_num=1,
        benchmark_metrics=True,
        show_control_hud=False,
    )
    runtime = LingBotWorldFastGenerationSession(
        config=config,
        height=480,
        width=832,
        latent_f=1,
        frame_tokens=1560,
        chunk_size=1,
        max_attention_size=32760,
        kv_cache_capacity_tokens=32760,
        cache_handle=7,
    )
    pipeline = MagicMock()
    pipeline.device = "cpu"
    pipeline.text_device = torch.device("cpu")
    pipeline.vae_encode_device = torch.device("cpu")
    pipeline.vae_decode_device = torch.device("cpu")
    pipeline.vae_device = torch.device("cpu")
    pipeline.config = SimpleNamespace(parallel_config=SimpleNamespace(device_ids=None))
    pipeline.dit = SimpleNamespace(local_attn_size=18, sink_size=6)
    pipeline._create_initialized_session.return_value = runtime
    pipeline._resolve_control.return_value = torch.zeros(1)

    streaming_runtime = MagicMock()
    streaming_session = SimpleNamespace(session_id="actor-session")
    streaming_runtime.create_session.return_value = streaming_session
    streaming_runtime.error.return_value = None
    streaming_runtime.try_submit_chunk.return_value = True
    streaming_runtime.poll_frames.return_value = [(0, [Image.new("RGB", (16, 16)) for _ in range(3)])]
    pipeline._get_streaming_runtime.return_value = streaming_runtime

    service = LingBotWorldFastService(pipeline)
    state = LingBotWorldFastSessionState(config=config, control_context=SimpleNamespace())
    statuses: list[dict[str, object]] = []

    def emit_status(stage: str, **data: object) -> None:
        statuses.append({"stage": stage, **data})

    with (
        patch.object(service, "_next_realtime_control", return_value=(object(), None)),
        patch.object(service, "_put_output"),
        patch(
            "telefuser.pipelines.lingbot_world_fast.service.start_runtime_measurement",
            wraps=start_runtime_measurement,
        ) as start_measurement,
    ):
        service._run_actor_worker_loop(state, state.control_context, MagicMock(), emit_status)

    runtime_ready = next(item for item in statuses if item.get("stage") == "runtime_ready")
    chunk_sent = next(item for item in statuses if item.get("stage") == "chunk_sent")

    assert runtime_ready["measurement"]["name"] == "runtime_creation"
    assert runtime_ready["runtime"]["kv_cache_capacity_tokens"] == 32760
    assert runtime_ready["runtime"]["kv_local_attn_size"] == 18
    assert chunk_sent["measurement"]["index"] == 0
    assert chunk_sent["measurement"]["frames"] == 3
    assert chunk_sent["measurement"]["compute_seconds"] >= 0
    assert chunk_sent["measurement"]["memory"] == []
    assert "encode_seconds" not in chunk_sent["measurement"]
    assert start_measurement.call_count == 2
    assert all(call.kwargs["capture_peak_memory"] is False for call in start_measurement.call_args_list)
