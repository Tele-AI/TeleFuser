import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from PIL import Image

from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService
from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionState,
)


def _state() -> LingBotWorldFastSessionState:
    return LingBotWorldFastSessionState(
        config=LingBotWorldFastSessionConfig(
            prompt="test",
            image=Image.new("RGB", (8, 8)),
            frame_num=9,
        )
    )


def test_online_worker_waits_for_external_action_before_generating() -> None:
    pipeline = MagicMock()
    pipeline._best_output_size.return_value = (8, 8)
    pipeline.check_resize_height_width.return_value = (8, 8)
    service = LingBotWorldFastService(pipeline)
    state = _state()
    runtime_ready = threading.Event()
    runtime = SimpleNamespace(
        active=True,
        width=8,
        height=8,
        latent_f=3,
        noise_chunks=[torch.zeros(1, 1, 3)],
        current_chunk_index=0,
    )

    def create_session(*args, **kwargs):
        runtime_ready.set()
        return runtime

    def generate_chunk(runtime_state, request, progress_callback=None):
        assert request.control_override is control
        runtime_state.active = False
        return SimpleNamespace(frames=[Image.new("RGB", (8, 8))])

    pipeline.create_session.side_effect = create_session
    pipeline.build_control_override.return_value = control = torch.ones(1)
    pipeline.side_effect = generate_chunk

    worker = threading.Thread(
        target=service._run_worker_loop,
        args=("session-a", state, MagicMock()),
        daemon=True,
    )
    worker.start()

    assert runtime_ready.wait(timeout=1.0)
    assert pipeline.call_count == 0

    state.pending_inputs.put({"type": "control", "control_tensor": control})
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert pipeline.call_count == 1
    pipeline.release_session.assert_called_once_with(runtime)


def test_direction_action_updates_state_and_wakes_worker() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state

    service.push_chunk(
        "session-a",
        {"type": "control", "direction": "up", "event": "press"},
    )

    assert state.pressed_controls == {"up"}
    assert state.pending_inputs.get_nowait() == {"type": "direction_control"}


def test_service_stop_closes_sessions_before_pipeline() -> None:
    pipeline = MagicMock()
    service = LingBotWorldFastService(pipeline)
    service._sessions = {"session-a": MagicMock(), "session-b": MagicMock()}
    service.close_session = MagicMock()

    service.stop()

    assert service.close_session.call_args_list == [
        (("session-a",),),
        (("session-b",),),
    ]
    pipeline.close.assert_called_once_with()
