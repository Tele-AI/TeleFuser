import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

    def control_context(*args, **kwargs):
        runtime_ready.set()
        return SimpleNamespace(
            control_type="cam",
            chunk_size=3,
            width=8,
            height=8,
            latent_frames=3,
        )

    def generate_chunk(runtime, request, progress_callback=None):
        assert request.control is deferred_control
        runtime.active = False
        return SimpleNamespace(frames=[Image.new("RGB", (8, 8))])

    pipeline.control_context.side_effect = control_context
    pipeline.side_effect = generate_chunk
    builder = MagicMock()
    control = torch.ones(1)
    deferred_control = MagicMock(return_value=control)
    builder.defer.return_value = deferred_control

    with patch("telefuser.pipelines.lingbot_world_fast.service.LingBotWorldFastControlBuilder", return_value=builder):
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
    builder.defer.assert_called_once_with({"type": "control", "control_tensor": control})
    pipeline.release_session.assert_called_once()


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
