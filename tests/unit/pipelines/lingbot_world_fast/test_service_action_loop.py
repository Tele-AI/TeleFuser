import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
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
        return SimpleNamespace(frames=[Image.new("RGB", (8, 8))], done=True)

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


def test_async_worker_initializes_session_before_finish_check() -> None:
    pipeline = MagicMock()
    pipeline.async_vae_enabled = True
    control_context = SimpleNamespace(
        control_type="cam",
        chunk_size=4,
        width=8,
        height=8,
        latent_frames=4,
    )
    pipeline.control_context.return_value = control_context
    service = LingBotWorldFastService(pipeline)
    state = LingBotWorldFastSessionState(
        config=LingBotWorldFastSessionConfig(
            prompt="test",
            image=Image.new("RGB", (8, 8)),
            frame_num=13,
            chunk_size=4,
            show_control_hud=False,
        ),
        control_context=control_context,
    )
    state.pending_inputs.put({"type": "control", "control_tensor": torch.ones(1)})
    handle = SimpleNamespace(chunk_id=0, is_last_clip=True)

    def submit_chunk(runtime, request, progress_callback=None):
        assert runtime.chunk_size == 0
        runtime.chunk_size = 4
        runtime.latent_f = 4
        runtime.current_chunk_index = 1
        return handle

    pipeline.submit_async_vae_chunk.side_effect = submit_chunk
    pipeline.wait_async_vae_chunk.return_value = [Image.new("RGB", (8, 8))]
    builder = MagicMock()
    builder.defer.return_value = MagicMock(return_value=torch.ones(1))

    with patch("telefuser.pipelines.lingbot_world_fast.service.LingBotWorldFastControlBuilder", return_value=builder):
        service._run_worker_loop("session-a", state, MagicMock())

    pipeline.submit_async_vae_chunk.assert_called_once()
    pipeline.wait_async_vae_chunk.assert_called_once_with(ANY, handle, progress_callback=ANY)
    pipeline.release_session.assert_called_once()


def test_direction_action_updates_state_and_wakes_worker() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state

    service.push_chunk(
        "session-a",
        {"type": "control", "direction": "up", "event": "press"},
    )

    assert state.pressed_controls == {"w"}
    assert state.pending_inputs.get_nowait() == {"type": "direction_control"}


def test_release_stops_control_without_scheduling_stationary_generation() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state
    state.pressed_controls.add("w")

    service.push_chunk("session-a", {"type": "control", "key": "ArrowUp", "event": "release"})

    assert state.pressed_controls == set()
    assert state.pending_inputs.empty()


def test_directional_chunks_match_source_video_rate_integration_and_boundary() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.pressed_controls.add("w")
    context = SimpleNamespace(control_type="cam", chunk_size=3)

    first = service._build_directional_control_chunk(state, context)
    second = service._build_directional_control_chunk(state, context)

    assert first is not None
    assert second is not None
    assert "previous_pose" not in first
    first_poses = np.asarray(first["poses"])
    second_poses = np.asarray(second["poses"])
    np.testing.assert_allclose(first_poses[:, 2, 3], [0.0, 0.2, 0.4], rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(second["previous_pose"])[2, 3], 0.4, rtol=0, atol=1e-6)
    np.testing.assert_allclose(second_poses[:, 2, 3], [0.6, 0.8, 1.0], rtol=0, atol=1e-6)


def test_wasd_ijkl_and_arrow_aliases_have_distinct_translation_and_rotation_controls() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    context = SimpleNamespace(control_type="cam", chunk_size=3)

    state.pressed_controls.add("j")
    yaw_chunk = service._build_directional_control_chunk(state, context)

    assert yaw_chunk is not None
    expected_yaw = np.deg2rad(-16.0)
    np.testing.assert_allclose(np.asarray(yaw_chunk["poses"])[-1, 0, 0], np.cos(expected_yaw), atol=1e-6)
    assert service._direction_from_chunk({"key": "ArrowLeft"}) == "j"
    assert service._direction_from_chunk({"key": "KeyA"}) == "a"
    assert service._direction_from_chunk({"key": "KeyI"}) == "i"


def test_control_hud_always_places_movement_and_rotation_at_bottom_corners() -> None:
    width, height = 832, 480
    frame = Image.new("RGB", (width, height))

    rendered = LingBotWorldFastService._overlay_control_hud([frame], controls=None)[0]

    changed = np.any(np.asarray(rendered) != np.asarray(frame), axis=2)
    changed_y, changed_x = np.nonzero(changed)
    assert changed_y.min() > height // 2
    assert np.any(changed[:, : width // 2])
    assert np.any(changed[:, width // 2 :])


def test_control_hud_labels_movement_and_rotation_panels() -> None:
    frame = Image.new("RGB", (832, 480))

    with patch.object(
        LingBotWorldFastService,
        "_draw_control_panel",
        wraps=LingBotWorldFastService._draw_control_panel,
    ) as draw_panel:
        LingBotWorldFastService._overlay_control_hud([frame], controls=["w"])

    assert [call.kwargs["label"] for call in draw_panel.call_args_list] == ["MOVE", "ROTATE"]


def test_preview_frame_includes_idle_control_hud() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.control_context = SimpleNamespace(width=832, height=480)

    with (
        patch.object(service, "_overlay_control_hud", return_value=[Image.new("RGB", (832, 480))]) as overlay,
        patch.object(service, "_encode_frames_to_b64", return_value=["encoded-frame"]),
        patch.object(service, "_put_output") as put_output,
    ):
        service._emit_preview_frame(state)

    overlay.assert_called_once()
    assert overlay.call_args.kwargs == {"controls": None}
    assert put_output.call_args.args[1]["frames_b64"] == ["encoded-frame"]


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


def test_create_session_rejects_invalid_pipeline_configuration() -> None:
    pipeline = MagicMock()
    pipeline.control_context.side_effect = ValueError("invalid session configuration")
    service = LingBotWorldFastService(pipeline)

    with pytest.raises(ValueError, match="invalid session"):
        service.create_session({"image": Image.new("RGB", (8, 8))})

    assert service._sessions == {}


def test_create_session_limits_stream_generation_to_20_seconds() -> None:
    pipeline = MagicMock()
    service = LingBotWorldFastService(pipeline)

    session_id = service.create_session(
        {
            "image": Image.new("RGB", (8, 8)),
            "fps": 16,
            "frame_num": 321,
        }
    )
    assert service._sessions[session_id].config.frame_num == 321
    service.close_session(session_id)

    with pytest.raises(ValueError, match="must not exceed 20 seconds"):
        service.create_session(
            {
                "image": Image.new("RGB", (8, 8)),
                "fps": 16,
                "frame_num": 333,
            }
        )


def test_create_session_uses_truncated_frame_count_for_duration_validation() -> None:
    pipeline = MagicMock()
    service = LingBotWorldFastService(pipeline)

    session_id = service.create_session(
        {
            "image": Image.new("RGB", (8, 8)),
            "fps": 16,
            "chunk_size": 3,
            "frame_num": 13,
            "max_duration_seconds": 0.5,
        }
    )

    assert service._sessions[session_id].config.frame_num == 9
    assert pipeline.control_context.call_args.args[0].frame_num == 9
    service.close_session(session_id)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", 0, "fps must be positive"),
        ("chunk_size", 0, "chunk_size must be positive"),
        ("chunk_size", -1, "chunk_size must be positive"),
        ("max_duration_seconds", 0, "max_duration_seconds must be positive"),
        ("max_duration_seconds", -1, "max_duration_seconds must be positive"),
    ],
)
def test_create_session_rejects_non_positive_stream_parameters(field: str, value: int, message: str) -> None:
    service = LingBotWorldFastService(MagicMock())

    request = {"image": Image.new("RGB", (8, 8)), field: value}
    if field == "max_duration_seconds":
        request["frame_num"] = 9
    with pytest.raises(ValueError, match=message):
        service.create_session(request)


def test_create_session_initializes_fixed_intrinsics_from_intrinsics_path() -> None:
    pipeline = MagicMock()
    service = LingBotWorldFastService(pipeline)
    intrinsics = np.asarray([[8.0, 8.0, 4.0, 4.0], [9.0, 9.0, 4.0, 4.0]])

    with patch("telefuser.pipelines.lingbot_world_fast.service.np.load", return_value=intrinsics) as load:
        session_id = service.create_session(
            {
                "image": Image.new("RGB", (8, 8)),
                "intrinsics_path": "/controls/intrinsics.npy",
            }
        )

    load.assert_called_once_with(Path("/controls/intrinsics.npy"))
    session_config = pipeline.control_context.call_args.args[0]
    assert session_config.intrinsics is intrinsics
    assert service._sessions[session_id].control_context is pipeline.control_context.return_value


def test_pull_chunks_drains_terminal_messages_after_session_becomes_inactive() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.active = False
    state.worker_thread = MagicMock()
    state.worker_thread.is_alive.return_value = True
    state.output_queue = asyncio.Queue()
    state.output_queue.put_nowait({"type": "preview"})
    state.output_queue.put_nowait({"type": "error"})
    state.output_queue.put_nowait({"type": "done"})
    service._sessions["session-a"] = state

    async def collect() -> list[dict]:
        return [chunk async for chunk in service.pull_chunks("session-a")]

    assert asyncio.run(collect()) == [{"type": "preview"}, {"type": "error"}]


def test_service_start_warms_the_pipeline_with_its_default_shape() -> None:
    pipeline = MagicMock()
    pipeline.config = SimpleNamespace(control_type="cam", orig_width=832, orig_height=480)
    service = LingBotWorldFastService(pipeline, default_session_config={"chunk_size": 4})

    service.start()

    warmup_config = pipeline.warmup.call_args.args[0]
    assert warmup_config.image.size == (832, 480)
    assert warmup_config.chunk_size == 4
    assert warmup_config.frame_num == 29
