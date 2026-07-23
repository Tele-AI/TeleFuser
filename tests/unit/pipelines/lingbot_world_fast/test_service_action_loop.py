import asyncio
import base64
import io
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService
from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastGenerationSession,
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


def test_online_worker_delegates_to_the_actor_scheduler() -> None:
    pipeline = MagicMock()
    pipeline._best_output_size.return_value = (8, 8)
    pipeline.check_resize_height_width.return_value = (8, 8)
    pipeline.control_context.return_value = SimpleNamespace(
        control_type="cam", chunk_size=3, width=8, height=8, latent_frames=3
    )
    service = LingBotWorldFastService(pipeline)
    state = _state()
    state.active = False

    with patch.object(service, "_run_actor_worker_loop") as run_realtime:
        service._run_worker_loop("session-a", state, MagicMock())

    run_realtime.assert_called_once()


def test_actor_worker_submits_control_and_emits_ordered_chunk() -> None:
    pipeline = MagicMock()
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(
            prompt="test",
            image=Image.new("RGB", (8, 8)),
            chunk_size=1,
            frame_num=1,
            show_control_hud=False,
        ),
        latent_f=1,
        chunk_size=1,
        width=8,
        height=8,
        cache_handle=7,
    )
    pipeline._create_initialized_session.return_value = runtime
    pipeline._resolve_control.return_value = torch.zeros(1)
    streaming_runtime = MagicMock()
    streaming_session = SimpleNamespace(session_id="actor-session")
    streaming_runtime.create_session.return_value = streaming_session
    streaming_runtime.error.return_value = None
    streaming_runtime.try_submit_chunk.return_value = True
    streaming_runtime.poll_frames.return_value = [(0, [Image.new("RGB", (8, 8))])]
    pipeline._get_streaming_runtime.return_value = streaming_runtime

    service = LingBotWorldFastService(pipeline)
    state = LingBotWorldFastSessionState(config=runtime.config)
    state.control_context = SimpleNamespace()
    control_builder = MagicMock()
    emit_status = MagicMock()
    first_control = (object(), ["w"])

    with (
        patch.object(service, "_next_realtime_control", return_value=first_control),
        patch.object(service, "_put_output") as put_output,
    ):
        service._run_actor_worker_loop(state, state.control_context, control_builder, emit_status)

    streaming_runtime.create_session.assert_called_once_with(runtime, progress_callback=emit_status)
    streaming_runtime.try_submit_chunk.assert_called_once_with(
        streaming_session, 0, pipeline._resolve_control.return_value
    )
    assert put_output.call_args.args[1]["index"] == 0
    assert put_output.call_args.args[1]["frames"][0].size == (8, 8)
    assert runtime.current_chunk_index == 1
    assert runtime.emitted_frames == 1
    assert state.streaming_session is streaming_session


def test_chunk_hud_and_metadata_use_the_control_snapshot_submitted_to_the_model() -> None:
    pipeline = MagicMock()
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(
            prompt="test",
            image=Image.new("RGB", (8, 8)),
            chunk_size=1,
            frame_num=1,
            show_control_hud=True,
        ),
        latent_f=1,
        chunk_size=1,
        width=8,
        height=8,
        cache_handle=7,
    )
    pipeline._create_initialized_session.return_value = runtime
    pipeline._resolve_control.return_value = torch.zeros(1)
    streaming_runtime = MagicMock()
    streaming_session = SimpleNamespace(session_id="actor-session")
    streaming_runtime.create_session.return_value = streaming_session
    streaming_runtime.error.return_value = None
    streaming_runtime.try_submit_chunk.return_value = True
    frames = [Image.new("RGB", (8, 8))]
    streaming_runtime.poll_frames.return_value = [(0, frames)]
    pipeline._get_streaming_runtime.return_value = streaming_runtime

    service = LingBotWorldFastService(pipeline)
    state = LingBotWorldFastSessionState(config=runtime.config, control_context=SimpleNamespace())
    emit_status = MagicMock()
    with (
        patch.object(service, "_next_realtime_control", return_value=(object(), ["w", "j"])),
        patch.object(service, "_overlay_control_hud", return_value=frames) as overlay,
        patch.object(service, "_put_output") as put_output,
    ):
        service._run_actor_worker_loop(state, state.control_context, MagicMock(), emit_status)

    overlay.assert_called_once_with(frames, ["w", "j"])
    assert put_output.call_args.args[1]["applied_controls"] == ["w", "j"]
    chunk_sent = [call for call in emit_status.call_args_list if call.args[0] == "chunk_sent"]
    assert chunk_sent[0].kwargs["controls"] == ["w", "j"]


def test_actor_worker_does_not_submit_after_close_during_initialization() -> None:
    pipeline = MagicMock()
    runtime = LingBotWorldFastGenerationSession(config=_state().config, cache_handle=7)
    service = LingBotWorldFastService(pipeline)
    state = _state()

    def initialize(*_args: object, **_kwargs: object) -> LingBotWorldFastGenerationSession:
        state.active = False
        return runtime

    pipeline._create_initialized_session.side_effect = initialize
    with patch.object(service, "_next_realtime_control", return_value=(object(), None)):
        service._run_actor_worker_loop(state, MagicMock(), MagicMock(), MagicMock())

    pipeline._get_streaming_runtime.assert_not_called()
    assert state.generation_session is runtime


def test_actor_worker_prefetches_directional_chunks_within_ingress_capacity() -> None:
    pipeline = MagicMock()
    runtime = LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8)), chunk_size=1),
        latent_f=2,
        chunk_size=1,
        cache_handle=7,
    )
    pipeline._create_initialized_session.return_value = runtime
    pipeline._resolve_control.return_value = torch.zeros(1)
    streaming_runtime = MagicMock()
    streaming_session = SimpleNamespace(session_id="actor-session")
    streaming_runtime.create_session.return_value = streaming_session
    streaming_runtime.error.return_value = None
    streaming_runtime.try_submit_chunk.return_value = True
    streaming_runtime.can_submit_chunk.return_value = True
    streaming_runtime.poll_frames.return_value = []

    service = LingBotWorldFastService(pipeline)
    state = LingBotWorldFastSessionState(config=runtime.config, control_context=SimpleNamespace())

    def stop_wait(*_args: object, **_kwargs: object) -> bool:
        state.active = False
        return True

    streaming_runtime.wait_until_idle.side_effect = stop_wait
    pipeline._get_streaming_runtime.return_value = streaming_runtime

    with patch.object(service, "_next_realtime_control", return_value=(object(), ["w"])):
        service._run_actor_worker_loop(state, state.control_context, MagicMock(), MagicMock())

    assert streaming_runtime.try_submit_chunk.call_args_list == [
        ((streaming_session, 0, pipeline._resolve_control.return_value), {}),
        ((streaming_session, 1, pipeline._resolve_control.return_value), {}),
    ]


def test_direction_action_updates_state_and_wakes_worker() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state

    service.push_chunk(
        "session-a",
        {"type": "control", "direction": "up", "event": "press"},
    )

    assert state.pressed_controls == {"w"}
    assert state.pending_direction_command is not None
    assert (state.pending_direction_command.revision, state.pending_direction_command.controls) == (1, frozenset({"w"}))
    assert state.pending_inputs.get_nowait() == {"type": "direction_control"}


def test_release_stops_control_without_scheduling_stationary_generation() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state
    state.pressed_controls.add("w")

    service.push_chunk("session-a", {"type": "control", "key": "ArrowUp", "event": "release"})

    assert state.pressed_controls == set()
    assert state.pending_direction_command is None
    assert state.pending_inputs.get_nowait() == {"type": "direction_control"}


def test_held_direction_does_not_synthesize_another_chunk_without_an_event() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.pressed_controls.add("j")
    control_builder = MagicMock()

    item = service._next_realtime_control(
        state,
        SimpleNamespace(control_type="cam", chunk_size=3),
        control_builder,
        chunk_index=1,
        emit_status=MagicMock(),
        block=False,
    )

    assert item is None
    control_builder.defer.assert_not_called()


def test_new_short_press_overwrites_stale_direction_snapshot() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.control_context = SimpleNamespace(control_type="cam", chunk_size=3)
    service._sessions["session-a"] = state

    for direction in ("left", "left", "left", "right"):
        service.push_chunk("session-a", {"type": "control", "direction": direction, "event": "press"})
        service.push_chunk("session-a", {"type": "control", "direction": direction, "event": "release"})

    assert state.pending_direction_command is not None
    assert state.pending_direction_command.controls == frozenset({"d"})
    assert state.overwritten_direction_commands == 3


def test_reset_clears_held_and_pending_direction_state() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state
    service.push_chunk("session-a", {"type": "control", "direction": "left", "event": "press"})

    service.push_chunk("session-a", {"type": "control", "direction": "up", "event": "reset"})

    assert state.pressed_controls == set()
    assert state.pending_direction_command is None
    assert state.control_initialized is False


def test_reset_controls_retains_pose_but_reset_pose_restores_identity() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.control_c2w[0][3] = 2.0
    state.control_pitch = 0.5
    state.control_initialized = True
    service._sessions["session-a"] = state

    service.push_chunk("session-a", {"type": "control", "control": "w", "event": "reset"})

    assert state.control_c2w[0][3] == 2.0
    assert state.control_pitch == 0.5
    assert state.control_initialized is True

    service.push_chunk("session-a", {"type": "control", "control": "w", "event": "reset_pose"})

    assert state.control_c2w == np.eye(4).tolist()
    assert state.control_pitch == 0.0
    assert state.control_initialized is False


def test_unsupported_direction_event_does_not_mutate_control_state() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state

    service.push_chunk("session-a", {"type": "control", "direction": "left", "event": "tap"})

    assert state.pressed_controls == set()
    assert state.pending_direction_command is None


def test_latest_short_press_is_applied_without_sticky_rotation() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.control_context = SimpleNamespace(control_type="cam", chunk_size=3)
    service._sessions["session-a"] = state
    control_builder = MagicMock()

    for direction in ("left", "backward"):
        service.push_chunk("session-a", {"type": "control", "direction": direction, "event": "press"})
        service.push_chunk("session-a", {"type": "control", "direction": direction, "event": "release"})

    first = service._next_realtime_control(state, state.control_context, control_builder, 0, MagicMock(), block=True)

    assert first is not None
    assert [call.args[0]["controls"] for call in control_builder.defer.call_args_list] == [["s"]]


def test_control_state_supports_combined_translation_and_rotation() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.control_context = SimpleNamespace(control_type="cam", chunk_size=3)
    service._sessions["session-a"] = state
    control_builder = MagicMock()

    service.push_chunk("session-a", {"type": "control_state", "controls": ["w"]})
    service.push_chunk("session-a", {"type": "control_state", "controls": ["w", "j"]})
    service._next_realtime_control(state, state.control_context, control_builder, 0, MagicMock(), block=True)

    assert state.pressed_controls == {"w", "j"}
    assert [call.args[0]["controls"] for call in control_builder.defer.call_args_list] == [["j", "w"]]


def test_held_direction_continues_only_after_a_chunk_is_completed() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.pressed_controls.add("j")
    control_builder = MagicMock()

    assert (
        service._next_realtime_control(
            state,
            SimpleNamespace(control_type="cam", chunk_size=3),
            control_builder,
            chunk_index=1,
            emit_status=MagicMock(),
            block=False,
        )
        is None
    )

    item = service._next_realtime_control(
        state,
        SimpleNamespace(control_type="cam", chunk_size=3),
        control_builder,
        chunk_index=1,
        emit_status=MagicMock(),
        block=True,
    )

    assert item is not None


def test_explicit_controls_keep_only_the_latest_pending_value() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state

    first = {"control_tensor": "first"}
    second = {"control_tensor": "second"}
    service.push_chunk("session-a", first)
    service.push_chunk("session-a", second)

    assert state.latest_explicit_control == second
    assert state.overwritten_explicit_controls == 1
    assert state.pending_inputs.qsize() == 1


def test_directional_chunks_match_source_video_rate_integration_and_boundary() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.pressed_controls.add("w")
    context = SimpleNamespace(control_type="cam", chunk_size=3)

    first = service._build_directional_control_chunk(state, context, {"w"})
    second = service._build_directional_control_chunk(state, context, {"w"})

    assert first is not None
    assert second is not None
    assert first["translation_scale"] == 3.0
    assert second["translation_scale"] == 3.0
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
    yaw_chunk = service._build_directional_control_chunk(state, context, {"j"})

    assert yaw_chunk is not None
    expected_yaw = np.deg2rad(-16.0)
    np.testing.assert_allclose(np.asarray(yaw_chunk["poses"])[-1, 0, 0], np.cos(expected_yaw), atol=1e-6)
    assert service._direction_from_chunk({"key": "ArrowLeft"}) == "a"
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
        patch.object(service, "_put_output") as put_output,
    ):
        service._emit_preview_frame(state)

    overlay.assert_called_once()
    assert overlay.call_args.kwargs == {"controls": None}
    assert "frames_b64" not in put_output.call_args.args[1]
    assert put_output.call_args.args[1]["frames"][0].size == (832, 480)


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


def test_load_image_accepts_browser_data_url() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 6), (10, 20, 30, 255)).save(buffer, format="PNG")
    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")

    image = LingBotWorldFastService._load_image({"image": f"data:image/png;base64,{image_data}"})

    assert image.mode == "RGB"
    assert image.size == (8, 6)
    assert image.getpixel((0, 0)) == (10, 20, 30)


def test_load_image_rejects_invalid_browser_data_url() -> None:
    with pytest.raises(ValueError, match="invalid base64"):
        LingBotWorldFastService._load_image({"image": "data:image/png;base64,not-valid"})


def test_create_session_initializes_fixed_intrinsics_from_intrinsics_path() -> None:
    pipeline = MagicMock()
    service = LingBotWorldFastService(pipeline)
    intrinsics = np.asarray([[8.0, 8.0, 4.0, 4.0], [9.0, 9.0, 4.0, 4.0]])

    with patch("telefuser.pipelines.lingbot_world_fast.service.np.load", return_value=intrinsics) as load:
        session_id = service.create_session(
            {
                "image": Image.new("RGB", (8, 8)),
                "intrinsics_path": "/controls/intrinsics.npy",
                "intrinsics_width": 8,
                "intrinsics_height": 8,
            }
        )

    load.assert_called_once_with(Path("/controls/intrinsics.npy"))
    session_config = pipeline.control_context.call_args.args[0]
    assert session_config.intrinsics is intrinsics
    assert session_config.intrinsics_width == 8
    assert session_config.intrinsics_height == 8
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


def test_output_queue_discards_stale_video_and_records_runtime_metrics() -> None:
    state = _state()
    state.output_queue = asyncio.Queue(maxsize=2)

    LingBotWorldFastService._enqueue_output(state, {"type": "chunk", "index": 0})
    LingBotWorldFastService._enqueue_output(state, {"type": "status", "stage": "generating_chunk"})
    LingBotWorldFastService._enqueue_output(state, {"type": "chunk", "index": 1})

    assert list(state.output_queue._queue) == [
        {"type": "status", "stage": "generating_chunk"},
        {"type": "chunk", "index": 1},
    ]
    assert LingBotWorldFastService._runtime_metrics(state)["dropped_video_payloads"] == 1
    assert LingBotWorldFastService._runtime_metrics(state)["output_queue_high_watermark"] == 2


def test_stream_progress_reports_duration_frames_and_chunks() -> None:
    service = LingBotWorldFastService(MagicMock(), max_generation_seconds=20.0)
    state = _state()
    runtime = LingBotWorldFastGenerationSession(config=state.config, emitted_frames=8, current_chunk_index=1)

    progress = service._stream_progress(state, runtime)

    assert progress == {
        "service_max_duration_seconds": 20.0,
        "target_duration_seconds": 0.5,
        "generated_duration_seconds": 0.5,
        "target_frames": 9,
        "generated_frames": 8,
        "fps": 16,
        "total_chunks": 1,
        "completed_chunks": 1,
    }


def test_close_session_waits_for_worker_to_release_generation_state() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    state.generation_session = MagicMock()
    state.worker_thread = MagicMock()
    state.worker_thread.is_alive.return_value = True
    service._sessions["session-a"] = state

    service.close_session("session-a")

    assert "session-a" in service._sessions
    service.pipeline.release_session.assert_not_called()
    assert state.pending_inputs.get_nowait() == {"type": "stop"}


def test_close_session_uses_a_bounded_worker_join_timeout() -> None:
    service = LingBotWorldFastService(MagicMock(), close_timeout=1.5)
    state = _state()
    state.worker_thread = MagicMock()
    state.worker_thread.is_alive.return_value = True
    service._sessions["session-a"] = state

    service.close_session("session-a")

    state.worker_thread.join.assert_called_once_with(timeout=1.5)
    assert "session-a" in service._sessions


def test_release_generation_session_closes_actor_session_once() -> None:
    pipeline = MagicMock()
    streaming_runtime = MagicMock()
    pipeline._get_streaming_runtime.return_value = streaming_runtime
    service = LingBotWorldFastService(pipeline)
    state = _state()
    state.generation_session = MagicMock()
    state.streaming_session = SimpleNamespace(session_id="actor-session")

    service._release_generation_session(state)
    service._release_generation_session(state)

    streaming_runtime.close_session.assert_called_once()
    pipeline.release_session.assert_not_called()
    assert state.streaming_session is None
    assert state.generation_session is None


def test_release_generation_session_retains_handles_when_drain_fails() -> None:
    pipeline = MagicMock()
    pipeline._get_streaming_runtime.return_value.close_session.side_effect = TimeoutError("drain failed")
    service = LingBotWorldFastService(pipeline)
    state = _state()
    generation_session = MagicMock()
    streaming_session = SimpleNamespace(session_id="actor-session")
    state.generation_session = generation_session
    state.streaming_session = streaming_session

    with pytest.raises(TimeoutError, match="drain failed"):
        service._release_generation_session(state)

    assert state.streaming_session is streaming_session
    assert state.generation_session is generation_session


def test_worker_emits_terminal_messages_when_cleanup_fails() -> None:
    service = LingBotWorldFastService(MagicMock())
    state = _state()
    service._sessions["session-a"] = state
    emitted: list[dict] = []

    with (
        patch.object(service, "_emit_preview_frame"),
        patch.object(service, "_run_actor_worker_loop"),
        patch.object(service, "_release_generation_session", side_effect=RuntimeError("release failed")),
        patch.object(service, "_put_output", side_effect=lambda _, payload: emitted.append(payload)),
    ):
        service._run_worker_loop("session-a", state, MagicMock())

    assert [payload["type"] for payload in emitted] == ["error", "status", "done"]
    assert emitted[0]["stage"] == "cleanup_failed"
    assert "session-a" not in service._sessions


def test_service_start_warms_the_pipeline_with_its_default_shape() -> None:
    pipeline = MagicMock()
    pipeline.config = SimpleNamespace(control_type="cam", orig_width=832, orig_height=480)
    service = LingBotWorldFastService(pipeline, default_session_config={"chunk_size": 4})

    service.start()

    warmup_config = pipeline.warmup.call_args.args[0]
    assert warmup_config.image.size == (832, 480)
    assert warmup_config.chunk_size == 4
    assert warmup_config.frame_num == 29
