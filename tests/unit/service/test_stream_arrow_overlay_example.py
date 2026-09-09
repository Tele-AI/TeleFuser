from __future__ import annotations

from examples.stream_server.stream_arrow_overlay import ArrowOverlayService


def test_missing_video_uses_generated_frames(tmp_path) -> None:
    service = ArrowOverlayService(video_path=str(tmp_path / "missing.mp4"))

    service.start()

    assert len(service._frames) == 96
    assert service._frames[0].shape == (480, 832, 3)


def test_control_state_updates_arrow_overlay() -> None:
    service = ArrowOverlayService()
    session_id = service.create_session({})

    service.push_chunk(session_id, {"type": "control_state", "controls": ["w", "d", "j"]})

    assert service._sessions[session_id].pressed_keys == {"ArrowUp", "ArrowRight"}

    service.push_chunk(session_id, {"type": "control_state", "controls": []})

    assert service._sessions[session_id].pressed_keys == set()

    service.close_session(session_id)
    assert not service.has_session(session_id)
