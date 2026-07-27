from __future__ import annotations

import json

import pytest

from telefuser.service.livekit.data_protocol import (
    TF_CONTROL_TOPIC,
    DataProtocolError,
    normalize_control_message,
    strip_media_fields,
)


def test_normalize_enveloped_control_message() -> None:
    payload = {
        "version": 1,
        "type": "control",
        "session_id": "session-1",
        "payload": {"event": "press", "key": "ArrowUp"},
    }

    chunk = normalize_control_message(
        json.dumps(payload),
        topic=TF_CONTROL_TOPIC,
        session_id="session-1",
        sender_identity="controller",
        controller_identity="controller",
    )

    assert chunk == {"type": "control", "event": "press", "key": "ArrowUp"}


def test_normalize_legacy_stop_message() -> None:
    chunk = normalize_control_message(
        {"type": "stop"},
        topic=TF_CONTROL_TOPIC,
        session_id="session-1",
        sender_identity="controller",
        controller_identity="controller",
    )

    assert chunk == {"type": "stop"}


def test_rejects_viewer_control_message() -> None:
    with pytest.raises(DataProtocolError, match="controller"):
        normalize_control_message(
            {"type": "stop"},
            topic=TF_CONTROL_TOPIC,
            session_id="session-1",
            sender_identity="viewer",
            controller_identity="controller",
        )


def test_rejects_session_mismatch() -> None:
    with pytest.raises(DataProtocolError, match="session_id"):
        normalize_control_message(
            {"version": 1, "type": "stop", "session_id": "other"},
            topic=TF_CONTROL_TOPIC,
            session_id="session-1",
            sender_identity="controller",
            controller_identity="controller",
        )


def test_strip_media_fields_handles_nested_data() -> None:
    metadata = strip_media_fields(
        {
            "type": "chunk",
            "frames_b64": ["frame"],
            "audio_b64": "audio",
            "data": {"frames_b64": ["nested"], "stage": "ok"},
            "index": 1,
        }
    )

    assert metadata == {"type": "chunk", "data": {"stage": "ok"}, "index": 1}


def test_strip_media_fields_preserves_numeric_frame_counts() -> None:
    metadata = strip_media_fields(
        {
            "type": "status",
            "stage": "chunk_decoded",
            "frames": 13,
            "data": {"stage": "nested", "frames": 7},
        }
    )

    assert metadata == {
        "type": "status",
        "stage": "chunk_decoded",
        "frames": 13,
        "data": {"stage": "nested", "frames": 7},
    }


@pytest.mark.parametrize(
    "message",
    [
        {"type": "control_state", "controls": ["a", "d", "i", "j", "k", "l", "s", "w"]},
        {"type": "control", "control": "up", "event": "reset"},
        {"type": "control", "control": "up", "event": "reset_pose"},
    ],
)
def test_normalize_livekit_demo_controls(message: dict) -> None:
    chunk = normalize_control_message(
        message,
        topic=TF_CONTROL_TOPIC,
        session_id="session-1",
        sender_identity="controller",
        controller_identity="controller",
    )

    assert chunk == message


@pytest.mark.parametrize(
    "payload",
    [
        {"controls": ["w", "w"]},
        {"controls": ["unsupported"]},
    ],
)
def test_rejects_invalid_enveloped_control_state(payload: dict) -> None:
    with pytest.raises(DataProtocolError):
        normalize_control_message(
            {"version": 1, "type": "control_state", "payload": payload},
            topic=TF_CONTROL_TOPIC,
            session_id="session-1",
            sender_identity="controller",
            controller_identity="controller",
        )


def test_accepts_control_when_livekit_omits_sender_participant() -> None:
    chunk = normalize_control_message(
        {"type": "control_state", "controls": ["w"]},
        topic=TF_CONTROL_TOPIC,
        session_id="session-1",
        sender_identity="",
        controller_identity="controller",
    )

    assert chunk == {"type": "control_state", "controls": ["w"]}
