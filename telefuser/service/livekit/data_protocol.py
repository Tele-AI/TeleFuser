"""LiveKit data topic and message validation."""

from __future__ import annotations

import json
from typing import Any

TF_CONTROL_TOPIC = "tf.control"
TF_STATUS_TOPIC = "tf.status"
TF_METRICS_TOPIC = "tf.metrics"
TF_ASSET_TOPIC = "tf.asset"

KNOWN_CONTROL_TYPES = frozenset({"control_state", "control", "prompt", "reset", "stop"})
KNOWN_CONTROLS = frozenset(
    {
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "KeyW",
        "KeyA",
        "KeyS",
        "KeyD",
        "KeyI",
        "KeyJ",
        "KeyK",
        "KeyL",
        "w",
        "a",
        "s",
        "d",
        "i",
        "j",
        "k",
        "l",
        "up",
        "down",
        "left",
        "right",
        "forward",
        "backward",
    }
)
KNOWN_CONTROL_EVENTS = frozenset({"press", "release", "keyup", "end", "reset", "reset_pose"})
MEDIA_KEYS = frozenset({"frames", "frames_b64", "audio_b64", "audio_sample_rate", "audio_channels"})


class DataProtocolError(ValueError):
    """Raised when a LiveKit data message violates the TeleFuser protocol."""


def normalize_control_message(
    message: bytes | str | dict[str, Any],
    *,
    topic: str,
    session_id: str,
    sender_identity: str,
    controller_identity: str,
    max_bytes: int = 12 * 1024,
) -> dict[str, Any]:
    """Validate and normalize one client-to-worker control message."""
    if topic != TF_CONTROL_TOPIC:
        raise DataProtocolError(f"Unsupported data topic: {topic}")
    if sender_identity and sender_identity != controller_identity:
        raise DataProtocolError("Only the controller may send control messages")

    decoded = _decode_json_message(message, max_bytes=max_bytes)
    if "version" in decoded:
        return _normalize_enveloped_message(decoded, session_id=session_id)
    return _normalize_legacy_message(decoded)


def strip_media_fields(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return chunk metadata without video/audio payload fields."""
    top_level_media_keys = MEDIA_KEYS if isinstance(chunk.get("frames"), (list, tuple)) else MEDIA_KEYS - {"frames"}
    metadata = {key: value for key, value in chunk.items() if key not in top_level_media_keys}
    data = metadata.get("data")
    if isinstance(data, dict):
        nested_media_keys = MEDIA_KEYS if isinstance(data.get("frames"), (list, tuple)) else MEDIA_KEYS - {"frames"}
        nested = {key: value for key, value in data.items() if key not in nested_media_keys}
        if nested:
            metadata["data"] = nested
        else:
            metadata.pop("data", None)
    return metadata


def _decode_json_message(message: bytes | str | dict[str, Any], *, max_bytes: int) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if isinstance(message, bytes):
        raw_size = len(message)
        raw = message.decode("utf-8")
    else:
        raw = message
        raw_size = len(raw.encode("utf-8"))

    if raw_size > max_bytes:
        raise DataProtocolError(f"Data message exceeds {max_bytes} bytes")

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataProtocolError("Data message must be JSON") from exc
    if not isinstance(decoded, dict):
        raise DataProtocolError("Data message must decode to a JSON object")
    return decoded


def _normalize_enveloped_message(message: dict[str, Any], *, session_id: str) -> dict[str, Any]:
    if message.get("version") != 1:
        raise DataProtocolError("Unsupported data protocol version")
    msg_session_id = message.get("session_id")
    if msg_session_id is not None and msg_session_id != session_id:
        raise DataProtocolError("Control message session_id does not match active session")

    msg_type = message.get("type")
    if msg_type not in KNOWN_CONTROL_TYPES:
        raise DataProtocolError(f"Unsupported control message type: {msg_type}")

    payload = message.get("payload") or {}
    if not isinstance(payload, dict):
        raise DataProtocolError("Control message payload must be an object")
    normalized = {"type": msg_type}
    normalized.update(payload)
    return _normalize_legacy_message(normalized)


def _normalize_legacy_message(message: dict[str, Any]) -> dict[str, Any]:
    msg_type = message.get("type")
    if msg_type not in KNOWN_CONTROL_TYPES:
        raise DataProtocolError(f"Unsupported control message type: {msg_type}")

    if msg_type == "control_state":
        controls = message.get("controls")
        if not isinstance(controls, list) or not all(isinstance(control, str) for control in controls):
            raise DataProtocolError("control_state controls must be a list of strings")
        if any(control not in KNOWN_CONTROLS for control in controls):
            raise DataProtocolError("control_state contains an unsupported control")
        if len(controls) != len(set(controls)):
            raise DataProtocolError("control_state controls must not contain duplicates")
    elif msg_type == "control":
        control = message.get("control", message.get("key"))
        event = str(message.get("event") or message.get("action") or "press").lower()
        if control not in KNOWN_CONTROLS:
            raise DataProtocolError(f"Unsupported control: {control}")
        if event not in KNOWN_CONTROL_EVENTS:
            raise DataProtocolError(f"Unsupported control event: {event}")
    return dict(message)
