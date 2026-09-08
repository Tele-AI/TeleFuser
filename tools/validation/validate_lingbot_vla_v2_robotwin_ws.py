"""Validate a running LingBot-VLA v2 RoboTwin WebSocket endpoint without simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

PROTOCOL_VERSION = "1.0"
ACTION_TYPE = "absolute_qpos"
ACTION_DTYPE = "float32"
DEFAULT_MAX_IMAGE_EDGE = 640
ACTION_ORDER = (
    "left_arm_joint_0",
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "left_gripper",
    "right_arm_joint_0",
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_gripper",
)
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
_ACTION_TIMING_FIELDS = (
    "decode_ms",
    "infer_ms",
    "lock_wait_ms",
    "pipeline_ms",
    "action_mapping_ms",
    "adapter_total_ms",
)


class ValidationFailure(RuntimeError):
    """Raised when the endpoint violates the RoboTwin action contract."""


def parse_state_json(value: str) -> list[float]:
    """Parse a finite 14-dimensional RoboTwin state."""
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("state must be valid JSON") from error
    if not isinstance(raw, list) or len(raw) != 14:
        raise argparse.ArgumentTypeError("state must be a JSON array containing exactly 14 values")
    state: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(float(item)):
            raise argparse.ArgumentTypeError("state values must be finite numbers")
        state.append(float(item))
    return state


def validate_metadata(metadata: Any) -> dict[str, Any]:
    """Validate and summarize the server's advertised action contract."""
    if not isinstance(metadata, dict):
        raise ValidationFailure("metadata frame must be a MessagePack object")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "robot_profile": "robotwin",
        "action_type": ACTION_TYPE,
        "action_dim": len(ACTION_ORDER),
        "action_dtype": ACTION_DTYPE,
        "action_order": list(ACTION_ORDER),
    }
    for field, expected_value in expected.items():
        if metadata.get(field) != expected_value:
            raise ValidationFailure(
                f"metadata {field} mismatch: expected {expected_value!r}, observed {metadata.get(field)!r}"
            )
    horizon = metadata.get("action_horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValidationFailure("metadata action_horizon must be a positive integer")
    policy_verified = metadata.get("policy_verified")
    verification_status = metadata.get("verification_status")
    if not isinstance(policy_verified, bool):
        raise ValidationFailure("metadata policy_verified must be boolean")
    if not isinstance(verification_status, str) or not verification_status:
        raise ValidationFailure("metadata verification_status must be a non-empty string")
    max_request_bytes = metadata.get("max_request_bytes")
    if isinstance(max_request_bytes, bool) or not isinstance(max_request_bytes, int) or max_request_bytes < 1:
        raise ValidationFailure("metadata max_request_bytes must be a positive integer")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "action_shape": [horizon, len(ACTION_ORDER)],
        "action_type": ACTION_TYPE,
        "action_dtype": ACTION_DTYPE,
        "policy_verified": policy_verified,
        "verification_status": verification_status,
        "max_request_bytes": max_request_bytes,
    }


def _validate_trace_fields(response: Mapping[str, Any], *, request_id: str, episode_id: str) -> None:
    if response.get("request_id") != request_id:
        raise ValidationFailure("response did not echo the request_id")
    if response.get("episode_id") != episode_id:
        raise ValidationFailure("response did not echo the episode_id")


def _validate_timings(response: Mapping[str, Any], required_fields: tuple[str, ...]) -> dict[str, float]:
    timings = response.get("server_timing")
    if not isinstance(timings, dict):
        raise ValidationFailure("response server_timing must be an object")
    validated: dict[str, float] = {}
    for field in required_fields:
        value = timings.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValidationFailure(f"response server_timing.{field} must be a finite number")
        if float(value) < 0:
            raise ValidationFailure(f"response server_timing.{field} must be non-negative")
        validated[field] = float(value)
    if "prev_total_ms" in timings:
        value = timings["prev_total_ms"]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValidationFailure("response server_timing.prev_total_ms must be a finite number")
        if float(value) < 0:
            raise ValidationFailure("response server_timing.prev_total_ms must be non-negative")
        validated["prev_total_ms"] = float(value)
    return validated


def validate_reset_response(response: Any, *, request_id: str, episode_id: str) -> dict[str, Any]:
    """Validate one reset acknowledgement."""
    if not isinstance(response, dict):
        raise ValidationFailure("reset response must be a MessagePack object")
    if response.get("action", object()) is not None:
        raise ValidationFailure("reset response action must be null")
    _validate_trace_fields(response, request_id=request_id, episode_id=episode_id)
    return {"server_timing_ms": _validate_timings(response, ("decode_ms", "infer_ms"))}


def validate_action_response(
    response: Any,
    *,
    expected_horizon: int,
    request_id: str,
    episode_id: str,
    seed: int,
) -> dict[str, Any]:
    """Validate and summarize one raw RoboTwin action response."""
    if not isinstance(response, dict):
        raise ValidationFailure("action response must be a MessagePack object")
    _validate_trace_fields(response, request_id=request_id, episode_id=episode_id)
    if response.get("seed") != seed:
        raise ValidationFailure("response did not echo the inference seed")
    actions = response.get("action")
    if not isinstance(actions, np.ndarray):
        raise ValidationFailure("response action must be a NumPy array")
    if actions.shape != (expected_horizon, len(ACTION_ORDER)):
        raise ValidationFailure(
            f"response action shape must be {(expected_horizon, len(ACTION_ORDER))}, got {actions.shape}"
        )
    if actions.dtype != np.dtype(np.float32):
        raise ValidationFailure(f"response action dtype must be float32, got {actions.dtype}")
    if not np.isfinite(actions).all():
        raise ValidationFailure("response action contains non-finite values")
    policy_verified = response.get("policy_verified")
    verification_status = response.get("verification_status")
    if not isinstance(policy_verified, bool):
        raise ValidationFailure("response policy_verified must be boolean")
    if not isinstance(verification_status, str) or not verification_status:
        raise ValidationFailure("response verification_status must be a non-empty string")

    contiguous = np.ascontiguousarray(actions, dtype="<f4")
    flat = contiguous.reshape(-1)
    return {
        "shape": list(actions.shape),
        "dtype": str(actions.dtype),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "mean": float(statistics.fmean(flat)),
        "l2_norm": float(np.linalg.norm(flat.astype(np.float64))),
        "sha256_float32_le": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "policy_verified": policy_verified,
        "verification_status": verification_status,
        "server_timing_ms": _validate_timings(response, _ACTION_TIMING_FIELDS),
    }


def require_exact_replay(records: list[dict[str, Any]]) -> None:
    """Require every recorded action digest to match the first response."""
    if len(records) < 2:
        raise ValidationFailure("exact replay validation requires at least two requests")
    digests = {record["action"]["sha256_float32_le"] for record in records}
    if len(digests) != 1:
        raise ValidationFailure(f"fixed-seed action replay diverged across {len(digests)} digests")


def _receive_binary(connection: Any, *, timeout_seconds: float) -> bytes:
    payload = connection.recv(timeout=timeout_seconds)
    if not isinstance(payload, bytes):
        raise ValidationFailure(f"server returned a text error frame: {payload}")
    return payload


def load_validation_image(image_path: Path, *, max_image_edge: int) -> tuple[np.ndarray, list[int]]:
    """Load an RGB image and bound its encoded request size while preserving aspect ratio."""
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        source_shape = [image.height, image.width, 3]
        if max(image.size) > max_image_edge:
            image.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8), source_shape


def run_validation(
    *,
    host: str,
    port: int,
    image_path: Path,
    task: str,
    state: list[float],
    seed: int,
    request_count: int,
    timeout_seconds: float,
    exact_replay: bool,
    max_image_edge: int = DEFAULT_MAX_IMAGE_EDGE,
) -> dict[str, Any]:
    """Connect to a resident policy and exercise reset plus fixed-seed inference."""
    from websockets.sync.client import connect

    from examples.lingbot_vla_v2.lingbot_vla_v2_robotwin_server import pack_message, unpack_message

    if request_count < 1:
        raise ValueError("request_count must be positive")
    if exact_replay and request_count < 2:
        raise ValueError("exact replay validation requires request_count >= 2")
    image, source_image_shape = load_validation_image(image_path, max_image_edge=max_image_edge)

    episode_id = f"validator-{time.time_ns()}"
    uri = f"ws://{host}:{port}/"
    records: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    with connect(
        uri,
        open_timeout=timeout_seconds,
        close_timeout=timeout_seconds,
        max_size=None,
        compression=None,
    ) as connection:
        metadata = unpack_message(_receive_binary(connection, timeout_seconds=timeout_seconds))
        metadata_summary = validate_metadata(metadata)
        expected_horizon = metadata["action_horizon"]

        reset_request_id = "validator-reset"
        connection.send(
            pack_message(
                {
                    "reset": True,
                    "robo_name": "robotwin",
                    "request_id": reset_request_id,
                    "episode_id": episode_id,
                }
            )
        )
        reset = unpack_message(_receive_binary(connection, timeout_seconds=timeout_seconds))
        reset_summary = validate_reset_response(reset, request_id=reset_request_id, episode_id=episode_id)

        base_request = {
            **{key: image for key in CAMERA_KEYS},
            "observation.state": np.asarray(state, dtype=np.float32),
            "task": task,
            "episode_id": episode_id,
            "seed": seed,
        }
        request_payload_bytes = len(pack_message({**base_request, "request_id": "validator-size-check"}))
        if request_payload_bytes > metadata["max_request_bytes"]:
            raise ValidationFailure(
                f"encoded request uses {request_payload_bytes} bytes, exceeding the server limit "
                f"of {metadata['max_request_bytes']} bytes"
            )
        for index in range(request_count):
            request_id = f"validator-{index}"
            request_started_at = time.perf_counter()
            connection.send(pack_message({**base_request, "request_id": request_id}))
            response = unpack_message(_receive_binary(connection, timeout_seconds=timeout_seconds))
            round_trip_ms = (time.perf_counter() - request_started_at) * 1000.0
            action_summary = validate_action_response(
                response,
                expected_horizon=expected_horizon,
                request_id=request_id,
                episode_id=episode_id,
                seed=seed,
            )
            records.append({"request_id": request_id, "round_trip_ms": round_trip_ms, "action": action_summary})

    if exact_replay:
        require_exact_replay(records)
    return {
        "passed": True,
        "endpoint": uri,
        "elapsed_seconds": time.perf_counter() - started_at,
        "configuration": {
            "image": str(image_path),
            "task": task,
            "seed": seed,
            "request_count": request_count,
            "exact_replay": exact_replay,
            "max_image_edge": max_image_edge,
            "source_image_shape": source_image_shape,
            "transmitted_image_shape": list(image.shape),
            "request_payload_bytes": request_payload_bytes,
        },
        "metadata": metadata,
        "metadata_summary": metadata_summary,
        "reset": reset_summary,
        "records": records,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9330)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("examples/data/lingbot_world_fast/image.jpg"),
    )
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--state-json", type=parse_state_json, default=[0.0] * 14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--requests", type=_positive_int, default=2)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--max-image-edge", type=_positive_int, default=DEFAULT_MAX_IMAGE_EDGE)
    parser.add_argument("--require-exact-replay", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Run the no-simulation RoboTwin WebSocket validation."""
    args = build_parser().parse_args()
    report = run_validation(
        host=args.host,
        port=args.port,
        image_path=args.image,
        task=args.task,
        state=args.state_json,
        seed=args.seed,
        request_count=args.requests,
        timeout_seconds=args.timeout_seconds,
        exact_replay=args.require_exact_replay,
        max_image_edge=args.max_image_edge,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
