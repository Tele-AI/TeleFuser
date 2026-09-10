"""Versioned JSON-safe serialization for public VLA contracts."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .contracts import ActionSpaceSpec, RobotActionChunk, RobotObservation, RobotState

VLA_WIRE_SCHEMA_VERSION = 1
VLA_WIRE_ENCODING = "json-base64-v1"
DEFAULT_MAX_TENSOR_BYTES = 64 * 1024 * 1024

_DTYPES: dict[str, torch.dtype] = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
_DTYPE_NAMES = {dtype: name for name, dtype in _DTYPES.items()}


def action_space_to_wire(spec: ActionSpaceSpec) -> dict[str, Any]:
    """Serialize an action-space contract without losing field meaning."""
    return {
        "schema": "telefuser.vla.action_space",
        "schema_version": VLA_WIRE_SCHEMA_VERSION,
        "representation": spec.representation,
        "dimension_names": list(spec.dimension_names),
        "units": list(spec.units),
        "frame": spec.frame,
        "control_hz": spec.control_hz,
        "normalized": spec.normalized,
        "normalization_profile": spec.normalization_profile,
    }


def action_space_from_wire(payload: Mapping[str, Any]) -> ActionSpaceSpec:
    """Deserialize and validate an action-space contract."""
    _require_schema(payload, "telefuser.vla.action_space")
    dimension_names_value = payload.get("dimension_names")
    units_value = payload.get("units")
    if not isinstance(dimension_names_value, list) or not isinstance(units_value, list):
        raise ValueError("VLA action-space dimension_names and units must be arrays")
    try:
        dimension_names = tuple(dimension_names_value)
        units = tuple(units_value)
        return ActionSpaceSpec(
            representation=payload["representation"],
            dimension_names=dimension_names,
            units=units,
            frame=payload.get("frame"),
            control_hz=payload.get("control_hz"),
            normalized=payload["normalized"],
            normalization_profile=payload.get("normalization_profile"),
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid VLA action-space payload") from error


def tensor_to_wire(value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    """Serialize a dense CPU copy of a tensor using explicit raw bytes."""
    tensor = torch.from_numpy(np.asarray(value).copy()) if isinstance(value, np.ndarray) else value
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("VLA tensor payload must be a torch.Tensor or numpy.ndarray")
    dtype_name = _DTYPE_NAMES.get(tensor.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported VLA tensor dtype: {tensor.dtype}")
    contiguous = tensor.detach().to(device="cpu").contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    return {
        "schema": "telefuser.vla.tensor",
        "schema_version": VLA_WIRE_SCHEMA_VERSION,
        "dtype": dtype_name,
        "shape": list(contiguous.shape),
        "data": base64.b64encode(raw).decode("ascii"),
    }


def tensor_from_wire(
    payload: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> torch.Tensor:
    """Deserialize a bounded tensor payload into independent CPU storage."""
    _require_schema(payload, "telefuser.vla.tensor")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    dtype_name = payload.get("dtype")
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported VLA tensor dtype: {dtype_name!r}")
    shape_value = payload.get("shape")
    if not isinstance(shape_value, list) or any(
        not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape_value
    ):
        raise ValueError("VLA tensor shape must contain non-negative integers")
    encoded = payload.get("data")
    if not isinstance(encoded, str):
        raise ValueError("VLA tensor data must be a Base64 string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("VLA tensor data is not valid Base64") from error
    if len(raw) > max_bytes:
        raise ValueError(f"VLA tensor exceeds the {max_bytes}-byte limit")
    dtype = _DTYPES[dtype_name]
    element_size = torch.empty((), dtype=dtype).element_size()
    element_count = math.prod(shape_value)
    if len(raw) != element_count * element_size:
        raise ValueError("VLA tensor byte length does not match dtype and shape")
    if element_count == 0:
        return torch.empty(tuple(shape_value), dtype=dtype)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(shape_value)).clone()


def robot_observation_to_wire(observation: RobotObservation) -> dict[str, Any]:
    """Serialize one robot observation and its named camera tensors."""
    if any(not isinstance(name, str) or not name for name in observation.images):
        raise ValueError("robot observation image names must be non-empty strings")
    images = {name: tensor_to_wire(_as_tensor(image)) for name, image in observation.images.items()}
    payload = {
        "schema": "telefuser.vla.robot_observation",
        "schema_version": VLA_WIRE_SCHEMA_VERSION,
        "state": {
            "values": tensor_to_wire(observation.state.values),
            "dimension_names": list(observation.state.dimension_names),
            "timestamp_ns": observation.state.timestamp_ns,
        },
        "images": images,
        "metadata": dict(observation.metadata),
    }
    _require_json_value(payload["metadata"], "observation metadata")
    return payload


def robot_observation_from_wire(
    payload: Mapping[str, Any],
    *,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> RobotObservation:
    """Deserialize one bounded robot observation."""
    _require_schema(payload, "telefuser.vla.robot_observation")
    state_payload = payload.get("state")
    images_payload = payload.get("images")
    if not isinstance(state_payload, Mapping) or not isinstance(images_payload, Mapping):
        raise ValueError("robot observation requires state and images objects")
    try:
        dimension_names = state_payload.get("dimension_names")
        if not isinstance(dimension_names, list):
            raise ValueError("robot state dimension_names must be an array")
        if any(not isinstance(name, str) or not name for name in images_payload):
            raise ValueError("robot observation image names must be non-empty strings")
        state = RobotState(
            values=tensor_from_wire(state_payload["values"], max_bytes=max_tensor_bytes),
            dimension_names=tuple(dimension_names),
            timestamp_ns=state_payload["timestamp_ns"],
        )
        images = {name: tensor_from_wire(image, max_bytes=max_tensor_bytes) for name, image in images_payload.items()}
        metadata = payload.get("metadata", {})
        _require_json_value(metadata, "observation metadata")
        return RobotObservation(state=state, images=images, metadata=metadata)
    except (KeyError, TypeError) as error:
        raise ValueError("invalid robot observation payload") from error


def robot_action_chunk_to_wire(chunk: RobotActionChunk) -> dict[str, Any]:
    """Serialize a semantically described robot action chunk."""
    metadata = dict(chunk.metadata)
    _require_json_value(metadata, "action chunk metadata")
    return {
        "schema": "telefuser.vla.robot_action_chunk",
        "schema_version": VLA_WIRE_SCHEMA_VERSION,
        "actions": tensor_to_wire(chunk.actions),
        "action_space": action_space_to_wire(chunk.action_space),
        "valid_length": chunk.valid_length,
        "observation_timestamp_ns": chunk.observation_timestamp_ns,
        "sequence_id": chunk.sequence_id,
        "episode_id": chunk.episode_id,
        "metadata": metadata,
    }


def robot_action_chunk_from_wire(
    payload: Mapping[str, Any],
    *,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> RobotActionChunk:
    """Deserialize and validate a robot action chunk."""
    _require_schema(payload, "telefuser.vla.robot_action_chunk")
    try:
        metadata = payload.get("metadata", {})
        _require_json_value(metadata, "action chunk metadata")
        return RobotActionChunk(
            actions=tensor_from_wire(payload["actions"], max_bytes=max_tensor_bytes),
            action_space=action_space_from_wire(payload["action_space"]),
            valid_length=payload["valid_length"],
            observation_timestamp_ns=payload["observation_timestamp_ns"],
            sequence_id=payload["sequence_id"],
            episode_id=payload["episode_id"],
            metadata=metadata,
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid robot action chunk payload") from error


def dumps_wire_message(payload: Mapping[str, Any]) -> str:
    """Encode one deterministic compact JSON WebSocket message."""
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def loads_wire_message(payload: str, *, max_message_bytes: int) -> dict[str, Any]:
    """Decode one size-bounded JSON WebSocket message."""
    if not isinstance(payload, str):
        raise TypeError("VLA WebSocket messages must be JSON text")
    if not isinstance(max_message_bytes, int) or isinstance(max_message_bytes, bool) or max_message_bytes < 1:
        raise ValueError("max_message_bytes must be a positive integer")
    if len(payload.encode("utf-8")) > max_message_bytes:
        raise ValueError(f"VLA WebSocket message exceeds the {max_message_bytes}-byte limit")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("VLA WebSocket message is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("VLA WebSocket message must be a JSON object")
    return decoded


def _require_schema(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{expected} payload must be an object")
    if payload.get("schema") != expected:
        raise ValueError(f"expected schema {expected!r}")
    if payload.get("schema_version") != VLA_WIRE_SCHEMA_VERSION:
        raise ValueError(f"unsupported VLA schema version: {payload.get('schema_version')!r}")


def _require_json_value(value: Any, name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON values") from error


def _as_tensor(value: Any) -> torch.Tensor | np.ndarray:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value
    raise TypeError("serialized robot images must be torch.Tensor or numpy.ndarray values")
