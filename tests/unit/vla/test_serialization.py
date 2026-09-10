"""Tests for stable JSON/Base64 VLA wire contracts."""

from __future__ import annotations

import pytest
import torch

from telefuser.vla import (
    ActionSpaceSpec,
    RobotActionChunk,
    RobotObservation,
    RobotState,
    action_space_from_wire,
    action_space_to_wire,
    robot_action_chunk_from_wire,
    robot_action_chunk_to_wire,
    robot_observation_from_wire,
    robot_observation_to_wire,
)
from telefuser.vla.serialization import dumps_wire_message, loads_wire_message, tensor_from_wire, tensor_to_wire

SPACE = ActionSpaceSpec("joint_position", ("a", "b"), ("radian", "radian"), "base", 20.0, False)


def test_action_space_observation_and_chunk_round_trip_without_semantic_loss() -> None:
    observation = RobotObservation(
        RobotState(torch.tensor([0.1, 0.2]), ("a", "b"), 123),
        {"front": torch.arange(12, dtype=torch.uint8).reshape(2, 2, 3)},
        {"source": "fake"},
    )
    chunk = RobotActionChunk(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), SPACE, 2, 123, 7, "episode")

    assert action_space_from_wire(action_space_to_wire(SPACE)) == SPACE
    restored_observation = robot_observation_from_wire(robot_observation_to_wire(observation))
    assert restored_observation.state.dimension_names == observation.state.dimension_names
    assert restored_observation.state.timestamp_ns == observation.state.timestamp_ns
    assert torch.equal(restored_observation.state.values, observation.state.values)
    assert torch.equal(restored_observation.images["front"], observation.images["front"])
    assert dict(restored_observation.metadata) == {"source": "fake"}
    restored_chunk = robot_action_chunk_from_wire(robot_action_chunk_to_wire(chunk))
    assert restored_chunk.action_space == SPACE
    assert torch.equal(restored_chunk.actions, chunk.actions)
    assert restored_chunk.sequence_id == 7


def test_wire_json_is_deterministic_and_size_bounded() -> None:
    encoded = dumps_wire_message({"z": 1, "a": 2})
    assert encoded == '{"a":2,"z":1}'
    assert loads_wire_message(encoded, max_message_bytes=len(encoded)) == {"a": 2, "z": 1}
    with pytest.raises(ValueError, match="exceeds"):
        loads_wire_message(encoded, max_message_bytes=len(encoded) - 1)
    with pytest.raises(ValueError, match="positive integer"):
        loads_wire_message(encoded, max_message_bytes=0)


def test_tensor_wire_rejects_schema_shape_base64_and_byte_limit_errors() -> None:
    payload = tensor_to_wire(torch.ones((2, 2), dtype=torch.float32))

    invalid_shape = dict(payload, shape=[2, -1])
    with pytest.raises(ValueError, match="shape"):
        tensor_from_wire(invalid_shape)

    invalid_data = dict(payload, data="not base64!")
    with pytest.raises(ValueError, match="Base64"):
        tensor_from_wire(invalid_data)

    with pytest.raises(ValueError, match="exceeds"):
        tensor_from_wire(payload, max_bytes=15)

    invalid_schema = dict(payload, schema_version=999)
    with pytest.raises(ValueError, match="unsupported"):
        tensor_from_wire(invalid_schema)


def test_action_space_wire_requires_arrays_not_ambiguous_strings() -> None:
    payload = action_space_to_wire(SPACE)
    payload["dimension_names"] = "ab"
    with pytest.raises(ValueError, match="must be arrays"):
        action_space_from_wire(payload)
