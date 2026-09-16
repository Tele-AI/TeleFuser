from __future__ import annotations

import pytest
import torch

from telefuser.vla import ActionSpaceSpec, RobotActionChunk, RobotState
from telefuser.vla.runtime import BoundedActionSafety, ChunkExecutor


def _space(*, representation: str = "joint_position") -> ActionSpaceSpec:
    return ActionSpaceSpec(
        representation=representation,
        dimension_names=("joint_0", "joint_1"),
        units=("radian", "radian"),
        frame="robot_joint",
        control_hz=20.0,
        normalized=False,
    )


def _chunk(space: ActionSpaceSpec | None = None) -> RobotActionChunk:
    return RobotActionChunk(
        actions=torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]),
        action_space=space or _space(),
        valid_length=3,
        observation_timestamp_ns=100,
        sequence_id=4,
        episode_id="episode",
        metadata={"source": "test"},
    )


def _state() -> RobotState:
    return RobotState(torch.zeros(2), ("joint_0", "joint_1"), timestamp_ns=100)


def test_action_space_rejects_semantic_mismatch_even_when_dimensions_match() -> None:
    with pytest.raises(ValueError, match="representation"):
        _space().require_compatible(_space(representation="joint_delta"))


def test_chunk_executor_trims_and_checks_age_and_bounds() -> None:
    executor = ChunkExecutor(
        _space(),
        execute_horizon=2,
        max_observation_age_ns=50,
        safety_policy=BoundedActionSafety(torch.full((2,), -1.0), torch.full((2,), 1.0)),
    )

    prepared = executor.prepare(_chunk(), _state(), now_ns=149)

    assert prepared.actions.shape == (2, 2)
    assert prepared.valid_length == 2
    assert [action.step_index for action in executor.iter_actions(prepared)] == [0, 1]
    with pytest.raises(ValueError, match="stale"):
        executor.prepare(_chunk(), _state(), now_ns=151)


def test_contracts_reject_invalid_tensor_shape_and_timestamp_type() -> None:
    with pytest.raises(ValueError, match="width"):
        RobotActionChunk(torch.zeros(2, 3), _space(), 2, 0, 0, "episode")
    with pytest.raises(ValueError, match="timestamp"):
        RobotState(torch.zeros(2), ("joint_0", "joint_1"), timestamp_ns="100")
