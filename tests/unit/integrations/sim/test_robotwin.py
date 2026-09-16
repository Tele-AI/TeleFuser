from __future__ import annotations

import numpy as np
import pytest
import torch

from telefuser.integrations.sim import RoboTwinSimulatorAdapter
from telefuser.vla import ActionSpaceSpec, RobotActionChunk, RobotObservation, RobotState

SPACE = ActionSpaceSpec("joint_position", ("joint",), ("radian",), "robot_joint", None, False)


def _observation() -> RobotObservation:
    return RobotObservation(RobotState(torch.zeros(1), ("joint",), 0), {})


def test_robotwin_adapter_executes_only_valid_chunk_actions_as_float32() -> None:
    executed: list[np.ndarray] = []
    adapter = RoboTwinSimulatorAdapter(
        SPACE,
        observe_fn=_observation,
        reset_fn=_observation,
        execute_fn=executed.append,
    )
    chunk = RobotActionChunk(torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.bfloat16), SPACE, 2, 0, 1, "episode")

    count = adapter.execute_chunk(chunk)

    assert count == 2
    assert [action.dtype for action in executed] == [np.float32, np.float32]
    assert [action.tolist() for action in executed] == [[1.0], [2.0]]
    assert adapter.observe() == _observation()


def test_robotwin_adapter_rejects_semantically_different_action() -> None:
    other = ActionSpaceSpec("joint_delta", ("joint",), ("radian",), "robot_joint", None, False)
    adapter = RoboTwinSimulatorAdapter(
        SPACE,
        observe_fn=_observation,
        reset_fn=_observation,
        execute_fn=lambda _action: None,
    )
    chunk = RobotActionChunk(torch.zeros(1, 1), other, 1, 0, 0, "episode")

    with pytest.raises(ValueError, match="representation"):
        adapter.execute_chunk(chunk)
