from __future__ import annotations

import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from telefuser.integrations.sim import MuJoCoJointBinding, MuJoCoSimulatorAdapter
from telefuser.vla import ActionSpaceSpec, ObservationSpaceSpec
from telefuser.vla.runtime import RobotAction

ACTION_NAMES = tuple(f"action_{index}" for index in range(14))
ACTION_SPACE = ActionSpaceSpec(
    "absolute_qpos",
    ACTION_NAMES,
    ("radian",) * 6 + ("normalized_position",) + ("radian",) * 6 + ("normalized_position",),
    "robot_joint",
    None,
    False,
)
OBSERVATION_SPACE = ObservationSpaceSpec(ACTION_NAMES)


def _model() -> mujoco.MjModel:
    joint_names = (
        *(f"left_{index}" for index in range(6)),
        "left_gripper_a",
        "left_gripper_b",
        *(f"right_{index}" for index in range(6)),
        "right_gripper_a",
        "right_gripper_b",
    )
    bodies = []
    for index, name in enumerate(joint_names):
        joint_type = "slide" if "gripper" in name else "hinge"
        joint_range = "0 0.05" if "gripper" in name else "-1 1"
        bodies.append(
            f'<body name="body_{index}" pos="{index * 0.04} 0 0.1">'
            f'<joint name="{name}" type="{joint_type}" axis="0 0 1" range="{joint_range}"/>'
            '<geom type="box" pos="0.015 0 0" size="0.02 0.01 0.01" mass="0.01"/>'
            "</body>"
        )
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><option gravity="0 0 0" timestep="0.002"/><worldbody>{"".join(bodies)}</worldbody></mujoco>'
    )


def _bindings() -> tuple[MuJoCoJointBinding, ...]:
    return (
        *(MuJoCoJointBinding(ACTION_NAMES[index], (f"left_{index}",)) for index in range(6)),
        MuJoCoJointBinding(ACTION_NAMES[6], ("left_gripper_a", "left_gripper_b"), True),
        *(MuJoCoJointBinding(ACTION_NAMES[index + 7], (f"right_{index}",)) for index in range(6)),
        MuJoCoJointBinding(ACTION_NAMES[13], ("right_gripper_a", "right_gripper_b"), True),
    )


def test_mujoco_adapter_executes_pd_action_and_returns_semantic_state() -> None:
    adapter = MuJoCoSimulatorAdapter(
        _model(),
        ACTION_SPACE,
        OBSERVATION_SPACE,
        _bindings(),
        steps_per_action=20,
    )
    initial = adapter.reset()
    target = torch.full((14,), 0.2)
    target[6] = 0.5
    target[13] = 0.75

    adapter.execute(RobotAction(target, ACTION_SPACE, sequence_id=0, step_index=0, episode_id="episode"))
    observed = adapter.observe()

    assert observed.state.dimension_names == ACTION_NAMES
    assert observed.state.values.shape == (14,)
    assert observed.metadata["simulator"] == "mujoco"
    assert observed.metadata["simulation_time_s"] > 0
    assert torch.linalg.vector_norm(observed.state.values - initial.state.values) > 0
    assert 0 < observed.state.values[6] <= 1
    assert 0 < observed.state.values[13] <= 1
    adapter.close()


def test_mujoco_adapter_rejects_binding_order_mismatch() -> None:
    bindings = list(_bindings())
    bindings[0], bindings[1] = bindings[1], bindings[0]

    with pytest.raises(ValueError, match="binding order"):
        MuJoCoSimulatorAdapter(_model(), ACTION_SPACE, OBSERVATION_SPACE, bindings)
