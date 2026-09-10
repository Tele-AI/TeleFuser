from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from telefuser.vla import (
    ActionSpaceSpec,
    ModelActionChunk,
    ModelObservation,
    RobotActionChunk,
    RobotObservation,
    RobotState,
    VLACapabilities,
    VLARegistry,
    VLASessionManager,
)


def _space(representation: str) -> ActionSpaceSpec:
    return ActionSpaceSpec(representation, ("joint",), ("radian",), None, None, False)


MODEL_SPACE = _space("joint_delta")
ROBOT_SPACE = _space("joint_position")


class _Policy:
    resets: list[str]

    def __init__(self) -> None:
        self.resets = []

    def capabilities(self) -> VLACapabilities:
        return VLACapabilities("fake", MODEL_SPACE, max_horizon=3)

    def predict(self, request):
        return ModelActionChunk(
            torch.tensor([[0.1], [0.2], [0.3]]),
            MODEL_SPACE,
            3,
            request.observation_timestamp_ns,
            request.sequence_id,
            request.episode_id,
        )

    def reset(self, episode_id: str) -> None:
        self.resets.append(episode_id)


@dataclass
class _Embodiment:
    embodiment_id: str = "fake-robot"
    model_action_space: ActionSpaceSpec = MODEL_SPACE
    robot_action_space: ActionSpaceSpec = ROBOT_SPACE

    def encode_observation(self, observation: RobotObservation) -> ModelObservation:
        return ModelObservation(observation.state.values, observation.images)

    def decode_actions(self, actions: ModelActionChunk, robot_state: RobotState) -> RobotActionChunk:
        return RobotActionChunk(
            actions.actions + robot_state.values,
            self.robot_action_space,
            actions.valid_length,
            actions.observation_timestamp_ns,
            actions.sequence_id,
            actions.episode_id,
        )


def _manager(policy: _Policy | None = None) -> tuple[VLASessionManager, _Policy]:
    policy = policy or _Policy()
    registry = VLARegistry()
    registry.register_policy("fake", policy)
    registry.register_embodiment("fake-robot", _Embodiment())
    return VLASessionManager(registry), policy


def _observation() -> RobotObservation:
    return RobotObservation(RobotState(torch.tensor([1.0]), ("joint",), 100), {})


def test_session_runs_full_semantic_dataflow_and_rejects_out_of_order_sequence() -> None:
    manager, _ = _manager()
    session = manager.open(
        "session",
        model_id="fake",
        embodiment_id="fake-robot",
        episode_id="episode",
        execute_horizon=2,
    )

    timings: dict[str, float] = {}
    chunk = session.predict(_observation(), "move", sequence_id=5, seed=7, timings=timings)

    assert torch.allclose(chunk.actions, torch.tensor([[1.1], [1.2]]))
    assert chunk.action_space == ROBOT_SPACE
    assert set(timings) == {
        "decode_actions_ms",
        "encode_observation_ms",
        "policy_ms",
        "prepare_actions_ms",
    }
    assert all(value >= 0 for value in timings.values())
    with pytest.raises(ValueError, match="must increase"):
        session.predict(_observation(), "move", sequence_id=5)


def test_session_reset_allows_new_sequence_and_close_releases_state() -> None:
    manager, policy = _manager()
    session = manager.open(
        "session", model_id="fake", embodiment_id="fake-robot", episode_id="episode", execute_horizon=1
    )
    session.predict(_observation(), "move", sequence_id=1)

    manager.reset("session", "episode-2")
    chunk = session.predict(_observation(), "move", sequence_id=0)
    manager.close("session")

    assert chunk.episode_id == "episode-2"
    assert policy.resets == ["episode", "episode-2"]
    assert manager.session_ids() == ()


def test_open_rejects_policy_and_embodiment_action_space_mismatch() -> None:
    policy = _Policy()
    registry = VLARegistry()
    registry.register_policy("fake", policy)
    registry.register_embodiment(
        "fake-robot",
        _Embodiment(model_action_space=_space("velocity")),
    )
    manager = VLASessionManager(registry)

    with pytest.raises(ValueError, match="representation"):
        manager.open("session", model_id="fake", embodiment_id="fake-robot", episode_id="episode")
