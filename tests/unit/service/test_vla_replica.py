"""CPU-only tests for worker-local VLA replica dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from telefuser.service.vla_replica import VLAReplicaProvider
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
    action_space_to_wire,
    robot_action_chunk_from_wire,
    robot_observation_to_wire,
)

MODEL_SPACE = ActionSpaceSpec("joint_delta", ("joint",), ("radian",), None, 10.0, False)
ROBOT_SPACE = ActionSpaceSpec("joint_position", ("joint",), ("radian",), None, 10.0, False)


class _Policy:
    def __init__(self) -> None:
        self.resets: list[str] = []

    def capabilities(self) -> VLACapabilities:
        return VLACapabilities("fake", MODEL_SPACE, max_horizon=2)

    def predict(self, request) -> ModelActionChunk:
        return ModelActionChunk(
            torch.tensor([[0.1], [0.2]]),
            MODEL_SPACE,
            2,
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
            ROBOT_SPACE,
            actions.valid_length,
            actions.observation_timestamp_ns,
            actions.sequence_id,
            actions.episode_id,
        )


def _provider() -> tuple[VLAReplicaProvider, _Policy]:
    policy = _Policy()
    registry = VLARegistry()
    registry.register_policy("fake", policy)
    registry.register_embodiment("fake-robot", _Embodiment())
    return VLAReplicaProvider(VLASessionManager(registry)), policy


def test_provider_dispatches_complete_worker_local_session() -> None:
    provider, policy = _provider()
    assert provider.metadata() == {"model_ids": ["fake"], "embodiment_ids": ["fake-robot"]}
    opened = provider.dispatch(
        "OPEN",
        {
            "session_id": "session",
            "model_id": "fake",
            "embodiment_id": "fake-robot",
            "episode_id": "episode",
            "expected_robot_action_space": action_space_to_wire(ROBOT_SPACE),
        },
    )
    assert opened["max_horizon"] == 2
    assert opened["robot_action_space"] == action_space_to_wire(ROBOT_SPACE)

    observation = RobotObservation(RobotState(torch.tensor([1.0]), ("joint",), 100), {})
    response = provider.dispatch(
        "PREDICT",
        {
            "session_id": "session",
            "sequence_id": 1,
            "observation_timestamp_ns": 100,
            "instruction": "move",
            "seed": 7,
            "observation": robot_observation_to_wire(observation),
        },
    )
    chunk = robot_action_chunk_from_wire(response["chunk"])
    assert chunk.actions.flatten().tolist() == pytest.approx([1.1, 1.2])

    provider.dispatch("RESET", {"session_id": "session", "episode_id": "episode-2"})
    provider.dispatch("CLOSE", {"session_id": "session"})
    assert policy.resets == ["episode", "episode-2"]


def test_provider_rejects_action_contract_mismatch_before_open() -> None:
    provider, _ = _provider()
    mismatch = ActionSpaceSpec("velocity", ("joint",), ("radian_per_second",), None, 10.0, False)
    with pytest.raises(ValueError, match="action space"):
        provider.dispatch(
            "OPEN",
            {
                "session_id": "session",
                "model_id": "fake",
                "embodiment_id": "fake-robot",
                "episode_id": "episode",
                "expected_robot_action_space": action_space_to_wire(mismatch),
            },
        )


def test_provider_close_releases_all_worker_sessions() -> None:
    provider, policy = _provider()
    for session_id in ("one", "two"):
        provider.dispatch(
            "OPEN",
            {
                "session_id": session_id,
                "model_id": "fake",
                "embodiment_id": "fake-robot",
                "episode_id": session_id,
            },
        )
    provider.close()
    assert provider.sessions.session_ids() == ()
    assert policy.resets == ["one", "two"]
