"""CPU-only pipeline fixture for replica-process VLA tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from telefuser.service.vla_replica import VLAReplicaProvider
from telefuser.vla import (
    ActionSpaceSpec,
    ModelActionChunk,
    ModelObservation,
    ObservationSpaceSpec,
    RobotActionChunk,
    RobotObservation,
    RobotState,
    VLACapabilities,
    VLARegistry,
    VLASessionManager,
)

MODEL_SPACE = ActionSpaceSpec("joint_delta", ("joint",), ("radian",), None, 10.0, False)
ROBOT_SPACE = ActionSpaceSpec("joint_position", ("joint",), ("radian",), None, 10.0, False)
OBSERVATION_SPACE = ObservationSpaceSpec(("joint",))

PIPELINE_CONTRACT = {
    "contract_version": "v1",
    "pipeline_name": "fake_vla_pipeline",
    "supported_tasks": ["vla_action"],
    "supported_media_types": ["structured"],
    "execution_mode": "serial_single_pipeline",
    "effective_max_concurrent_tasks": 1,
    "entrypoints": {"get_pipeline": "get_pipeline", "run_with_file": "run_structured"},
    "task_contracts": {"vla_action": {"media_type": "structured", "required_inputs": [], "optional_inputs": []}},
}


class _Policy:
    def capabilities(self) -> VLACapabilities:
        return VLACapabilities("fake", MODEL_SPACE, max_horizon=2)

    def predict(self, request) -> ModelActionChunk:
        return ModelActionChunk(
            torch.tensor([[0.25], [0.5]]),
            MODEL_SPACE,
            2,
            request.observation_timestamp_ns,
            request.sequence_id,
            request.episode_id,
        )

    def reset(self, episode_id: str) -> None:
        pass


@dataclass
class _Embodiment:
    embodiment_id: str = "fake-robot"
    model_action_space: ActionSpaceSpec = MODEL_SPACE
    robot_action_space: ActionSpaceSpec = ROBOT_SPACE
    observation_space: ObservationSpaceSpec = OBSERVATION_SPACE

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


def get_pipeline(parallelism: int = 1) -> object:
    if parallelism != 1:
        raise ValueError("fake VLA pipeline only supports parallelism=1")
    return object()


def run_structured(_pipeline: object, **_kwargs: object) -> dict[str, object]:
    return {}


def get_vla_provider(_pipeline: object) -> VLAReplicaProvider:
    registry = VLARegistry()
    registry.register_policy("fake", _Policy())
    registry.register_embodiment("fake-robot", _Embodiment())
    return VLAReplicaProvider(VLASessionManager(registry))
