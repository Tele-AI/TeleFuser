"""Embodiment protocol separating model coordinates from robot controls."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import (
    ActionSpaceSpec,
    ModelActionChunk,
    ModelObservation,
    ObservationSpaceSpec,
    RobotActionChunk,
    RobotObservation,
    RobotState,
)


@runtime_checkable
class EmbodimentAdapter(Protocol):
    """Encode robot observations and decode model actions for one embodiment."""

    @property
    def embodiment_id(self) -> str: ...

    @property
    def model_action_space(self) -> ActionSpaceSpec: ...

    @property
    def robot_action_space(self) -> ActionSpaceSpec: ...

    @property
    def observation_space(self) -> ObservationSpaceSpec: ...

    def encode_observation(self, observation: RobotObservation) -> ModelObservation: ...

    def decode_actions(self, actions: ModelActionChunk, robot_state: RobotState) -> RobotActionChunk: ...
