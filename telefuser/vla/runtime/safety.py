"""Safety checks applied after embodiment action decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from ..contracts import RobotActionChunk, RobotState


@runtime_checkable
class ActionSafetyPolicy(Protocol):
    """Validate a robot action chunk before it reaches a simulator."""

    def validate(self, chunk: RobotActionChunk, robot_state: RobotState) -> None: ...


class FiniteActionSafety:
    """Reject non-finite state and action values."""

    def validate(self, chunk: RobotActionChunk, robot_state: RobotState) -> None:
        """Validate finite values in the executable portion of a chunk."""
        if not torch.isfinite(robot_state.values).all():
            raise ValueError("robot state must contain only finite values")
        if not torch.isfinite(chunk.actions[: chunk.valid_length]).all():
            raise ValueError("robot actions must contain only finite values")


@dataclass(frozen=True)
class BoundedActionSafety(FiniteActionSafety):
    """Apply per-dimension bounds and an optional first-step delta limit."""

    lower: torch.Tensor
    upper: torch.Tensor
    max_initial_delta: torch.Tensor | None = None

    def validate(self, chunk: RobotActionChunk, robot_state: RobotState) -> None:
        """Validate finite values, limits, and the transition from current state."""
        super().validate(chunk, robot_state)
        dimension = chunk.action_space.dimension
        if self.lower.shape != (dimension,) or self.upper.shape != (dimension,):
            raise ValueError("safety bounds must match the robot action dimension")
        if torch.any(self.lower > self.upper):
            raise ValueError("safety lower bounds must not exceed upper bounds")
        actions = chunk.actions[: chunk.valid_length]
        lower = self.lower.to(device=actions.device, dtype=actions.dtype)
        upper = self.upper.to(device=actions.device, dtype=actions.dtype)
        if torch.any(actions < lower) or torch.any(actions > upper):
            raise ValueError("robot actions exceed configured safety bounds")
        if self.max_initial_delta is None:
            return
        if self.max_initial_delta.shape != (dimension,):
            raise ValueError("max_initial_delta must match the robot action dimension")
        if robot_state.values.shape != (dimension,):
            raise ValueError("robot state dimension must match action delta safety checks")
        maximum = self.max_initial_delta.to(device=actions.device, dtype=actions.dtype)
        state = robot_state.values.to(device=actions.device, dtype=actions.dtype)
        if torch.any(torch.abs(actions[0] - state) > maximum):
            raise ValueError("first robot action exceeds configured state delta limits")
