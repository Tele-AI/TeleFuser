"""Dependency-free boundary between the VLA runtime and RoboTwin processes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from telefuser.vla.contracts import ActionSpaceSpec, RobotActionChunk, RobotObservation
from telefuser.vla.runtime.executor import ChunkExecutor, RobotAction


class RoboTwinSimulatorAdapter:
    """Validate semantic actions before invoking RoboTwin-owned callbacks.

    The callbacks keep RoboTwin, SAPIEN, and Vulkan imports in the remote
    simulator process rather than making them TeleFuser dependencies.
    """

    def __init__(
        self,
        action_space: ActionSpaceSpec,
        *,
        observe_fn: Callable[[], RobotObservation],
        execute_fn: Callable[[np.ndarray], None],
        reset_fn: Callable[[], RobotObservation],
    ) -> None:
        self.action_space = action_space
        self._observe_fn = observe_fn
        self._execute_fn = execute_fn
        self._reset_fn = reset_fn

    def observe(self) -> RobotObservation:
        """Read one observation from the RoboTwin process."""
        observation = self._observe_fn()
        if not isinstance(observation, RobotObservation):
            raise TypeError("RoboTwin observe callback must return RobotObservation")
        return observation

    def execute(self, action: RobotAction) -> None:
        """Submit one validated float32 action vector to RoboTwin."""
        if not isinstance(action, RobotAction):
            raise TypeError("RoboTwin execute expects RobotAction")
        self.action_space.require_compatible(action.action_space, context="RoboTwin action space")
        values = action.values.detach().to(device="cpu", dtype=torch.float32).numpy()
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.shape != (self.action_space.dimension,):
            raise ValueError(f"RoboTwin action must have shape ({self.action_space.dimension},), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("RoboTwin action must contain only finite values")
        self._execute_fn(values)

    def execute_chunk(self, chunk: RobotActionChunk) -> int:
        """Execute the valid portion of an already prepared chunk."""
        self.action_space.require_compatible(chunk.action_space, context="RoboTwin action space")
        count = 0
        for action in ChunkExecutor.iter_actions(chunk):
            self.execute(action)
            count += 1
        return count

    def reset(self) -> RobotObservation:
        """Reset RoboTwin and validate its initial observation."""
        observation = self._reset_fn()
        if not isinstance(observation, RobotObservation):
            raise TypeError("RoboTwin reset callback must return RobotObservation")
        return observation
