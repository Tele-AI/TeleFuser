"""Preparation of semantic robot action chunks for execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import torch

from ..contracts import ActionSpaceSpec, RobotActionChunk, RobotState
from .safety import ActionSafetyPolicy, FiniteActionSafety


@dataclass(frozen=True)
class RobotAction:
    """One executable action extracted from a validated chunk."""

    values: torch.Tensor
    action_space: ActionSpaceSpec
    sequence_id: int
    step_index: int
    episode_id: str


class ChunkExecutor:
    """Trim, validate, and expose actions without simulator-specific logic."""

    def __init__(
        self,
        expected_action_space: ActionSpaceSpec,
        *,
        execute_horizon: int | None = None,
        safety_policy: ActionSafetyPolicy | None = None,
        max_observation_age_ns: int | None = None,
    ) -> None:
        if execute_horizon is not None and (
            not isinstance(execute_horizon, int) or isinstance(execute_horizon, bool) or execute_horizon < 1
        ):
            raise ValueError("execute_horizon must be positive")
        if max_observation_age_ns is not None and (
            not isinstance(max_observation_age_ns, int)
            or isinstance(max_observation_age_ns, bool)
            or max_observation_age_ns < 1
        ):
            raise ValueError("max_observation_age_ns must be positive")
        self.expected_action_space = expected_action_space
        self.execute_horizon = execute_horizon
        self.safety_policy = safety_policy or FiniteActionSafety()
        self.max_observation_age_ns = max_observation_age_ns

    def prepare(
        self,
        chunk: RobotActionChunk,
        robot_state: RobotState,
        *,
        now_ns: int | None = None,
    ) -> RobotActionChunk:
        """Return a bounded, safe chunk ready for simulator consumption."""
        self.expected_action_space.require_compatible(chunk.action_space, context="robot action space")
        if self.max_observation_age_ns is not None:
            if now_ns is None:
                raise ValueError("now_ns is required when max_observation_age_ns is configured")
            if not isinstance(now_ns, int) or isinstance(now_ns, bool) or now_ns < 0:
                raise ValueError("now_ns must be a non-negative integer")
            if now_ns < chunk.observation_timestamp_ns:
                raise ValueError("now_ns cannot precede the observation timestamp")
            if now_ns - chunk.observation_timestamp_ns > self.max_observation_age_ns:
                raise ValueError("robot action chunk is stale")
        valid_length = chunk.valid_length
        if self.execute_horizon is not None:
            valid_length = min(valid_length, self.execute_horizon)
        prepared = replace(chunk, actions=chunk.actions[:valid_length], valid_length=valid_length)
        self.safety_policy.validate(prepared, robot_state)
        return prepared

    @staticmethod
    def iter_actions(chunk: RobotActionChunk) -> Iterator[RobotAction]:
        """Yield only the validated portion of a prepared action chunk."""
        for step_index in range(chunk.valid_length):
            yield RobotAction(
                values=chunk.actions[step_index],
                action_space=chunk.action_space,
                sequence_id=chunk.sequence_id,
                step_index=step_index,
                episode_id=chunk.episode_id,
            )
