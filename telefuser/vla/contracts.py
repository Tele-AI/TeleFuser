"""Semantic contracts shared by VLA policies, embodiments, and simulators."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ActionSpaceSpec:
    """Describe action meaning independently of its tensor shape."""

    representation: str
    dimension_names: tuple[str, ...]
    units: tuple[str, ...]
    frame: str | None
    control_hz: float | None
    normalized: bool
    normalization_profile: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.representation, str) or not self.representation:
            raise ValueError("action representation must be a non-empty string")
        if not isinstance(self.dimension_names, tuple):
            raise ValueError("dimension_names must be a tuple")
        if not self.dimension_names or any(not isinstance(name, str) or not name for name in self.dimension_names):
            raise ValueError("dimension_names must contain non-empty strings")
        if len(set(self.dimension_names)) != len(self.dimension_names):
            raise ValueError("dimension_names must be unique")
        if not isinstance(self.units, tuple):
            raise ValueError("units must be a tuple")
        if len(self.units) != len(self.dimension_names) or any(
            not isinstance(unit, str) or not unit for unit in self.units
        ):
            raise ValueError("units must contain one non-empty value per action dimension")
        if self.frame is not None and (not isinstance(self.frame, str) or not self.frame):
            raise ValueError("frame must be None or a non-empty string")
        if self.control_hz is not None:
            if isinstance(self.control_hz, bool) or not isinstance(self.control_hz, (int, float)):
                raise ValueError("control_hz must be None or a positive finite number")
            if not math.isfinite(self.control_hz) or self.control_hz <= 0:
                raise ValueError("control_hz must be None or a positive finite number")
        if not isinstance(self.normalized, bool):
            raise ValueError("normalized must be a boolean")
        if self.normalized and (not isinstance(self.normalization_profile, str) or not self.normalization_profile):
            raise ValueError("normalized action spaces require a normalization_profile")
        if not self.normalized and self.normalization_profile is not None:
            raise ValueError("unnormalized action spaces cannot declare a normalization_profile")

    @property
    def dimension(self) -> int:
        """Return the action vector width."""
        return len(self.dimension_names)

    def require_compatible(self, actual: "ActionSpaceSpec", *, context: str = "action space") -> None:
        """Reject semantic mismatches without inferring meaning from tensor width."""
        fields = (
            "representation",
            "dimension_names",
            "units",
            "frame",
            "normalized",
            "normalization_profile",
        )
        mismatches = [name for name in fields if getattr(self, name) != getattr(actual, name)]
        if self.control_hz is not None and actual.control_hz is not None and self.control_hz != actual.control_hz:
            mismatches.append("control_hz")
        if mismatches:
            raise ValueError(f"{context} mismatch in fields: {mismatches}")


@dataclass(frozen=True)
class ActionChunk:
    """A time-ordered action tensor with explicit semantics and provenance."""

    actions: torch.Tensor
    action_space: ActionSpaceSpec
    valid_length: int
    observation_timestamp_ns: int
    sequence_id: int
    episode_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if self.actions.ndim != 2:
            raise ValueError(f"actions must have shape [T,A], got {tuple(self.actions.shape)}")
        if self.actions.shape[1] != self.action_space.dimension:
            raise ValueError(
                f"actions width must match the action space dimension {self.action_space.dimension}, "
                f"got {self.actions.shape[1]}"
            )
        if not isinstance(self.valid_length, int) or isinstance(self.valid_length, bool):
            raise ValueError("valid_length must be an integer within the action tensor horizon")
        if not 1 <= self.valid_length <= self.actions.shape[0]:
            raise ValueError("valid_length must be within the action tensor horizon")
        if not isinstance(self.observation_timestamp_ns, int) or isinstance(self.observation_timestamp_ns, bool):
            raise ValueError("observation_timestamp_ns must be a non-negative integer")
        if self.observation_timestamp_ns < 0:
            raise ValueError("observation_timestamp_ns must be a non-negative integer")
        if not isinstance(self.sequence_id, int) or isinstance(self.sequence_id, bool):
            raise ValueError("sequence_id must be a non-negative integer")
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be a non-negative integer")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def horizon(self) -> int:
        """Return the allocated action horizon."""
        return int(self.actions.shape[0])

    def trim(self, length: int) -> "ActionChunk":
        """Return the same semantic chunk limited to at most ``length`` actions."""
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise ValueError("chunk trim length must be positive")
        valid_length = min(length, self.valid_length)
        return replace(self, actions=self.actions[:valid_length], valid_length=valid_length)


@dataclass(frozen=True)
class ModelActionChunk(ActionChunk):
    """Action chunk expressed in a model-owned action space."""


@dataclass(frozen=True)
class RobotActionChunk(ActionChunk):
    """Action chunk expressed in an embodiment-owned robot action space."""


@dataclass(frozen=True)
class RobotState:
    """One robot state vector with explicit dimension order."""

    values: torch.Tensor
    dimension_names: tuple[str, ...]
    timestamp_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("robot state values must be a torch.Tensor")
        if not isinstance(self.dimension_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.dimension_names
        ):
            raise ValueError("robot state dimension_names must be a tuple of non-empty strings")
        if self.values.ndim != 1 or self.values.shape[0] != len(self.dimension_names):
            raise ValueError("robot state values must be one-dimensional and match dimension_names")
        if len(set(self.dimension_names)) != len(self.dimension_names):
            raise ValueError("robot state dimension_names must be unique")
        if not isinstance(self.timestamp_ns, int) or isinstance(self.timestamp_ns, bool) or self.timestamp_ns < 0:
            raise ValueError("robot state timestamp_ns must be a non-negative integer")


@dataclass(frozen=True)
class RobotObservation:
    """Simulator observation before embodiment-specific model encoding."""

    state: RobotState
    images: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", MappingProxyType(dict(self.images)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class ModelObservation:
    """Observation representation accepted by one VLA policy."""

    state: Any
    images: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", MappingProxyType(dict(self.images)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class VLARequest:
    """One policy prediction request after embodiment encoding."""

    observation: ModelObservation
    instruction: str
    episode_id: str
    sequence_id: int
    observation_timestamp_ns: int
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(self.sequence_id, int) or isinstance(self.sequence_id, bool) or self.sequence_id < 0:
            raise ValueError("sequence_id must be a non-negative integer")
        if (
            not isinstance(self.observation_timestamp_ns, int)
            or isinstance(self.observation_timestamp_ns, bool)
            or self.observation_timestamp_ns < 0
        ):
            raise ValueError("observation_timestamp_ns must be a non-negative integer")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError("seed must be an integer or None")


@dataclass(frozen=True)
class VLACapabilities:
    """Static policy behavior used for session compatibility checks."""

    model_id: str
    output_action_space: ActionSpaceSpec
    max_horizon: int
    stateful: bool = False
    supports_seed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(self.max_horizon, int) or isinstance(self.max_horizon, bool) or self.max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        if not isinstance(self.stateful, bool) or not isinstance(self.supports_seed, bool):
            raise ValueError("stateful and supports_seed must be booleans")
