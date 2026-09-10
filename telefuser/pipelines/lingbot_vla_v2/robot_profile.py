"""RobotWin feature mapping for LingBot-VLA v2 inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from telefuser.vla.contracts import (
    ActionSpaceSpec,
    ModelActionChunk,
    ModelObservation,
    RobotActionChunk,
    RobotObservation,
    RobotState,
)

ROBOTWIN_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ROBOTWIN_STATE_DIM = 14
CANONICAL_DIM = 55
ARM_SLICE = slice(0, 12)
EFFECTOR_SLICE = slice(28, 30)
ROBOTWIN_ACTION_ORDER = (
    "left_arm_joint_0",
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "left_gripper",
    "right_arm_joint_0",
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_gripper",
)
LINGBOT_VLA_V2_ACTION_SPACE = ActionSpaceSpec(
    representation="canonical_normalized",
    dimension_names=tuple(f"canonical_action_{index}" for index in range(CANONICAL_DIM)),
    units=("normalized",) * CANONICAL_DIM,
    frame=None,
    control_hz=None,
    normalized=True,
    normalization_profile="lingbot_vla_v2_canonical",
)
ROBOTWIN_ACTION_SPACE = ActionSpaceSpec(
    representation="absolute_qpos",
    dimension_names=ROBOTWIN_ACTION_ORDER,
    units=("radian",) * 6 + ("normalized_position",) + ("radian",) * 6 + ("normalized_position",),
    frame="robot_joint",
    control_hz=None,
    normalized=False,
)


@dataclass(frozen=True)
class LingBotVlaV2ActionChunk:
    """Structured RobotWin action chunk returned by the SDK."""

    fields: Mapping[str, torch.Tensor]
    raw_actions: torch.Tensor
    action_mask: torch.Tensor
    horizon: int
    robot_profile: str = "robotwin"
    policy_verified: bool = False
    verification_status: str = "unverified_official_6b_base"
    canonical_normalized_actions: torch.Tensor | None = None


class RobotWinProfile:
    """Map RobotWin observations and actions to LingBot's canonical space."""

    name = "robotwin"
    embodiment_id = "robotwin"
    camera_keys = ROBOTWIN_CAMERA_KEYS
    canonical_dim = CANONICAL_DIM
    raw_state_dim = ROBOTWIN_STATE_DIM
    _REQUIRED_STATS = (
        "observation.state.arm.position",
        "observation.state.effector.position",
        "action.arm.position",
        "action.effector.position",
    )

    def __init__(self, norm_stats: Mapping[str, Mapping[str, object]]) -> None:
        self._stats = {
            key: {
                stat_name: torch.as_tensor(stat_value, dtype=torch.float64) for stat_name, stat_value in values.items()
            }
            for key, values in norm_stats.items()
        }
        self._validate_stats()

    @classmethod
    def from_json(cls, path: str | Path) -> "RobotWinProfile":
        """Load RobotWin normalization statistics from an upstream-format JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        norm_stats = payload.get("norm_stats")
        if not isinstance(norm_stats, dict):
            raise ValueError("RobotWin normalization file must contain a norm_stats object")
        return cls(norm_stats)

    @classmethod
    def default(cls) -> "RobotWinProfile":
        """Load the RobotWin statistics bundled with TeleFuser."""
        path = Path(__file__).with_name("assets") / "robotwin_norm_stats.json"
        return cls.from_json(path)

    @property
    def action_mask(self) -> torch.Tensor:
        """Return the canonical dimensions used by RobotWin actions."""
        mask = torch.zeros(self.canonical_dim, dtype=torch.bool)
        mask[ARM_SLICE] = True
        mask[EFFECTOR_SLICE] = True
        return mask

    @property
    def model_action_space(self) -> ActionSpaceSpec:
        """Return the semantic action space accepted from LingBot-VLA v2."""
        return LINGBOT_VLA_V2_ACTION_SPACE

    @property
    def robot_action_space(self) -> ActionSpaceSpec:
        """Return the semantic action space emitted for RoboTwin."""
        return ROBOTWIN_ACTION_SPACE

    def encode_observation(self, observation: RobotObservation) -> ModelObservation:
        """Validate a RoboTwin observation while retaining raw state for the model processor."""
        if not isinstance(observation, RobotObservation):
            raise TypeError("observation must be a RobotObservation")
        if observation.state.dimension_names != ROBOTWIN_ACTION_ORDER:
            raise ValueError("RobotWin state dimension order does not match the robot profile")
        missing = [key for key in self.camera_keys if key not in observation.images]
        if missing:
            raise ValueError(f"RobotWin observation is missing camera keys: {missing}")
        return ModelObservation(
            state=observation.state.values,
            images={key: observation.images[key] for key in self.camera_keys},
            metadata=observation.metadata,
        )

    def decode_actions(
        self,
        actions: ModelActionChunk,
        robot_state: RobotState,
    ) -> RobotActionChunk:
        """Map a semantic LingBot chunk to an absolute RoboTwin joint chunk."""
        self.model_action_space.require_compatible(actions.action_space, context="LingBot model action space")
        if robot_state.dimension_names != ROBOTWIN_ACTION_ORDER:
            raise ValueError("RobotWin state dimension order does not match the robot profile")
        structured = self.structure_actions(actions.actions[: actions.valid_length])
        return RobotActionChunk(
            actions=structured.raw_actions,
            action_space=self.robot_action_space,
            valid_length=structured.horizon,
            observation_timestamp_ns=actions.observation_timestamp_ns,
            sequence_id=actions.sequence_id,
            episode_id=actions.episode_id,
            metadata=actions.metadata,
        )

    def normalize_state(self, raw_state: torch.Tensor | Sequence[float]) -> torch.Tensor:
        """Convert one raw 14-D RobotWin state to normalized canonical 55-D space."""
        state = torch.as_tensor(raw_state, dtype=torch.float32, device="cpu")
        if state.shape != (self.raw_state_dim,):
            raise ValueError(f"RobotWin state must have shape ({self.raw_state_dim},), got {tuple(state.shape)}")
        if not torch.isfinite(state).all():
            raise ValueError("RobotWin state must contain only finite values")

        arm = torch.cat((state[0:6], state[7:13]))
        effector = state[[6, 13]]
        canonical = torch.zeros(self.canonical_dim, dtype=torch.float32)
        canonical[ARM_SLICE] = self._normalize("observation.state.arm.position", arm)
        canonical[EFFECTOR_SLICE] = self._normalize("observation.state.effector.position", effector)
        return canonical

    def structure_actions(
        self,
        canonical_normalized_actions: torch.Tensor,
        *,
        include_canonical: bool = False,
    ) -> LingBotVlaV2ActionChunk:
        """Convert a normalized canonical action chunk to RobotWin action fields."""
        actions = torch.as_tensor(canonical_normalized_actions, dtype=torch.float32, device="cpu")
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError("RobotWin structured output currently supports a single observation")
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[-1] != self.canonical_dim:
            raise ValueError(
                f"canonical actions must have shape [H,{self.canonical_dim}] or [1,H,{self.canonical_dim}], "
                f"got {tuple(actions.shape)}"
            )
        if not torch.isfinite(actions).all():
            raise ValueError("canonical actions must contain only finite values")

        arm = self._unnormalize("action.arm.position", actions[:, ARM_SLICE])
        effector = self._unnormalize("action.effector.position", actions[:, EFFECTOR_SLICE])
        raw = torch.empty(actions.shape[0], self.raw_state_dim, dtype=torch.float32)
        raw[:, 0:6] = arm[:, 0:6]
        raw[:, 6] = effector[:, 0]
        raw[:, 7:13] = arm[:, 6:12]
        raw[:, 13] = effector[:, 1]
        fields = MappingProxyType(
            {
                "action.arm.position": arm,
                "action.effector.position": effector,
                "action": raw,
            }
        )
        return LingBotVlaV2ActionChunk(
            fields=fields,
            raw_actions=raw,
            action_mask=self.action_mask,
            horizon=int(actions.shape[0]),
            canonical_normalized_actions=actions.clone() if include_canonical else None,
        )

    def _validate_stats(self) -> None:
        expected_dims = {
            "observation.state.arm.position": 12,
            "observation.state.effector.position": 2,
            "action.arm.position": 12,
            "action.effector.position": 2,
        }
        missing = [key for key in self._REQUIRED_STATS if key not in self._stats]
        if missing:
            raise ValueError(f"RobotWin normalization statistics are missing keys: {missing}")
        for key, expected_dim in expected_dims.items():
            values = self._stats[key]
            for stat_name in ("q01", "q99"):
                if stat_name not in values or values[stat_name].shape != (expected_dim,):
                    shape = None if stat_name not in values else tuple(values[stat_name].shape)
                    raise ValueError(
                        f"RobotWin statistic {key}.{stat_name} must have shape ({expected_dim},), got {shape}"
                    )

    def _normalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        low = self._stats[key]["q01"]
        high = self._stats[key]["q99"]
        normalized = (value.to(dtype=torch.float64) - low) / (high - low + 1e-6) * 2.0 - 1.0
        return normalized.to(dtype=value.dtype)

    def _unnormalize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        low = self._stats[key]["q01"]
        high = self._stats[key]["q99"]
        unnormalized = (value.to(dtype=torch.float64) + 1.0) / 2.0 * (high - low + 1e-6) + low
        return unnormalized.to(dtype=value.dtype)
