"""Semantic VLA policy wrapper for the existing LingBot-VLA v2 pipeline."""

from __future__ import annotations

from typing import Any, Protocol

from telefuser.vla.contracts import ModelActionChunk, VLACapabilities, VLARequest
from telefuser.vla.registry import VLARegistry
from telefuser.vla.session import VLASessionManager

from .data import LingBotVlaV2Observation
from .robot_profile import LINGBOT_VLA_V2_ACTION_SPACE, RobotWinProfile

LINGBOT_VLA_V2_MODEL_ID = "lingbot-vla-v2"


class _LingBotPipeline(Protocol):
    def __call__(self, observation: LingBotVlaV2Observation, seed: int | None = None) -> Any: ...


class LingBotVlaV2VLAPolicy:
    """Adapt the stable LingBot pipeline API to the shared semantic contract."""

    def __init__(self, pipeline: _LingBotPipeline, *, max_horizon: int = 50) -> None:
        if not isinstance(max_horizon, int) or isinstance(max_horizon, bool) or max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        self.pipeline = pipeline
        self._capabilities = VLACapabilities(
            model_id=LINGBOT_VLA_V2_MODEL_ID,
            output_action_space=LINGBOT_VLA_V2_ACTION_SPACE,
            max_horizon=max_horizon,
            stateful=False,
            supports_seed=True,
        )

    def capabilities(self) -> VLACapabilities:
        """Describe the canonical LingBot action output."""
        return self._capabilities

    def predict(self, request: VLARequest) -> ModelActionChunk:
        """Run the existing pipeline and attach semantic request provenance."""
        observation = LingBotVlaV2Observation(
            task=request.instruction,
            state=request.observation.state,
            images=request.observation.images,
        )
        output = self.pipeline(observation, seed=request.seed)
        if output.horizon > self._capabilities.max_horizon:
            raise RuntimeError(
                f"LingBot pipeline returned horizon {output.horizon}, exceeding declared maximum "
                f"{self._capabilities.max_horizon}"
            )
        return ModelActionChunk(
            actions=output.canonical_normalized_actions,
            action_space=self._capabilities.output_action_space,
            valid_length=output.horizon,
            observation_timestamp_ns=request.observation_timestamp_ns,
            sequence_id=request.sequence_id,
            episode_id=request.episode_id,
            metadata={
                "checkpoint_variant": output.checkpoint_variant,
                "policy_verified": output.policy_verified,
                "verification_status": output.verification_status,
            },
        )

    def reset(self, episode_id: str) -> None:
        """Validate RESET for the currently stateless LingBot base policy."""
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")


def create_lingbot_vla_v2_session_manager(
    pipeline: _LingBotPipeline,
    *,
    profile: RobotWinProfile | None = None,
) -> VLASessionManager:
    """Register the shared LingBot policy and RoboTwin embodiment once."""
    resolved_profile = profile or RobotWinProfile.default()
    registry = VLARegistry()
    registry.register_policy(LINGBOT_VLA_V2_MODEL_ID, LingBotVlaV2VLAPolicy(pipeline))
    registry.register_embodiment(resolved_profile.embodiment_id, resolved_profile)
    return VLASessionManager(registry)
