from __future__ import annotations

import torch

from telefuser.pipelines.lingbot_vla_v2 import (
    LINGBOT_VLA_V2_ACTION_SPACE,
    LingBotVlaV2CanonicalActionChunk,
    LingBotVlaV2VLAPolicy,
)
from telefuser.vla import ModelObservation, VLARequest


class _Pipeline:
    def __init__(self) -> None:
        self.seed: int | None = None

    def __call__(self, observation, seed: int | None = None) -> LingBotVlaV2CanonicalActionChunk:
        self.seed = seed
        assert observation.task == "pick"
        return LingBotVlaV2CanonicalActionChunk(
            canonical_normalized_actions=torch.zeros(4, 55),
            horizon=4,
            action_dim=55,
            checkpoint_variant="base",
            policy_verified=False,
            verification_status="unverified",
        )


def test_lingbot_policy_wraps_existing_pipeline_without_changing_its_api() -> None:
    pipeline = _Pipeline()
    policy = LingBotVlaV2VLAPolicy(pipeline, max_horizon=4)
    request = VLARequest(
        ModelObservation(torch.zeros(14), {"camera": object()}),
        "pick",
        "episode",
        sequence_id=3,
        observation_timestamp_ns=123,
        seed=9,
    )

    chunk = policy.predict(request)

    assert pipeline.seed == 9
    assert chunk.actions.shape == (4, 55)
    assert chunk.action_space == LINGBOT_VLA_V2_ACTION_SPACE
    assert chunk.sequence_id == 3
    assert chunk.observation_timestamp_ns == 123
    assert chunk.metadata["verification_status"] == "unverified"
