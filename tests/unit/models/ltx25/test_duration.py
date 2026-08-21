"""LTX-2.5 DurationHead loading and frame-grid contracts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from telefuser.models.ltx25.duration import (
    LTX25DurationHead,
    ltx25_duration_checkpoint_key_coverage,
    seconds_to_num_frames,
)

_MODEL_ROOT = os.environ.get("LTX25_MODEL_ROOT")


def test_duration_head_accepts_either_connector_modality() -> None:
    model = LTX25DurationHead(
        video_cross_attention_dim=4,
        audio_cross_attention_dim=2,
        pooler_hidden_dim=4,
        num_pooler_heads=2,
    )

    assert model(video_tokens=torch.zeros(1, 3, 4)).shape == (1,)
    assert model(audio_tokens=torch.zeros(1, 3, 2)).shape == (1,)
    with pytest.raises(ValueError, match="video_tokens or audio_tokens"):
        model()


@pytest.mark.parametrize(
    ("seconds", "frame_rate", "expected"),
    [(0.2, 24.0, 25), (1.0, 24.0, 25), (1.5, 24.0, 33), (100.0, 24.0, 473)],
)
def test_duration_frame_resolution_matches_upstream_causal_grid(
    seconds: float, frame_rate: float, expected: int
) -> None:
    assert seconds_to_num_frames(seconds, frame_rate=frame_rate) == expected


@pytest.mark.skipif(_MODEL_ROOT is None, reason="LTX25_MODEL_ROOT is not configured")
def test_duration_checkpoint_has_full_strict_coverage() -> None:
    checkpoint = Path(_MODEL_ROOT) / "model_patches/ltx-2.5-duration-head-bf16.safetensors"
    model = LTX25DurationHead.from_checkpoint(checkpoint, device="cpu")

    unexpected, missing = ltx25_duration_checkpoint_key_coverage(checkpoint, set(model.state_dict()))

    assert not unexpected
    assert not missing
