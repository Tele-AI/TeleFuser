"""LTX-2.5 latent-state construction contracts."""

from __future__ import annotations

import torch

from telefuser.models.ltx25.diff_vae.types import VideoLatentShape
from telefuser.pipelines.ltx25_distilled.latent import (
    AudioLatentShape,
    AudioLatentTools,
    AudioPatchifier,
    VideoLatentPatchifier,
    VideoLatentTools,
)


def test_video_state_keeps_upstream_float32_pixel_positions() -> None:
    tools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(patch_size=1),
        target_shape=VideoLatentShape(batch=1, channels=128, frames=2, height=1, width=1),
        fps=24.0,
    )

    state = tools.create_initial_state(device=torch.device("cpu"), dtype=torch.bfloat16)

    assert state.positions.dtype == torch.float32
    assert state.keyframes_mask is not None
    torch.testing.assert_close(state.keyframes_mask, torch.tensor([[[1.0], [0.0]]]))
    torch.testing.assert_close(
        state.positions[:, 0, :, :],
        torch.tensor([[[0.0, 1.0 / 24.0], [1.0 / 24.0, 9.0 / 24.0]]]),
    )
    torch.testing.assert_close(
        state.positions[:, 1:, :, :],
        torch.tensor([[[[0.0, 32.0], [0.0, 32.0]], [[0.0, 32.0], [0.0, 32.0]]]]),
    )


def test_audio_state_keeps_upstream_float32_time_positions() -> None:
    tools = AudioLatentTools(
        patchifier=AudioPatchifier(patch_size=1),
        target_shape=AudioLatentShape(batch=1, channels=8, frames=2, mel_bins=16),
    )

    state = tools.create_initial_state(device=torch.device("cpu"), dtype=torch.bfloat16)

    assert state.positions.dtype == torch.float32
    torch.testing.assert_close(state.positions, torch.tensor([[[[0.0, 0.01], [0.01, 0.05]]]]))
