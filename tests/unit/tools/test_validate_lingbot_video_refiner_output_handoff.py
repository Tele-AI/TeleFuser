from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.validation.validate_lingbot_video_refiner_output_handoff import (
    _write_video_tensor,
    build_refiner_handoff_report,
    decoded_frame_quality_metrics,
    side_by_side_frames,
    write_side_by_side_video,
)


def test_final_handoff_report_includes_input_and_final_output_metrics() -> None:
    memory_input = torch.tensor([[[[[0.0]]]]])
    mp4_input = torch.tensor([[[[[0.25]]]]])
    memory_output = torch.full((1, 3, 1, 1, 1), 0.5)
    mp4_output = torch.full((1, 3, 1, 1, 1), 0.75)

    report = build_refiner_handoff_report(
        memory_input=memory_input,
        mp4_input=mp4_input,
        memory_metadata={"frames": 5},
        mp4_metadata={"frames": 5},
        memory_output=memory_output,
        mp4_output=mp4_output,
        memory_seconds=1.25,
        mp4_seconds=1.5,
    )

    assert report["comparison_baseline"] == "source_mp4_round_trip"
    assert report["mp4_round_trip_is_lossy"]
    assert report["metadata_match"]
    assert report["in_memory_to_mp4_input"]["max_abs"] == 0.25
    assert report["final_output"]["max_abs"] == 0.25
    assert report["final_output_quality"]["shape_match"]
    assert report["final_output_quality"]["psnr_db"] is not None
    assert report["memory_refiner_seconds"] == 1.25
    assert report["mp4_refiner_seconds"] == 1.5


def test_decoded_frame_quality_metrics_are_exact_for_identical_video() -> None:
    frames = torch.linspace(0.0, 1.0, 3 * 2 * 4 * 4).reshape(1, 3, 2, 4, 4)

    metrics = decoded_frame_quality_metrics(frames, frames.clone())

    assert metrics == {"shape_match": True, "psnr_db": None, "ssim": 1.0}


def test_side_by_side_frames_concatenates_video_width() -> None:
    reference = torch.zeros(1, 3, 2, 4, 5)
    candidate = torch.ones_like(reference)

    comparison = side_by_side_frames(reference, candidate)

    assert comparison.shape == (1, 3, 2, 4, 10)
    assert torch.equal(comparison[..., :5], reference)
    assert torch.equal(comparison[..., 5:], candidate)


def test_video_writers_pass_normalized_float_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from diffusers import utils

    captured: list[np.ndarray] = []

    def capture_export(frames: list[np.ndarray], output_path: str, fps: int) -> str:
        del output_path, fps
        captured.extend(frames)
        return "unused.mp4"

    monkeypatch.setattr(utils, "export_to_video", capture_export)
    frames = torch.full((1, 3, 2, 2, 2), 0.5)

    write_side_by_side_video(frames, frames, tmp_path / "comparison.mp4")
    _write_video_tensor(frames, tmp_path / "base.mp4")

    assert len(captured) == 4
    assert captured[0].dtype == np.float32
    assert float(captured[0][0, 0, 0]) == 0.5
    assert captured[2].dtype == np.float32
    assert float(captured[2][0, 0, 0]) == 0.5
