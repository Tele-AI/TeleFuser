from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.validation.validate_lingbot_video_refiner_handoff import exact_handoff_failures, validate_refiner_handoff


def test_refiner_mp4_handoff_tool_matches_source_loader(tmp_path: Path) -> None:
    pytest.importorskip("av")
    from diffusers.utils import export_to_video

    path = tmp_path / "base.mp4"
    frames = [np.full((16, 16, 3), value * 20, dtype=np.uint8) for value in range(10)]
    export_to_video(frames, str(path), fps=48)

    report = validate_refiner_handoff(
        path,
        upstream_root=Path("work_dirs/lingbot-video-master"),
        height=16,
        width=16,
        sample_fps=24,
        vae_tc=4,
    )

    assert report["metadata_match"]
    assert report["source_mp4_to_telefuser_mp4"] == {
        "shape_match": True,
        "dtype_match": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "relative_l2": 0.0,
        "cosine": 1.0,
        "exact_mismatch_count": 0,
    }

    in_memory_video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0).float().div(255.0)
    report_with_memory = validate_refiner_handoff(
        path,
        upstream_root=Path("work_dirs/lingbot-video-master"),
        height=16,
        width=16,
        sample_fps=24,
        vae_tc=4,
        in_memory_video=in_memory_video,
        in_memory_fps=48.0,
    )

    assert report_with_memory["in_memory_metadata_match"]
    assert report_with_memory["in_memory_to_mp4"]["shape_match"]


def test_exact_handoff_failures_accepts_source_compatibility() -> None:
    report = {
        "metadata_match": True,
        "source_mp4_to_telefuser_mp4": {
            "shape_match": True,
            "dtype_match": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "relative_l2": 0.0,
            "exact_mismatch_count": 0,
        },
    }

    assert exact_handoff_failures(report) == []


def test_exact_handoff_failures_ignores_native_handoff_quality_but_reports_source_drift() -> None:
    report = {
        "metadata_match": False,
        "source_mp4_to_telefuser_mp4": {
            "shape_match": True,
            "dtype_match": False,
            "max_abs": 0.25,
            "mean_abs": 0.0,
            "relative_l2": 0.5,
            "exact_mismatch_count": 1,
        },
        "in_memory_to_mp4": {"relative_l2": 0.75},
    }

    assert exact_handoff_failures(report) == [
        "metadata_match",
        "source_mp4_to_telefuser_mp4.dtype_match",
        "source_mp4_to_telefuser_mp4.max_abs",
        "source_mp4_to_telefuser_mp4.relative_l2",
        "source_mp4_to_telefuser_mp4.exact_mismatch_count",
    ]
