"""LTX-2.5 artifact-comparison contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.validation.compare_ltx25_artifacts import compare_captures


def _write_capture(
    root: Path,
    artifacts: dict[str, torch.Tensor],
    *,
    checkpoints: dict[str, object] | None = None,
    audio: dict[str, object] | None = None,
) -> None:
    root.mkdir(exist_ok=True)
    descriptors = {}
    for name, value in artifacts.items():
        path = root / f"{name}.pt"
        torch.save(value, path)
        descriptors[name] = {"path": path.name}
    (root / "capture_manifest.json").write_text(
        json.dumps(
            {
                "request": {"seed": 7},
                "checkpoints": checkpoints or {},
                "audio": audio or {},
                "artifacts": descriptors,
            }
        ),
        encoding="utf-8",
    )


def test_comparison_requires_exact_noise_and_accepts_close_float_artifacts(tmp_path: Path) -> None:
    golden = tmp_path / "golden"
    candidate = tmp_path / "candidate"
    _write_capture(golden, {"stage1_initial_video_noise": torch.ones(2), "latent": torch.ones(2)})
    _write_capture(candidate, {"stage1_initial_video_noise": torch.ones(2), "latent": torch.tensor([1.0, 1.0001])})
    report = compare_captures(golden, candidate)
    assert report["passed"]

    _write_capture(candidate, {"stage1_initial_video_noise": torch.tensor([1.0, 0.0]), "latent": torch.ones(2)})
    report = compare_captures(golden, candidate)
    assert not report["passed"]
    assert "stage1_initial_video_noise" in report["failures"]


def test_comparison_accepts_exact_non_contract_tensors_without_metric_margin(tmp_path: Path) -> None:
    golden = tmp_path / "golden"
    candidate = tmp_path / "candidate"
    _write_capture(golden, {"decoded_rgb": torch.full((1024,), 0.5, dtype=torch.bfloat16)})
    _write_capture(candidate, {"decoded_rgb": torch.full((1024,), 0.5, dtype=torch.bfloat16)})

    report = compare_captures(golden, candidate, cosine_threshold=1.1)

    assert report["passed"]
    assert report["tensors"]["decoded_rgb"]["exact"]
    assert report["tensors"]["decoded_rgb"]["ssim"] == 1.0


def test_comparison_requires_matching_checkpoint_and_audio_contracts(tmp_path: Path) -> None:
    """Golden checkpoint and decoded-audio metadata are exact capture contracts."""
    golden = tmp_path / "golden"
    candidate = tmp_path / "candidate"
    artifacts = {"decoded_waveform": torch.ones(2, 32)}
    _write_capture(
        golden,
        artifacts,
        checkpoints={"transformer": {"sha256": "golden"}},
        audio={"sample_rate": 48000},
    )
    _write_capture(
        candidate,
        artifacts,
        checkpoints={"transformer": {"sha256": "candidate"}},
        audio={"sample_rate": 44100},
    )

    report = compare_captures(golden, candidate)

    assert not report["passed"]
    assert not report["checkpoint_match"]
    assert not report["audio_match"]
    assert "checkpoints" in report["failures"]
    assert "audio" in report["failures"]


def test_comparison_allows_upstream_only_prompt_diagnostics(tmp_path: Path) -> None:
    golden = tmp_path / "golden"
    candidate = tmp_path / "candidate"
    _write_capture(
        golden,
        {
            "latent": torch.ones(2),
            "gemma_token_ids": torch.ones(2, dtype=torch.long),
            "gemma_hidden_state_0": torch.ones(2),
        },
    )
    _write_capture(candidate, {"latent": torch.ones(2)})

    report = compare_captures(golden, candidate)

    assert report["passed"]
    assert not report["artifact_set_match"]
    assert report["artifact_contract_match"]
    assert not report["missing_required_from_candidate"]
