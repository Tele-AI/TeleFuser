from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.validation.compare_lingbot_vla_v2_quantization import (
    action_error_metrics,
    compare_quantized_actions,
)


def _artifact(path: Path, action: np.ndarray, *, profile: str = "bf16", input_sha: str = "input") -> Path:
    np.savez(path, canonical_normalized_actions=action)
    metadata = {
        "checkpoint_manifest_sha256": "checkpoint",
        "processor_manifest_sha256": "processor",
        "norm_stats_sha256": "norm",
        "input_sha256": input_sha,
        "seed": 7,
        "num_steps": 10,
        "attention_backend": "eager",
        "moe_backend": "deterministic_torch_reference",
        "quantization": {"profile": profile, "enabled": profile != "bf16"},
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_action_error_metrics_reports_cosine_and_relative_l2() -> None:
    reference = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    candidate = reference + 0.1

    metrics = action_error_metrics(reference, candidate)

    assert metrics["shape"] == [2, 2]
    assert metrics["max_abs"] == pytest.approx(0.1)
    assert 0.0 < metrics["relative_l2"] < 0.1
    assert 0.99 < metrics["cosine"] < 1.0


def test_comparison_is_report_only_without_thresholds(tmp_path: Path) -> None:
    reference = _artifact(tmp_path / "bf16.npz", np.ones((50, 55), dtype=np.float32))
    candidate = _artifact(
        tmp_path / "torchao.npz",
        np.full((50, 55), 1.01, dtype=np.float32),
        profile="torchao-fp8",
    )

    report = compare_quantized_actions(reference, candidate)

    assert report["passed"] is True
    assert report["mode"] == "report_only"
    assert report["metadata"]["candidate_quantization"]["profile"] == "torchao-fp8"


def test_comparison_enforces_threshold_and_exact_replay(tmp_path: Path) -> None:
    reference_action = np.arange(20, dtype=np.float32).reshape(4, 5)
    candidate_action = reference_action + 0.01
    reference = _artifact(tmp_path / "bf16.npz", reference_action)
    candidate = _artifact(tmp_path / "nf4.npz", candidate_action, profile="bnb-nf4")
    replay = _artifact(tmp_path / "nf4_replay.npz", candidate_action.copy(), profile="bnb-nf4")

    report = compare_quantized_actions(
        reference,
        candidate,
        candidate_replay_path=replay,
        min_cosine=0.99,
        max_relative_l2=0.01,
        require_exact_replay=True,
    )

    assert report["passed"] is True
    assert report["mode"] == "thresholded"
    assert report["checks"]["exact_replay"] is True


def test_comparison_rejects_mismatched_input_identity(tmp_path: Path) -> None:
    action = np.ones((2, 3), dtype=np.float32)
    reference = _artifact(tmp_path / "bf16.npz", action, input_sha="one")
    candidate = _artifact(tmp_path / "quant.npz", action, profile="torchao-fp8", input_sha="two")

    with pytest.raises(ValueError, match="identity metadata differs"):
        compare_quantized_actions(reference, candidate)


def test_comparison_rejects_bf16_candidate(tmp_path: Path) -> None:
    action = np.ones((2, 3), dtype=np.float32)
    reference = _artifact(tmp_path / "reference.npz", action)
    candidate = _artifact(tmp_path / "candidate.npz", action)

    with pytest.raises(ValueError, match="non-BF16"):
        compare_quantized_actions(reference, candidate)
