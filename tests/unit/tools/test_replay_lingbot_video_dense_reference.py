"""Unit tests for the capture-driven Dense replay validator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPOSITORY_ROOT / "tools" / "validation" / "replay_lingbot_video_dense_reference.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("replay_lingbot_video_dense_reference_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tensor_metrics_reports_exact_replay() -> None:
    module = _load_tool_module()
    tensor = torch.tensor([1.0, -2.0])

    metrics = module.tensor_metrics(tensor, tensor.clone())

    assert metrics["shape_match"]
    assert metrics["max_abs"] == 0.0
    assert metrics["mean_abs"] == 0.0
    assert metrics["relative_l2"] == 0.0
    assert metrics["cosine"] == pytest.approx(1.0)
    assert metrics["exact_mismatch_count"] == 0


def test_tensor_metrics_reports_shape_mismatch() -> None:
    module = _load_tool_module()

    metrics = module.tensor_metrics(torch.ones(2), torch.ones(3))

    assert metrics == {"shape_match": False, "reference_numel": 2, "candidate_numel": 3}


def test_discover_reference_dirs_returns_sorted_capture_runs(tmp_path: Path) -> None:
    module = _load_tool_module()
    late = tmp_path / "t2v" / "example_2" / "run-00"
    early = tmp_path / "t2i" / "example_1" / "run-00"
    late.mkdir(parents=True)
    early.mkdir(parents=True)
    (late / "metadata.json").write_text("{}", encoding="utf-8")
    (early / "metadata.json").write_text("{}", encoding="utf-8")

    references = module.discover_reference_dirs(tmp_path)

    assert references == [early, late]


def test_discover_reference_dirs_rejects_empty_root(tmp_path: Path) -> None:
    module = _load_tool_module()

    with pytest.raises(FileNotFoundError, match="No capture metadata"):
        module.discover_reference_dirs(tmp_path)


def test_exact_replay_failures_accepts_exact_metrics() -> None:
    module = _load_tool_module()

    assert (
        module.exact_replay_failures({"metrics": {"decoded_frames": {"shape_match": True, "exact_mismatch_count": 0}}})
        == []
    )


def test_exact_replay_failures_reports_shape_and_value_mismatches() -> None:
    module = _load_tool_module()
    report = {
        "reports": [
            {"metrics": {"shape": {"shape_match": False}}},
            {"metrics": {"values": {"shape_match": True, "exact_mismatch_count": 1}}},
        ]
    }

    assert module.exact_replay_failures(report) == ["report[0].shape", "report[1].values"]
