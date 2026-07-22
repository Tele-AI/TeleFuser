"""Unit tests for strict MoE oracle-parity verdicts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPOSITORY_ROOT / "tools" / "validation" / "run_lingbot_video_moe_parity.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("run_lingbot_video_moe_parity_test", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_parity_failures_accepts_zero_drift() -> None:
    module = _load_tool_module()

    assert module.exact_parity_failures({"max_abs": 0.0, "mean_abs": 0.0, "relative_l2": 0.0, "cosine": 1.0}) == []


def test_exact_parity_failures_reports_drift() -> None:
    module = _load_tool_module()

    assert module.exact_parity_failures({"max_abs": 0.25, "mean_abs": 0.0, "relative_l2": 0.125, "cosine": 0.99}) == [
        "max_abs",
        "relative_l2",
    ]
