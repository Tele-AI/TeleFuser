from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validation import analyze_abot_serving_trace as analysis


def _prometheus_snapshot(*, items: float, executions: float, b1: float, b2_cumulative: float) -> str:
    return "\n".join(
        (
            f"telefuser_serving_batch_items_total {items}",
            f"telefuser_serving_batches_total {executions}",
            f'telefuser_serving_batch_size_bucket{{le="1.0"}} {b1}',
            f'telefuser_serving_batch_size_bucket{{le="2.0"}} {b2_cumulative}',
            f'telefuser_serving_batch_size_bucket{{le="3.0"}} {b2_cumulative}',
            f'telefuser_serving_batch_size_bucket{{le="4.0"}} {b2_cumulative}',
            "",
        )
    )


def test_batch_report_distinguishes_item_and_execution_percentages(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "serving_metrics"
    prometheus_dir = metrics_dir / "prometheus"
    prometheus_dir.mkdir(parents=True)
    (prometheus_dir / "000001.prom").write_text(
        _prometheus_snapshot(items=0, executions=0, b1=0, b2_cumulative=0), encoding="utf-8"
    )
    # Ten B=1 executions plus two B=2 executions produce fourteen items but
    # only twelve executions.  The B=2 item and execution shares differ.
    (prometheus_dir / "000002.prom").write_text(
        _prometheus_snapshot(items=14, executions=12, b1=10, b2_cumulative=14), encoding="utf-8"
    )
    records = [
        {"sequence": 1, "offset_seconds": 0.0, "prometheus": {"path": "prometheus/000001.prom"}},
        {"sequence": 2, "offset_seconds": 10.0, "prometheus": {"path": "prometheus/000002.prom"}},
    ]
    (metrics_dir / "serving-metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    snapshots = analysis.load_snapshots(metrics_dir)
    summary = analysis.summarize_interval(
        phase="test",
        requested_start_seconds=0,
        requested_end_seconds=10,
        before=snapshots[0],
        after=snapshots[1],
    )

    assert summary.batch_items == 14
    assert summary.batch_executions == 12
    assert summary.b1_execution_equivalents == 10
    assert summary.b2_execution_equivalents == 2
    assert summary.b2_item_share_percent == pytest.approx(4 / 14 * 100)
    assert summary.b2_execution_share_percent == pytest.approx(2 / 12 * 100)
    assert summary.mean_execution_batch_size == pytest.approx(14 / 12)
