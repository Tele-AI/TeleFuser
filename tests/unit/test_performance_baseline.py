"""CPU tests for the performance baseline schema and regression gates."""

from __future__ import annotations

import json

import pytest

from telefuser.metrics.baseline import (
    BaselineFile,
    BenchmarkRecord,
    OutputConsistency,
    RegressionRule,
    compare_baselines,
    compare_numeric_outputs,
    compare_output_hashes,
)


def _records(*, latency: float = 100.0, fps: float = 20.0) -> list[BenchmarkRecord]:
    return [
        BenchmarkRecord(
            workload_id="wan-t2v",
            kind="t2v",
            first_frame_latency_ms=latency,
            chunk_latencies_ms=(10.0, 20.0, 30.0),
            steady_fps=fps,
            gpu_peak_memory_mb=4096.0,
            throughput_per_second=2.0,
            output_consistency=OutputConsistency(
                compared=1,
                exact_matches=1,
                max_abs_error=0.0,
                mean_abs_error=0.0,
                min_cosine_similarity=1.0,
            ),
        ),
        BenchmarkRecord(
            workload_id="wan-t2v",
            kind="t2v",
            first_frame_latency_ms=latency + 10,
            chunk_latencies_ms=(20.0, 40.0),
            steady_fps=fps - 1,
            gpu_peak_memory_mb=4096.0,
            throughput_per_second=2.0,
        ),
        BenchmarkRecord(
            workload_id="wan-t2v",
            kind="t2v",
            first_frame_latency_ms=999.0,
            warmup=True,
        ),
    ]


def test_baseline_summarizes_metrics_and_excludes_warmup(tmp_path):
    baseline = BaselineFile.from_records(_records(), project_commit="abc123")
    summary = baseline.workloads["wan-t2v"]

    assert summary.measured_records == 2
    assert summary.first_frame_latency_ms is not None
    assert summary.first_frame_latency_ms.count == 2
    assert summary.chunk_latency_ms is not None
    assert summary.chunk_latency_ms.p50 == pytest.approx(20.0)
    assert summary.gpu_peak_memory_mb is not None
    assert summary.gpu_peak_memory_mb.maximum == 4096.0

    path = tmp_path / "baseline.json"
    baseline.write(path)
    loaded = BaselineFile.read(path)
    assert loaded.project_commit == "abc123"
    assert loaded.to_dict() == json.loads(path.read_text())


def test_record_aliases_normalize_aiperf_fields():
    record = BenchmarkRecord.from_mapping(
        {
            "workload": "stream",
            "workload_kind": "realtime_stream",
            "first_frame_latency_s": 0.25,
            "stream_fps": 24,
            "peak_memory_mib": 100,
            "chunk_measurements": [
                {"compute_seconds": 0.01},
                {"latency_ms": 20},
                {"compute_seconds": 99, "is_warmup": True},
            ],
        }
    )

    assert record.first_frame_latency_ms == 250.0
    assert record.steady_fps == 24.0
    assert record.gpu_peak_memory_mb == pytest.approx(104.8576)
    assert record.chunk_latencies_ms == (10.0, 20.0)


def test_regression_detects_latency_fps_and_output_changes():
    baseline = BaselineFile.from_records(_records())
    candidate = BaselineFile.from_records(_records(latency=130.0, fps=17.0))

    report = compare_baselines(baseline, candidate, rule=RegressionRule(relative_tolerance=0.05))

    assert not report.passed
    assert {issue.metric for issue in report.issues} >= {
        "first_frame_latency_ms.p95",
        "steady_fps.p50",
    }


def test_regression_detects_success_rate_drop():
    baseline = BaselineFile.from_records(_records())
    failed_records = _records()
    failed_records[1] = BenchmarkRecord(
        workload_id="wan-t2v",
        kind="t2v",
        success=False,
    )
    candidate = BaselineFile.from_records(failed_records)

    report = compare_baselines(baseline, candidate)

    assert any(issue.metric == "success_rate" for issue in report.issues)


def test_cli_records_require_supported_workload_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        BenchmarkRecord(workload_id="bad", kind="unknown")


def test_baseline_rejects_runs_with_only_warmup_records():
    with pytest.raises(ValueError, match="at least one measured record"):
        BaselineFile.from_records([BenchmarkRecord(workload_id="demo", kind="t2i", warmup=True)])


def test_output_consistency_helpers_cover_numeric_and_digest_outputs():
    numeric = compare_numeric_outputs([0.0, 1.0, 2.0], [0.0, 1.0, 2.00001])
    assert numeric.compared == 1
    assert numeric.exact_matches == 0
    assert numeric.max_abs_error == pytest.approx(1e-5)
    assert numeric.min_cosine_similarity is not None
    assert numeric.min_cosine_similarity > 0.9999

    digest = compare_output_hashes("reference", "reference")
    assert digest.exact_match_rate == 1.0
