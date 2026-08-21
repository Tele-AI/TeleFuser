"""Tests for LTX-2.5 benchmark report aggregation."""

from __future__ import annotations

from tools.validation.benchmark_ltx25_telefuser import summarize_samples


def test_summarize_samples_preserves_raw_samples_and_reports_p50() -> None:
    samples = [{"seconds": 3.0}, {"seconds": 1.0}, {"seconds": 2.0}]

    result = summarize_samples(samples)

    assert result["samples"] == samples
    assert result["count"] == 3
    assert result["min_seconds"] == 1.0
    assert result["max_seconds"] == 3.0
    assert result["mean_seconds"] == 2.0
    assert result["p50_seconds"] == 2.0
