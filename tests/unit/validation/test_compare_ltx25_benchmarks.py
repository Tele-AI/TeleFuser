"""Tests for the LTX-2.5 benchmark comparison gate."""

from __future__ import annotations

from tools.validation.compare_ltx25_benchmarks import compare_benchmarks


def _report(implementation: str, *, cold: float, warm: float, count: int = 5) -> dict[str, object]:
    return {
        "implementation": implementation,
        "request": {"seed": 42, "offload": "cpu"},
        "runtime": {
            "torch_version": "2.11.0",
            "cuda_version": "12.8",
            "natten_version": "0.21.6",
            "natten_has_libnatten": True,
            "gpus": [{"name": "H100"}],
        },
        "cold": {"end_to_end": {"p50_seconds": cold, "count": count}},
        "warm": {"p50_seconds": warm, "count": count},
    }


def test_compare_benchmarks_passes_only_for_clear_speedup() -> None:
    upstream = _report("upstream", cold=10.0, warm=8.0)
    candidate = _report("telefuser", cold=9.7, warm=7.7)

    result = compare_benchmarks(upstream, candidate)

    assert result["passed"]
    assert result["measurements"]["cold_end_to_end"]["status"] == "passed"
    assert result["measurements"]["warm_end_to_end"]["status"] == "passed"


def test_compare_benchmarks_marks_noise_band_inconclusive() -> None:
    upstream = _report("upstream", cold=10.0, warm=8.0)
    candidate = _report("telefuser", cold=10.1, warm=8.0)

    result = compare_benchmarks(upstream, candidate)

    assert not result["passed"]
    assert result["measurements"]["cold_end_to_end"]["status"] == "inconclusive"


def test_compare_benchmarks_rejects_mismatched_runtime_or_sample_count() -> None:
    upstream = _report("upstream", cold=10.0, warm=8.0)
    candidate = _report("telefuser", cold=9.0, warm=7.0, count=4)
    candidate["runtime"] = {**candidate["runtime"], "natten_has_libnatten": False}  # type: ignore[index]

    result = compare_benchmarks(upstream, candidate)

    assert not result["passed"]
    assert not result["sufficient_samples"]
    assert not result["runtime_match"]["natten_has_libnatten"]
