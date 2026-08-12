from __future__ import annotations

import pytest

from tools.validation.benchmark_lingbot_vla_v2_runtime import percentile, summarize
from tools.validation.compare_lingbot_vla_v2_runtime_benchmarks import compare_reports, render_markdown


def _report(implementation: str, latency: float) -> dict:
    summary = {
        "mean_seconds": latency,
        "p50_seconds": latency,
        "p95_seconds": latency,
        "p99_seconds": latency,
    }
    return {
        "benchmark": "lingbot_vla_v2_upstream_telefuser_runtime",
        "implementation": implementation,
        "implementation_commit": "commit",
        "model_root": "/models/vla",
        "qwen3vl_root": "/models/qwen",
        "input_artifact": "/inputs/parity.npz",
        "seed": 7,
        "warmup_runs": 3,
        "measured_runs": 20,
        "device": "cuda:0",
        "device_name": "H100",
        "environment": {
            "python_version": "3.10.12",
            "torch_version": "2.11.0+cu130",
            "cuda_version": "13.0",
            "transformers_version": "4.57.3",
            "platform": "Linux-test",
        },
        "attention_backend": "eager",
        "moe_backend": "robby_triton",
        "load_seconds": 7.0,
        "gpu_peak_allocated_mib": 12000.0,
        "core_model_latency": summary,
        "runtime_request_latency": summary,
    }


def test_summary_includes_p99_and_throughput() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    result = summarize(values)

    assert percentile(values, 0.5) == 2.5
    assert result["p99_seconds"] == pytest.approx(3.97)
    assert result["throughput_requests_per_second"] == 0.4


def test_compare_reports_calculates_telefuser_change() -> None:
    report = compare_reports(_report("official_upstream", 1.0), _report("telefuser", 0.8))

    comparison = report["core_model_latency"]["mean_seconds"]
    assert comparison["telefuser_change_percent"] == pytest.approx(-20.0)
    assert comparison["speedup_upstream_over_telefuser"] == pytest.approx(1.25)
    assert "TeleFuser (ms)" in render_markdown(report)


def test_compare_reports_rejects_backend_mismatch() -> None:
    upstream = _report("official_upstream", 1.0)
    telefuser = _report("telefuser", 0.8)
    telefuser["moe_backend"] = "fused_fallback"

    with pytest.raises(ValueError, match="moe_backend"):
        compare_reports(upstream, telefuser)


def test_compare_reports_rejects_environment_mismatch() -> None:
    upstream = _report("official_upstream", 1.0)
    telefuser = _report("telefuser", 0.8)
    telefuser["environment"] = {**telefuser["environment"], "torch_version": "different"}

    with pytest.raises(ValueError, match="environment"):
        compare_reports(upstream, telefuser)
