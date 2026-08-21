"""Decide an LTX-2.5 performance gate from matched upstream and TeleFuser reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_NOISE_BAND = 0.02
_RUNTIME_FIELDS = ("torch_version", "cuda_version", "natten_version", "natten_has_libnatten", "gpus")


def _load_report(path: Path) -> dict[str, Any]:
    """Load one benchmark report and validate that it is a JSON object."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"LTX-2.5 benchmark report must be a JSON object: {path}")
    return report


def _summary(report: dict[str, Any], *path: str) -> dict[str, Any]:
    value: Any = report
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"LTX-2.5 benchmark report is missing {'/'.join(path)}")
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError(f"LTX-2.5 benchmark summary must be an object: {'/'.join(path)}")
    if not isinstance(value.get("p50_seconds"), (int, float)) or not isinstance(value.get("count"), int):
        raise ValueError(f"LTX-2.5 benchmark summary is malformed: {'/'.join(path)}")
    return value


def _compare_measurement(candidate: float, upstream: float) -> dict[str, float | str]:
    if upstream <= 0:
        raise ValueError("upstream p50 must be positive")
    relative_delta = candidate / upstream - 1.0
    if relative_delta > _NOISE_BAND:
        status = "failed"
    elif relative_delta < -_NOISE_BAND:
        status = "passed"
    else:
        status = "inconclusive"
    return {
        "upstream_p50_seconds": upstream,
        "telefuser_p50_seconds": candidate,
        "relative_delta": relative_delta,
        "status": status,
    }


def compare_benchmarks(
    upstream: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_samples: int = 5,
) -> dict[str, Any]:
    """Apply the frozen request/runtime and 2%-noise performance contracts."""
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if upstream.get("implementation") != "upstream":
        raise ValueError("upstream report implementation must be 'upstream'")
    if candidate.get("implementation") != "telefuser":
        raise ValueError("candidate report implementation must be 'telefuser'")

    request_match = upstream.get("request") == candidate.get("request")
    upstream_runtime = upstream.get("runtime", {})
    candidate_runtime = candidate.get("runtime", {})
    if not isinstance(upstream_runtime, dict) or not isinstance(candidate_runtime, dict):
        raise ValueError("benchmark reports must contain runtime objects")
    runtime_match = {field: upstream_runtime.get(field) == candidate_runtime.get(field) for field in _RUNTIME_FIELDS}

    phases = {
        "cold_end_to_end": (_summary(upstream, "cold", "end_to_end"), _summary(candidate, "cold", "end_to_end")),
        "warm_end_to_end": (_summary(upstream, "warm"), _summary(candidate, "warm")),
    }
    sample_counts = {
        name: {"upstream": upstream_summary["count"], "telefuser": candidate_summary["count"]}
        for name, (upstream_summary, candidate_summary) in phases.items()
    }
    sufficient_samples = all(count >= minimum_samples for counts in sample_counts.values() for count in counts.values())
    measurements = {
        name: _compare_measurement(candidate_summary["p50_seconds"], upstream_summary["p50_seconds"])
        for name, (upstream_summary, candidate_summary) in phases.items()
    }
    statuses = [measurement["status"] for measurement in measurements.values()]
    passed = (
        request_match
        and all(runtime_match.values())
        and sufficient_samples
        and all(status == "passed" for status in statuses)
    )
    return {
        "request_match": request_match,
        "runtime_match": runtime_match,
        "minimum_samples": minimum_samples,
        "sample_counts": sample_counts,
        "sufficient_samples": sufficient_samples,
        "noise_band": _NOISE_BAND,
        "measurements": measurements,
        "passed": passed,
    }


def main() -> None:
    """Compare matched raw LTX-2.5 benchmark reports and write a decision artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("telefuser", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_benchmarks(
        _load_report(args.upstream), _load_report(args.telefuser), minimum_samples=args.minimum_samples
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
