"""Compare matched upstream and TeleFuser LingBot-VLA v2 runtime benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_LATENCY_KEYS = ("mean_seconds", "p50_seconds", "p95_seconds", "p99_seconds")


def _validate_pair(upstream: dict[str, Any], telefuser: dict[str, Any]) -> None:
    required_equal = (
        "benchmark",
        "model_root",
        "qwen3vl_root",
        "input_artifact",
        "seed",
        "warmup_runs",
        "measured_runs",
        "device",
        "device_name",
        "environment",
        "attention_backend",
        "moe_backend",
    )
    mismatches = [key for key in required_equal if upstream.get(key) != telefuser.get(key)]
    if mismatches:
        raise ValueError(f"benchmark conditions differ for: {', '.join(mismatches)}")
    if upstream.get("implementation") != "official_upstream" or telefuser.get("implementation") != "telefuser":
        raise ValueError("expected official_upstream and telefuser benchmark reports")


def _compare_latency(upstream: dict[str, Any], telefuser: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in _LATENCY_KEYS:
        upstream_value = float(upstream[key])
        telefuser_value = float(telefuser[key])
        result[key] = {
            "upstream_seconds": upstream_value,
            "telefuser_seconds": telefuser_value,
            "telefuser_minus_upstream_seconds": telefuser_value - upstream_value,
            "telefuser_change_percent": (telefuser_value / upstream_value - 1.0) * 100.0,
            "speedup_upstream_over_telefuser": upstream_value / telefuser_value,
        }
    return result


def compare_reports(upstream: dict[str, Any], telefuser: dict[str, Any]) -> dict[str, Any]:
    """Validate matched conditions and produce bounded comparison facts."""
    _validate_pair(upstream, telefuser)
    return {
        "schema_version": 1,
        "comparison": "lingbot_vla_v2_upstream_vs_telefuser_runtime",
        "conditions": {
            key: upstream[key]
            for key in (
                "model_root",
                "qwen3vl_root",
                "input_artifact",
                "seed",
                "warmup_runs",
                "measured_runs",
                "device",
                "device_name",
                "environment",
                "attention_backend",
                "moe_backend",
            )
        },
        "commits": {
            "upstream": upstream["implementation_commit"],
            "telefuser": telefuser["implementation_commit"],
        },
        "core_model_latency": _compare_latency(upstream["core_model_latency"], telefuser["core_model_latency"]),
        "runtime_request_latency": _compare_latency(
            upstream["runtime_request_latency"], telefuser["runtime_request_latency"]
        ),
        "load_seconds": {
            "upstream": upstream["load_seconds"],
            "telefuser": telefuser["load_seconds"],
            "comparable": False,
            "reason": "The two implementations construct processors and framework objects at different boundaries.",
        },
        "gpu_peak_allocated_mib": {
            "upstream": upstream["gpu_peak_allocated_mib"],
            "telefuser": telefuser["gpu_peak_allocated_mib"],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact human-readable benchmark table."""
    lines = [
        "# LingBot-VLA v2 Upstream vs TeleFuser Runtime",
        "",
        "| Scope | Metric | Upstream (ms) | TeleFuser (ms) | Change |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope_key, scope_label in (
        ("core_model_latency", "Core model"),
        ("runtime_request_latency", "Runtime request"),
    ):
        for metric in ("mean_seconds", "p50_seconds", "p95_seconds", "p99_seconds"):
            values = report[scope_key][metric]
            lines.append(
                f"| {scope_label} | {metric.removesuffix('_seconds')} | "
                f"{values['upstream_seconds'] * 1000:.3f} | "
                f"{values['telefuser_seconds'] * 1000:.3f} | "
                f"{values['telefuser_change_percent']:+.2f}% |"
            )
    conditions = report["conditions"]
    lines.extend(
        [
            "",
            f"GPU: `{conditions['device_name']}`. Warmup: {conditions['warmup_runs']}; measured runs: "
            f"{conditions['measured_runs']}; attention: `{conditions['attention_backend']}`; "
            f"MoE: `{conditions['moe_backend']}`.",
            "",
            "Negative change means TeleFuser is faster. Core model excludes transfers; runtime request includes "
            "CPU/GPU transfers, seeded noise construction, validation, and CPU action delivery. Image preprocessing "
            "is excluded on both sides by using the frozen parity tensors.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--telefuser", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    upstream = json.loads(args.upstream.read_text(encoding="utf-8"))
    telefuser = json.loads(args.telefuser.read_text(encoding="utf-8"))
    report = compare_reports(upstream, telefuser)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": True, "output": str(args.output_json)}))


if __name__ == "__main__":
    main()
