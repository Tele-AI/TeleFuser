#!/usr/bin/env python3
"""Summarize and visualize a captured ABot-World serving metrics trace.

The serving collector writes cumulative Prometheus snapshots once per sample.
This tool turns those snapshots into a batch-execution table and a small
dependency-light PNG suitable for an experiment notebook or paper appendix.

The distinction between *items* and *executions* matters here: every member of
a coalesced B=N execution emits one model-output event.  Therefore the
``batch_size`` histogram counts session-chunk items, whereas
``telefuser_serving_batches_total`` is incremented by ``1/N`` per member and
is an execution-equivalent counter.  The report makes both denominators
explicit rather than accidentally calling the B=2 item percentage a B=2
execution percentage.

Example:

    PYTHONPATH=$PWD python tools/validation/analyze_abot_serving_trace.py \
      --serving-metrics-dir /path/to/serving_metrics \
      --result /path/to/result.json \
      --gpu-metrics /path/to/gpu_metrics/gpu-metrics.jsonl \
      --output-dir /tmp/abot-peak16-analysis

It never contacts a server or GPU.  ``Pillow`` is used only for the PNG; JSON
and CSV summaries are still written when it is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

_BATCH_BUCKET_LABELS = ("1.0", "2.0", "3.0", "4.0")
_BATCH_SIZES = (1, 2, 3, 4)


@dataclass(frozen=True)
class BatchCounters:
    """Cumulative counters at one Prometheus scrape point."""

    offset_seconds: float
    sequence: int
    batch_items: float
    batch_executions: float
    bucket_items: tuple[float, float, float, float]


@dataclass(frozen=True)
class PhaseSummary:
    """A counter-delta summary for a named workload phase."""

    phase: str
    requested_start_seconds: float
    requested_end_seconds: float
    sampled_start_seconds: float
    sampled_end_seconds: float
    batch_items: float
    batch_executions: float
    b1_items: float
    b2_items: float
    b3_items: float
    b4_items: float
    b_gt4_items: float
    b1_execution_equivalents: float
    b2_execution_equivalents: float
    b3_execution_equivalents: float
    b4_execution_equivalents: float
    b2_execution_share_percent: float
    b2_item_share_percent: float
    mean_execution_batch_size: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serving-metrics-dir",
        required=True,
        type=Path,
        help="Directory containing serving-metrics.jsonl and prometheus/ snapshots.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        help="Optional benchmark result.json; its phase boundaries split the report.",
    )
    parser.add_argument(
        "--gpu-metrics",
        type=Path,
        help="Optional gpu-metrics.jsonl captured on the same monotonic timeline.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty directory for summary.json, phase-batches.csv, and PNG.",
    )
    return parser.parse_args(argv)


def _metric_value(metrics: dict[str, float], name: str) -> float:
    """Read a numeric exposition line, treating not-yet-created metrics as zero."""

    return float(metrics.get(name, 0.0))


def _read_prometheus_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        name, value_text = line.rsplit(" ", 1)
        if not name.startswith("telefuser_serving_"):
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[name] = value
    return metrics


def load_snapshots(serving_metrics_dir: Path) -> list[BatchCounters]:
    """Load valid Prometheus snapshots in capture order."""

    root = serving_metrics_dir.expanduser().resolve()
    jsonl_path = root / "serving-metrics.jsonl"
    if not jsonl_path.is_file():
        raise ValueError(f"missing serving metrics JSONL: {jsonl_path}")
    snapshots: list[BatchCounters] = []
    for raw in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        prom = record.get("prometheus")
        if not isinstance(prom, dict) or not isinstance(prom.get("path"), str):
            continue
        prom_path = root / prom["path"]
        if not prom_path.is_file():
            continue
        metrics = _read_prometheus_metrics(prom_path)
        buckets = tuple(
            _metric_value(metrics, f'telefuser_serving_batch_size_bucket{{le="{label}"}}')
            for label in _BATCH_BUCKET_LABELS
        )
        snapshots.append(
            BatchCounters(
                offset_seconds=float(record["offset_seconds"]),
                sequence=int(record["sequence"]),
                batch_items=_metric_value(metrics, "telefuser_serving_batch_items_total"),
                batch_executions=_metric_value(metrics, "telefuser_serving_batches_total"),
                bucket_items=buckets,
            )
        )
    if len(snapshots) < 2:
        raise ValueError(f"need at least two valid Prometheus snapshots under {root}")
    if any(later.offset_seconds < earlier.offset_seconds for earlier, later in zip(snapshots, snapshots[1:])):
        raise ValueError("serving metrics snapshots are not ordered by monotonic offset")
    return snapshots


def _nearest_snapshot(snapshots: list[BatchCounters], offset_seconds: float) -> BatchCounters:
    return min(snapshots, key=lambda item: abs(item.offset_seconds - offset_seconds))


def _counter_delta(after: float, before: float, label: str) -> float:
    delta = after - before
    if delta < -1e-6:
        raise ValueError(f"counter reset or reversed phase for {label}: {before} -> {after}")
    return max(0.0, delta)


def _bucket_deltas(after: BatchCounters, before: BatchCounters) -> tuple[float, float, float, float, float]:
    cumulative = [
        _counter_delta(current, prior, f"batch histogram <= {size}")
        for size, current, prior in zip(_BATCH_SIZES, after.bucket_items, before.bucket_items)
    ]
    b1 = cumulative[0]
    b2 = cumulative[1] - cumulative[0]
    b3 = cumulative[2] - cumulative[1]
    b4 = cumulative[3] - cumulative[2]
    items = _counter_delta(after.batch_items, before.batch_items, "batch items")
    b_gt4 = items - cumulative[3]
    if min(b2, b3, b4, b_gt4) < -1e-6:
        raise ValueError("batch histogram is not monotonically cumulative")
    return b1, max(0.0, b2), max(0.0, b3), max(0.0, b4), max(0.0, b_gt4)


def summarize_interval(
    *,
    phase: str,
    requested_start_seconds: float,
    requested_end_seconds: float,
    before: BatchCounters,
    after: BatchCounters,
) -> PhaseSummary:
    """Compute observed execution-equivalent batch distribution over an interval."""

    items = _counter_delta(after.batch_items, before.batch_items, "batch items")
    executions = _counter_delta(after.batch_executions, before.batch_executions, "batch executions")
    b1, b2, b3, b4, b_gt4 = _bucket_deltas(after, before)
    b1 + b2 / 2.0 + b3 / 3.0 + b4 / 4.0
    # The metric contains explicit 1--4 buckets.  Items above four do not have
    # a one-to-one bucket (5/6 are together), so do not invent an exact count.
    # They are intentionally left out of the named B=1..4 execution bars.
    mean_batch = items / executions if executions else 0.0
    return PhaseSummary(
        phase=phase,
        requested_start_seconds=requested_start_seconds,
        requested_end_seconds=requested_end_seconds,
        sampled_start_seconds=before.offset_seconds,
        sampled_end_seconds=after.offset_seconds,
        batch_items=items,
        batch_executions=executions,
        b1_items=b1,
        b2_items=b2,
        b3_items=b3,
        b4_items=b4,
        b_gt4_items=b_gt4,
        b1_execution_equivalents=b1,
        b2_execution_equivalents=b2 / 2.0,
        b3_execution_equivalents=b3 / 3.0,
        b4_execution_equivalents=b4 / 4.0,
        b2_execution_share_percent=(100.0 * (b2 / 2.0) / executions if executions else 0.0),
        b2_item_share_percent=(100.0 * b2 / items if items else 0.0),
        mean_execution_batch_size=mean_batch,
    )


def _load_phase_bounds(result_path: Path) -> list[tuple[str, float, float]]:
    payload = json.loads(result_path.expanduser().resolve().read_text(encoding="utf-8"))
    phase_results = payload.get("phase_results")
    if not isinstance(phase_results, list):
        raise ValueError(f"result has no phase_results list: {result_path}")
    bounds: list[tuple[str, float, float]] = []
    for entry in phase_results:
        if not isinstance(entry, dict):
            continue
        name = entry.get("phase")
        start = entry.get("started_offset_seconds")
        end = entry.get("completed_offset_seconds")
        if isinstance(name, str) and isinstance(start, int | float) and isinstance(end, int | float):
            bounds.append((name, float(start), float(end)))
    if not bounds:
        raise ValueError(f"result has no usable phase bounds: {result_path}")
    return bounds


def _summaries_from_result(snapshots: list[BatchCounters], result_path: Path | None) -> list[PhaseSummary]:
    if result_path is None:
        return [
            summarize_interval(
                phase="entire_capture",
                requested_start_seconds=snapshots[0].offset_seconds,
                requested_end_seconds=snapshots[-1].offset_seconds,
                before=snapshots[0],
                after=snapshots[-1],
            )
        ]
    summaries: list[PhaseSummary] = []
    for name, start, end in _load_phase_bounds(result_path):
        summaries.append(
            summarize_interval(
                phase=name,
                requested_start_seconds=start,
                requested_end_seconds=end,
                before=_nearest_snapshot(snapshots, start),
                after=_nearest_snapshot(snapshots, end),
            )
        )
    return summaries


def _load_gpu_phase_means(
    gpu_metrics_path: Path | None,
    summaries: Iterable[PhaseSummary],
) -> dict[str, dict[str, float]]:
    """Return only physical-GPU facts present in the optional NVML artifact."""

    if gpu_metrics_path is None:
        return {}
    samples = [
        json.loads(raw)
        for raw in gpu_metrics_path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    result: dict[str, dict[str, float]] = {}
    for summary in summaries:
        selected = [
            gpu
            for sample in samples
            if summary.sampled_start_seconds <= float(sample.get("offset_seconds", -1)) < summary.sampled_end_seconds
            for gpu in sample.get("gpus", [])
            if isinstance(gpu, dict)
        ]
        if not selected:
            continue
        utilization = [float(gpu["gpu_utilization_percent"]) for gpu in selected]
        memory_gib = [float(gpu["memory_used_bytes"]) / (1024**3) for gpu in selected]
        result[summary.phase] = {
            "physical_gpu_samples": float(len(selected)),
            "mean_gpu_utilization_percent": sum(utilization) / len(utilization),
            "mean_memory_used_gib": sum(memory_gib) / len(memory_gib),
        }
    return result


def _write_csv(path: Path, summaries: Iterable[PhaseSummary]) -> None:
    rows = [asdict(summary) for summary in summaries]
    if not rows:
        return
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _draw_png(path: Path, summaries: list[PhaseSummary]) -> str | None:
    """Render B=1..4 execution-equivalent proportions using Pillow when present."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    width = max(960, 190 + 150 * len(summaries))
    height = 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title = "ABot serving: observed model-batch execution distribution"
    draw.text((28, 20), title, fill="#111827", font=font)
    draw.text(
        (28, 40),
        "Bars use execution-equivalents (B=2 items / 2); B>4 is omitted from colored B=1..4 bars.",
        fill="#4b5563",
        font=font,
    )
    colors = ("#e67e22", "#3b82f6", "#16a34a", "#8b5cf6")
    labels = ("B=1", "B=2", "B=3", "B=4")
    chart_left, chart_top, chart_bottom = 72, 98, 430
    chart_height = chart_bottom - chart_top
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="#6b7280", width=1)
    draw.line((chart_left, chart_bottom, width - 30, chart_bottom), fill="#6b7280", width=1)
    for percent in range(0, 101, 20):
        y = chart_bottom - chart_height * percent / 100.0
        draw.line((chart_left, y, width - 30, y), fill="#e5e7eb", width=1)
        draw.text((20, y - 5), f"{percent}%", fill="#4b5563", font=font)
    group_width = max(88, (width - chart_left - 40) / max(1, len(summaries)))
    for index, summary in enumerate(summaries):
        values = (
            summary.b1_execution_equivalents,
            summary.b2_execution_equivalents,
            summary.b3_execution_equivalents,
            summary.b4_execution_equivalents,
        )
        total = summary.batch_executions
        x = chart_left + index * group_width + group_width * 0.16
        usable_width = group_width * 0.68
        cursor = chart_bottom
        for value, color in zip(values, colors):
            height_px = chart_height * value / total if total else 0.0
            draw.rectangle((x, cursor - height_px, x + usable_width, cursor), fill=color)
            cursor -= height_px
        short_name = summary.phase[:18]
        draw.text((x, chart_bottom + 10), short_name, fill="#111827", font=font)
        draw.text((x, chart_bottom + 24), f"mean B={summary.mean_execution_batch_size:.3f}", fill="#4b5563", font=font)
        draw.text(
            (x, chart_bottom + 38), f"B2 exec={summary.b2_execution_share_percent:.1f}%", fill="#4b5563", font=font
        )
    legend_x = 30
    for label, color in zip(labels, colors):
        draw.rectangle((legend_x, 485, legend_x + 13, 498), fill=color)
        draw.text((legend_x + 18, 486), label, fill="#111827", font=font)
        legend_x += 100
    draw.text(
        (30, 520),
        "A half execution-equivalent can appear at a capture boundary because each B=2 member is forwarded separately.",
        fill="#4b5563",
        font=font,
    )
    image.save(path)
    return str(path)


def _print_table(summaries: list[PhaseSummary]) -> None:
    print("phase                       executions  mean-B  B1-exec  B2-exec  B2-exec%  B2-item%  B>=3-items")
    for item in summaries:
        print(
            f"{item.phase[:27]:27} {item.batch_executions:10.1f} {item.mean_execution_batch_size:7.3f}"
            f" {item.b1_execution_equivalents:8.1f} {item.b2_execution_equivalents:8.1f}"
            f" {item.b2_execution_share_percent:9.2f}% {item.b2_item_share_percent:8.2f}%"
            f" {item.b3_items + item.b4_items + item.b_gt4_items:11.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = load_snapshots(args.serving_metrics_dir)
    summaries = _summaries_from_result(snapshots, args.result)
    whole_capture = summarize_interval(
        phase="entire_capture",
        requested_start_seconds=snapshots[0].offset_seconds,
        requested_end_seconds=snapshots[-1].offset_seconds,
        before=snapshots[0],
        after=snapshots[-1],
    )
    gpu_phase_means = _load_gpu_phase_means(args.gpu_metrics, summaries)
    _write_csv(output_dir / "phase-batches.csv", summaries)
    png_path = _draw_png(output_dir / "batch-distribution.png", summaries)
    report = {
        "schema_version": 1,
        "metric_semantics": {
            "batch_items": "Session-chunk items; every member of a coalesced execution contributes one.",
            "batch_executions": "Execution-equivalent total; serving increments 1/B per emitted B-member.",
            "batch_histogram": "Counts items, not executions. B=2 execution share is B2-items / 2 / executions.",
        },
        "capture": {
            "first_offset_seconds": snapshots[0].offset_seconds,
            "last_offset_seconds": snapshots[-1].offset_seconds,
            "valid_prometheus_snapshots": len(snapshots),
        },
        "entire_capture": asdict(whole_capture),
        "phases": [asdict(summary) for summary in summaries],
        "physical_gpu_phase_means": gpu_phase_means,
        "png": png_path,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_table([whole_capture])
    if args.result is not None:
        print()
        _print_table(summaries)
    print(f"Wrote analysis: {output_dir}")
    if png_path is None:
        print("Pillow unavailable: wrote JSON/CSV but no PNG.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ABot serving trace analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
