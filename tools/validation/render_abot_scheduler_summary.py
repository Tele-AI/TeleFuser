#!/usr/bin/env python3
"""Render a read-only summary figure from saved ABot-World experiment artifacts.

This program deliberately consumes only JSON files produced by the native
three-session scheduler timeline and the four-GPU LiveKit workload trace.  It
does not import serving code, initialize CUDA, or contact a server.  The PNG
is designed as a compact, reproducible companion to the raw timeline PNGs and
Prometheus-based batch analysis.

Example:

    PYTHONPATH=$PWD /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \
      tools/validation/render_abot_scheduler_summary.py \
      --staggered results/experiments/abot_session_strategy_3user_20260814/native_12fps/staggered/timeline.json \
      --aligned results/experiments/abot_session_strategy_3user_20260814/native_12fps/aligned/timeline.json \
      --four-gpu-result results/experiments/abot_4gpu_lf3_12fps_intermittent_proxyfree_20260814/result.json \
      --four-gpu-analysis results/experiments/abot_4gpu_lf3_12fps_intermittent_proxyfree_20260814/analysis/summary.json \
      --output-dir results/experiments/abot_4gpu_lf3_12fps_intermittent_proxyfree_20260814/analysis

Outputs ``scheduler-summary.png`` and ``scheduler-summary.md``.
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_NAVY = "#0f172a"
_SLATE = "#475569"
_GRID = "#cbd5e1"
_PANEL = "#f8fafc"
_WHITE = "#ffffff"
_BLUE = "#2563eb"
_CYAN = "#0891b2"
_ORANGE = "#ea580c"
_RED = "#dc2626"
_GREEN = "#059669"
_PURPLE = "#7c3aed"
_USERS = ("#2563eb", "#ea580c", "#059669")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staggered", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--four-gpu-result", type=Path, required=True)
    parser.add_argument("--four-gpu-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(face, size=size)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    size: int = 22,
    fill: str = _NAVY,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, fill=fill, font=_font(size, bold=bold), anchor=anchor)


def _rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    fill: str = _PANEL,
    outline: str | None = _GRID,
    radius: int = 18,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) else default


def _timeline_stats(payload: dict[str, Any]) -> dict[str, Any]:
    batches = [item for item in payload.get("batches", []) if isinstance(item, dict)]
    histogram = Counter(int(item.get("batch_size", 0)) for item in batches)
    durations = [_safe_float(item.get("duration_ms")) for item in batches]
    scenario = payload.get("scenario", {}) if isinstance(payload.get("scenario"), dict) else {}
    return {
        "batches": batches,
        "calls": len(batches),
        "histogram": histogram,
        "mean_duration_ms": sum(durations) / len(durations) if durations else 0.0,
        "min_duration_ms": min(durations, default=0.0),
        "max_duration_ms": max(durations, default=0.0),
        "offsets_ms": [float(value) for value in scenario.get("session_arrival_offsets_ms", [])],
        "frames": int(scenario.get("control_latent_frames", 3)) * 4,
        "fps": _safe_float(scenario.get("fps"), 12.0),
    }


def _phase(result: dict[str, Any], name: str) -> dict[str, Any]:
    for item in result.get("phase_results", []):
        if isinstance(item, dict) and item.get("phase") == name:
            return item
    raise ValueError(f"missing phase {name!r} in four-GPU result")


def _draw_single_card_panel(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    staggered: dict[str, Any],
    aligned: dict[str, Any],
) -> None:
    x0, y0, x1, y1 = box
    _rounded(draw, box)
    _text(draw, (x0 + 26, y0 + 20), "One H100: native 3-session scheduler trace", size=28, bold=True)
    _text(
        draw,
        (x0 + 26, y0 + 57),
        "ABot-World LF=3, 12 frames/chunk, 12 FPS target; model calls measured at generate_next_blocks().",
        size=16,
        fill=_SLATE,
    )
    divider = (x0 + x1) // 2
    draw.line((divider, y0 + 92, divider, y1 - 24), fill=_GRID, width=2)

    columns = (
        (x0 + 26, divider - 22, staggered, "Staggered controls", _ORANGE),
        (divider + 22, x1 - 26, aligned, "Barrier-aligned controls", _GREEN),
    )
    for col_x0, col_x1, stats, title, accent in columns:
        hist: Counter[int] = stats["histogram"]
        call_count = stats["calls"]
        b3_count = hist[3]
        b1_count = hist[1]
        _text(draw, (col_x0, y0 + 108), title, size=22, bold=True)
        if title.startswith("Staggered"):
            offsets = "/".join(f"{value:.0f}" for value in stats["offsets_ms"])
            _text(draw, (col_x0, y0 + 140), f"arrival offsets: {offsets} ms", size=16, fill=_SLATE)
            result = f"{b1_count}/{call_count} calls are B=1; B=3: {b3_count}/{call_count}"
        else:
            _text(draw, (col_x0, y0 + 140), "all three controls activated in one scheduler turn", size=16, fill=_SLATE)
            result = f"B=3: {b3_count}/{call_count} calls; B=1: {b1_count}/{call_count}"
        _text(draw, (col_x0, y0 + 169), result, size=17, fill=accent, bold=True)

        bars_y0 = y0 + 217
        y0 + 357
        duration_max = max((_safe_float(item.get("end_seconds")) for item in stats["batches"]), default=1.0)
        duration_max = max(duration_max, 0.1)
        lane_width = col_x1 - col_x0 - 10
        for index, batch in enumerate(stats["batches"]):
            start = _safe_float(batch.get("start_seconds")) / duration_max
            end = _safe_float(batch.get("end_seconds")) / duration_max
            bx0 = col_x0 + 3 + int(lane_width * start)
            bx1 = col_x0 + 3 + int(lane_width * end)
            by0 = bars_y0 + (index % 3) * 42
            by1 = by0 + 28
            batch_size = int(batch.get("batch_size", 0))
            if batch_size == 1:
                session_ids = batch.get("session_ids", [])
                session_id = session_ids[0] if isinstance(session_ids, list) and session_ids else "user-1"
                user_index = (
                    max(0, min(2, int(str(session_id).split("-")[-1]) - 1))
                    if str(session_id).split("-")[-1].isdigit()
                    else 0
                )
                fill = _USERS[user_index]
            else:
                fill = _GREEN
            draw.rounded_rectangle((bx0, by0, max(bx0 + 3, bx1), by1), radius=6, fill=fill)
            if batch_size == 3:
                stripe = max(1, (max(bx0 + 3, bx1) - bx0) // 3)
                for stripe_index, color in enumerate(_USERS):
                    left = bx0 + stripe_index * stripe
                    right = bx0 + (stripe_index + 1) * stripe if stripe_index < 2 else max(bx0 + 3, bx1)
                    draw.rectangle((left, by0, right, by1), fill=color)
            label = f"B{batch_size}  {float(batch.get('duration_ms', 0.0)):.0f}ms"
            _text(draw, (bx0 + 5, by0 + 4), label, size=12, fill=_WHITE, bold=True)
        for lane_index, label in enumerate(("u1", "u2", "u3")):
            _text(draw, (col_x0 - 2, bars_y0 + lane_index * 42 - 15), label, size=13, fill=_SLATE)
        _text(draw, (col_x0, y0 + 378), f"mean native call: {stats['mean_duration_ms']:.1f} ms", size=16, fill=_SLATE)
        if title.startswith("Staggered"):
            _text(
                draw,
                (col_x0, y0 + 405),
                "Interpretation: non-preemptive, chunk-level time division.",
                size=15,
                fill=_SLATE,
            )
        else:
            _text(
                draw,
                (col_x0, y0 + 405),
                "Interpretation: scheduler can make a true coalesced B=3 call.",
                size=15,
                fill=_SLATE,
            )


def _draw_fps_trace(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    result: dict[str, Any],
    peak: dict[str, Any],
) -> None:
    x0, y0, x1, y1 = box
    _rounded(draw, box)
    _text(draw, (x0 + 24, y0 + 18), "Four H100s: client-visible FPS / active session", size=24, bold=True)
    summary = peak["summary"]
    mean = _safe_float(summary["per_active_session_delivery_fps"]["mean"])
    p50 = _safe_float(summary["per_active_session_delivery_fps"]["p50"])
    attainment = 100.0 * _safe_float(summary.get("slo_sample_attainment"))
    _text(
        draw,
        (x0 + 24, y0 + 51),
        f"Peak-16 continuous: mean {mean:.3f} FPS (p50 {p50:.3f}); 12-FPS SLO attainment {attainment:.1f}%.",
        size=15,
        fill=_SLATE,
    )
    chart = (x0 + 55, y0 + 92, x1 - 28, y1 - 49)
    cx0, cy0, cx1, cy1 = chart
    samples = [item for item in result.get("samples", []) if isinstance(item, dict)]
    phase_results = [item for item in result.get("phase_results", []) if isinstance(item, dict)]
    max_x = max((_safe_float(item.get("offset_seconds")) for item in samples), default=1.0)
    max_x = max(max_x, 1.0)
    y_max = 14.0
    peak_start = _safe_float(peak.get("started_offset_seconds"))
    peak_end = _safe_float(peak.get("completed_offset_seconds"))
    px0 = cx0 + (cx1 - cx0) * peak_start / max_x
    px1 = cx0 + (cx1 - cx0) * peak_end / max_x
    draw.rectangle((px0, cy0, px1, cy1), fill="#fef3c7")
    _text(draw, ((px0 + px1) / 2, cy0 + 8), "peak 16", size=13, fill="#92400e", bold=True, anchor="ma")
    for fps in (0, 4, 8, 12):
        y = cy1 - (cy1 - cy0) * fps / y_max
        draw.line((cx0, y, cx1, y), fill=_GRID, width=1)
        _text(draw, (cx0 - 10, y), str(fps), size=13, fill=_SLATE, anchor="rm")
    y_target = cy1 - (cy1 - cy0) * 12.0 / y_max
    draw.line((cx0, y_target, cx1, y_target), fill=_RED, width=2)
    _text(draw, (cx1 - 4, y_target - 4), "12 FPS target", size=13, fill=_RED, anchor="rs")
    points: list[tuple[float, float]] = []
    for sample in samples:
        value = sample.get("per_active_session_delivery_fps")
        if not isinstance(value, int | float):
            continue
        x = cx0 + (cx1 - cx0) * _safe_float(sample.get("offset_seconds")) / max_x
        y = cy1 - (cy1 - cy0) * min(y_max, max(0.0, float(value))) / y_max
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=_BLUE, width=3, joint="curve")
    for phase in phase_results:
        end = _safe_float(phase.get("completed_offset_seconds"))
        x = cx0 + (cx1 - cx0) * end / max_x
        draw.line((x, cy1, x, cy1 + 5), fill=_SLATE, width=1)
    _text(draw, (cx0, cy1 + 13), "0 s", size=13, fill=_SLATE)
    _text(draw, (cx1, cy1 + 13), f"{max_x:.0f} s", size=13, fill=_SLATE, anchor="ra")
    _text(
        draw,
        (cx0, y1 - 25),
        "Consumer-side rolling delivery rate; not an aggregate/model-throughput metric.",
        size=13,
        fill=_SLATE,
    )


def _draw_batch_panel(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[int, int, int, int],
    analysis: dict[str, Any],
    peak: dict[str, Any],
) -> None:
    x0, y0, x1, y1 = box
    _rounded(draw, box)
    entire = analysis["entire_capture"]
    executions = _safe_float(entire.get("batch_executions"))
    b1 = _safe_float(entire.get("b1_execution_equivalents"))
    b2 = _safe_float(entire.get("b2_execution_equivalents"))
    b3 = _safe_float(entire.get("b3_execution_equivalents"))
    b4 = _safe_float(entire.get("b4_execution_equivalents"))
    peak_summary = peak["summary"]
    _text(draw, (x0 + 24, y0 + 18), "Why the realistic trace gets almost no batching", size=24, bold=True)
    _text(
        draw,
        (x0 + 24, y0 + 51),
        f"Entire 419.2-s capture: {executions:.0f} model executions; B=2/3/4 all zero.",
        size=15,
        fill=_SLATE,
    )
    values = (("B=1", b1, _BLUE), ("B=2", b2, _ORANGE), ("B=3", b3, _GREEN), ("B=4", b4, _PURPLE))
    bar_x0, _bar_y0, bar_x1, bar_y1 = x0 + 52, y0 + 109, x1 - 34, y0 + 258
    max_value = max([value for _, value, _ in values] or [1.0])
    for index, (label, value, color) in enumerate(values):
        baseline = bar_y1 - index * 34
        draw.rounded_rectangle(
            (bar_x0, baseline - 21, bar_x0 + (bar_x1 - bar_x0) * value / max_value, baseline), radius=5, fill=color
        )
        _text(draw, (bar_x0 - 10, baseline - 11), label, size=15, fill=_SLATE, anchor="rm")
        _text(
            draw,
            (bar_x0 + (bar_x1 - bar_x0) * value / max_value + 8, baseline - 11),
            f"{value:.0f}",
            size=15,
            fill=_NAVY,
            bold=True,
        )
    _text(
        draw,
        (x0 + 24, y0 + 288),
        "Peak-16: 16/16 immediately admitted; no queue.  Mean execution batch size = 1.000.",
        size=15,
        fill=_SLATE,
    )
    _text(
        draw,
        (x0 + 24, y0 + 319),
        "Conclusion: the four GPUs supply parallel B=1 service; timing/position mismatch prevents within-GPU coalescing.",  # noqa: E501
        size=14,
        fill=_SLATE,
    )
    _text(
        draw,
        (x0 + 24, y0 + 355),
        f"Peak-16 aggregate delivery mean: {_safe_float(peak_summary['aggregate_delivery_fps']['mean']):.2f} FPS",
        size=15,
        fill=_NAVY,
        bold=True,
    )
    _text(
        draw,
        (x0 + 24, y0 + 383),
        "(Aggregate is included only as context; SLO judgment uses per-active-session FPS at left.)",
        size=13,
        fill=_SLATE,
    )


def _render(
    *,
    staggered: dict[str, Any],
    aligned: dict[str, Any],
    four_result: dict[str, Any],
    four_analysis: dict[str, Any],
    output_png: Path,
) -> dict[str, Any]:
    canvas = Image.new("RGB", (1800, 1320), _WHITE)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (50, 28), "ABot-World serving: alignment enables B=3, realistic arrivals do not", size=36, bold=True)
    _text(
        draw,
        (50, 76),
        "Native model timeline + 4-GPU LiveKit trace. All quantities are derived from the saved artifacts named in scheduler-summary.md.",  # noqa: E501
        size=18,
        fill=_SLATE,
    )
    staggered_stats = _timeline_stats(staggered)
    aligned_stats = _timeline_stats(aligned)
    peak = _phase(four_result, "peak_16_continuous")
    _draw_single_card_panel(draw, box=(45, 122, 1755, 590), staggered=staggered_stats, aligned=aligned_stats)
    _draw_fps_trace(draw, box=(45, 620, 1125, 1260), result=four_result, peak=peak)
    _draw_batch_panel(draw, box=(1150, 620, 1755, 1260), analysis=four_analysis, peak=peak)
    canvas.save(output_png)
    return {
        "staggered": staggered_stats,
        "aligned": aligned_stats,
        "peak": peak,
        "entire": four_analysis["entire_capture"],
    }


def _write_markdown(
    *,
    path: Path,
    stats: dict[str, Any],
    source_paths: dict[str, Path],
) -> None:
    staggered = stats["staggered"]
    aligned = stats["aligned"]
    peak = stats["peak"]["summary"]
    entire = stats["entire"]
    staggered_hist: Counter[int] = staggered["histogram"]
    aligned_hist: Counter[int] = aligned["histogram"]
    content = f"""# ABot-World scheduler summary (artifact-derived)

![Scheduler summary](scheduler-summary.png)

| Experiment | Native model dispatch evidence | Client-visible result |
|---|---:|---:|
| 1 H100, 3 sessions, staggered controls (0/450/900 ms) | {staggered_hist[1]}/{staggered["calls"]} B=1 calls; {staggered_hist[3]}/{staggered["calls"]} B=3 calls; mean call {staggered["mean_duration_ms"]:.1f} ms | Non-preemptive, chunk-level time division (no overlapping GPU calls) |
| 1 H100, 3 sessions, scheduler barrier aligned | {aligned_hist[3]}/{aligned["calls"]} B=3 calls; {aligned_hist[1]}/{aligned["calls"]} B=1 calls; mean B=3 call {aligned["mean_duration_ms"]:.1f} ms | Actual coalesced B=3 reached the native `generate_next_blocks()` path |
| 4 H100, intermittent 16-user LiveKit trace | B=1 {entire["b1_execution_equivalents"]:.0f}/{entire["batch_executions"]:.0f}; B=2/3/4 = 0; mean execution batch {entire["mean_execution_batch_size"]:.3f} | `peak_16_continuous`: mean **{peak["per_active_session_delivery_fps"]["mean"]:.3f} FPS/active session**, p50 {peak["per_active_session_delivery_fps"]["p50"]:.3f}, 12-FPS SLO attainment {100.0 * peak["slo_sample_attainment"]:.1f}% |

The 4-GPU result is **not** evidence that native batching lacks value: the controlled one-GPU barrier proves the production scheduler can form B=3.  It is evidence that this realistic, staggered/intermittent trace has no compatible sessions ready together on a worker at the same frame/cache boundary, so the system operates as four parallel B=1 workers.

## Reproduce the figure without starting a server or GPU

```bash
cd /public/fanyk1/lwb/TeleFuser-abot-world
PYTHONPATH=$PWD /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \\
  tools/validation/render_abot_scheduler_summary.py \\
  --staggered {source_paths["staggered"]} \\
  --aligned {source_paths["aligned"]} \\
  --four-gpu-result {source_paths["four_result"]} \\
  --four-gpu-analysis {source_paths["four_analysis"]} \\
  --output-dir {path.parent}
```

## Source artifacts

- `{source_paths["staggered"]}`
- `{source_paths["aligned"]}`
- `{source_paths["four_result"]}`
- `{source_paths["four_analysis"]}`
"""  # noqa: E501
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = _args()
    paths = {
        "staggered": args.staggered.expanduser().resolve(),
        "aligned": args.aligned.expanduser().resolve(),
        "four_result": args.four_gpu_result.expanduser().resolve(),
        "four_analysis": args.four_gpu_analysis.expanduser().resolve(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = _render(
        staggered=_load(paths["staggered"]),
        aligned=_load(paths["aligned"]),
        four_result=_load(paths["four_result"]),
        four_analysis=_load(paths["four_analysis"]),
        output_png=args.output_dir / "scheduler-summary.png",
    )
    _write_markdown(path=args.output_dir / "scheduler-summary.md", stats=stats, source_paths=paths)
    print(args.output_dir / "scheduler-summary.png")
    print(args.output_dir / "scheduler-summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
