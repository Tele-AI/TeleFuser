#!/usr/bin/env python3
"""Render an evidence-preserving physical-GPU ABot dispatch timeline.

The input is the parent-owned JSONL produced by
``TELEFUSER_LIVEKIT_DISPATCH_TRACE_PATH``.  Each line corresponds to one real
``generate_next_block(s)`` invocation in a model worker, rather than a sampled
Prometheus counter or an inferred batch.  The renderer only reads saved
artifacts; it never contacts the serving system or initializes CUDA.

Example:

    PYTHONPATH=$PWD python tools/validation/render_abot_dispatch_timeline.py \
      --dispatch-trace results/experiments/run/dispatch-trace.jsonl \
      --result results/experiments/run/result.json \
      --output-dir results/experiments/run/dispatch-analysis

Outputs:

* ``dispatch-timeline.png``: physical-GPU timeline with workload-phase bands
  and a labelled zoom;
* ``dispatches.csv``: one human-readable row per actual model dispatch;
* ``stage-projections.csv``: the stage-duration partition used for the inner
  rectangle strips (explicitly a visual projection, not kernel timestamps);
* ``phase-summary.csv``: dispatch/batch accounting aligned to workload phases;
* ``summary.json`` and ``summary.md``: physical-GPU accounting and provenance.

The outer rectangles use the measured host wall-clock interval from model
dispatch start through completion.  The narrow coloured stage strip inside a
rectangle is a sequential projection of CUDA-measured DiT/LightVAE stage
durations into that wall-clock interval.  It is deliberately not presented as
an Nsight kernel trace; the raw JSONL remains the source of truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

_NAVY = "#0f172a"
_SLATE = "#475569"
_GRID = "#cbd5e1"
_PANEL = "#f8fafc"
_WHITE = "#ffffff"
_BATCH_COLORS = {1: "#2563eb", 2: "#059669", 3: "#d97706", 4: "#dc2626"}
_PHASE_COLORS = (
    "#dbeafe",
    "#dcfce7",
    "#fef3c7",
    "#fee2e2",
    "#f3e8ff",
    "#cffafe",
    "#fae8ff",
    "#ffedd5",
    "#e0e7ff",
    "#ecfccb",
)
_STAGE_COLORS = {
    "input_prepare": "#94a3b8",
    "cache_collate": "#7c3aed",
    "denoise": "#0ea5e9",
    "cache_scatter": "#a855f7",
    "vae_decode": "#f97316",
    "postprocess": "#64748b",
}
_STAGE_LABELS = {
    "input_prepare": "input",
    "cache_collate": "KV collect",
    "denoise": "DiT",
    "cache_scatter": "KV scatter",
    "vae_decode": "LightVAE",
    "postprocess": "post",
}
_STAGE_KEYS = tuple(_STAGE_COLORS)


@dataclass(frozen=True)
class Dispatch:
    """One completed or failed real model invocation."""

    sequence: int
    worker_id: str
    configured_gpu_id: str
    gpu_id: str
    logical_cuda_device: str
    selected: float
    started: float
    completed: float
    started_unix: float | None
    duration: float
    batch_size: int
    control_latent_frames: int
    sessions: tuple[dict[str, Any], ...]
    stages: dict[str, float]
    vae_mode: str
    vae_effective_batch_size: int
    vae_invocations: int
    outcome: str
    error: str | None


@dataclass(frozen=True)
class WorkloadPhase:
    """A benchmark phase mapped onto the dispatch trace clock."""

    index: int
    name: str
    started: float
    completed: float
    target_users: int | None
    active_input_fraction: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch-trace", type=Path, required=True, help="Parent-owned dispatch-trace.jsonl.")
    parser.add_argument(
        "--result",
        type=Path,
        help="Optional black-box result.json; maps opaque session IDs to wave-xxx users.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--zoom-seconds",
        type=float,
        default=12.0,
        help="Duration of auto-selected dense labelled window (default: 12).",
    )
    parser.add_argument(
        "--zoom-start-seconds",
        type=float,
        help="Optional time relative to first dispatch, overriding automatic window selection.",
    )
    return parser.parse_args()


def _number(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _finite_or_none(value: object) -> float | None:
    candidate = _number(value, math.nan)
    return candidate if math.isfinite(candidate) else None


def _integer(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


def _short_user(value: str) -> str:
    if value.startswith("wave-"):
        return "u" + value.removeprefix("wave-")
    # Public TurboServe-derived lifecycle trace IDs are stable source-session
    # IDs plus a generation (for example ``ts-00079-g02``). Keeping the
    # generation avoids visually merging a departed/re-arrived user in the
    # labelled 12-second zoom of a 30-minute replay.
    parts = value.split("-")
    if len(parts) == 3 and parts[0] == "ts" and parts[1].isdigit() and parts[2].startswith("g"):
        source = str(int(parts[1]))
        generation = parts[2].removeprefix("g")
        return f"u{source}g{generation}"
    return value[:8]


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)


def _text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    size: int = 18,
    fill: str = _NAVY,
    bold: bool = False,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=_font(size, bold=bold), fill=fill, anchor=anchor)


def _load_dispatches(path: Path) -> list[Dispatch]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"missing dispatch trace: {source}")
    dispatches: list[Dispatch] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
        if not isinstance(event, dict) or event.get("event_type") != "model_dispatch":
            continue
        sessions = event.get("sessions", [])
        if not isinstance(sessions, list) or not all(isinstance(item, dict) for item in sessions):
            raise ValueError(f"model_dispatch at line {line_number} has invalid sessions")
        started = _number(event.get("model_started_monotonic_seconds"), math.nan)
        completed = _number(event.get("model_completed_monotonic_seconds"), math.nan)
        duration = _number(event.get("model_duration_seconds"), math.nan)
        if not all(math.isfinite(item) for item in (started, completed, duration)) or completed < started:
            raise ValueError(f"model_dispatch at line {line_number} has invalid timing")
        stages_raw = event.get("stages_seconds", {})
        stages = (
            {key: max(0.0, _number(stages_raw.get(key))) for key in _STAGE_KEYS}
            if isinstance(stages_raw, dict)
            else {key: 0.0 for key in _STAGE_KEYS}
        )
        gpu = event.get("gpu", {})
        configured_gpu_id = "unknown"
        gpu_id = "unknown"
        logical_cuda_device = "unknown"
        if isinstance(gpu, dict):
            configured = gpu.get("configured_gpu_id")
            physical = gpu.get("physical_gpu_id")
            logical = gpu.get("logical_cuda_device", gpu.get("cuda_device_index"))
            configured_gpu_id = str(configured) if configured is not None else "unknown"
            logical_cuda_device = str(logical) if logical is not None else "unknown"
            gpu_id = str(
                physical if physical is not None else configured if configured is not None else logical_cuda_device
            )
        vae = event.get("vae_decode", {})
        if not isinstance(vae, dict):
            vae = {}
        dispatches.append(
            Dispatch(
                sequence=_integer(event.get("parent_sequence"), line_number),
                worker_id=str(event.get("worker_id", "unknown")),
                configured_gpu_id=configured_gpu_id,
                gpu_id=gpu_id,
                logical_cuda_device=logical_cuda_device,
                selected=_number(event.get("selected_monotonic_seconds"), started),
                started=started,
                completed=completed,
                started_unix=_finite_or_none(event.get("model_started_unix_seconds")),
                duration=max(0.0, duration),
                batch_size=max(1, _integer(event.get("batch_size"), len(sessions) or 1)),
                control_latent_frames=_integer(event.get("control_latent_frames"), 0),
                sessions=tuple(dict(item) for item in sessions),
                stages=stages,
                vae_mode=str(vae.get("mode_name", vae.get("mode", "unknown"))),
                vae_effective_batch_size=_integer(vae.get("effective_batch_size"), 0),
                vae_invocations=_integer(vae.get("invocations"), 0),
                outcome=str(event.get("outcome", "unknown")),
                error=str(event["error"]) if event.get("error") is not None else None,
            )
        )
    if not dispatches:
        raise ValueError(f"no model_dispatch events in {source}")
    return sorted(dispatches, key=lambda item: (item.started, item.sequence))


def _session_labels(result_path: Path | None) -> dict[str, str]:
    if result_path is None:
        return {}
    result = json.loads(result_path.expanduser().resolve().read_text(encoding="utf-8"))
    events = result.get("events", []) if isinstance(result, dict) else []
    labels: dict[str, str] = {}
    if not isinstance(events, list):
        return labels
    for event in events:
        if not isinstance(event, dict) or event.get("event") != "session_created":
            continue
        session_id = event.get("server_session_id")
        user = event.get("session")
        if isinstance(session_id, str) and isinstance(user, str):
            labels[session_id] = user
    return labels


def _optional_int(value: object) -> int | None:
    return _integer(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    return _number(value, math.nan) if math.isfinite(_number(value, math.nan)) else None


def _load_workload_phases(
    result_path: Path | None,
    *,
    dispatch_origin_unix: float | None,
) -> list[WorkloadPhase]:
    """Map measured benchmark phase boundaries onto the dispatch trace clock.

    The benchmark records phase offsets from its own Unix-clock origin, while
    dispatch JSONL records model-start Unix timestamps.  This is an exact clock
    conversion when both are present; no sampled counter alignment is used.
    """

    if result_path is None or dispatch_origin_unix is None:
        return []
    payload = json.loads(result_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    trace_clock = payload.get("trace_clock")
    origin_unix = None
    if isinstance(trace_clock, dict):
        origin_unix = _optional_float(trace_clock.get("origin_unix_seconds"))
    if origin_unix is None:
        origin_unix = _optional_float(payload.get("started_at_unix_seconds"))
    phase_results = payload.get("phase_results")
    if origin_unix is None or not isinstance(phase_results, list):
        return []

    scenario = payload.get("scenario")
    declared = scenario.get("phases", []) if isinstance(scenario, dict) else []
    declared_by_name = {
        str(item.get("name")): item for item in declared if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    phases: list[WorkloadPhase] = []
    for index, raw in enumerate(phase_results, start=1):
        if not isinstance(raw, dict) or not isinstance(raw.get("phase"), str):
            continue
        start_offset = _optional_float(raw.get("started_offset_seconds"))
        completed_offset = _optional_float(raw.get("completed_offset_seconds"))
        if start_offset is None or completed_offset is None or completed_offset < start_offset:
            continue
        name = str(raw["phase"])
        declared_phase = declared_by_name.get(name, {})
        phase_summary = raw.get("summary")
        if not isinstance(phase_summary, dict):
            phase_summary = {}
        target_users = _optional_int(phase_summary.get("target_users"))
        if target_users is None and isinstance(declared_phase, dict):
            target_users = _optional_int(declared_phase.get("target_users"))
        active_input_fraction = None
        if isinstance(declared_phase, dict):
            active_input_fraction = _optional_float(declared_phase.get("active_input_fraction"))
        phases.append(
            WorkloadPhase(
                index=index,
                name=name,
                started=origin_unix + start_offset - dispatch_origin_unix,
                completed=origin_unix + completed_offset - dispatch_origin_unix,
                target_users=target_users,
                active_input_fraction=active_input_fraction,
            )
        )
    return phases


def _phase_name_at(phases: Iterable[WorkloadPhase], value: float) -> str:
    for phase in phases:
        if phase.started <= value <= phase.completed:
            return phase.name
    return ""


def _relative(dispatches: Iterable[Dispatch], origin: float) -> list[Dispatch]:
    return [
        Dispatch(
            **{
                **item.__dict__,
                "selected": item.selected - origin,
                "started": item.started - origin,
                "completed": item.completed - origin,
            }
        )
        for item in dispatches
    ]


def _gpu_sort_key(value: str) -> tuple[int, int, str]:
    return (0, int(value), value) if value.isdigit() else (1, 0, value)


def _worker_order(dispatches: Iterable[Dispatch]) -> list[str]:
    return sorted({item.gpu_id for item in dispatches}, key=_gpu_sort_key)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def _auto_zoom_start(dispatches: list[Dispatch], window: float) -> float:
    if not dispatches:
        return 0.0
    candidates = sorted({max(0.0, item.started) for item in dispatches})
    best_start = candidates[0]
    best_score = (-1, -1, -1, -1.0)
    for start in candidates:
        end = start + window
        selected = [item for item in dispatches if item.completed >= start and item.started <= end]
        score = (
            sum(item.batch_size > 1 for item in selected),
            sum(max(0, item.batch_size - 1) for item in selected),
            len(selected),
            sum(min(item.completed, end) - max(item.started, start) for item in selected),
        )
        if score > best_score:
            best_score = score
            best_start = start
    return best_start


def _interval_union_seconds(items: Iterable[Dispatch]) -> float:
    """Return physical-GPU busy time without double-counting overlaps."""

    intervals = sorted((item.started, item.completed) for item in items)
    if not intervals:
        return 0.0
    total = 0.0
    left, right = intervals[0]
    for next_left, next_right in intervals[1:]:
        if next_left <= right:
            right = max(right, next_right)
            continue
        total += right - left
        left, right = next_left, next_right
    return total + right - left


def _worker_summary(dispatches: list[Dispatch], workers: list[str], span: float) -> list[dict[str, Any]]:
    """Summarize one physical GPU lane per row.

    The legacy parameter name workers now contains physical GPU IDs.
    """

    grouped: dict[str, list[Dispatch]] = defaultdict(list)
    for dispatch in dispatches:
        grouped[dispatch.gpu_id].append(dispatch)
    rows: list[dict[str, Any]] = []
    for gpu_id in workers:
        items = grouped[gpu_id]
        batch_counts = Counter(item.batch_size for item in items)
        raw_busy = sum(item.duration for item in items)
        busy = _interval_union_seconds(items)
        rows.append(
            {
                "gpu_id": gpu_id,
                "worker_ids": sorted({item.worker_id for item in items}),
                "configured_gpu_ids": sorted({item.configured_gpu_id for item in items}),
                "logical_cuda_devices": sorted({item.logical_cuda_device for item in items}),
                "dispatches": len(items),
                "busy_seconds": round(busy, 6),
                "overlap_seconds": round(max(0.0, raw_busy - busy), 6),
                "busy_fraction_of_trace": round(busy / span, 6) if span > 0 else 0.0,
                "mean_duration_seconds": round(statistics.fmean(item.duration for item in items), 6) if items else 0.0,
                "p95_duration_seconds": round(_percentile([item.duration for item in items], 0.95), 6),
                "batches_by_size": {str(size): batch_counts[size] for size in sorted(batch_counts)},
                "batch_items": sum(item.batch_size for item in items),
            }
        )
    return rows


def _x(value: float, start: float, end: float, x0: int, x1: int) -> int:
    if end <= start:
        return x0
    return x0 + round((max(start, min(end, value)) - start) / (end - start) * (x1 - x0))


def _draw_axis(
    draw: ImageDraw.ImageDraw,
    *,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    start: float,
    end: float,
    ticks: int,
) -> None:
    for index in range(ticks + 1):
        value = start + (end - start) * index / ticks
        px = _x(value, start, end, x0, x1)
        draw.line((px, y0, px, y1), fill=_GRID, width=1)
        _text(draw, (px, y1 + 8), f"{value:.0f}s", size=14, fill=_SLATE, anchor="ma")


def _session_text(dispatch: Dispatch, labels: dict[str, str]) -> str:
    members: list[str] = []
    for session in dispatch.sessions:
        session_id = str(session.get("session_id", "?"))
        user = _short_user(labels.get(session_id, session_id))
        chunk_index = _integer(session.get("chunk_index"), -1)
        members.append(f"{user}@c{chunk_index}")
    return f"B{dispatch.batch_size}\n" + "+".join(members)


def _draw_stage_strip(
    draw: ImageDraw.ImageDraw,
    *,
    dispatch: Dispatch,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    total = sum(dispatch.stages.values())
    if total <= 0:
        draw.rectangle((x0, y0, x1, y1), fill=_SLATE)
        return
    cursor = x0
    for index, key in enumerate(_STAGE_KEYS):
        seconds = dispatch.stages.get(key, 0.0)
        fraction = seconds / total
        right = x1 if index == len(_STAGE_KEYS) - 1 else min(x1, cursor + max(1, round(width * fraction)))
        if right > cursor:
            draw.rectangle((cursor, y0, right, y1), fill=_STAGE_COLORS[key])
        cursor = right


def _draw_phase_bands(
    draw: ImageDraw.ImageDraw,
    *,
    phases: Iterable[WorkloadPhase],
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    start: float,
    end: float,
) -> None:
    """Draw a narrow workload-phase band on the same clock as dispatches."""

    for phase in phases:
        if phase.completed < start or phase.started > end:
            continue
        left = _x(phase.started, start, end, x0, x1)
        right = max(left + 1, _x(phase.completed, start, end, x0, x1))
        color = _PHASE_COLORS[(phase.index - 1) % len(_PHASE_COLORS)]
        draw.rectangle((left, y0, right, y1), fill=color)
        draw.line((left, y0, left, y1), fill=_SLATE, width=1)
        if right - left >= 30:
            _text(
                draw,
                ((left + right) // 2, (y0 + y1) // 2),
                f"P{phase.index}",
                size=13,
                fill=_NAVY,
                bold=True,
                anchor="mm",
            )


def _draw_timeline(
    dispatches: list[Dispatch],
    labels: dict[str, str],
    path: Path,
    phases: list[WorkloadPhase],
    *,
    zoom_start: float,
    zoom_seconds: float,
) -> None:
    workers = _worker_order(dispatches)
    trace_end = max(item.completed for item in dispatches)
    width, height = 2600, 1720
    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)
    _text(draw, (72, 48), "ABot-World real physical-GPU dispatch timeline", size=38, bold=True)
    _text(
        draw,
        (72, 96),
        "One rectangle = one real generate_next_block(s) invocation; time origin is the first model dispatch.",
        size=19,
        fill=_SLATE,
    )
    _text(
        draw,
        (72, 124),
        "Outline colour: batch size. Inner strip: input / KV / DiT / LightVAE / postprocess durations "
        "projected into dispatch wall time.",
        size=17,
        fill=_SLATE,
    )

    overview = (62, 178, width - 62, 668)
    draw.rounded_rectangle(overview, radius=18, fill=_PANEL, outline=_GRID, width=2)
    _text(draw, (overview[0] + 24, overview[1] + 18), f"Full trace — 0 to {trace_end:.1f}s", size=26, bold=True)
    ov_x0, ov_x1 = overview[0] + 178, overview[2] - 28
    ov_phase_y0, ov_phase_y1 = overview[1] + 62, overview[1] + 82
    ov_y0, ov_y1 = overview[1] + 98, overview[3] - 58
    _draw_phase_bands(
        draw, phases=phases, x0=ov_x0, x1=ov_x1, y0=ov_phase_y0, y1=ov_phase_y1, start=0.0, end=max(1.0, trace_end)
    )
    _draw_axis(draw, x0=ov_x0, x1=ov_x1, y0=ov_y0, y1=ov_y1, start=0.0, end=max(1.0, trace_end), ticks=10)
    lane_height = max(64, (ov_y1 - ov_y0) // max(1, len(workers)))
    for index, worker in enumerate(workers):
        lane_top = ov_y0 + index * lane_height + 12
        lane_bottom = min(ov_y1 - 6, lane_top + lane_height - 24)
        lane_workers = ",".join(sorted({item.worker_id for item in dispatches if item.gpu_id == worker}))
        _text(
            draw,
            (overview[0] + 22, (lane_top + lane_bottom) // 2),
            f"GPU {worker}\n{lane_workers}",
            size=16,
            fill=_SLATE,
            anchor="lm",
        )
        draw.line((ov_x0, lane_bottom + 8, ov_x1, lane_bottom + 8), fill=_GRID, width=1)
        for item in dispatches:
            if item.gpu_id != worker:
                continue
            left = _x(item.started, 0.0, max(1.0, trace_end), ov_x0, ov_x1)
            right = max(left + 2, _x(item.completed, 0.0, max(1.0, trace_end), ov_x0, ov_x1))
            color = _BATCH_COLORS.get(item.batch_size, "#7c3aed")
            draw.rounded_rectangle((left, lane_top, right, lane_bottom), radius=4, fill=color)

    zoom_end = min(trace_end, zoom_start + zoom_seconds)
    if zoom_end <= zoom_start:
        zoom_start, zoom_end = max(0.0, trace_end - zoom_seconds), trace_end
    panel = (62, 720, width - 62, height - 66)
    draw.rounded_rectangle(panel, radius=18, fill=_PANEL, outline=_GRID, width=2)
    _text(
        draw,
        (panel[0] + 24, panel[1] + 18),
        f"Labelled dense window — {zoom_start:.2f}s to {zoom_end:.2f}s",
        size=26,
        bold=True,
    )
    _text(
        draw,
        (panel[0] + 24, panel[1] + 52),
        "Each label is B + user/session alias + chunk index; no inferred batches.",
        size=17,
        fill=_SLATE,
    )
    z_x0, z_x1 = panel[0] + 178, panel[2] - 28
    z_phase_y0, z_phase_y1 = panel[1] + 78, panel[1] + 98
    z_y0, z_y1 = panel[1] + 116, panel[3] - 76
    _draw_phase_bands(
        draw, phases=phases, x0=z_x0, x1=z_x1, y0=z_phase_y0, y1=z_phase_y1, start=zoom_start, end=zoom_end
    )
    _draw_axis(draw, x0=z_x0, x1=z_x1, y0=z_y0, y1=z_y1, start=zoom_start, end=zoom_end, ticks=12)
    lane_height = max(95, (z_y1 - z_y0) // max(1, len(workers)))
    for index, worker in enumerate(workers):
        lane_top = z_y0 + index * lane_height + 16
        lane_bottom = min(z_y1 - 8, lane_top + lane_height - 30)
        lane_workers = ",".join(sorted({item.worker_id for item in dispatches if item.gpu_id == worker}))
        _text(
            draw,
            (panel[0] + 22, (lane_top + lane_bottom) // 2),
            f"GPU {worker}\n{lane_workers}",
            size=18,
            fill=_SLATE,
            anchor="lm",
        )
        draw.line((z_x0, lane_bottom + 10, z_x1, lane_bottom + 10), fill=_GRID, width=1)
        for item in dispatches:
            if item.gpu_id != worker or item.completed < zoom_start or item.started > zoom_end:
                continue
            left = _x(item.started, zoom_start, zoom_end, z_x0, z_x1)
            right = max(left + 3, _x(item.completed, zoom_start, zoom_end, z_x0, z_x1))
            color = _BATCH_COLORS.get(item.batch_size, "#7c3aed")
            draw.rounded_rectangle((left, lane_top, right, lane_bottom), radius=7, fill=color, outline=_NAVY, width=1)
            strip_top = max(lane_top + 4, lane_bottom - 12)
            _draw_stage_strip(draw, dispatch=item, box=(left + 1, strip_top, right - 1, lane_bottom - 2))
            if right - left >= 42:
                label = _session_text(item, labels)
                label_size = 13 if right - left < 90 else 15
                _text(draw, (left + 4, lane_top + 5), label, size=label_size, fill=_WHITE, bold=True)

    legend_x = 86
    legend_y = height - 38
    for batch_size in (1, 2, 3, 4):
        draw.rounded_rectangle(
            (legend_x, legend_y - 12, legend_x + 22, legend_y + 10), radius=4, fill=_BATCH_COLORS[batch_size]
        )
        _text(draw, (legend_x + 29, legend_y - 11), f"B{batch_size}", size=15, fill=_SLATE)
        legend_x += 84
    _text(draw, (legend_x + 6, legend_y - 11), "stage strip:", size=15, fill=_SLATE)
    legend_x += 112
    for key in _STAGE_KEYS:
        draw.rectangle((legend_x, legend_y - 12, legend_x + 18, legend_y + 10), fill=_STAGE_COLORS[key])
        _text(draw, (legend_x + 24, legend_y - 11), _STAGE_LABELS[key], size=14, fill=_SLATE)
        legend_x += 24 + int(draw.textlength(_STAGE_LABELS[key], font=_font(14))) + 24
    image.save(path)


def _write_csv(path: Path, dispatches: list[Dispatch], labels: dict[str, str], phases: list[WorkloadPhase]) -> None:
    fields = (
        "sequence",
        "worker_id",
        "gpu_id",
        "configured_gpu_id",
        "logical_cuda_device",
        "workload_phase",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "batch_size",
        "control_latent_frames",
        "users",
        "session_ids",
        "chunk_indexes",
        "frame_positions_before",
        "frame_positions_after",
        "denoise_seconds",
        "vae_decode_seconds",
        "postprocess_seconds",
        "vae_mode",
        "vae_effective_batch_size",
        "vae_invocations",
        "outcome",
        "error",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for item in dispatches:
            session_ids = [str(session.get("session_id", "")) for session in item.sessions]
            phase_name = _phase_name_at(phases, item.started)
            writer.writerow(
                {
                    "sequence": item.sequence,
                    "worker_id": item.worker_id,
                    "gpu_id": item.gpu_id,
                    "configured_gpu_id": item.configured_gpu_id,
                    "logical_cuda_device": item.logical_cuda_device,
                    "workload_phase": phase_name,
                    "start_seconds": f"{item.started:.6f}",
                    "end_seconds": f"{item.completed:.6f}",
                    "duration_seconds": f"{item.duration:.6f}",
                    "batch_size": item.batch_size,
                    "control_latent_frames": item.control_latent_frames,
                    "users": "+".join(labels.get(value, value) for value in session_ids),
                    "session_ids": "+".join(session_ids),
                    "chunk_indexes": "+".join(
                        str(_integer(session.get("chunk_index"), -1)) for session in item.sessions
                    ),
                    "frame_positions_before": "+".join(
                        str(_integer(session.get("next_latent_frame_before"), -1)) for session in item.sessions
                    ),
                    "frame_positions_after": "+".join(
                        str(_integer(session.get("next_latent_frame_after"), -1)) for session in item.sessions
                    ),
                    "denoise_seconds": f"{item.stages.get('denoise', 0.0):.6f}",
                    "vae_decode_seconds": f"{item.stages.get('vae_decode', 0.0):.6f}",
                    "postprocess_seconds": f"{item.stages.get('postprocess', 0.0):.6f}",
                    "vae_mode": item.vae_mode,
                    "vae_effective_batch_size": item.vae_effective_batch_size,
                    "vae_invocations": item.vae_invocations,
                    "outcome": item.outcome,
                    "error": item.error or "",
                }
            )


def _phase_summary_rows(
    phases: Iterable[WorkloadPhase],
    dispatches: Iterable[Dispatch],
) -> list[dict[str, Any]]:
    dispatch_list = list(dispatches)
    rows: list[dict[str, Any]] = []
    for phase in phases:
        # Attribute a dispatch to the phase in which its model invocation starts.
        # This makes phase batch counts mutually exclusive and sum to the global
        # histogram, unlike interval-overlap accounting at a phase boundary.
        items = [item for item in dispatch_list if phase.started <= item.started <= phase.completed]
        histogram = Counter(item.batch_size for item in items)
        executions = len(items)
        batch_items = sum(item.batch_size for item in items)
        rows.append(
            {
                "phase_id": f"P{phase.index}",
                "phase": phase.name,
                "start_seconds": round(phase.started, 6),
                "end_seconds": round(phase.completed, 6),
                "duration_seconds": round(max(0.0, phase.completed - phase.started), 6),
                "target_users": phase.target_users,
                "active_input_fraction": phase.active_input_fraction,
                "dispatches": executions,
                "batch_items": batch_items,
                "mean_batch_size": round(batch_items / executions, 6) if executions else 0.0,
                "batches_b1": histogram[1],
                "batches_b2": histogram[2],
                "batches_b3": histogram[3],
                "batches_b4": histogram[4],
            }
        )
    return rows


def _write_phase_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = (
        "phase_id",
        "phase",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "target_users",
        "active_input_fraction",
        "dispatches",
        "batch_items",
        "mean_batch_size",
        "batches_b1",
        "batches_b2",
        "batches_b3",
        "batches_b4",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_stage_projection_csv(path: Path, dispatches: Iterable[Dispatch], labels: dict[str, str]) -> None:
    """Write the exact stage-strip partition used in the PNG.

    CUDA event durations are real measured durations, but the start/end columns
    are scaled into the enclosing host dispatch span solely for visualization.
    """

    fields = (
        "sequence",
        "physical_gpu_id",
        "worker_id",
        "batch_size",
        "users",
        "chunk_indexes",
        "stage",
        "measured_stage_seconds",
        "projected_start_seconds",
        "projected_end_seconds",
        "projection_note",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for dispatch in dispatches:
            total = sum(dispatch.stages.values())
            scale = dispatch.duration / total if total > 0 else 0.0
            cursor = dispatch.started
            session_ids = [str(session.get("session_id", "")) for session in dispatch.sessions]
            users = "+".join(labels.get(value, value) for value in session_ids)
            chunks = "+".join(str(_integer(session.get("chunk_index"), -1)) for session in dispatch.sessions)
            for index, stage in enumerate(_STAGE_KEYS):
                measured = dispatch.stages.get(stage, 0.0)
                projected_end = dispatch.completed if index == len(_STAGE_KEYS) - 1 else cursor + measured * scale
                writer.writerow(
                    {
                        "sequence": dispatch.sequence,
                        "physical_gpu_id": dispatch.gpu_id,
                        "worker_id": dispatch.worker_id,
                        "batch_size": dispatch.batch_size,
                        "users": users,
                        "chunk_indexes": chunks,
                        "stage": stage,
                        "measured_stage_seconds": f"{measured:.9f}",
                        "projected_start_seconds": f"{cursor:.9f}",
                        "projected_end_seconds": f"{projected_end:.9f}",
                        "projection_note": "scaled into model dispatch wall interval; not kernel timestamps",
                    }
                )
                cursor = projected_end


def _write_summary(
    path: Path,
    *,
    source: Path,
    rows: list[dict[str, Any]],
    phase_rows: list[dict[str, Any]],
    dispatches: list[Dispatch],
    span: float,
    zoom_start: float,
    zoom_seconds: float,
) -> None:
    batch_histogram = Counter(item.batch_size for item in dispatches)
    payload = {
        "schema_version": 1,
        "source_dispatch_trace": str(source.resolve()),
        "time_origin": "first model_started_monotonic_seconds in the input trace",
        "trace_span_seconds": round(span, 6),
        "dispatches": len(dispatches),
        "batch_histogram": {str(key): batch_histogram[key] for key in sorted(batch_histogram)},
        "physical_gpus": rows,
        "workload_phases": phase_rows,
        "labelled_zoom": {"start_seconds": round(zoom_start, 6), "duration_seconds": round(zoom_seconds, 6)},
        "interpretation": {
            "rectangle": "Measured model dispatch host wall-clock start-to-completion interval.",
            "stage_strip": "Sequential visual projection of measured stage durations; not an Nsight kernel trace.",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, *, summary: dict[str, Any], output_dir: Path) -> None:
    rows = summary["physical_gpus"]
    lines = [
        "# Physical-GPU ABot dispatch timeline",
        "",
        "This report is computed solely from the real parent-owned dispatch JSONL.",
        "Each row is one model invocation, not a sampled counter.",
        "",
        f"- Trace span: {summary['trace_span_seconds']:.3f} s",
        f"- Dispatches: {summary['dispatches']}",
        f"- Batch histogram: {summary['batch_histogram']}",
        "",
        "| Physical GPU | Workers | Dispatches | Busy time | Busy fraction | Batches by size |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['gpu_id']} | {', '.join(row['worker_ids'])} | {row['dispatches']} | "
            f"{row['busy_seconds']:.3f}s | {100 * row['busy_fraction_of_trace']:.1f}% | "
            f"{row['batches_by_size']} |"
        )
    phase_rows = summary["workload_phases"]
    if phase_rows:
        lines.extend(
            [
                "",
                "## Workload phases aligned to the dispatch clock",
                "",
                "| ID | Phase | Time | Users | Active input | Dispatches | B1 | B2 |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in phase_rows:
            input_fraction = row["active_input_fraction"]
            input_text = f"{100 * input_fraction:.0f}%" if input_fraction is not None else "—"
            lines.append(
                f"| {row['phase_id']} | {row['phase']} | {row['start_seconds']:.1f}–{row['end_seconds']:.1f}s | "
                f"{row['target_users'] if row['target_users'] is not None else '—'} | {input_text} | "
                f"{row['dispatches']} | {row['batches_b1']} | {row['batches_b2']} |"
            )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `dispatch-timeline.png`: physical-GPU lanes, phase bands, and labelled zoom.",
            "- `dispatches.csv`: raw dispatch rows with workload user/phase mapping.",
            "- `phase-summary.csv`: phase-aligned batch distribution.",
            "- `stage-projections.csv`: visual stage-strip partition (not kernel timestamps).",
            "",
            "The coloured stage strip is a visual projection of measured stage durations within the actual "
            "wall-clock dispatch interval; it is not a kernel-level Nsight timeline.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    if args.zoom_seconds <= 0:
        raise ValueError("--zoom-seconds must be positive")
    source = args.dispatch_trace.expanduser().resolve()
    raw_dispatches = _load_dispatches(source)
    origin = min(item.started for item in raw_dispatches)
    dispatches = _relative(raw_dispatches, origin)
    dispatch_origin_unix = raw_dispatches[0].started_unix
    phases = _load_workload_phases(args.result, dispatch_origin_unix=dispatch_origin_unix)
    phase_rows = _phase_summary_rows(phases, dispatches)
    labels = _session_labels(args.result)
    span = max(item.completed for item in dispatches)
    zoom_start = args.zoom_start_seconds
    if zoom_start is None:
        zoom_start = _auto_zoom_start(dispatches, args.zoom_seconds)
    zoom_start = max(0.0, min(float(zoom_start), max(0.0, span - min(args.zoom_seconds, span))))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = _worker_order(dispatches)
    rows = _worker_summary(dispatches, workers, span)
    _write_csv(output_dir / "dispatches.csv", dispatches, labels, phases)
    _write_stage_projection_csv(output_dir / "stage-projections.csv", dispatches, labels)
    _write_phase_csv(output_dir / "phase-summary.csv", phase_rows)
    summary_path = output_dir / "summary.json"
    _write_summary(
        summary_path,
        source=source,
        rows=rows,
        phase_rows=phase_rows,
        dispatches=dispatches,
        span=span,
        zoom_start=zoom_start,
        zoom_seconds=min(args.zoom_seconds, span),
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _write_markdown(output_dir / "summary.md", summary=summary, output_dir=output_dir)
    _draw_timeline(
        dispatches,
        labels,
        output_dir / "dispatch-timeline.png",
        phases=phases,
        zoom_start=zoom_start,
        zoom_seconds=min(args.zoom_seconds, span),
    )
    print(
        json.dumps({"output_dir": str(output_dir), "dispatches": len(dispatches), "span_seconds": span}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
