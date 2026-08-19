#!/usr/bin/env python3
"""Render client-side ABot demand onsets and active intervals for one trace.

A demand onset is either a session arriving with active input enabled, or a
user resuming input after an explicit idle interval. The runner does not write
every heartbeat to ``result.json``; the horizontal lane segments therefore
represent the continuous active-demand intervals between those onsets and the
corresponding pause/departure. Orange spans come from actual parent-owned
model-dispatch records whose batch size is greater than one.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

_NAVY = "#0f172a"
_SLATE = "#475569"
_GRID = "#cbd5e1"
_PANEL = "#f8fafc"
_WHITE = "#ffffff"
_ACTIVE = "#188038"
_RESUME = "#1a73e8"
_INTERVAL = "#94a3b8"
_BATCH = "#f9ab00"
_BATCH_BORDER = "#d93025"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(face, size=size)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _events(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = result.get("events")
    if not isinstance(raw, list):
        raise ValueError("result.events is missing")
    return [event for event in raw if isinstance(event, dict) and isinstance(event.get("offset_seconds"), int | float)]


def _session_for_event(event: dict[str, Any]) -> str | None:
    for key in ("session", "trace_session_id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dispatch-trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _time_to_x(value: float, *, left: int, width: int, last_offset: float) -> int:
    return left + round(max(0.0, min(value, last_offset)) / last_offset * width)


def _circle(draw: ImageDraw.ImageDraw, x: int, y: int, color: str) -> None:
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=_WHITE, width=1)


def _triangle(draw: ImageDraw.ImageDraw, x: int, y: int, color: str) -> None:
    radius = 6
    draw.polygon(((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)), fill=color, outline=_WHITE)


def _deduplicated(values: list[tuple[float, str, str]]) -> list[tuple[float, str, str]]:
    seen: set[tuple[float, str, str]] = set()
    result: list[tuple[float, str, str]] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def main() -> None:
    args = _parse_args()
    result = _load_json(args.result.expanduser().resolve())
    events = sorted(_events(result), key=lambda event: float(event["offset_seconds"]))
    clock = result.get("trace_clock")
    if not isinstance(clock, dict) or not isinstance(clock.get("origin_unix_seconds"), int | float):
        raise ValueError("result.trace_clock.origin_unix_seconds is required")
    origin_unix = float(clock["origin_unix_seconds"])

    arrivals: dict[str, float] = {}
    onsets: list[tuple[float, str, str]] = []
    intervals: dict[str, list[list[float | None]]] = defaultdict(list)
    active_open: set[str] = set()
    for event in events:
        timestamp = float(event["offset_seconds"])
        event_name = event.get("event")
        session = _session_for_event(event)
        if session is None:
            continue
        if event_name == "lifecycle_session_arrival_scheduled":
            arrivals.setdefault(session, timestamp)
            if event.get("input_enabled") is True:
                onsets.append((timestamp, session, "arrival_active"))
                intervals[session].append([timestamp, None])
                active_open.add(session)
        elif event_name == "input_resumed":
            if session not in active_open:
                onsets.append((timestamp, session, "resumed"))
                intervals[session].append([timestamp, None])
                active_open.add(session)
        elif event_name == "input_paused" and session in active_open:
            intervals[session][-1][1] = timestamp
            active_open.remove(session)
        elif event_name in {"lifecycle_session_departure_scheduled", "session_stopped"}:
            if session in active_open:
                intervals[session][-1][1] = timestamp
                active_open.remove(session)

    if not arrivals:
        raise ValueError("result contains no lifecycle session arrivals")
    last_offset = max(float(event["offset_seconds"]) for event in events)
    for session_intervals in intervals.values():
        for interval in session_intervals:
            if interval[1] is None:
                interval[1] = last_offset
    onsets = _deduplicated(onsets)
    lanes = sorted(arrivals, key=lambda session: (arrivals[session], session))
    lane_index = {session: index for index, session in enumerate(lanes)}

    trace_lines = args.dispatch_trace.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    dispatches = [json.loads(line) for line in trace_lines[1:] if line.strip()]
    batched = [record for record in dispatches if int(record.get("batch_size", 0)) > 1]

    active_delta: list[tuple[float, int]] = []
    for session_intervals in intervals.values():
        for start, end in session_intervals:
            active_delta.append((float(start), 1))
            active_delta.append((float(end), -1))
    active_delta.sort(key=lambda item: (item[0], item[1]))

    margin_left = 250
    margin_right = 55
    margin_top = 115
    count_height = 180
    lane_height = 18
    lane_top = margin_top + count_height + 70
    width = 3000
    plot_width = width - margin_left - margin_right
    height = lane_top + max(1, len(lanes)) * lane_height + 105
    image = Image.new("RGB", (width, height), _WHITE)
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, height), fill=_WHITE)
    draw.text(
        (margin_left, 28), "ABot client demand onsets and active intervals", font=_font(30, bold=True), fill=_NAVY
    )
    subtitle = (
        "Green circle: arrival with active input; blue triangle: resume; gray: active-demand interval; "
        "orange: actual B>1 dispatch"
    )
    draw.text((margin_left, 67), subtitle, font=_font(15), fill=_SLATE)

    count_top = margin_top
    count_bottom = count_top + count_height
    lane_bottom = lane_top + len(lanes) * lane_height
    draw.rectangle((margin_left, count_top, width - margin_right, count_bottom), fill=_PANEL, outline=_GRID)
    draw.rectangle((margin_left, lane_top, width - margin_right, lane_bottom), fill=_PANEL, outline=_GRID)

    for seconds in range(0, int(last_offset) + 1, 120):
        x = _time_to_x(seconds, left=margin_left, width=plot_width, last_offset=last_offset)
        draw.line((x, count_top, x, lane_bottom), fill="#e2e8f0", width=1)
        draw.text((x, lane_bottom + 16), f"{seconds}s", font=_font(12), fill=_SLATE, anchor="ma")

    active_count = 0
    max_seen_active = 0
    for _, delta in active_delta:
        active_count += delta
        max_seen_active = max(max_seen_active, active_count)
    chart_max = max(1, max_seen_active)
    for value in range(chart_max + 1):
        y = count_bottom - round(value / chart_max * (count_height - 25))
        draw.line((margin_left, y, width - margin_right, y), fill="#e2e8f0", width=1)
        draw.text((margin_left - 10, y), str(value), font=_font(12), fill=_SLATE, anchor="rm")
    draw.text(
        (margin_left - 18, count_top + count_height // 2),
        "active users",
        font=_font(14, bold=True),
        fill=_NAVY,
        anchor="ms",
    )

    for record in batched:
        start = float(record["model_started_unix_seconds"]) - origin_unix
        end = float(record["model_completed_unix_seconds"]) - origin_unix
        x0 = _time_to_x(start, left=margin_left, width=plot_width, last_offset=last_offset)
        x1 = _time_to_x(end, left=margin_left, width=plot_width, last_offset=last_offset)
        draw.rectangle(
            (x0, count_top, max(x0 + 1, x1), count_bottom),
            fill="#fef3c7",
            outline=_BATCH_BORDER,
            width=1,
        )
        draw.text((x0 + 3, count_top + 4), f"B={record['batch_size']}", font=_font(12, bold=True), fill=_BATCH_BORDER)

    previous_time = 0.0
    active_count = 0
    previous_x = _time_to_x(previous_time, left=margin_left, width=plot_width, last_offset=last_offset)
    previous_y = count_bottom
    for timestamp, delta in active_delta:
        x = _time_to_x(timestamp, left=margin_left, width=plot_width, last_offset=last_offset)
        draw.line((previous_x, previous_y, x, previous_y), fill="#355c7d", width=3)
        active_count += delta
        y = count_bottom - round(active_count / chart_max * (count_height - 25))
        draw.line((x, previous_y, x, y), fill="#355c7d", width=3)
        previous_x = x
        previous_y = y
    draw.line((previous_x, previous_y, width - margin_right, previous_y), fill="#355c7d", width=3)

    for session, lane in lane_index.items():
        y = lane_top + lane * lane_height + lane_height // 2
        if lane % 2 == 0:
            draw.rectangle(
                (margin_left, y - lane_height // 2, width - margin_right, y + lane_height // 2), fill="#ffffff"
            )
        draw.text((margin_left - 10, y), session, font=_font(9), fill=_SLATE, anchor="rm")
        for start, end in intervals.get(session, []):
            x0 = _time_to_x(float(start), left=margin_left, width=plot_width, last_offset=last_offset)
            x1 = _time_to_x(float(end), left=margin_left, width=plot_width, last_offset=last_offset)
            draw.line((x0, y, x1, y), fill=_INTERVAL, width=3)

    for record in batched:
        start = float(record["model_started_unix_seconds"]) - origin_unix
        x0 = _time_to_x(start, left=margin_left, width=plot_width, last_offset=last_offset)
        draw.line((x0, lane_top, x0, lane_bottom), fill=_BATCH_BORDER, width=2)

    for timestamp, session, kind in onsets:
        y = lane_top + lane_index[session] * lane_height + lane_height // 2
        x = _time_to_x(timestamp, left=margin_left, width=plot_width, last_offset=last_offset)
        if kind == "arrival_active":
            _circle(draw, x, y, _ACTIVE)
        else:
            _triangle(draw, x, y, _RESUME)

    draw.text(
        (margin_left - 18, (lane_top + lane_bottom) // 2),
        "logical user generation",
        font=_font(14, bold=True),
        fill=_NAVY,
        anchor="ms",
    )
    draw.text(
        (width // 2, height - 24), "seconds since workload start", font=_font(15, bold=True), fill=_NAVY, anchor="ms"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image.save(args.output_dir / "demand-scatter.png")
    with (args.output_dir / "demand-events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["offset_seconds", "session", "event"])
        writer.writeheader()
        for timestamp, session, kind in onsets:
            writer.writerow({"offset_seconds": f"{timestamp:.6f}", "session": session, "event": kind})
    summary = {
        "demand_onsets": len(onsets),
        "arrival_active_onsets": sum(kind == "arrival_active" for _, _, kind in onsets),
        "resume_onsets": sum(kind == "resumed" for _, _, kind in onsets),
        "active_intervals": sum(len(value) for value in intervals.values()),
        "batched_dispatches": len(batched),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
