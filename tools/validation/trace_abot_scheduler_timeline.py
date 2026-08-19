#!/usr/bin/env python3
"""Produce deterministic, inspectable ABot scheduler timelines on CPU.

This is a *scheduler-semantics* experiment, not a model-performance benchmark.
It drives the production :class:`ABotWorldLiveKitService` with a small CPU fake
pipeline, so the scheduler thread, readiness predicates, batching window,
playout pacing, and per-session ordering are the real implementation.  The
fake pipeline only replaces DiT/VAE execution with a controlled sleep and
records the actual calls the service makes.

The default ``both`` run writes two complementary traces:

* ``staggered``: three independently arriving clients stay phase-shifted and
  demonstrate the service's singleton/time-sliced dispatches;
* ``aligned``: the three clients become ready in one scheduler turn and
  demonstrate a real ``generate_next_blocks(..., B=3)`` dispatch.

Example (no GPU/model checkpoint required)::

    python tools/validation/trace_abot_scheduler_timeline.py \
      --scenario both --output-dir /tmp/abot-scheduler-timeline

Each scenario directory contains ``timeline.json``, ``events.csv``,
``batches.csv``, ``chunks.csv``, and ``timeline.png``.  The JSON and CSVs are
intended for paper plots and independent analysis; the PNG is deliberately
dependency-free (Pillow only) so it can be inspected on a bare serving node.
"""

# ruff: noqa: I001
from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch
from PIL import Image, ImageDraw, ImageFont

# Validation tools are often invoked by absolute path from a server that also
# has another TeleFuser checkout installed editable.  Prefer this checkout so
# the trace probes the service implementation adjacent to this file.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from telefuser.pipelines.abot_world.interactive import ABotWorldSessionLifecycle
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService


_SCENARIOS = ("staggered", "aligned")
_BATCH_COLORS = {
    1: "#e67e22",  # orange: serialized singleton turn
    2: "#3b82f6",  # blue: B=2 coalesced turn
    3: "#16a34a",  # green: B=3 coalesced turn
}


@dataclass(frozen=True)
class TimelineConfig:
    """Configuration shared by the deterministic scheduler traces."""

    sessions: int = 3
    chunks_per_session: int = 3
    fps: int = 12
    frames_per_chunk: int = 12
    control_latent_frames: int = 3
    batching_window_ms: float = 2.0
    fake_batch_overhead_ms: float = 4.0
    fake_per_item_ms: float = 8.0
    stagger_offsets_ms: tuple[float, ...] = (0.0, 260.0, 520.0)
    output_timeout_seconds: float = 20.0

    @property
    def chunk_playout_seconds(self) -> float:
        return self.frames_per_chunk / self.fps


class _TimelineRecorder:
    """Thread-safe wall-clock recorder shared by fake pipeline and clients."""

    def __init__(self) -> None:
        self.origin = time.monotonic()
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self._next_batch_id = 0

    def elapsed(self, timestamp: float | None = None) -> float:
        return (time.monotonic() if timestamp is None else timestamp) - self.origin

    def event(self, kind: str, *, session_id: str | None = None, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "time_seconds": round(self.elapsed(), 6),
            "kind": kind,
            "session_id": session_id or "",
            **fields,
        }
        with self._lock:
            self.events.append(payload)

    def begin_batch(self, session_ids: list[str], control_latent_frames: int) -> tuple[int, float]:
        started = time.monotonic()
        with self._lock:
            batch_id = self._next_batch_id
            self._next_batch_id += 1
        self.event(
            "batch_start",
            batch_id=batch_id,
            batch_size=len(session_ids),
            session_ids="|".join(session_ids),
            control_latent_frames=control_latent_frames,
        )
        return batch_id, started

    def end_batch(
        self,
        *,
        batch_id: int,
        started: float,
        session_ids: list[str],
        control_latent_frames: int,
    ) -> None:
        ended = time.monotonic()
        batch = {
            "batch_id": batch_id,
            "start_seconds": round(self.elapsed(started), 6),
            "end_seconds": round(self.elapsed(ended), 6),
            "duration_ms": round((ended - started) * 1000.0, 3),
            "batch_size": len(session_ids),
            "session_ids": list(session_ids),
            "control_latent_frames": control_latent_frames,
        }
        with self._lock:
            self.batches.append(batch)
        self.event(
            "batch_end",
            batch_id=batch_id,
            batch_size=len(session_ids),
            session_ids="|".join(session_ids),
            duration_ms=batch["duration_ms"],
        )

    def chunk(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.chunks.append(payload)


class _FakePipelineSession:
    """Small subset of a resident ABot session used by the service scheduler."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.next_latent_frame = 0
        self.first_frame_latent = torch.zeros(1, 1, 1, 1, 1)
        self.self_cache = [
            {
                "local_end_index": torch.zeros(1, dtype=torch.long),
                "global_end_index": torch.zeros(1, dtype=torch.long),
            }
        ]
        self.lifecycle = ABotWorldSessionLifecycle.READY
        self.closed = False

    @property
    def is_resident(self) -> bool:
        return self.lifecycle != ABotWorldSessionLifecycle.SUSPENDED


class _TimelineFakePipeline:
    """CPU pipeline whose batch calls are recorded after real service selection."""

    def __init__(self, recorder: _TimelineRecorder, config: TimelineConfig) -> None:
        self.config = SimpleNamespace(width=8, height=8)
        self.device = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.denoise_stage = SimpleNamespace(
            dit=SimpleNamespace(
                patch_size=(1, 2, 2),
                dim=8,
                num_heads=2,
                num_layers=2,
                local_attn_size=18,
                text_len=8,
            )
        )
        self.recorder = recorder
        self.timeline_config = config
        self.closed = False

    def preload_models(self) -> None:
        return None

    def create_interactive_session(
        self,
        image: Image.Image,
        prompt: str,
        *,
        seed: int,
        session_id: str | None = None,
    ) -> _FakePipelineSession:
        del image, prompt, seed
        if session_id is None:
            raise ValueError("timeline harness requires a stable session id")
        return _FakePipelineSession(session_id)

    def _generate(
        self,
        sessions: list[_FakePipelineSession],
        controls: list[dict[str, bool]],
        *,
        control_latent_frames: int,
    ) -> list[list[Image.Image]]:
        if len(sessions) != len(controls):
            raise AssertionError("session/control cardinality mismatch")
        session_ids = [session.session_id for session in sessions]
        batch_id, started = self.recorder.begin_batch(session_ids, control_latent_frames)
        # A controllable synthetic kernel model: it makes B=3 visibly shorter
        # than three B=1 calls, while remaining clearly labelled synthetic.
        sleep_seconds = (
            self.timeline_config.fake_batch_overhead_ms + self.timeline_config.fake_per_item_ms * len(sessions)
        ) / 1000.0
        time.sleep(sleep_seconds)
        results: list[list[Image.Image]] = []
        for index, session in enumerate(sessions):
            session.next_latent_frame += control_latent_frames
            session.self_cache[0]["local_end_index"] += control_latent_frames
            color = (30, (batch_id * 47 + index * 71) % 255, 70)
            results.append(
                [Image.new("RGB", (8, 8), color=color) for _ in range(self.timeline_config.frames_per_chunk)]
            )
        self.recorder.end_batch(
            batch_id=batch_id,
            started=started,
            session_ids=session_ids,
            control_latent_frames=control_latent_frames,
        )
        return results

    def generate_next_block(
        self,
        session: _FakePipelineSession,
        controls: dict[str, bool],
        *,
        control_latent_frames: int,
    ) -> list[Image.Image]:
        return self._generate([session], [controls], control_latent_frames=control_latent_frames)[0]

    def generate_next_blocks(
        self,
        sessions: list[_FakePipelineSession],
        controls: list[dict[str, bool]],
        *,
        control_latent_frames: int,
    ) -> list[list[Image.Image]]:
        return self._generate(sessions, controls, control_latent_frames=control_latent_frames)

    def suspend_interactive_session(self, session: _FakePipelineSession) -> None:
        session.lifecycle = ABotWorldSessionLifecycle.SUSPENDED

    def restore_interactive_session(self, session: _FakePipelineSession) -> None:
        session.lifecycle = ABotWorldSessionLifecycle.READY

    def close_interactive_session(self, session: _FakePipelineSession) -> None:
        session.closed = True

    def close(self) -> None:
        self.closed = True


@dataclass
class _Consumer:
    session_id: str
    state: Any
    thread: threading.Thread
    error: BaseException | None = None


def _consumer_loop(
    *,
    service: ABotWorldLiveKitService,
    consumer: _Consumer,
    recorder: _TimelineRecorder,
    config: TimelineConfig,
) -> None:
    """Consume chunks at the requested playback cadence, like a real publisher."""
    received = 0
    try:
        preview = consumer.state.output_queue.get(timeout=config.output_timeout_seconds)
        if preview.get("type") != "preview":
            raise RuntimeError(f"{consumer.session_id}: expected preview, got {preview.get('type')!r}")
        recorder.event("preview_dequeued", session_id=consumer.session_id)
        with service._scheduler_condition:  # noqa: SLF001 - consumer notification mirrors pull_chunks().
            service._scheduler_condition.notify_all()  # noqa: SLF001
        while received < config.chunks_per_session:
            payload = consumer.state.output_queue.get(timeout=config.output_timeout_seconds)
            with service._scheduler_condition:  # noqa: SLF001
                service._scheduler_condition.notify_all()  # noqa: SLF001
            if payload.get("type") == "error":
                raise RuntimeError(f"{consumer.session_id}: service error: {payload.get('error')}")
            if payload.get("type") != "chunk":
                continue
            dequeued_at = recorder.elapsed()
            scheduler = dict(payload.get("scheduler", {}))
            chunk = {
                "session_id": consumer.session_id,
                "chunk_index": int(payload.get("index", -1)),
                "dequeued_seconds": round(dequeued_at, 6),
                "frames": len(payload.get("frames", [])),
                "batch_size": int(scheduler.get("batch_size", 0)),
                "queue_wait_ms": round(float(scheduler.get("queue_wait_seconds", 0.0)) * 1000.0, 3),
                "compute_ms": round(float(scheduler.get("compute_seconds", 0.0)) * 1000.0, 3),
            }
            recorder.chunk(chunk)
            recorder.event(
                "chunk_dequeued",
                session_id=consumer.session_id,
                chunk_index=chunk["chunk_index"],
                batch_size=chunk["batch_size"],
            )
            received += 1
            if received == config.chunks_per_session:
                # Avoid a free-running extra prefetch after the measured tail.
                service.push_chunk(consumer.session_id, {"type": "control_state", "controls": []})
                recorder.event("control_released", session_id=consumer.session_id)
            recorder.event("playback_start", session_id=consumer.session_id, chunk_index=chunk["chunk_index"])
            time.sleep(config.chunk_playout_seconds)
            recorder.event("playback_end", session_id=consumer.session_id, chunk_index=chunk["chunk_index"])
    except BaseException as exc:  # pragma: no cover - surfaced deterministically by caller.
        consumer.error = exc


def _make_consumer(
    service: ABotWorldLiveKitService,
    session_id: str,
    recorder: _TimelineRecorder,
    config: TimelineConfig,
) -> _Consumer:
    state = service._session(session_id)  # noqa: SLF001 - this is an in-process scheduler probe.
    if state is None:
        raise KeyError(session_id)
    placeholder = _Consumer(session_id=session_id, state=state, thread=threading.Thread())
    thread = threading.Thread(
        target=_consumer_loop,
        kwargs={"service": service, "consumer": placeholder, "recorder": recorder, "config": config},
        name=f"abot-timeline-consumer-{session_id}",
        daemon=True,
    )
    placeholder.thread = thread
    thread.start()
    return placeholder


def _arrival_offsets(scenario: Literal["staggered", "aligned"], config: TimelineConfig) -> tuple[float, ...]:
    if scenario == "aligned":
        return (0.0,) * config.sessions
    if len(config.stagger_offsets_ms) != config.sessions:
        raise ValueError(
            f"--stagger-offsets-ms must contain exactly {config.sessions} values, got {len(config.stagger_offsets_ms)}"
        )
    offsets = tuple(value / 1000.0 for value in config.stagger_offsets_ms)
    if offsets[0] != 0.0 or any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("stagger offsets must start at 0 and be non-decreasing")
    return offsets


def run_scenario(
    scenario: Literal["staggered", "aligned"],
    config: TimelineConfig,
) -> dict[str, Any]:
    """Run one trace against the production scheduler with the CPU fake backend."""
    if config.sessions != 3:
        raise ValueError("this focused harness intentionally requires exactly three sessions")
    if config.chunks_per_session < 1 or config.fps < 1 or config.frames_per_chunk < 1:
        raise ValueError("chunks-per-session, fps, and frames-per-chunk must be positive")
    if config.control_latent_frames not in {1, 2, 3}:
        raise ValueError("control-latent-frames must be one of 1, 2, 3")
    if config.batching_window_ms < 0 or config.fake_batch_overhead_ms < 0 or config.fake_per_item_ms < 0:
        raise ValueError("timing values must be non-negative")

    recorder = _TimelineRecorder()
    pipeline = _TimelineFakePipeline(recorder, config)
    service = ABotWorldLiveKitService(
        pipeline,
        default_fps=config.fps,
        default_session_config={"prompt": "ABot scheduler timeline probe"},
        output_queue_size=4,
        control_idle_timeout=30.0,
        idle_suspension_seconds=600.0,
        max_batch_size=3,
        batching_window_ms=config.batching_window_ms,
        scheduler_mode="batched",
    )
    session_ids: list[str] = []
    consumers: list[_Consumer] = []
    offsets = _arrival_offsets(scenario, config)
    try:
        service.configure_session_capacity(config.sessions)
        service.start()
        if scenario == "aligned":
            # The pause is a harness-level barrier: session creation and control
            # activation complete before the real scheduler examines readiness.
            # It does not change scheduler selection or service code.
            with service._scheduler_condition:  # noqa: SLF001
                service._scheduler_paused = True  # noqa: SLF001
                recorder.event("scheduler_barrier_closed")

        for index, offset in enumerate(offsets):
            target = recorder.origin + offset
            remaining = target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            session_id = f"user-{index + 1}"
            service.create_session(
                {
                    "session_id": session_id,
                    "image": Image.new("RGB", (8, 8), color=(30, index * 50, 90)),
                    "prompt": "ABot scheduler timeline probe",
                    "seed": 100 + index,
                    "fps": config.fps,
                    "control_latent_frames": config.control_latent_frames,
                    "delivery_mode": "latest",
                }
            )
            session_ids.append(session_id)
            recorder.event("session_created", session_id=session_id, arrival_offset_ms=round(offset * 1000.0, 3))
            consumers.append(_make_consumer(service, session_id, recorder, config))
            if scenario == "staggered":
                service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
                recorder.event("control_activated", session_id=session_id)

        if scenario == "aligned":
            for session_id in session_ids:
                service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
                recorder.event("control_activated", session_id=session_id)
            with service._scheduler_condition:  # noqa: SLF001
                service._scheduler_paused = False  # noqa: SLF001
                service._scheduler_condition.notify_all()  # noqa: SLF001
                recorder.event("scheduler_barrier_opened")

        join_timeout = config.output_timeout_seconds + config.chunks_per_session * config.chunk_playout_seconds + 5.0
        for consumer in consumers:
            consumer.thread.join(timeout=join_timeout)
            if consumer.thread.is_alive():
                raise TimeoutError(f"{consumer.session_id} did not consume requested chunks")
            if consumer.error is not None:
                raise consumer.error
    finally:
        for session_id in session_ids:
            service.close_session(session_id, timeout=5.0)
        service.stop(close_pipeline=True)

    batches = sorted(recorder.batches, key=lambda value: int(value["batch_id"]))
    chunks = sorted(recorder.chunks, key=lambda value: (str(value["session_id"]), int(value["chunk_index"])))
    events = sorted(recorder.events, key=lambda value: float(value["time_seconds"]))
    histogram = Counter(int(batch["batch_size"]) for batch in batches)
    expected_chunks = config.sessions * config.chunks_per_session
    if len(chunks) != expected_chunks:
        raise RuntimeError(f"expected {expected_chunks} measured chunks, received {len(chunks)}")
    overlaps = sum(
        1
        for earlier, later in zip(batches, batches[1:])
        if float(later["start_seconds"]) < float(earlier["end_seconds"])
    )
    session_chunk_counts = Counter(str(chunk["session_id"]) for chunk in chunks)
    summary = {
        "batch_calls": len(batches),
        "batch_items": sum(int(batch["batch_size"]) for batch in batches),
        "batch_size_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "mean_batch_size": round(
            sum(int(batch["batch_size"]) for batch in batches) / len(batches) if batches else 0.0,
            6,
        ),
        "singleton_dispatch_fraction": round(
            histogram[1] / len(batches) if batches else 0.0,
            6,
        ),
        "maximum_observed_batch_size": max(histogram, default=0),
        "overlapping_gpu_batch_calls": overlaps,
        "serialized_scheduler_thread": overlaps == 0,
        "per_session_measured_chunks": dict(sorted(session_chunk_counts.items())),
        "first_batch_size": int(batches[0]["batch_size"]) if batches else 0,
        "classification": (
            "time_sliced_singletons"
            if batches and all(int(batch["batch_size"]) == 1 for batch in batches)
            else "coalesced_microbatching"
        ),
    }
    return {
        "schema_version": 1,
        "backend": {
            "kind": "cpu_fake_pipeline_with_production_abot_service_scheduler",
            "claim": (
                "Batch membership/timing comes from ABotWorldLiveKitService. "
                "The synthetic sleep is not a DiT/VAE performance measurement."
            ),
            "synthetic_kernel_model": {
                "batch_overhead_ms": config.fake_batch_overhead_ms,
                "per_item_ms": config.fake_per_item_ms,
                "batch_duration_ms_formula": "overhead_ms + per_item_ms * batch_size",
            },
        },
        "scenario": {
            "name": scenario,
            "scheduler_mode": "batched",
            "max_batch_size": 3,
            "delivery_mode": "latest",
            "session_arrival_offsets_ms": [round(offset * 1000.0, 3) for offset in offsets],
            "aligned_activation_barrier": scenario == "aligned",
            "plausible_chunk_playout_ms": round(config.chunk_playout_seconds * 1000.0, 3),
            **asdict(config),
        },
        "summary": summary,
        "batches": batches,
        "chunks": chunks,
        "events": events,
        "service_runtime_metrics": service.runtime_metrics(),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "|".join(value) if isinstance(value, list) else value for key, value in row.items()})


def _draw_timeline(result: dict[str, Any], path: Path) -> None:
    """Render a paper-friendly lane chart without requiring matplotlib."""
    batches = list(result["batches"])
    events = list(result["events"])
    scenario = dict(result["scenario"])
    sessions = [f"user-{index}" for index in range(1, 4)]
    maximum = max(
        [0.001]
        + [float(batch["end_seconds"]) for batch in batches]
        + [float(event["time_seconds"]) for event in events]
    )
    maximum = max(maximum * 1.08, 0.1)
    width, left, right = 1600, 170, 60
    top, lane_height, bottom = 110, 90, 95
    height = top + lane_height * len(sessions) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bold = ImageFont.load_default()
    chart_width = width - left - right

    def x(value: float) -> int:
        return left + round(chart_width * value / maximum)

    title = (
        f"ABot scheduler timeline — {scenario['name']} | "
        f"hist={result['summary']['batch_size_histogram']} | "
        f"mean B={result['summary']['mean_batch_size']}"
    )
    draw.text((20, 16), title, fill="black", font=bold)
    subtitle = (
        "Native ABot model and production service scheduler"
        if str(result["backend"]["kind"]).startswith("native_")
        else "CPU fake compute; batch membership is selected by the production ABotWorldLiveKitService scheduler"
    )
    draw.text((20, 36), subtitle, fill="#444444", font=font)
    draw.text(
        (20, 56),
        "orange=B1 singleton, blue=B2, green=B3; vertical ticks: A=arrival, C=control",
        fill="#444444",
        font=font,
    )

    for index, session_id in enumerate(sessions):
        center_y = top + index * lane_height + lane_height // 2
        draw.line((left, center_y, width - right, center_y), fill="#d1d5db", width=1)
        draw.text((18, center_y - 6), session_id, fill="black", font=font)

    # Grid and ticks are generated from the observed wall-clock span.
    tick_count = 8
    for index in range(tick_count + 1):
        value = maximum * index / tick_count
        px = x(value)
        draw.line((px, top - 12, px, height - bottom + 6), fill="#f0f0f0", width=1)
        draw.text((px - 14, height - bottom + 18), f"{value * 1000:.0f}", fill="#555555", font=font)
    draw.text((width // 2 - 42, height - 28), "elapsed milliseconds", fill="#333333", font=font)

    for batch in batches:
        start = x(float(batch["start_seconds"]))
        end = max(start + 4, x(float(batch["end_seconds"])))
        batch_size = int(batch["batch_size"])
        color = _BATCH_COLORS.get(batch_size, "#8b5cf6")
        for session_id in batch["session_ids"]:
            lane = sessions.index(session_id)
            center_y = top + lane * lane_height + lane_height // 2
            draw.rounded_rectangle((start, center_y - 18, end, center_y + 18), radius=4, fill=color, outline="#1f2937")
            if end - start >= 28:
                draw.text((start + 4, center_y - 5), f"B{batch_size}", fill="white", font=font)

    for event in events:
        session_id = str(event.get("session_id", ""))
        if session_id not in sessions:
            continue
        if event["kind"] not in {"session_created", "control_activated", "control_released"}:
            continue
        lane = sessions.index(session_id)
        center_y = top + lane * lane_height + lane_height // 2
        px = x(float(event["time_seconds"]))
        label = {"session_created": "A", "control_activated": "C", "control_released": "R"}[str(event["kind"])]
        color = {"A": "#111827", "C": "#7c3aed", "R": "#dc2626"}[label]
        draw.line((px, center_y - 32, px, center_y + 32), fill=color, width=2)
        draw.text((px + 3, center_y - 33), label, fill=color, font=font)

    image.save(path)


def _write_result(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "timeline.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "events.csv", list(result["events"]))
    _write_csv(output_dir / "batches.csv", list(result["batches"]))
    _write_csv(output_dir / "chunks.csv", list(result["chunks"]))
    _write_csv(output_dir / "summary.csv", [dict(result["summary"])])
    _draw_timeline(result, output_dir / "timeline.png")


def _draw_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 900, 340
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (20, 18), "ABot service scheduler: phase alignment changes observed microbatching", fill="black", font=font
    )
    max_calls = max([1] + [int(row["batch_calls"]) for row in rows])
    for index, row in enumerate(rows):
        x0 = 90 + index * 390
        y_base = 275
        bar_width = 58
        histogram = {int(key): int(value) for key, value in dict(row["batch_size_histogram"]).items()}
        draw.text((x0, 55), str(row["scenario"]), fill="black", font=font)
        for batch_size in (1, 2, 3):
            count = histogram.get(batch_size, 0)
            bar_height = int(175 * count / max_calls)
            left = x0 + (batch_size - 1) * 90
            color = _BATCH_COLORS[batch_size]
            draw.rectangle((left, y_base - bar_height, left + bar_width, y_base), fill=color, outline="#1f2937")
            draw.text((left + 20, y_base - bar_height - 17), str(count), fill="black", font=font)
            draw.text((left + 19, y_base + 8), f"B{batch_size}", fill="#333333", font=font)
        draw.text((x0, 306), f"mean batch={float(row['mean_batch_size']):.2f}", fill="#333333", font=font)
    draw.text(
        (20, height - 18),
        "Counts are actual service pipeline calls; only compute duration is synthetic.",
        fill="#555555",
        font=font,
    )
    image.save(path)


def _parse_offsets(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("stagger offsets must be comma-separated milliseconds") from exc
    if len(parsed) != 3 or any(not math.isfinite(part) or part < 0 for part in parsed):
        raise argparse.ArgumentTypeError("stagger offsets must contain three non-negative finite milliseconds")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=("staggered", "aligned", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunks-per-session", type=int, default=3)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames-per-chunk", type=int, default=12)
    parser.add_argument("--control-latent-frames", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--batching-window-ms", type=float, default=2.0)
    parser.add_argument("--fake-batch-overhead-ms", type=float, default=4.0)
    parser.add_argument("--fake-per-item-ms", type=float, default=8.0)
    parser.add_argument("--stagger-offsets-ms", type=_parse_offsets, default=(0.0, 260.0, 520.0))
    parser.add_argument("--output-timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()
    if args.chunks_per_session < 1 or args.fps < 1 or args.frames_per_chunk < 1:
        parser.error("chunks-per-session, fps, and frames-per-chunk must be positive")
    if args.batching_window_ms < 0 or args.fake_batch_overhead_ms < 0 or args.fake_per_item_ms < 0:
        parser.error("timing arguments must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    config = TimelineConfig(
        chunks_per_session=args.chunks_per_session,
        fps=args.fps,
        frames_per_chunk=args.frames_per_chunk,
        control_latent_frames=args.control_latent_frames,
        batching_window_ms=args.batching_window_ms,
        fake_batch_overhead_ms=args.fake_batch_overhead_ms,
        fake_per_item_ms=args.fake_per_item_ms,
        stagger_offsets_ms=args.stagger_offsets_ms,
        output_timeout_seconds=args.output_timeout_seconds,
    )
    scenarios: tuple[Literal["staggered", "aligned"], ...] = _SCENARIOS if args.scenario == "both" else (args.scenario,)  # type: ignore[assignment]
    comparison_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = run_scenario(scenario, config)
        _write_result(args.output_dir / scenario, result)
        comparison_rows.append({"scenario": scenario, **result["summary"]})
        print(
            json.dumps(
                {
                    "scenario": scenario,
                    "summary": result["summary"],
                    "timeline": str((args.output_dir / scenario / "timeline.png").resolve()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _write_csv(args.output_dir / "comparison.csv", comparison_rows)
    if len(comparison_rows) > 1:
        _draw_comparison(comparison_rows, args.output_dir / "comparison.png")


if __name__ == "__main__":
    main()
