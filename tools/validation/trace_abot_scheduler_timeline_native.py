#!/usr/bin/env python3
"""Trace three native ABot sessions through the production service scheduler.

This companion to ``trace_abot_scheduler_timeline.py`` is deliberately small:
it uses the exact same ``ABotWorldLiveKitService`` and the native
``ABotWorldInteractivePipeline``.  The only instrumentation is an instance
wrapper around ``generate_next_blocks`` that records the actual session IDs,
batch size, and wall-clock duration of each model invocation.  It does not
change service or model source code.

Unlike the CPU semantic probe, session initialization is completed before the
trace clock starts.  The experiment therefore isolates the relevant event for
continuous batching: when independently retained sessions become *control
ready*.  ``aligned`` uses a scheduler barrier to make the three controls ready
in one scheduler turn; ``staggered`` activates those controls at the supplied
offsets.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.abot_world._loader import DEFAULT_PROMPT, get_pipeline
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService
from tools.validation import trace_abot_scheduler_timeline as common


def _record_native_pipeline_calls(pipeline: ABotWorldInteractivePipeline, recorder: common._TimelineRecorder) -> None:
    """Record actual native B=1/B>1 calls without altering source code."""
    original = pipeline.generate_next_blocks

    def traced(
        sessions: list[Any],
        actions: list[Any],
        *,
        control_latent_frames: int = 3,
    ) -> list[list[Image.Image]]:
        session_ids = [str(session.session_id) for session in sessions]
        batch_id, started = recorder.begin_batch(session_ids, control_latent_frames)
        try:
            return original(sessions, actions, control_latent_frames=control_latent_frames)
        except Exception as exc:
            recorder.event(
                "batch_error",
                batch_id=batch_id,
                batch_size=len(session_ids),
                session_ids="|".join(session_ids),
                error=repr(exc),
            )
            raise
        finally:
            recorder.end_batch(
                batch_id=batch_id,
                started=started,
                session_ids=session_ids,
                control_latent_frames=control_latent_frames,
            )

    # Instance assignment means generate_next_block() also reaches this
    # wrapper, because its implementation calls self.generate_next_blocks().
    pipeline.generate_next_blocks = traced  # type: ignore[method-assign]


def _wait_for_previews(recorder: common._TimelineRecorder, expected: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with recorder._lock:  # noqa: SLF001 - harness-owned recorder.
            seen = sum(event["kind"] == "preview_dequeued" for event in recorder.events)
        if seen >= expected:
            return
        time.sleep(0.002)
    raise TimeoutError(f"only {seen}/{expected} session previews were consumed")


def _reset_trace_clock(recorder: common._TimelineRecorder) -> None:
    """Exclude model preload/session initialization from scheduling evidence."""
    with recorder._lock:  # noqa: SLF001 - harness-owned recorder.
        recorder.origin = time.monotonic()
        recorder.events.clear()
        recorder.batches.clear()
        recorder.chunks.clear()
        recorder._next_batch_id = 0  # noqa: SLF001


def _run_native_scenario(
    scenario: Literal["staggered", "aligned"],
    *,
    config: common.TimelineConfig,
    model_root: Path,
    image_path: Path,
    device_id: int,
) -> dict[str, Any]:
    recorder = common._TimelineRecorder()
    pipeline = get_pipeline(
        model_root=model_root,
        device_id=device_id,
        pipeline_class=ABotWorldInteractivePipeline,
    )
    _record_native_pipeline_calls(pipeline, recorder)
    service = ABotWorldLiveKitService(
        pipeline,
        default_fps=config.fps,
        default_session_config={"image_path": str(image_path), "prompt": DEFAULT_PROMPT},
        output_queue_size=4,
        control_idle_timeout=30.0,
        idle_suspension_seconds=600.0,
        max_batch_size=3,
        batching_window_ms=config.batching_window_ms,
        scheduler_mode="batched",
    )
    session_ids: list[str] = []
    consumers: list[common._Consumer] = []
    offsets = common._arrival_offsets(scenario, config)
    runtime_metrics: dict[str, Any] = {}
    try:
        # This experiment has a known three-session B=3 target.  Do not add a
        # fourth hidden capacity-profiling session to a trace intended to show
        # exactly three retained sessions.
        service._capacity_profile = {"effective_capacity": config.sessions}  # noqa: SLF001
        service.start()
        for index in range(config.sessions):
            session_id = f"user-{index + 1}"
            service.create_session(
                {
                    "session_id": session_id,
                    "image_path": str(image_path),
                    "prompt": DEFAULT_PROMPT,
                    "seed": 100 + index,
                    "fps": config.fps,
                    "control_latent_frames": config.control_latent_frames,
                    "delivery_mode": "latest",
                }
            )
            session_ids.append(session_id)
            consumers.append(common._make_consumer(service, session_id, recorder, config))
        _wait_for_previews(recorder, config.sessions, config.output_timeout_seconds)
        _reset_trace_clock(recorder)
        for session_id in session_ids:
            recorder.event("session_prepared", session_id=session_id)

        if scenario == "aligned":
            # This only defers the existing scheduler thread.  After resuming,
            # normal batch-key and deadline checks select the actual B=3/B=1.
            with service._scheduler_condition:  # noqa: SLF001
                service._scheduler_paused = True  # noqa: SLF001
                recorder.event("scheduler_barrier_closed")
            for session_id in session_ids:
                service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
                recorder.event("control_activated", session_id=session_id, arrival_offset_ms=0.0)
            with service._scheduler_condition:  # noqa: SLF001
                service._scheduler_paused = False  # noqa: SLF001
                service._scheduler_condition.notify_all()  # noqa: SLF001
                recorder.event("scheduler_barrier_opened")
        else:
            for session_id, offset in zip(session_ids, offsets):
                target = recorder.origin + offset
                remaining = target - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                service.push_chunk(session_id, {"type": "control_state", "controls": ["KeyW"]})
                recorder.event("control_activated", session_id=session_id, arrival_offset_ms=round(offset * 1000.0, 3))

        join_timeout = config.output_timeout_seconds + config.chunks_per_session * config.chunk_playout_seconds + 30.0
        for consumer in consumers:
            consumer.thread.join(timeout=join_timeout)
            if consumer.thread.is_alive():
                raise TimeoutError(f"{consumer.session_id} did not consume requested chunks")
            if consumer.error is not None:
                raise consumer.error
        runtime_metrics = service.runtime_metrics()
    finally:
        for session_id in session_ids:
            service.close_session(session_id, timeout=20.0)
        service.stop(close_pipeline=True)

    batches = sorted(recorder.batches, key=lambda value: int(value["batch_id"]))
    chunks = sorted(recorder.chunks, key=lambda value: (str(value["session_id"]), int(value["chunk_index"])))
    events = sorted(recorder.events, key=lambda value: float(value["time_seconds"]))
    expected_chunks = config.sessions * config.chunks_per_session
    if len(chunks) != expected_chunks:
        raise RuntimeError(f"expected {expected_chunks} measured chunks, received {len(chunks)}")
    histogram = Counter(int(batch["batch_size"]) for batch in batches)
    overlaps = sum(
        1
        for earlier, later in zip(batches, batches[1:])
        if float(later["start_seconds"]) < float(earlier["end_seconds"])
    )
    per_session = Counter(str(chunk["session_id"]) for chunk in chunks)
    summary = {
        "batch_calls": len(batches),
        "batch_items": sum(int(batch["batch_size"]) for batch in batches),
        "batch_size_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "mean_batch_size": round(
            sum(int(batch["batch_size"]) for batch in batches) / len(batches) if batches else 0.0,
            6,
        ),
        "singleton_dispatch_fraction": round(histogram[1] / len(batches) if batches else 0.0, 6),
        "maximum_observed_batch_size": max(histogram, default=0),
        "overlapping_gpu_batch_calls": overlaps,
        "serialized_scheduler_thread": overlaps == 0,
        "per_session_measured_chunks": dict(sorted(per_session.items())),
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
            "kind": "native_abot_world_model_with_production_abot_service_scheduler",
            "claim": (
                "Batch membership and duration are measured around the native "
                "ABotWorldInteractivePipeline.generate_next_blocks call."
            ),
            "model_root": str(model_root),
            "image_path": str(image_path),
            "device_id": device_id,
        },
        "scenario": {
            "name": scenario,
            "scheduler_mode": "batched",
            "max_batch_size": 3,
            "delivery_mode": "latest",
            "session_arrival_offsets_ms": [round(offset * 1000.0, 3) for offset in offsets],
            "aligned_activation_barrier": scenario == "aligned",
            "session_preparation": "all three retained sessions are initialized before trace time zero",
            "nominal_chunk_playout_ms": round(config.chunk_playout_seconds * 1000.0, 3),
            **asdict(config),
        },
        "summary": summary,
        "batches": batches,
        "chunks": chunks,
        "events": events,
        "service_runtime_metrics": runtime_metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("staggered", "aligned", "both"), default="both")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunks-per-session", type=int, default=3)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frames-per-chunk", type=int, default=12)
    parser.add_argument("--control-latent-frames", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--batching-window-ms", type=float, default=2.0)
    parser.add_argument("--stagger-offsets-ms", type=common._parse_offsets, default=(0.0, 450.0, 900.0))
    parser.add_argument("--output-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if not args.model_root.is_dir() or not args.image.is_file():
        parser.error("--model-root must be a directory and --image must be an existing image")
    if args.chunks_per_session < 1 or args.fps < 1 or args.frames_per_chunk < 1:
        parser.error("chunks-per-session, fps, and frames-per-chunk must be positive")
    if args.batching_window_ms < 0:
        parser.error("batching-window-ms must be non-negative")
    return args


def main() -> None:
    args = _parse_args()
    config = common.TimelineConfig(
        chunks_per_session=args.chunks_per_session,
        fps=args.fps,
        frames_per_chunk=args.frames_per_chunk,
        control_latent_frames=args.control_latent_frames,
        batching_window_ms=args.batching_window_ms,
        stagger_offsets_ms=args.stagger_offsets_ms,
        output_timeout_seconds=args.output_timeout_seconds,
    )
    scenarios: tuple[Literal["staggered", "aligned"], ...] = (
        common._SCENARIOS if args.scenario == "both" else (args.scenario,)
    )  # type: ignore[assignment]
    comparison_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = _run_native_scenario(
            scenario,
            config=config,
            model_root=args.model_root.resolve(),
            image_path=args.image.resolve(),
            device_id=args.device_id,
        )
        common._write_result(args.output_dir / scenario, result)
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
    common._write_csv(args.output_dir / "comparison.csv", comparison_rows)
    if len(comparison_rows) > 1:
        common._draw_comparison(comparison_rows, args.output_dir / "comparison.png")


if __name__ == "__main__":
    main()
