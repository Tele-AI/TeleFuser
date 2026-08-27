"""Drive the ABot continuous-batching service with bursty interactive clients.

Unlike ``benchmark_abot_turboserve.py``, this exercises the same service
scheduler used by the LiveKit adapter: sessions arrive over time, controls are
updated independently, and each client consumes frames at its requested FPS.
Run one process per GPU for multi-replica measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from examples.abot_world._loader import DEFAULT_PROMPT, get_pipeline
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))]


async def _client(
    service: ABotWorldLiveKitService,
    *,
    session_index: int,
    args: argparse.Namespace,
    started_at: float,
) -> dict[str, Any]:
    rng = random.Random(args.seed + session_index)
    await asyncio.sleep(rng.uniform(0.0, args.arrival_window_seconds))
    session_id = f"concurrent-{session_index}"
    service.create_session(
        {
            "session_id": session_id,
            "image_path": str(args.image),
            "prompt": args.prompt,
            "seed": args.seed + session_index,
            "fps": args.fps,
            "control_latent_frames": args.control_latent_frames,
            "delivery_mode": args.delivery_mode,
        }
    )
    created_at = time.monotonic()
    produced_chunks = 0
    produced_frames = 0
    scheduler_waits: list[float] = []
    compute_times: list[float] = []
    batch_sizes: list[int] = []
    first_chunk_at: float | None = None
    received_at: list[float] = []
    displayed_at: list[float] = []
    first_frame_at: float | None = None
    displayed_frames = 0
    consumer_done = asyncio.Event()

    async def consume() -> None:
        nonlocal first_chunk_at, first_frame_at, produced_chunks, produced_frames, displayed_frames
        async for payload in service.pull_chunks(session_id):
            if payload.get("type") != "chunk":
                continue
            received = time.monotonic()
            received_at.append(received)
            first_chunk_at = received if first_chunk_at is None else first_chunk_at
            frames = payload.get("frames", [])
            produced_chunks += 1
            produced_frames += len(frames)
            scheduler = payload.get("scheduler", {})
            scheduler_waits.append(float(scheduler.get("queue_wait_seconds", 0.0)))
            compute_times.append(float(scheduler.get("compute_seconds", 0.0)))
            batch_sizes.append(int(scheduler.get("batch_size", 1)))
            # Count a frame only after the client has consumed and displayed it.
            # This is the user-visible end-to-end measurement, not model output.
            for _ in frames:
                if args.consumer_playback_fps > 0:
                    await asyncio.sleep(1.0 / args.consumer_playback_fps)
                displayed = time.monotonic()
                displayed_at.append(displayed)
                first_frame_at = displayed if first_frame_at is None else first_frame_at
                displayed_frames += 1
        consumer_done.set()

    consumer = asyncio.create_task(consume(), name=f"abot-consumer-{session_id}")
    deadline = started_at + args.duration_seconds
    controls = ["W"]
    try:
        while time.monotonic() < deadline:
            service.push_chunk(session_id, {"type": "control_state", "controls": controls})
            await asyncio.sleep(rng.uniform(args.control_update_min_seconds, args.control_update_max_seconds))
            # Brief releases are common in real keyboard input and create an
            # independently changing active set for the scheduler.
            if rng.random() < args.idle_probability:
                service.push_chunk(session_id, {"type": "control_state", "controls": []})
                await asyncio.sleep(rng.uniform(args.idle_min_seconds, args.idle_max_seconds))
            controls = rng.choice((["W"], ["W", "A"], ["W", "D"], ["S"], ["I"], ["W", "J"]))
        metrics = service.runtime_metrics(session_id)
    finally:
        service.close_session(session_id)
        await asyncio.wait_for(consumer_done.wait(), timeout=args.close_timeout_seconds)
        await consumer
    consumer_completed_at = time.monotonic()
    intervals = [later - earlier for earlier, later in zip(received_at, received_at[1:])]
    return {
        "session_id": session_id,
        "arrival_offset_seconds": created_at - started_at,
        "first_chunk_seconds": (first_chunk_at - created_at) if first_chunk_at is not None else None,
        "received_chunks": produced_chunks,
        "received_frames": produced_frames,
        "consumer_displayed_frames": displayed_frames,
        "consumer_end_to_end_seconds": consumer_completed_at - created_at,
        "consumer_end_to_end_fps": (
            displayed_frames / (consumer_completed_at - created_at) if consumer_completed_at > created_at else 0.0
        ),
        "consumer_first_frame_seconds": (first_frame_at - created_at) if first_frame_at is not None else None,
        "scheduler_queue_wait_seconds": scheduler_waits,
        "compute_seconds": compute_times,
        "batch_sizes": batch_sizes,
        "chunk_interarrival_seconds": intervals,
        "service_metrics": metrics,
    }


async def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    pipeline = get_pipeline(model_root=args.model_root, pipeline_class=ABotWorldInteractivePipeline)
    service = ABotWorldLiveKitService(
        pipeline,
        default_fps=args.fps,
        default_session_config={
            "image_path": str(args.image),
            "prompt": args.prompt,
            "seed": args.seed,
            "control_latent_frames": args.control_latent_frames,
        },
        max_batch_size=args.max_batch_size,
        batching_window_ms=args.batching_window_ms,
        scheduler_mode=args.scheduler_mode,
        output_queue_size=args.output_queue_size,
        idle_suspension_seconds=args.idle_suspension_seconds,
    )
    try:
        service.start()
        service.configure_session_capacity(args.sessions)
        started_at = time.monotonic()
        results = await asyncio.gather(
            *(_client(service, session_index=index, args=args, started_at=started_at) for index in range(args.sessions))
        )
        elapsed = time.monotonic() - started_at
    finally:
        service.stop()
    waits = [value for result in results for value in result["scheduler_queue_wait_seconds"]]
    computes = [value for result in results for value in result["compute_seconds"]]
    batches = [value for result in results for value in result["batch_sizes"]]
    interarrivals = [value for result in results for value in result["chunk_interarrival_seconds"]]
    first_chunks = [result["first_chunk_seconds"] for result in results if result["first_chunk_seconds"] is not None]
    total_frames = sum(result["received_frames"] for result in results)
    displayed_frames = sum(result["consumer_displayed_frames"] for result in results)
    consumer_fps = [result["consumer_end_to_end_fps"] for result in results]
    consumer_first_frames = [
        result["consumer_first_frame_seconds"]
        for result in results
        if result["consumer_first_frame_seconds"] is not None
    ]
    return {
        "scenario": {
            "kind": "interactive_continuous_service",
            "scheduler_mode": args.scheduler_mode,
            "sessions": args.sessions,
            "arrival_window_seconds": args.arrival_window_seconds,
            "duration_seconds": args.duration_seconds,
            "target_fps_per_session": args.fps,
            "consumer_playback_fps": args.consumer_playback_fps,
            "control_latent_frames": args.control_latent_frames,
            "delivery_mode": args.delivery_mode,
            "max_batch_size": args.max_batch_size,
            "batching_window_ms": args.batching_window_ms,
        },
        "elapsed_seconds": elapsed,
        "received_frames": total_frames,
        "consumer_displayed_frames": displayed_frames,
        "consumer_end_to_end_fps": _summary(consumer_fps),
        "consumer_first_frame_seconds": _summary(consumer_first_frames),
        "first_chunk_seconds": _summary(first_chunks),
        "scheduler_queue_wait_seconds": _summary(waits),
        "model_compute_seconds": _summary(computes),
        "chunk_interarrival_seconds": _summary(interarrivals),
        "observed_batch_sizes": dict(sorted(Counter(batches).items())),
        "mean_observed_batch_size": statistics.fmean(batches) if batches else 0.0,
        "sessions_detail": results,
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "maximum": max(values) if values else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--arrival-window-seconds", type=float, default=1.5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--consumer-playback-fps",
        type=float,
        default=8.0,
        help="Client display cadence used for the end-to-end FPS metric; defaults to the 8-FPS target.",
    )
    parser.add_argument("--control-latent-frames", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--scheduler-mode", choices=("round_robin", "batched"), default="round_robin")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batching-window-ms", type=float, default=2.0)
    parser.add_argument("--output-queue-size", type=int, default=4)
    parser.add_argument("--delivery-mode", choices=("latest", "lossless"), default="latest")
    parser.add_argument("--control-update-min-seconds", type=float, default=1.0)
    parser.add_argument("--control-update-max-seconds", type=float, default=1.0)
    parser.add_argument("--idle-probability", type=float, default=0.0)
    parser.add_argument("--idle-min-seconds", type=float, default=0.05)
    parser.add_argument("--idle-max-seconds", type=float, default=0.25)
    parser.add_argument("--idle-suspension-seconds", type=float, default=5.0)
    parser.add_argument("--close-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sessions < 1 or args.duration_seconds <= 0 or args.arrival_window_seconds < 0:
        parser.error("sessions and duration must be positive; arrival window must be non-negative")
    if args.consumer_playback_fps < 0:
        parser.error("consumer playback FPS must be non-negative")
    if args.control_update_min_seconds <= 0 or args.control_update_max_seconds < args.control_update_min_seconds:
        parser.error("control update range must be positive and ordered")
    if (
        not 0 <= args.idle_probability <= 1
        or args.idle_min_seconds < 0
        or args.idle_max_seconds < args.idle_min_seconds
    ):
        parser.error("invalid idle burst configuration")
    return args


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_benchmark(args))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "sessions_detail"}, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
