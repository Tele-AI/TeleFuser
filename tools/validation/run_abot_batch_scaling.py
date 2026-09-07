"""Run a reproducible single-GPU ABot continuous-batching scaling sweep.

This is the first motivation experiment for world-model serving.  It uses the
same ``ABotWorldLiveKitService`` scheduler as LiveKit, keeps all sessions
continuously active, and measures the steady-state cost of different retained
session counts and batch caps after a warm-up round.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService


def _loader_module() -> Any:
    loader_path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location("abot_batch_scaling_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * q + 0.999999) - 1))]


def _mean(values: Iterable[float | int]) -> float:
    materialized = [float(value) for value in values]
    return statistics.fmean(materialized) if materialized else 0.0


def _drain_initial_previews(service: ABotWorldLiveKitService, session_ids: list[str]) -> None:
    for session_id in session_ids:
        state = service._session(session_id)  # Session service boundary, kept local for an offline benchmark.
        if state is None:
            raise KeyError(session_id)
        preview = state.output_queue.get(timeout=30.0)
        if preview.get("type") != "preview":
            raise RuntimeError(f"Expected preview output for {session_id}, got {preview.get('type')!r}")


def _run_point(
    *,
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    args: argparse.Namespace,
    sessions: int,
    max_batch_size: int,
) -> dict[str, Any]:
    service = ABotWorldLiveKitService(
        pipeline,
        default_fps=args.fps,
        default_session_config={"image_path": str(args.image), "prompt": args.prompt, "seed": args.seed},
        max_batch_size=max_batch_size,
        batching_window_ms=args.batching_window_ms,
        output_queue_size=max(128, args.chunks_per_session + args.warmup_chunks + 8),
        control_idle_timeout=args.control_idle_timeout_seconds,
        idle_suspension_seconds=max(args.idle_suspension_seconds, args.duration_hint_seconds),
    )
    # The offline sweep explicitly controls admission and never calls the
    # runtime capacity profiler; its warmup is measured separately below.
    service._capacity_profile = {"effective_capacity": sessions}  # noqa: SLF001
    session_ids: list[str] = []
    try:
        service.start()
        for index in range(sessions):
            session_ids.append(
                service.create_session(
                    {
                        "session_id": f"s{sessions}-b{max_batch_size}-{index}",
                        "image_path": str(args.image),
                        "prompt": args.prompt,
                        "seed": args.seed + index,
                        "fps": args.fps,
                        "control_latent_frames": args.control_latent_frames,
                        "delivery_mode": "lossless",
                    }
                )
            )
        _drain_initial_previews(service, session_ids)
        for session_id in session_ids:
            service.push_chunk(session_id, {"type": "control_state", "controls": ["W"]})

        outputs: dict[str, list[dict[str, Any]]] = {session_id: [] for session_id in session_ids}
        warmup_remaining = {session_id: args.warmup_chunks for session_id in session_ids}
        measured_remaining = {session_id: args.chunks_per_session for session_id in session_ids}
        measurement_started_at: float | None = None
        while any(value > 0 for value in measured_remaining.values()):
            made_progress = False
            for session_id in session_ids:
                if measured_remaining[session_id] <= 0 and warmup_remaining[session_id] <= 0:
                    continue
                state = service._session(session_id)  # noqa: SLF001
                if state is None:
                    raise RuntimeError(f"Benchmark session disappeared: {session_id}")
                payload = state.output_queue.get(timeout=args.chunk_timeout_seconds)
                if payload.get("type") == "error":
                    error = str(payload.get("error", "ABot scheduler failed"))
                    if "out of memory" in error.lower():
                        raise torch.OutOfMemoryError(error)
                    raise RuntimeError(error)
                if payload.get("type") != "chunk":
                    continue
                made_progress = True
                if warmup_remaining[session_id] > 0:
                    warmup_remaining[session_id] -= 1
                    if all(value == 0 for value in warmup_remaining.values()):
                        measurement_started_at = time.monotonic()
                    continue
                outputs[session_id].append(payload)
                measured_remaining[session_id] -= 1
            if not made_progress:
                raise RuntimeError("No ABot chunks were produced during benchmark")
        ended_at = time.monotonic()
        if measurement_started_at is None:
            measurement_started_at = ended_at

        samples = [payload for payloads in outputs.values() for payload in payloads]
        scheduler = [payload.get("scheduler", {}) for payload in samples]
        compute = [float(item.get("compute_seconds", 0.0)) for item in scheduler]
        queue_wait = [float(item.get("queue_wait_seconds", 0.0)) for item in scheduler]
        batch_sizes = [int(item.get("batch_size", 1)) for item in scheduler]
        denoise = [float(item.get("denoise_seconds", 0.0)) for item in scheduler]
        vae_decode = [float(item.get("vae_decode_seconds", 0.0)) for item in scheduler]
        total_frames = sum(len(payload.get("frames", [])) for payload in samples)
        elapsed = ended_at - measurement_started_at
        per_session_frames = [
            sum(len(payload.get("frames", [])) for payload in outputs[session_id]) for session_id in session_ids
        ]
        return {
            "sessions": sessions,
            "max_batch_size": max_batch_size,
            "control_latent_frames": args.control_latent_frames,
            "chunks_per_session": args.chunks_per_session,
            "warmup_chunks": args.warmup_chunks,
            "elapsed_seconds": elapsed,
            "total_frames": total_frames,
            "aggregate_fps": total_frames / elapsed if elapsed else 0.0,
            "per_session_fps": _mean(frame / elapsed for frame in per_session_frames) if elapsed else 0.0,
            "mean_batch_size": _mean(batch_sizes),
            "max_observed_batch_size": max(batch_sizes, default=0),
            "p50_compute_seconds": _percentile(compute, 0.50),
            "p95_compute_seconds": _percentile(compute, 0.95),
            "p50_queue_wait_seconds": _percentile(queue_wait, 0.50),
            "p95_queue_wait_seconds": _percentile(queue_wait, 0.95),
            "p95_chunk_latency_seconds": _percentile([left + right for left, right in zip(queue_wait, compute)], 0.95),
            "mean_denoise_seconds": _mean(denoise),
            "mean_vae_decode_seconds": _mean(vae_decode),
            "denoise_share": _mean(denoise) / _mean(compute) if _mean(compute) else 0.0,
            "vae_decode_share": _mean(vae_decode) / _mean(compute) if _mean(compute) else 0.0,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(pipeline.device)),
            "service_runtime_metrics": service.runtime_metrics(),
        }
    finally:
        service.stop(close_pipeline=False)
        torch.cuda.empty_cache()


def _parse_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item.strip()]
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--sessions", type=_parse_ints, default=[1, 2, 4])
    parser.add_argument("--max-batch-sizes", type=_parse_ints, default=[1, 2, 4])
    parser.add_argument("--chunks-per-session", type=int, default=4)
    parser.add_argument("--warmup-chunks", type=int, default=1)
    parser.add_argument("--control-latent-frames", choices=(1, 2, 3), type=int, default=2)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--batching-window-ms", type=float, default=2.0)
    parser.add_argument("--idle-suspension-seconds", type=float, default=600.0)
    parser.add_argument("--control-idle-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--duration-hint-seconds", type=float, default=600.0)
    parser.add_argument("--chunk-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.chunks_per_session < 1 or args.warmup_chunks < 0:
        parser.error("chunks-per-session must be positive and warmup-chunks non-negative")
    if args.control_idle_timeout_seconds <= 0:
        parser.error("control-idle-timeout-seconds must be positive")
    return args


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader = _loader_module()
    pipeline = loader.get_pipeline(
        model_root=args.model_root,
        pipeline_class=ABotWorldInteractivePipeline,
    )
    image = Image.open(args.image).convert("RGB")
    results: list[dict[str, Any]] = []
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"

    def write_results() -> None:
        json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not results:
            return
        fieldnames = sorted({key for row in results for key in row if key != "service_runtime_metrics"})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key in fieldnames} for row in results)

    try:
        for sessions in args.sessions:
            for max_batch_size in args.max_batch_sizes:
                if max_batch_size > sessions:
                    continue
                print(f"running sessions={sessions} max_batch_size={max_batch_size}", flush=True)
                try:
                    row = _run_point(
                        pipeline=pipeline,
                        image=image,
                        args=args,
                        sessions=sessions,
                        max_batch_size=max_batch_size,
                    )
                except torch.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    row = {
                        "sessions": sessions,
                        "max_batch_size": max_batch_size,
                        "control_latent_frames": args.control_latent_frames,
                        "status": "oom",
                        "error": str(exc).splitlines()[0],
                    }
                    print(f"OOM sessions={sessions} max_batch_size={max_batch_size}", flush=True)
                results.append(row)
                write_results()
    finally:
        pipeline.close()
    write_results()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
