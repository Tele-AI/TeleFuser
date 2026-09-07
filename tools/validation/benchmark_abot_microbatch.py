"""Measure synchronous ABot-World retained-session microbatch scaling on one GPU.

This intentionally bypasses the service scheduler.  For every requested batch
size B it creates B independent retained sessions, warms them up, and then
executes one ``generate_next_blocks`` call per sample.  Each measured call
generates a continuation chunk of ``4 * control_latent_frames`` video frames for every session.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline


def _load_example_loader() -> Any:
    path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location("abot_microbatch_loader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot loader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_batch_sizes(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return values


def _sample_stats(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean": statistics.fmean(samples),
        "stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def _run_point(
    pipeline: ABotWorldInteractivePipeline,
    image: Image.Image,
    args: argparse.Namespace,
    batch_size: int,
) -> dict[str, Any]:
    sessions = []
    device = torch.device(pipeline.device)
    try:
        for index in range(batch_size):
            sessions.append(
                pipeline.create_interactive_session(
                    image,
                    args.prompt,
                    seed=args.seed + index,
                    session_id=f"microbatch-b{batch_size}-s{index}",
                )
            )
        controls = [{"W": True} for _ in sessions]

        # Exclude the first generation from warmup and timing.  It exercises a
        # distinct first-frame path, but it still emits one configured chunk.
        # Every chunk therefore emits exactly 4 pixel frames per requested
        # continuation latent, per session.
        initial_frames = pipeline.generate_next_blocks(
            sessions, controls, control_latent_frames=args.control_latent_frames
        )
        expected_frames = 4 * args.control_latent_frames
        if any(len(item) != expected_frames for item in initial_frames):
            raise RuntimeError(
                f"initial chunk did not emit {expected_frames} frames per session: "
                f"{[len(item) for item in initial_frames]}"
            )
        for _ in range(args.warmup_chunks):
            frames = pipeline.generate_next_blocks(sessions, controls, control_latent_frames=args.control_latent_frames)
            expected_frames = 4 * args.control_latent_frames
            if any(len(item) != expected_frames for item in frames):
                raise RuntimeError(
                    f"warmup did not emit {expected_frames} frames per session: {[len(item) for item in frames]}"
                )

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples: list[float] = []
        denoise_samples: list[float] = []
        vae_samples: list[float] = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            frames = pipeline.generate_next_blocks(sessions, controls, control_latent_frames=args.control_latent_frames)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started_at
            expected_frames = 4 * args.control_latent_frames
            if any(len(item) != expected_frames for item in frames):
                raise RuntimeError(
                    f"sample did not emit {expected_frames} frames per session: {[len(item) for item in frames]}"
                )
            samples.append(elapsed)
            stage_metrics = pipeline.last_stage_metrics()
            denoise_samples.append(float(stage_metrics.get("denoise_seconds", 0.0)))
            vae_samples.append(float(stage_metrics.get("vae_decode_seconds", 0.0)))

        timing = _sample_stats(samples)
        chunk_time = timing["mean"]
        return {
            "status": "ok",
            "batch": batch_size,
            "warmup_chunks": args.warmup_chunks,
            "repeats": args.repeats,
            "control_latent_frames": args.control_latent_frames,
            "frames_per_session_per_chunk": 4 * args.control_latent_frames,
            "chunk_time_seconds": chunk_time,
            "chunk_time_stats_seconds": timing,
            "mean_denoise_seconds": statistics.fmean(denoise_samples),
            "mean_vae_decode_seconds": statistics.fmean(vae_samples),
            "mean_other_seconds": chunk_time - statistics.fmean(denoise_samples) - statistics.fmean(vae_samples),
            "aggregate_fps": float(4 * args.control_latent_frames * batch_size) / chunk_time,
            "fps_per_session": float(4 * args.control_latent_frames) / chunk_time,
            "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    finally:
        for session in sessions:
            pipeline.close_interactive_session(session)
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=[1, 2, 3, 4])
    parser.add_argument("--warmup-chunks", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--control-latent-frames", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup_chunks < 1 or args.repeats < 2:
        parser.error("warmup-chunks must be at least 1 and repeats at least 2")
    return args


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader = _load_example_loader()
    pipeline = loader.get_pipeline(model_root=args.model_root, pipeline_class=ABotWorldInteractivePipeline)
    image = Image.open(args.image).convert("RGB")
    results: list[dict[str, Any]] = []
    try:
        for batch_size in args.batch_sizes:
            print(f"running batch={batch_size}", flush=True)
            try:
                row = _run_point(pipeline, image, args, batch_size)
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                row = {"status": "oom", "batch": batch_size, "error": str(exc).splitlines()[0]}
            results.append(row)
            (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        pipeline.close()

    fields = ["batch", "status", "chunk_time_seconds", "aggregate_fps", "fps_per_session", "gpu_peak_memory_bytes"]
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
