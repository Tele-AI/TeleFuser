"""Profile ABot TurboServe continuous batching with retained causal sessions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline


def _loader_module():
    loader_path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location("abot_turboserve_benchmark_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot example loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))]


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    loader = _loader_module()
    pipeline = loader.get_pipeline(
        model_root=args.model_root,
        pipeline_class=ABotWorldInteractivePipeline,
    )
    pipeline.preload_models()
    image = Image.open(args.image).convert("RGB")
    sessions = [
        pipeline.create_interactive_session(
            image,
            args.prompt,
            seed=args.seed + index,
            session_id=f"benchmark-{index}",
        )
        for index in range(args.sessions)
    ]
    chunk_latencies: list[float] = []
    stage_samples: list[dict[str, float | int]] = []
    total_frames = 0
    started_at = time.monotonic()
    try:
        for _chunk_index in range(args.chunks):
            for offset in range(0, len(sessions), args.batch_size):
                batch = sessions[offset : offset + args.batch_size]
                batch_started_at = time.monotonic()
                outputs = pipeline.generate_next_blocks(
                    batch,
                    [{"W": True} for _ in batch],
                    control_latent_frames=args.control_latent_frames,
                )
                chunk_latencies.append(time.monotonic() - batch_started_at)
                stage_samples.append(pipeline.last_stage_metrics())
                total_frames += sum(len(frames) for frames in outputs)
        elapsed = time.monotonic() - started_at
        expected_latents = args.chunks * args.control_latent_frames
        if any(session.next_latent_frame != expected_latents for session in sessions):
            raise RuntimeError("One or more ABot sessions did not advance through every requested chunk")
        return {
            "sessions": args.sessions,
            "chunks_per_session": args.chunks,
            "configured_batch_size": args.batch_size,
            "control_latent_frames": args.control_latent_frames,
            "total_frames": total_frames,
            "elapsed_seconds": elapsed,
            "frames_per_second": total_frames / elapsed if elapsed else 0.0,
            "batch_latency_seconds": {
                "count": len(chunk_latencies),
                "mean": statistics.fmean(chunk_latencies),
                "p50": _percentile(chunk_latencies, 0.50),
                "p95": _percentile(chunk_latencies, 0.95),
                "p99": _percentile(chunk_latencies, 0.99),
                "maximum": max(chunk_latencies),
            },
            "stage_samples": stage_samples,
        }
    finally:
        for session in sessions:
            pipeline.close_interactive_session(session)
        pipeline.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--chunks", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--control-latent-frames", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sessions < 1 or args.chunks < 1 or not 1 <= args.batch_size <= args.sessions:
        parser.error("sessions/chunks must be positive and batch-size in [1, sessions]")
    return args


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
