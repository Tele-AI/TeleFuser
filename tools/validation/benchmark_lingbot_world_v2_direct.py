"""Benchmark LingBot-World v2 through the direct pipeline-service output path."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from telefuser.service.livekit.pipeline_adapter import LiveKitPipelineAdapter
from telefuser.service.security.security_validator import SecurityLevel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--control-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-num", type=int, default=4)
    parser.add_argument("--prompt", default="walk forward through the scene")
    parser.add_argument("--frame-num", type=int, default=957)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    p90_index = int(0.9 * (len(ordered) - 1))
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


async def _send_controls(
    adapter: LiveKitPipelineAdapter,
    session_id: str,
    events: list[dict[str, Any]],
    started_at: float,
) -> None:
    for event in events:
        delay = started_at + float(event["delay_s"]) - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        adapter.push_chunk(session_id, dict(event["message"]))


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    trace = json.loads(Path(args.control_trace).read_text())
    events = trace["events"]
    adapter = LiveKitPipelineAdapter(security_level=SecurityLevel.NONE)
    adapter.start(args.pipeline, skip_validation=True, gpu_num=args.gpu_num)
    capacity = adapter.configure_session_capacity(2)
    session_id = adapter.create_session(
        {
            "prompt": args.prompt,
            "image_path": str(Path(args.image).resolve()),
            "fps": args.fps,
            "chunk_size": args.chunk_size,
            "frame_num": args.frame_num,
            "max_duration_seconds": 60.0,
            "sample_shift": 10.0,
            "control_mode": "cam",
            "show_control_hud": False,
            "benchmark_metrics": True,
        }
    )
    started_at = time.monotonic()
    sender = asyncio.create_task(_send_controls(adapter, session_id, events, started_at))
    frames = 0
    chunk_profiles: list[dict[str, Any]] = []
    runtime: dict[str, Any] | None = None
    first_preview_at: float | None = None
    first_generated_frame_at: float | None = None
    try:
        async for payload in adapter.pull_chunks(session_id):
            payload_type = payload.get("type")
            if payload_type in {"preview", "chunk"}:
                payload_frames = payload.get("frames", [])
                if payload_type == "chunk":
                    frames += len(payload_frames)
                    if payload_frames and first_generated_frame_at is None:
                        first_generated_frame_at = time.monotonic()
                elif payload_frames and first_preview_at is None:
                    first_preview_at = time.monotonic()
            if payload_type != "status":
                continue
            if payload.get("stage") == "runtime_ready":
                runtime = payload.get("runtime")
            measurement = payload.get("measurement")
            if isinstance(measurement, dict) and "index" in measurement:
                chunk_profiles.append(measurement)
        await sender
    finally:
        if not sender.done():
            sender.cancel()
        await adapter.aclose()

    elapsed = time.monotonic() - started_at
    steady = [profile for profile in chunk_profiles if int(profile["index"]) > 0]
    compute_seconds = [float(profile["compute_seconds"]) for profile in steady]
    phase_names = sorted(
        {
            name
            for profile in steady
            for name, value in profile.get("phases", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        "environment": {
            "sys_executable": sys.executable,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "request": {
            "pipeline": str(Path(args.pipeline).resolve()),
            "image": str(Path(args.image).resolve()),
            "control_trace": str(Path(args.control_trace).resolve()),
            "prompt": args.prompt,
            "frame_num": args.frame_num,
            "fps": args.fps,
            "chunk_size": args.chunk_size,
            "gpu_num": args.gpu_num,
        },
        "transport": "direct pipeline service; no LiveKit room, pacing, codec, or client",
        "capacity": capacity,
        "runtime": runtime,
        "result": {
            "frames": frames,
            "chunks": len(chunk_profiles),
            "steady_chunks": len(steady),
            "elapsed_seconds": elapsed,
            "first_preview_seconds": None if first_preview_at is None else first_preview_at - started_at,
            "first_generated_frame_seconds": (
                None if first_generated_frame_at is None else first_generated_frame_at - started_at
            ),
            "steady_compute_seconds": sum(compute_seconds),
            "steady_compute_fps": sum(float(profile["frames"]) for profile in steady) / sum(compute_seconds),
            "chunk_compute_seconds": _summary(compute_seconds),
            "phases": {
                name: _summary([float(profile["phases"][name]) for profile in steady if name in profile["phases"]])
                for name in phase_names
            },
        },
        "chunk_profiles": chunk_profiles,
    }


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_run(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["result"], indent=2, sort_keys=True))
    print(f"Artifact: {output}")


if __name__ == "__main__":
    main()
