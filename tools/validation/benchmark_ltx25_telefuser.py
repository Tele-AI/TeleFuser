"""Record synchronized cold and warm LTX-2.5 TeleFuser pipeline measurements."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from telefuser.metrics.runtime import collect_runtime_environment, finish_runtime_measurement, start_runtime_measurement
from telefuser.pipelines.ltx25_distilled import LTX25DistilledPipeline, LTX25ImageCondition


def _measure(device: torch.device, operation: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Execute one operation with synchronized timing and allocator peaks."""
    measurement = start_runtime_measurement([device], capture_peak_memory=True)
    with torch.inference_mode():
        result = operation()
    return result, finish_runtime_measurement(measurement)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return stable raw timings plus p50 summary for one measured phase."""
    seconds = [float(sample["seconds"]) for sample in samples]
    if not seconds:
        raise ValueError("benchmark requires at least one measured sample")
    return {
        "samples": samples,
        "count": len(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "mean_seconds": statistics.mean(seconds),
        "p50_seconds": statistics.median(seconds),
    }


def _request_image(path: Path | None, frame_index: int, strength: float) -> tuple[LTX25ImageCondition, ...]:
    if path is None:
        return ()
    if not path.is_file():
        raise FileNotFoundError(f"LTX-2.5 conditioning image does not exist: {path}")
    return (LTX25ImageCondition(Image.open(path).convert("RGB"), frame_idx=frame_index, strength=strength),)


def _request_kwargs(args: argparse.Namespace, images: tuple[LTX25ImageCondition, ...]) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "images": images,
    }


def _build_pipeline(args: argparse.Namespace) -> LTX25DistilledPipeline:
    return LTX25DistilledPipeline.from_model_root(
        args.model_root,
        device=args.device,
        video_vae=args.video_vae,
        offload=args.offload,
    )


def main() -> None:
    """Run the requested cold/warm benchmark and write one provenance-rich report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--video-vae", choices=("diff", "conv"), default="diff")
    parser.add_argument("--offload", choices=("none", "cpu"), required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-frame-index", type=int, default=0)
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LTX-2.5 formal benchmarking requires a CUDA device")
    images = _request_image(args.image, args.image_frame_index, args.image_strength)
    request = _request_kwargs(args, images)

    cold_construction: list[dict[str, Any]] = []
    cold_generation: list[dict[str, Any]] = []
    for _ in range(args.runs):
        cold_pipeline, construction = _measure(device, lambda: _build_pipeline(args))
        _, generation = _measure(device, lambda pipeline=cold_pipeline: pipeline(**request))
        cold_construction.append(construction)
        cold_generation.append(generation)
        del cold_pipeline
        torch.cuda.empty_cache()

    warm_result = _measure(device, lambda: _build_pipeline(args))
    warm_pipeline, warm_construction = warm_result
    warmup: list[dict[str, Any]] = []
    for _ in range(args.warmup):
        _, measurement = _measure(device, lambda pipeline=warm_pipeline: pipeline(**request))
        warmup.append(measurement)
    samples: list[dict[str, Any]] = []
    for _ in range(args.runs):
        _, measurement = _measure(device, lambda pipeline=warm_pipeline: pipeline(**request))
        samples.append(measurement)

    try:
        import natten

        natten_capability: bool | None = bool(getattr(natten, "HAS_LIBNATTEN", False))
        natten_version: str | None = getattr(natten, "__version__", None)
    except ImportError:
        natten_capability, natten_version = None, None
    report = {
        "implementation": "telefuser",
        "runtime": {
            **collect_runtime_environment([device], repo_root=Path.cwd()),
            "natten_version": natten_version,
            "natten_has_libnatten": natten_capability,
        },
        "request": {
            "prompt": args.prompt,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "frame_rate": args.frame_rate,
            "video_vae": args.video_vae,
            "offload": args.offload,
            "image": None
            if args.image is None
            else {
                "path": str(args.image.resolve()),
                "frame_index": args.image_frame_index,
                "strength": args.image_strength,
            },
        },
        "cold": {
            "pipeline_construction": summarize_samples(cold_construction),
            "end_to_end": summarize_samples(cold_generation),
        },
        "warm_pipeline_construction": warm_construction,
        "warmup": warmup,
        "warm": summarize_samples(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
