"""Record synchronized cold and warm LTX-2.5 upstream pipeline measurements.

Run with the pinned upstream LTX source on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from telefuser.metrics.runtime import collect_runtime_environment, finish_runtime_measurement, start_runtime_measurement


def _measure(device: torch.device, operation: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Execute one operation with synchronized timing and allocator peaks."""
    measurement = start_runtime_measurement([device], capture_peak_memory=True)
    with torch.inference_mode():
        result = operation()
    return result, finish_runtime_measurement(measurement)


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
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


def main() -> None:
    """Run the requested upstream cold/warm benchmark and write one report."""
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
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(f"LTX-2.5 conditioning image does not exist: {args.image}")

    from ltx_pipelines.distilled import DistilledPipeline  # type: ignore[import-not-found]
    from ltx_pipelines.utils.args import ImageConditioningInput  # type: ignore[import-not-found]
    from ltx_pipelines.utils.model_paths import ModelPaths  # type: ignore[import-not-found]
    from ltx_pipelines.utils.types import OffloadMode  # type: ignore[import-not-found]

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LTX-2.5 formal benchmarking requires a CUDA device")
    video_vae_name = (
        "ltx-2.5-video-vae-bf16.safetensors" if args.video_vae == "diff" else "ltx-2.5-video-vae-conv-bf16.safetensors"
    )
    paths = ModelPaths.from_split(
        transformer_path=str(args.model_root / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        text_encoder_path=str(args.model_root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        video_vae_path=str(args.model_root / "vae" / video_vae_name),
        audio_vae_path=str(args.model_root / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        duration_head_path=str(args.model_root / "model_patches/ltx-2.5-duration-head-bf16.safetensors"),
    )
    images = (
        []
        if args.image is None
        else [
            ImageConditioningInput(
                path=str(args.image.resolve()), frame_idx=args.image_frame_index, strength=args.image_strength
            )
        ]
    )
    request = {
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "frame_rate": args.frame_rate,
        "images": images,
        "num_frames": args.num_frames,
    }

    def build_pipeline() -> Any:
        return DistilledPipeline(
            model_paths=paths,
            spatial_upsampler_path=str(
                args.model_root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
            ),
            loras=(),
            device=device,
            offload_mode=OffloadMode(args.offload),
        )

    cold_construction: list[dict[str, Any]] = []
    cold_generation: list[dict[str, Any]] = []
    for _ in range(args.runs):
        cold_pipeline, construction = _measure(device, build_pipeline)
        _, generation = _measure(device, lambda pipeline=cold_pipeline: pipeline(**request))
        cold_construction.append(construction)
        cold_generation.append(generation)
        del cold_pipeline
        torch.cuda.empty_cache()

    warm_result = _measure(device, build_pipeline)
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
        "implementation": "upstream",
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
            "pipeline_construction": _summarize_samples(cold_construction),
            "end_to_end": _summarize_samples(cold_generation),
        },
        "warm_pipeline_construction": warm_construction,
        "warmup": warmup,
        "warm": _summarize_samples(samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
