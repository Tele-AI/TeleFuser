"""Benchmark a native LingBot-Video base or base-plus-refiner run on one GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.lingbot_video.lingbot_video_dense_1_3b import build_pipeline as build_dense_pipeline
from examples.lingbot_video.lingbot_video_moe_30b import build_pipeline as build_moe_pipeline
from examples.lingbot_video.lingbot_video_moe_30b import build_refiner
from telefuser.pipelines.lingbot_video import (
    LingBotVideoRequest,
    default_negative_caption,
    load_lingbot_video_prompt,
    load_refiner_first_frame,
    num_frames_from_duration,
    prepare_refiner_video,
)

BenchmarkSamples = dict[str, list[float]]


def _resolve_negative_caption(negative_caption: str | None, num_frames: int) -> str:
    """Keep benchmark sampling aligned with the pipeline's source defaults."""
    return default_negative_caption(num_frames) if negative_caption is None else negative_caption


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure(samples: BenchmarkSamples, name: str, operation: Callable[[], Any]) -> Any:
    _synchronize()
    start = time.perf_counter()
    result = operation()
    _synchronize()
    samples[name].append(time.perf_counter() - start)
    return result


def _instrument_method(
    samples_provider: Callable[[], BenchmarkSamples], owner: object, attribute: str, metric_name: str
) -> None:
    original = getattr(owner, attribute)

    def measured(*args: object, **kwargs: object) -> Any:
        return _measure(samples_provider(), metric_name, lambda: original(*args, **kwargs))

    setattr(owner, attribute, measured)


def _image_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()


def _summary(samples: BenchmarkSamples) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "count": len(values),
            "total_seconds": sum(values),
            "mean_seconds": sum(values) / len(values),
            "max_seconds": max(values),
        }
        for name, values in sorted(samples.items())
    }


def _encode_output(frames: torch.Tensor, output_path: Path, fps: int) -> None:
    video = frames[0].permute(1, 2, 3, 0).float().clamp(0.0, 1.0).cpu().numpy()
    if video.shape[0] == 1:
        Image.fromarray((video[0] * 255).round().astype("uint8")).save(output_path)
        return
    from diffusers.utils import export_to_video

    export_to_video(list(video), str(output_path), fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--caption-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variant", choices=("dense", "moe"), default="dense")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--negative-caption", default=None)
    parser.add_argument("--image")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refiner-height", type=int, default=1088)
    parser.add_argument("--refiner-width", type=int, default=1920)
    parser.add_argument("--refiner-steps", type=int, default=8)
    parser.add_argument("--refiner-guidance-scale", type=float, default=3.0)
    parser.add_argument("--refiner-shift", type=float, default=3.0)
    parser.add_argument("--refiner-t-thresh", type=float, default=0.85)
    parser.add_argument("--refiner-tail-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-cpu-offload", action="store_true")
    parser.add_argument("--warmup", type=int, default=1, help="Number of unreported warmup generations.")
    parser.add_argument("--runs", type=int, default=1, help="Number of measured generations.")
    args = parser.parse_args()
    if args.refine and args.variant != "moe":
        raise ValueError("--refine requires --variant moe")
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")

    caption, duration = load_lingbot_video_prompt(args.caption_json)
    num_frames = (
        args.num_frames
        if args.num_frames is not None
        else num_frames_from_duration(duration)
        if duration is not None
        else 121
    )
    negative_caption = _resolve_negative_caption(args.negative_caption, num_frames)
    setup_samples: BenchmarkSamples = defaultdict(list)
    warmup_samples: BenchmarkSamples = defaultdict(list)
    measured_samples: BenchmarkSamples = defaultdict(list)
    active_samples = measured_samples

    def get_active_samples() -> BenchmarkSamples:
        return active_samples

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pipeline_builder = build_dense_pipeline if args.variant == "dense" else build_moe_pipeline
    pipeline = _measure(
        setup_samples,
        "base_load",
        lambda: pipeline_builder(
            args.model_dir,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            cpu_offload=False if args.disable_cpu_offload else None,
        ),
    )
    if pipeline.text_stage is None or pipeline.denoising_stage is None or pipeline.vae_decode_stage is None:
        raise RuntimeError("LingBot runtime did not create all required base stages")
    _instrument_method(get_active_samples, pipeline.text_stage, "encode", "base_text_encode")
    _instrument_method(get_active_samples, pipeline.denoising_stage, "predict_noise_with_cfg", "base_denoise_step")
    if pipeline.vae_encode_stage is not None:
        _instrument_method(get_active_samples, pipeline.vae_encode_stage, "encode", "base_vae_encode")
    _instrument_method(get_active_samples, pipeline.vae_decode_stage, "decode", "base_vae_decode")
    source_image = _image_to_tensor(args.image) if args.image else None
    refiner_prompt_conditions_reused = False
    refiner = None

    def generate_once() -> torch.Tensor:
        nonlocal refiner_prompt_conditions_reused, refiner
        generation = _measure(
            active_samples,
            "base_total",
            lambda: pipeline.generate(
                LingBotVideoRequest(
                    caption=caption,
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    image=source_image,
                ),
                negative_caption=negative_caption,
                generator=torch.Generator("cuda").manual_seed(args.seed),
            ),
        )
        frames = generation.output
        if not args.refine:
            return frames
        if pipeline.text_stage is None:
            raise RuntimeError("LingBot refiner requires the base text stage")
        if generation.prompt_conditions.has_visual_condition:
            positive, positive_mask = _measure(
                active_samples, "refiner_text_encode", lambda: pipeline.text_stage.encode(caption)
            )
            negative, negative_mask = _measure(
                active_samples, "refiner_text_encode", lambda: pipeline.text_stage.encode(negative_caption)
            )
        else:
            positive = generation.prompt_conditions.positive_prompt_embeds
            negative = generation.prompt_conditions.negative_prompt_embeds
            positive_mask = generation.prompt_conditions.positive_attention_mask
            negative_mask = generation.prompt_conditions.negative_attention_mask
            refiner_prompt_conditions_reused = True
        _measure(active_samples, "base_release", pipeline.release_gpu_resources)
        if refiner is None:
            refiner = _measure(
                setup_samples,
                "refiner_load",
                lambda: build_refiner(args.model_dir, cpu_offload=not args.disable_cpu_offload),
            )
            _instrument_method(
                get_active_samples,
                refiner.denoising_stage,
                "predict_noise_with_cfg",
                "refiner_denoise_step",
            )
            _instrument_method(get_active_samples, refiner.vae_encode_stage, "encode", "refiner_vae_encode")
            _instrument_method(get_active_samples, refiner.vae_decode_stage, "decode", "refiner_vae_decode")
        lowres_video, _ = _measure(
            active_samples,
            "refiner_prepare",
            lambda: prepare_refiner_video(
                frames, source_fps=24.0, height=args.refiner_height, width=args.refiner_width
            ),
        )
        clean_first_frame = (
            _measure(
                active_samples,
                "refiner_first_frame_prepare",
                lambda: load_refiner_first_frame(
                    args.image,
                    target_height=args.refiner_height,
                    target_width=args.refiner_width,
                    geometry_height=args.height,
                    geometry_width=args.width,
                ),
            )
            if args.image
            else None
        )
        return _measure(
            active_samples,
            "refiner_total",
            lambda: refiner.refine(
                lowres_video,
                positive,
                negative,
                positive_mask,
                negative_mask,
                num_inference_steps=args.refiner_steps,
                guidance_scale=args.refiner_guidance_scale,
                shift=args.refiner_shift,
                t_thresh=args.refiner_t_thresh,
                tail_steps=args.refiner_tail_steps,
                clean_first_frame=clean_first_frame,
                generator=torch.Generator("cuda").manual_seed(args.seed),
            ),
        )

    for _ in range(args.warmup):
        active_samples = warmup_samples
        generate_once()
    frames = None
    for _ in range(args.runs):
        active_samples = measured_samples
        frames = generate_once()
    if frames is None:
        raise RuntimeError("benchmark did not produce a measured output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _measure(measured_samples, "output_encode", lambda: _encode_output(frames, args.output, fps=24))
    report = {
        "model_dir": str(args.model_dir),
        "variant": args.variant,
        "refine": args.refine,
        "height": args.height,
        "width": args.width,
        "num_frames": num_frames,
        "steps": args.steps,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "cpu_offload": not args.disable_cpu_offload if args.variant == "moe" else False,
        "refiner_prompt_conditions_reused": refiner_prompt_conditions_reused,
        "setup_metrics": _summary(setup_samples),
        "warmup_metrics": _summary(warmup_samples),
        "metrics": _summary(measured_samples),
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
