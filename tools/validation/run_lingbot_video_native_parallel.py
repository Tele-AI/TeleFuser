"""Run LingBot-Video through the native TeleFuser multi-process worker topology."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.lingbot_video.lingbot_video_dense_1_3b import build_pipeline as build_dense_pipeline
from examples.lingbot_video.lingbot_video_moe_30b import build_pipeline as build_moe_pipeline
from examples.lingbot_video.lingbot_video_moe_30b import build_refiner
from telefuser.core.config import ParallelConfig
from telefuser.pipelines.lingbot_video import (
    LingBotVideoRequest,
    load_lingbot_video_prompt,
    prepare_refiner_video,
)


def _write_video(frames: torch.Tensor, output: Path, fps: int) -> None:
    """Encode normalized RGB frames from the rank-zero worker result."""
    from diffusers.utils import export_to_video

    output.parent.mkdir(parents=True, exist_ok=True)
    video = frames[0].permute(1, 2, 3, 0).float().clamp(0.0, 1.0).cpu().numpy()
    export_to_video(list(video), str(output), fps=fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--caption-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variant", choices=("dense", "moe"), default="dense")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refiner-height", type=int, default=1088)
    parser.add_argument("--refiner-width", type=int, default=1920)
    parser.add_argument("--refiner-steps", type=int, default=8)
    parser.add_argument("--refiner-guidance-scale", type=float, default=3.0)
    parser.add_argument("--refiner-shift", type=float, default=3.0)
    parser.add_argument("--refiner-t-thresh", type=float, default=0.85)
    parser.add_argument("--refiner-tail-steps", type=int, default=2)
    parser.add_argument(
        "--refiner-batch-cfg",
        action="store_true",
        help="Use batched CFG for refiner; disabled by default so 1088p SP4 fits on four H100s.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    if args.refine and args.variant != "moe":
        raise ValueError("--refine requires --variant moe")

    if torch.cuda.device_count() < 4:
        raise RuntimeError("native LingBot SP4 validation requires four visible CUDA devices")
    caption, _ = load_lingbot_video_prompt(args.caption_json)
    pipeline_builder = build_dense_pipeline if args.variant == "dense" else build_moe_pipeline
    pipeline = pipeline_builder(
        args.model_dir,
        cpu_offload=False,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        parallel_config=ParallelConfig(
            device_ids=[0, 1, 2, 3],
            sp_ulysses_degree=4,
            enable_fsdp=True,
        ),
    )
    try:
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        generation = pipeline.generate(
            LingBotVideoRequest(
                caption=caption,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
            ),
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )
        output = generation.output
        base_elapsed_seconds = time.perf_counter() - started
        refiner_elapsed_seconds = None
        if args.refine:
            pipeline.release_gpu_resources()
            refiner = build_refiner(
                args.model_dir,
                cpu_offload=False,
                parallel_config=ParallelConfig(
                    device_ids=[0, 1, 2, 3],
                    sp_ulysses_degree=4,
                    enable_fsdp=True,
                ),
                batch_cfg=args.refiner_batch_cfg,
            )
            try:
                lowres_video, _ = prepare_refiner_video(
                    output,
                    source_fps=args.fps,
                    height=args.refiner_height,
                    width=args.refiner_width,
                )
                refiner_started = time.perf_counter()
                output = refiner.refine(
                    lowres_video,
                    generation.prompt_conditions.positive_prompt_embeds,
                    generation.prompt_conditions.negative_prompt_embeds,
                    generation.prompt_conditions.positive_attention_mask,
                    generation.prompt_conditions.negative_attention_mask,
                    num_inference_steps=args.refiner_steps,
                    guidance_scale=args.refiner_guidance_scale,
                    shift=args.refiner_shift,
                    t_thresh=args.refiner_t_thresh,
                    tail_steps=args.refiner_tail_steps,
                    generator=torch.Generator("cuda").manual_seed(args.seed),
                )
                refiner_elapsed_seconds = time.perf_counter() - refiner_started
            finally:
                refiner.close()
        _write_video(output, args.output, args.fps)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "world_size": 4,
                    "variant": args.variant,
                    "refine": args.refine,
                    "sp_ulysses_degree": 4,
                    "fsdp": True,
                    "batch_cfg": True,
                    "refiner_batch_cfg": args.refiner_batch_cfg if args.refine else None,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "steps": args.steps,
                    "base_elapsed_seconds": base_elapsed_seconds,
                    "refiner_elapsed_seconds": refiner_elapsed_seconds,
                    "peak_rank0_gpu_memory_mib": torch.cuda.max_memory_allocated(0) / 1024**2,
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
