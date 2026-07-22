"""Run one source-equivalent LingBot-Video request with FSDP and Ulysses SP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.lingbot_video.lingbot_video_dense_1_3b import build_pipeline as build_dense_pipeline
from examples.lingbot_video.lingbot_video_moe_30b import build_pipeline as build_moe_pipeline
from telefuser.core.config import ParallelConfig
from telefuser.pipelines.lingbot_video import (
    LingBotVideoRequest,
    load_lingbot_video_prompt,
)


def _write_video(frames: torch.Tensor, output: Path, fps: int) -> None:
    """Encode normalized TeleFuser RGB frames without a second uint8 conversion."""
    from diffusers.utils import export_to_video

    output.parent.mkdir(parents=True, exist_ok=True)
    video = frames[0].permute(1, 2, 3, 0).float().clamp(0.0, 1.0).cpu().numpy()
    export_to_video(list(video), str(output), fps=fps)


def _max_across_ranks(value: float, device: torch.device) -> float:
    """Return a scalar maximum across the default NCCL group."""
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--caption-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variant", choices=("dense", "moe"), default="dense")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        world_size = dist.get_world_size()
        if world_size < 2:
            raise ValueError("distributed LingBot validation requires at least two ranks")
        parallel_config = ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ulysses_degree=world_size,
            enable_fsdp=True,
        )
        caption, _ = load_lingbot_video_prompt(args.caption_json)
        torch.cuda.reset_peak_memory_stats(device)
        pipeline_builder = build_dense_pipeline if args.variant == "dense" else build_moe_pipeline
        pipeline = pipeline_builder(
            args.model_dir,
            cpu_offload=False,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            parallel_config=parallel_config,
        )
        dist.barrier()
        started = time.perf_counter()
        output = pipeline.generate(
            LingBotVideoRequest(
                caption=caption,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
            ),
            generator=torch.Generator(device=device).manual_seed(args.seed),
        ).output
        elapsed_seconds = _max_across_ranks(time.perf_counter() - started, device)
        reference = output.clone() if dist.get_rank() == 0 else torch.empty_like(output)
        dist.broadcast(reference, src=0)
        max_abs = _max_across_ranks(float((output.float() - reference.float()).abs().max().item()), device)
        peak_memory_mib = _max_across_ranks(torch.cuda.max_memory_allocated(device) / 1024**2, device)
        if dist.get_rank() == 0:
            _write_video(output, args.output, args.fps)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(
                    {
                        "variant": args.variant,
                        "world_size": world_size,
                        "sp_ulysses_degree": world_size,
                        "fsdp": True,
                        "height": args.height,
                        "width": args.width,
                        "num_frames": args.num_frames,
                        "steps": args.steps,
                        "rank_output_max_abs": max_abs,
                        "elapsed_seconds": elapsed_seconds,
                        "peak_gpu_memory_mib": peak_memory_mib,
                        "output": str(args.output),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
