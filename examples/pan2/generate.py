# SPDX-License-Identifier: Apache-2.0
"""Generate a PAN2 video from a prompt and optional first frame."""

import argparse
from pathlib import Path

from PIL import Image
from common import build_pipeline
from diffusers.utils import export_to_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image")
    parser.add_argument("--negative_prompt")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--sol", action="store_true")
    parser.add_argument("--sol_fp8", action="store_true")
    parser.add_argument("--output_path", default="work_dirs/pan2.mp4")
    args = parser.parse_args()
    pipeline = build_pipeline(
        args.model_root,
        device_ids=args.device_ids,
        sp=args.sp,
        tp=args.tp,
        fp8=args.fp8,
        sol=args.sol,
        sol_fp8=args.sol_fp8,
    )
    try:
        image = Image.open(args.image).convert("RGB") if args.image else None
        frames = pipeline(
            args.prompt,
            image=image,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            fps=args.fps,
            seed=args.seed,
        )
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(list(frames.numpy()), str(path), fps=args.fps)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
