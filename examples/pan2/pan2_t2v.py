"""Run PAN 2 through TeleFuser's NVFP4 loader."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffusers.utils import export_to_video

from telefuser.pipelines.pan2 import PAN2Pipeline, PAN2PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="IFM/PAN2")
    parser.add_argument("--prompt", default="A small red car drives through a rainy city street.")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("pan2.mp4"))
    parser.add_argument("--nvfp4", action="store_true")
    parser.add_argument("--layerwise-nvfp4-dir")
    args = parser.parse_args()

    pipe = PAN2Pipeline.from_pretrained(
        PAN2PipelineConfig(
            model_id=args.model_id,
            nvfp4=args.nvfp4,
            cpu_offload=True,
            layerwise_nvfp4_dir=args.layerwise_nvfp4_dir,
        )
    )
    request = {"prompt": args.prompt, "num_inference_steps": args.steps, "output_type": "np"}
    if args.image:
        request["image"] = str(args.image)
    result = pipe(**request)
    frames = getattr(result, "frames", getattr(result, "videos", result))
    export_to_video(frames[0] if isinstance(frames, list) else frames, str(args.output), fps=24)
    print(args.output)


if __name__ == "__main__":
    main()
