# SPDX-License-Identifier: Apache-2.0
"""Generate an image or edit up to sixteen references with MiniMax-H3-Image."""

import argparse
from pathlib import Path

from PIL import Image
from common import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_root", required=True)
    parser.add_argument("--text_encoder_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", nargs="*", default=[])
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--min_generation_area", type=int, default=2048 * 2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--sol", action="store_true")
    parser.add_argument("--sol_fp8", action="store_true")
    parser.add_argument("--output_path", default="work_dirs/h3-image.png")
    args = parser.parse_args()
    pipeline = build_pipeline(
        args.model_root,
        args.text_encoder_path,
        device_ids=args.device_ids,
        sp=args.sp,
        tp=args.tp,
        fp8=args.fp8,
        sol=args.sol,
        sol_fp8=args.sol_fp8,
    )
    try:
        images = [Image.open(path) for path in args.image]
        output = pipeline(
            args.prompt,
            image=images,
            height=args.height,
            width=args.width,
            min_generation_area=args.min_generation_area,
            seed=args.seed,
        )
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        output.save(path)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
