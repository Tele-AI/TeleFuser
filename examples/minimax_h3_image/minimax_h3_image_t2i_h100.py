# SPDX-License-Identifier: Apache-2.0
"""Generate an image or edit up to sixteen references with MiniMax-H3-Image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from examples.minimax_h3_image.common import build_pipeline
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    from common import build_pipeline
from telefuser.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline

# Configuration
TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")
PPL_CONFIG: dict[str, Any] = {
    "name": "minimax_h3_image_t2i_h100",
    "model_root": TF_MODEL_ZOO_PATH + "/MiniMaxAI_MiniMax-H3-Image",
    "prompt": "A red fox in a snowy forest, detailed natural photograph.",
    "height": 2048,
    "width": 2048,
    "min_generation_area": 4194304,
    "seed": 0,
    "output_path": "work_dirs/h3-image.png",
}


# Model loading
def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    *,
    text_encoder_path: str | None = None,
    device_ids: list[int] | None = None,
    sp: int | None = None,
    tp: int = 1,
    fp8: bool = False,
    sol: bool = False,
    sol_fp8: bool = False,
) -> MiniMaxH3ImagePipeline:
    """Load the released checkpoint with TeleFuser model CPU offload and SP/TP workers."""
    if parallelism < 1 or tp < 1:
        raise ValueError("parallelism and tp must be positive")
    devices = list(range(parallelism)) if device_ids is None else device_ids
    if not devices or len(set(devices)) != len(devices) or any(d < 0 for d in devices):
        raise ValueError("device_ids must contain distinct non-negative CUDA device IDs")
    if device_ids is not None and parallelism not in (1, len(devices)):
        raise ValueError("gpu_num must match the number of explicit device_ids")
    sp = len(devices) // tp if sp is None else sp
    if sp < 1 or sp * tp != len(devices):
        raise ValueError("The number of devices must equal sp * tp")
    return build_pipeline(
        model_root,
        text_encoder_path or str(Path(model_root) / "text_encoder"),
        device_ids=devices,
        sp=sp,
        tp=tp,
        fp8=fp8,
        sol=sol,
        sol_fp8=sol_fp8,
    )


# Inference
def run(
    pipeline: MiniMaxH3ImagePipeline,
    prompt: str = PPL_CONFIG["prompt"],
    seed: int = PPL_CONFIG["seed"],
    *,
    image: list[str] | None = None,
    height: int = PPL_CONFIG["height"],
    width: int = PPL_CONFIG["width"],
    min_generation_area: int = PPL_CONFIG["min_generation_area"],
) -> Image.Image:
    """Run the official conditioning and sampling flow; return the generated output."""
    images = []
    for image_path in image or []:
        with Image.open(image_path) as source:
            images.append(source.copy())
    return pipeline(
        prompt=prompt,
        image=images,
        height=height,
        width=width,
        min_generation_area=min_generation_area,
        seed=seed,
    )


def run_with_file(
    pipeline: MiniMaxH3ImagePipeline,
    prompt: str = PPL_CONFIG["prompt"],
    seed: int = PPL_CONFIG["seed"],
    output_path: str = PPL_CONFIG["output_path"],
    **kwargs: Any,
) -> dict[str, str]:
    """Run inference and save the result without stopping the reusable pipeline."""
    output = run(pipeline, prompt=prompt, seed=seed, **kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.save(path)
    return {"output_path": str(path)}


# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--text_encoder_path", help="Defaults to MODEL_ROOT/text_encoder")
    parser.add_argument("--prompt", default=PPL_CONFIG["prompt"])
    parser.add_argument("--image", nargs="*", default=[])
    parser.add_argument("--height", type=int, default=PPL_CONFIG["height"])
    parser.add_argument("--width", type=int, default=PPL_CONFIG["width"])
    parser.add_argument("--min_generation_area", type=int, default=PPL_CONFIG["min_generation_area"])
    parser.add_argument("--seed", type=int, default=PPL_CONFIG["seed"])
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of visible CUDA devices")
    parser.add_argument("--device_ids", type=int, nargs="+", help="Optional explicit visible device IDs")
    parser.add_argument("--sp", type=int, help="Ulysses degree; defaults to gpu_num / tp")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--sol", action="store_true")
    parser.add_argument("--sol_fp8", action="store_true")
    parser.add_argument("--output_path", default=PPL_CONFIG["output_path"])
    args = parser.parse_args()
    pipeline = get_pipeline(
        parallelism=args.gpu_num,
        model_root=args.model_root,
        text_encoder_path=args.text_encoder_path,
        device_ids=args.device_ids,
        sp=args.sp,
        tp=args.tp,
        fp8=args.fp8,
        sol=args.sol,
        sol_fp8=args.sol_fp8,
    )
    try:
        result = run_with_file(
            pipeline,
            prompt=args.prompt,
            output_path=args.output_path,
            image=args.image,
            height=args.height,
            width=args.width,
            min_generation_area=args.min_generation_area,
            seed=args.seed,
        )
        print(f"Output saved to {result['output_path']}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
