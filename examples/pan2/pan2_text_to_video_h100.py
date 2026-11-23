# SPDX-License-Identifier: Apache-2.0
"""Generate a PAN2 video from a prompt and optional first frame."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

try:
    from examples.pan2.common import build_pipeline
except ModuleNotFoundError as exc:
    if exc.name != "examples":
        raise
    from common import build_pipeline
from telefuser.pipelines.pan2 import PAN2Pipeline
from telefuser.utils.video import save_video

# Configuration
TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")
PPL_CONFIG: dict[str, Any] = {
    "name": "pan2_text_to_video_h100",
    "model_root": TF_MODEL_ZOO_PATH + "/IFM_PAN2",
    "prompt": "A red fox walks through a snowy forest, tracking camera, natural light.",
    "height": 704,
    "width": 1248,
    "num_frames": 121,
    "num_inference_steps": 50,
    "guidance_scale": 2.5,
    "fps": 24.0,
    "seed": 0,
    "output_path": "work_dirs/pan2.mp4",
}


# Model loading
def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    *,
    device_ids: list[int] | None = None,
    sp: int | None = None,
    tp: int = 1,
    fp8: bool = False,
    sol: bool = False,
) -> PAN2Pipeline:
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
        device_ids=devices,
        sp=sp,
        tp=tp,
        fp8=fp8,
        sol=sol,
    )


# Inference
def run(
    pipeline: PAN2Pipeline,
    prompt: str = PPL_CONFIG["prompt"],
    seed: int = PPL_CONFIG["seed"],
    *,
    image: str | None = None,
    negative_prompt: str | None = None,
    height: int = PPL_CONFIG["height"],
    width: int = PPL_CONFIG["width"],
    num_frames: int = PPL_CONFIG["num_frames"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    guidance_scale: float = PPL_CONFIG["guidance_scale"],
    fps: float = PPL_CONFIG["fps"],
) -> torch.Tensor:
    """Run the official conditioning and sampling flow; return the generated output."""
    first_frame = None
    if image:
        with Image.open(image) as source:
            first_frame = source.convert("RGB")
    return pipeline(
        prompt=prompt,
        image=first_frame,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        fps=fps,
        seed=seed,
    )


def run_with_file(
    pipeline: PAN2Pipeline,
    prompt: str = PPL_CONFIG["prompt"],
    seed: int = PPL_CONFIG["seed"],
    output_path: str = PPL_CONFIG["output_path"],
    **kwargs: Any,
) -> dict[str, str]:
    """Run inference and save the result without stopping the reusable pipeline."""
    output = run(pipeline, prompt=prompt, seed=seed, **kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = (output.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    save_video(list(frames), str(path), fps=kwargs.get("fps", PPL_CONFIG["fps"]))
    return {"output_path": str(path)}


# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--prompt", default=PPL_CONFIG["prompt"])
    parser.add_argument("--image")
    parser.add_argument("--negative_prompt")
    parser.add_argument("--height", type=int, default=PPL_CONFIG["height"])
    parser.add_argument("--width", type=int, default=PPL_CONFIG["width"])
    parser.add_argument("--num_frames", type=int, default=PPL_CONFIG["num_frames"])
    parser.add_argument("--num_inference_steps", type=int, default=PPL_CONFIG["num_inference_steps"])
    parser.add_argument("--guidance_scale", type=float, default=PPL_CONFIG["guidance_scale"])
    parser.add_argument("--fps", type=float, default=PPL_CONFIG["fps"])
    parser.add_argument("--seed", type=int, default=PPL_CONFIG["seed"])
    parser.add_argument("--gpu_num", type=int, default=1, help="Number of visible CUDA devices")
    parser.add_argument("--device_ids", type=int, nargs="+", help="Optional explicit visible device IDs")
    parser.add_argument("--sp", type=int, help="Ulysses degree; defaults to gpu_num / tp")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--sol", action="store_true")
    parser.add_argument("--output_path", default=PPL_CONFIG["output_path"])
    args = parser.parse_args()
    pipeline = get_pipeline(
        parallelism=args.gpu_num,
        model_root=args.model_root,
        device_ids=args.device_ids,
        sp=args.sp,
        tp=args.tp,
        fp8=args.fp8,
        sol=args.sol,
    )
    try:
        result = run_with_file(
            pipeline,
            prompt=args.prompt,
            output_path=args.output_path,
            image=args.image,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            fps=args.fps,
            seed=args.seed,
        )
        print(f"Output saved to {result['output_path']}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
