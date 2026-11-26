"""Generate LTX-Video 2B text-to-video with TeleFuser NVFP4 on RTX 5090."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from diffusers.utils import export_to_video

from telefuser.pipelines.ltx_video_2b import LTXVideo2BPipeline

PPL_CONFIG: dict[str, Any] = {
    "model_root": os.environ.get("TF_MODEL_ZOO_PATH", ".") + "/ltx-video-2b",
    "prompt": (
        "A woman with long brown hair and light skin smiles at another woman with long blonde hair. "
        "The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her "
        "right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. "
        "The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. "
        "The scene appears to be real-life footage"
    ),
    "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
    "height": 512,
    "width": 768,
    "num_frames": 161,
    "num_inference_steps": 50,
    "fps": 24,
    "seed": 123,
}


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    *,
    quantization: str = "nvfp4",
    nvfp4_profile: str = "quality",
) -> LTXVideo2BPipeline:
    """Load one reusable TeleFuser pipeline; LTX-Video 2B is single-GPU here."""
    if parallelism != 1:
        raise ValueError("This LTX-Video 2B integration currently supports one GPU")
    return LTXVideo2BPipeline.from_pretrained(
        model_root,
        device="cuda",
        torch_dtype=torch.bfloat16,
        quantization=quantization,
        nvfp4_profile=nvfp4_profile,
    )


def run(pipeline: LTXVideo2BPipeline, prompt: str = PPL_CONFIG["prompt"], **kwargs: Any):
    """Run the official LTX workload and return float RGB frames in [0, 1]."""
    return pipeline(prompt, **kwargs)


def run_with_file(
    pipeline: LTXVideo2BPipeline,
    prompt: str = PPL_CONFIG["prompt"],
    output_path: str = "work_dirs/ltx-video-2b-nvfp4.mp4",
    **kwargs: Any,
) -> dict[str, str]:
    frames = run(pipeline, prompt, **kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=kwargs.get("frame_rate", PPL_CONFIG["fps"]))
    return {"output_path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--prompt", default=PPL_CONFIG["prompt"])
    parser.add_argument("--negative_prompt", default=PPL_CONFIG["negative_prompt"])
    parser.add_argument("--height", type=int, default=PPL_CONFIG["height"])
    parser.add_argument("--width", type=int, default=PPL_CONFIG["width"])
    parser.add_argument("--num_frames", type=int, default=PPL_CONFIG["num_frames"])
    parser.add_argument("--num_inference_steps", type=int, default=PPL_CONFIG["num_inference_steps"])
    parser.add_argument("--fps", type=int, default=PPL_CONFIG["fps"])
    parser.add_argument("--seed", type=int, default=PPL_CONFIG["seed"])
    parser.add_argument("--quantization", choices=("nvfp4", "bf16"), default="nvfp4")
    parser.add_argument("--nvfp4_profile", choices=("quality", "full"), default="quality")
    parser.add_argument("--gpu_num", type=int, default=1)
    parser.add_argument("--output_path", default="work_dirs/ltx-video-2b-nvfp4.mp4")
    args = parser.parse_args()
    pipeline = get_pipeline(
        args.gpu_num, args.model_root, quantization=args.quantization, nvfp4_profile=args.nvfp4_profile
    )
    result = run_with_file(
        pipeline,
        prompt=args.prompt,
        output_path=args.output_path,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.fps,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )
    print(f"Output saved to {result['output_path']}")


if __name__ == "__main__":
    main()
