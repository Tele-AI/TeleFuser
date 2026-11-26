"""Run the official Diffusers LTX-Video text-to-video workload.

Workload source: https://huggingface.co/docs/diffusers/api/pipelines/ltx_video
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video

from telefuser.pipelines.ltx_video_2b import LTXVideo2BPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Lightricks/LTX-Video")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=161)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--prompt",
        default=(
            "A woman with long brown hair and light skin smiles at another woman with long blonde hair. "
            "The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her "
            "right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. "
            "The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. "
            "The scene appears to be real-life footage"
        ),
    )
    parser.add_argument("--negative-prompt", default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--decode-timestep", type=float, default=0.03)
    parser.add_argument("--decode-noise-scale", type=float, default=0.025)
    parser.add_argument("--quantization", choices=("nvfp4", "bf16"), default="nvfp4")
    parser.add_argument(
        "--nvfp4-profile",
        choices=("quality", "full"),
        default="quality",
        help="Quality quantizes FFN GEMMs only; full quantizes every supported Linear.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ltx-nvfp4"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_id = str(args.model_dir or args.model_id)
    pipe = LTXVideo2BPipeline.from_pretrained(
        model_id,
        quantization=args.quantization,
        nvfp4_profile=args.nvfp4_profile,
        torch_dtype=torch.bfloat16,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=3.0,
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        seed=args.seed,
    )
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    # Diffusers' ``output_type="np"`` returns float frames in [0, 1], which
    # export_to_video scales itself. Avoid converting to uint8 beforehand,
    # otherwise export_to_video would multiply by 255 a second time.
    output = args.output_dir / f"ltx-{args.quantization}.mp4"
    export_to_video(frames, str(output), fps=args.fps)
    report = {
        "model": model_id,
        "quantization": args.quantization,
        "nvfp4_profile": args.nvfp4_profile if args.quantization == "nvfp4" else None,
        "runtime": "TeleFuser LTXVideo2BPipeline",
        "nvfp4_linear_count": pipe.nvfp4_linear_count,
        "offload": args.quantization == "bf16",
        "text_encoder_quantization": "bf16_encode_then_cpu" if args.quantization == "nvfp4" else "bf16_cpu_offload",
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": len(frames),
        "steps": args.steps,
        "fps": args.fps,
        "seed": args.seed,
        "decode_timestep": args.decode_timestep,
        "decode_noise_scale": args.decode_noise_scale,
        "latency_seconds": latency,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "generated_fps": len(frames) / latency,
        "realtime_ratio": len(frames) / latency / args.fps,
        "output": str(output),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
