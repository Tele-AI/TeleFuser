"""Run an official Diffusers LTX-Video workload with TorchAO NVFP4."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video
from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
from torchao.quantization import quantize_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Lightricks/LTX-Video")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--prompt",
        default="A close-up cinematic shot of a chef plating a colorful gourmet dish in a bright kitchen.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ltx-nvfp4"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_id = str(args.model_dir or args.model_id)
    pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    nvfp4 = NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False)
    quantize_(pipe.transformer, nvfp4)
    # LTX's T5 encoder is ~19 GB in BF16; quantizing it too is what makes a
    # complete single-card pipeline possible on a 24 GiB Blackwell GPU.
    if pipe.text_encoder is not None:
        quantize_(pipe.text_encoder, nvfp4)
    # Quantization is performed on CPU to avoid materializing BF16 weights and
    # quantized copies simultaneously on the 24 GiB device.
    pipe.to("cuda")
    for module in (pipe.transformer, pipe.text_encoder):
        if module is None:
            continue
        for parameter in module.parameters():
            tensor = parameter.data
            if hasattr(tensor, "act_quant_kwargs") and tensor.act_quant_kwargs is not None:
                tensor.act_quant_kwargs.use_triton_kernel = True
    pipe.transformer.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    start = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=3.0,
        generator=generator,
        output_type="np",
    )
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    frames = result.frames[0]
    if isinstance(frames, np.ndarray):
        frames = np.clip(frames * 255, 0, 255).astype(np.uint8)
    output = args.output_dir / "ltx-nvfp4.mp4"
    export_to_video(frames, str(output), fps=args.fps)
    report = {
        "model": model_id,
        "quantization": "torchao-nvfp4-dynamic-activation-weight",
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": len(frames),
        "steps": args.steps,
        "fps": args.fps,
        "seed": args.seed,
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
