"""Compare official PAN 2 BF16 with TeleFuser NVFP4.

The two runs use identical prompt, seed, resolution, frame count and sampler
steps. Outputs are written beside a JSON report containing wall latency,
peak CUDA memory, generated FPS and realtime ratio.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video

from telefuser.pipelines.pan2 import PAN2Pipeline, PAN2PipelineConfig


def _frames(result: object) -> np.ndarray:
    value = getattr(result, "frames", None)
    if value is None:
        value = getattr(result, "videos", result)
    if isinstance(value, dict) and "frames" in value:
        value = value["frames"]
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 5:
        value = value[0]
    if value.ndim != 4:
        raise ValueError(f"PAN2 returned unexpected frames shape: {value.shape}")
    return value


def _run(label: str, pipe: object, args: argparse.Namespace) -> dict[str, object]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    start = time.perf_counter()
    request = dict(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        generator=generator,
        output_type="np",
    )
    if args.negative_prompt:
        request["negative_prompt"] = args.negative_prompt
    result = pipe(**request)
    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    frames = _frames(result)
    output = args.output_dir / f"{label}.mp4"
    export_to_video(frames, str(output), fps=args.fps)
    generated_seconds = frames.shape[0] / args.fps
    return {
        "label": label,
        "latency_seconds": latency,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "frames": int(frames.shape[0]),
        "video_seconds": generated_seconds,
        "generated_video_fps": frames.shape[0] / latency,
        "realtime_ratio": generated_seconds / latency,
        "output": str(output),
    }


def _load_offloaded_official(model_id: str):
    pipe = ModularPipeline.from_pretrained(model_id)
    pipe.load_components(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if hasattr(pipe, "disable_safety_checker"):
        pipe.disable_safety_checker()
    for component in (getattr(pipe, "transformer", None), getattr(pipe, "text_encoder", None)):
        if component is not None and hasattr(component, "enable_group_offload"):
            component.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                use_stream=True,
            )
    pipe.to("cpu")
    if hasattr(pipe, "disable_safety_checker"):
        pipe.disable_safety_checker()
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="IFM/PAN2")
    parser.add_argument("--prompt", default="A red car drives through a rainy city street.")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pan2-benchmark"))
    parser.add_argument("--layerwise-nvfp4-dir", default=None)
    parser.add_argument("--only-telefuser", action="store_true")
    parser.add_argument("--prompt-json", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    args = parser.parse_args()
    if args.prompt_json:
        records = json.loads(args.prompt_json.read_text())
        record = records[args.prompt_index]
        args.prompt = record.get("refined_prompt") or record.get("prompt") or record.get("original_prompt")
        args.negative_prompt = record.get("negative_prompt")
        if record.get("duration_sec"):
            args.num_frames = int(4 * np.ceil(record["duration_sec"] * args.fps / 4) + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    if not args.only_telefuser:
        official = _load_offloaded_official(args.model_id)
        runs.append(_run("official-bf16", official, args))
        del official
        torch.cuda.empty_cache()

    telefuser = PAN2Pipeline.from_pretrained(
        PAN2PipelineConfig(
            model_id=args.model_id,
            nvfp4=True,
            cpu_offload=True,
            layerwise_nvfp4_dir=args.layerwise_nvfp4_dir,
        )
    )
    runs.append(_run("telefuser-nvfp4", telefuser, args))
    report = {"hardware": torch.cuda.get_device_name(), "config": dict(vars(args)), "runs": runs}
    report["config"]["output_dir"] = str(report["config"]["output_dir"])
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
