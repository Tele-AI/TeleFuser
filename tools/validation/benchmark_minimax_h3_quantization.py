# SPDX-License-Identifier: Apache-2.0
"""Benchmark MiniMax H3 dense/Sol and BF16/FP8 single-GPU profiles."""

from __future__ import annotations

import argparse
import json
import time
from importlib import metadata
from pathlib import Path

from examples.minimax_h3.common import load_minimax_h3_pipeline, save_generation
from telefuser.core.config import AttnImplType


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3")
    parser.add_argument(
        "--backend",
        choices=("bf16", "bf16-sol", "fp8", "fp8-sol", "torchao-fp8", "bnb-nf4"),
        required=True,
    )
    parser.add_argument("--prompt", default="Steam rises from the ramen while the family talks in the background.")
    parser.add_argument("--prompt-file", type=Path, help="JSON file containing a top-level prompt string.")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path)
    args = parser.parse_args()
    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = json.loads(args.prompt_file.read_text(encoding="utf-8"))["prompt"]

    uses_sol = args.backend.endswith("-sol")
    quantization = "tf-kernel-fp8" if args.backend in {"fp8", "fp8-sol"} else args.backend
    if args.backend in {"bf16", "bf16-sol"}:
        quantization = None
    load_started = time.perf_counter()
    pipeline = load_minimax_h3_pipeline(
        args.model_root,
        partition="FL2VA",
        device=args.device,
        num_inference_steps=args.steps,
        attn_impl=AttnImplType.SOL_ATTN if uses_sol else AttnImplType.FLASH_ATTN_4,
        sol_fp8=args.backend == "fp8-sol",
        quantization=quantization,
    )
    load_seconds = time.perf_counter() - load_started
    try:
        generation_started = time.perf_counter()
        result = pipeline(
            task="t2va",
            prompt=prompt,
            conditions=[],
            target={
                "short_edge": 768,
                "aspect_ratio": args.aspect_ratio,
                "duration_seconds": args.duration,
            },
            seed=args.seed,
        )
        generation_seconds = time.perf_counter() - generation_started
        save_started = time.perf_counter()
        save_generation(result, args.output)
        save_seconds = time.perf_counter() - save_started
    finally:
        pipeline.stop()

    denoising_seconds = float(result.runtime_metrics["denoising_seconds"])
    report = {
        "backend": args.backend,
        "model_root": str(Path(args.model_root)),
        "output": str(args.output),
        "prompt": prompt,
        "duration_seconds": args.duration,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "aspect_ratio": args.aspect_ratio,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "denoising_steps_per_second": args.steps / denoising_seconds,
        "generated_video_seconds_per_second": args.duration / generation_seconds,
        "save_seconds": save_seconds,
        "runtime_metrics": result.runtime_metrics,
        "versions": {
            "torch": _package_version("torch"),
            "torchao": _package_version("torchao"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "telefuser": _package_version("telefuser"),
        },
    }
    metrics_path = args.metrics_json or args.output.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
