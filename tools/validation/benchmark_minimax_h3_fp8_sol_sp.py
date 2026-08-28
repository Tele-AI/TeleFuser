# SPDX-License-Identifier: Apache-2.0
"""Benchmark matched MiniMax H3 baseline and FP8 Sol profiles."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import threading
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from examples.minimax_h3.common import save_generation
from examples.minimax_h3.minimax_h3_fl2va_h100 import PPL_CONFIG, get_pipeline, run
from telefuser.core.config import AttnImplType
from telefuser.pipelines.minimax_h3.pipeline import MiniMaxH3Generation


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _device_memory_mib(device_indices: list[int]) -> list[int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    values: dict[int, int] = {}
    for line in output.splitlines():
        index, used_mib = (int(value.strip()) for value in line.split(","))
        values[index] = used_mib
    return [values[index] for index in device_indices]


def _sample_device_memory(stop: threading.Event, peaks_mib: list[int], device_indices: list[int]) -> None:
    while not stop.is_set():
        current = _device_memory_mib(device_indices)
        for index, used_mib in enumerate(current):
            peaks_mib[index] = max(peaks_mib[index], used_mib)
        stop.wait(0.1)


def _physical_device_indices(gpu_num: int) -> list[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        indices = [int(value.strip()) for value in visible.split(",") if value.strip()]
        if len(indices) < gpu_num:
            raise ValueError("CUDA_VISIBLE_DEVICES exposes fewer GPUs than --gpu-num")
        return indices[:gpu_num]
    return list(range(gpu_num))


def _generate(pipeline: object, args: argparse.Namespace) -> MiniMaxH3Generation:
    return run(
        pipeline,
        prompt=args.prompt,
        seed=args.seed,
        aspect_ratio=args.aspect_ratio,
        target_video_length=args.duration,
        mode="t2va",
    )


def _save_arrays(result: MiniMaxH3Generation, output: Path) -> tuple[Path, Path]:
    frames_path = output.with_suffix(".frames.npy")
    audio_path = output.with_suffix(".audio.npy")
    frames = result.video[0].mul(255).clamp(0, 255).to(dtype=torch.uint8).numpy()
    np.save(frames_path, frames, allow_pickle=False)
    np.save(audio_path, result.audio[0].float().numpy(), allow_pickle=False)
    return frames_path, audio_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--profile", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--gpu-num", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--prompt", default=PPL_CONFIG["prompt"])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--sol-dense-steps", type=int, default=10)
    parser.add_argument("--sol-dense-layers", type=int, default=2)
    parser.add_argument("--sol-tau", type=float, default=1.0)
    parser.add_argument("--sol-threshold-type", choices=("exact", "diag"), default="exact")
    parser.add_argument("--sol-fp8-layer-start", type=int, default=0)
    parser.add_argument("--sol-fp8-layer-end", type=int)
    parser.add_argument("--sol-fp8-smoothing", choices=("none", "k", "kv"), default="kv")
    parser.add_argument(
        "--sol-fp8-v-bias-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path)
    args = parser.parse_args()

    optimized = args.profile == "optimized"
    load_started = time.perf_counter()
    pipeline = get_pipeline(
        args.gpu_num,
        args.model_root,
        num_inference_steps=args.steps,
        enable_fsdp=False,
        online_adaln_cache=True,
        attn_impl=AttnImplType.SOL_ATTN if optimized else AttnImplType.FLASH_ATTN_4,
        sol_fp8=optimized,
        sol_dense_steps=args.sol_dense_steps,
        sol_dense_layers=args.sol_dense_layers,
        sol_tau=args.sol_tau,
        sol_threshold_type=args.sol_threshold_type,
        sol_fp8_layer_start=args.sol_fp8_layer_start,
        sol_fp8_layer_end=args.sol_fp8_layer_end,
        sol_fp8_smoothing=args.sol_fp8_smoothing,
        sol_fp8_v_bias_correction=args.sol_fp8_v_bias_correction,
        quantization="tf-kernel-fp8" if optimized else None,
    )
    load_seconds = time.perf_counter() - load_started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        warmup_seconds = None
        if not args.no_warmup:
            warmup_started = time.perf_counter()
            warmup = _generate(pipeline, args)
            warmup_seconds = time.perf_counter() - warmup_started
            del warmup
            gc.collect()

        device_indices = _physical_device_indices(args.gpu_num)
        resident_memory_mib = _device_memory_mib(device_indices)
        peaks_mib = resident_memory_mib.copy()
        stop_sampling = threading.Event()
        sampler = threading.Thread(
            target=_sample_device_memory,
            args=(stop_sampling, peaks_mib, device_indices),
            daemon=True,
        )
        sampler.start()
        try:
            generation_started = time.perf_counter()
            result = _generate(pipeline, args)
            generation_seconds = time.perf_counter() - generation_started
        finally:
            stop_sampling.set()
            sampler.join()

        save_started = time.perf_counter()
        save_generation(result, args.output)
        frames_path, audio_path = _save_arrays(result, args.output)
        save_seconds = time.perf_counter() - save_started
    finally:
        pipeline.stop()

    denoising_seconds = float(result.runtime_metrics["denoising_seconds"])
    report = {
        "schema_version": 1,
        "profile": args.profile,
        "parallelism": {
            "world_size": args.gpu_num,
            "ulysses_degree": args.gpu_num // 2 if args.gpu_num == 4 else args.gpu_num,
            "tp_degree": 2 if args.gpu_num == 4 else 1,
        },
        "attention": "SOL_ATTN" if optimized else "FLASH_ATTN_4",
        "linear_quantization": "tf-kernel-fp8" if optimized else None,
        "sol": (
            {
                "fp8_qkv": True,
                "dense_steps": args.sol_dense_steps,
                "dense_layers": args.sol_dense_layers,
                "tau": args.sol_tau,
                "threshold_type": args.sol_threshold_type,
                "fp8_layer_start": args.sol_fp8_layer_start,
                "fp8_layer_end": args.sol_fp8_layer_end,
                "smoothing": args.sol_fp8_smoothing,
                "v_bias_correction": args.sol_fp8_v_bias_correction,
            }
            if optimized
            else None
        ),
        "model_root": str(Path(args.model_root)),
        "prompt": args.prompt,
        "duration_seconds": args.duration,
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "aspect_ratio": args.aspect_ratio,
        "output": str(args.output),
        "frames": str(frames_path),
        "audio": str(audio_path),
        "packed_sequence_length": result.packed_sequence_length,
        "video_shape": list(result.video.shape),
        "audio_shape": list(result.audio.shape),
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "generation_seconds": generation_seconds,
        "save_seconds": save_seconds,
        "denoising_steps_per_second": args.steps / denoising_seconds,
        "generated_video_seconds_per_second": args.duration / generation_seconds,
        "resident_memory_mib": resident_memory_mib,
        "peak_memory_mib": peaks_mib,
        "peak_memory_total_mib": sum(peaks_mib),
        "runtime_metrics": result.runtime_metrics,
        "versions": {
            "torch": _package_version("torch"),
            "telefuser": _package_version("telefuser"),
            "tf-kernel": _package_version("tf-kernel"),
        },
    }
    metrics_path = args.metrics_json or args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
