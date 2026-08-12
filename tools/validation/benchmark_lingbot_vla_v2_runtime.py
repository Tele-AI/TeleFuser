"""Benchmark upstream or TeleFuser LingBot-VLA v2 inference without parity capture hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize synchronized wall-clock latency samples in seconds."""
    if not values:
        raise ValueError("summary requires at least one value")
    total = sum(values)
    return {
        "count": len(values),
        "total_seconds": total,
        "mean_seconds": statistics.fmean(values),
        "stdev_seconds": statistics.pstdev(values),
        "min_seconds": min(values),
        "p50_seconds": percentile(values, 0.50),
        "p90_seconds": percentile(values, 0.90),
        "p95_seconds": percentile(values, 0.95),
        "p99_seconds": percentile(values, 0.99),
        "max_seconds": max(values),
        "throughput_requests_per_second": len(values) / total,
    }


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_cpu_inputs(path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with np.load(path) as artifact:
        required = {
            "images",
            "img_masks",
            "lang_tokens",
            "lang_masks",
            "state",
            "image_grid_thw",
            "initial_noise",
        }
        missing = sorted(required.difference(artifact.files))
        if missing:
            raise ValueError(f"parity input artifact is missing arrays: {missing}")
        tensors = {
            "images": torch.from_numpy(artifact["images"]).to(dtype=torch.bfloat16),
            "img_masks": torch.from_numpy(artifact["img_masks"]).to(dtype=torch.bool),
            "lang_tokens": torch.from_numpy(artifact["lang_tokens"]).to(dtype=torch.long),
            "lang_masks": torch.from_numpy(artifact["lang_masks"]).to(dtype=torch.bool),
            "state": torch.from_numpy(artifact["state"]).to(dtype=torch.bfloat16),
            "image_grid_thw": torch.from_numpy(artifact["image_grid_thw"]).to(dtype=torch.long),
        }
        noise = torch.from_numpy(artifact["initial_noise"]).to(dtype=torch.bfloat16)
    return tensors, noise


def _to_device(tensors: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device=device) for name, tensor in tensors.items()}


def _run_samples(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    runs: int,
) -> tuple[dict[str, float | int], torch.Tensor]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize(device)
    samples: list[float] = []
    output: torch.Tensor | None = None
    for _ in range(runs):
        torch.cuda.synchronize(device)
        started_at = time.perf_counter()
        output = operation()
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started_at)
    assert output is not None
    return summarize(samples), output


def _output_summary(output: torch.Tensor) -> dict[str, Any]:
    snapshot = output.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(snapshot.shape) != (1, 50, 55):
        raise RuntimeError(f"unexpected action shape: {tuple(snapshot.shape)}")
    if not torch.isfinite(snapshot).all():
        raise RuntimeError("benchmark produced non-finite actions")
    array = snapshot.numpy()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "sha256_float32_le": hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest(),
    }


def _load_upstream(args: argparse.Namespace, device: torch.device) -> tuple[Any, dict[str, Any]]:
    import capture_lingbot_vla_v2_upstream as upstream_capture

    upstream_root = args.upstream_root.resolve()
    commit = upstream_capture._git_commit(upstream_root)
    if commit != upstream_capture.UPSTREAM_COMMIT:
        raise RuntimeError(f"expected upstream commit {upstream_capture.UPSTREAM_COMMIT}, got {commit}")
    sys.path.insert(0, str(upstream_root))
    config = upstream_capture._build_config(args.qwen3vl_root.resolve())
    model = upstream_capture._load_official_model(args.model_root.resolve(), config, device)

    import lingbotvla.models.vla.lingbot_vla.qwen2_action_expert as upstream_moe

    backend = "robby_triton" if upstream_moe.robby_moe_forward is not None else "fused_fallback"
    return model, {
        "implementation": "official_upstream",
        "implementation_commit": commit,
        "attention_backend": "eager",
        "moe_backend": backend,
    }


def _load_telefuser(args: argparse.Namespace, device: torch.device) -> tuple[Any, dict[str, Any]]:
    from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline

    pipeline = get_lingbot_vla_v2_pipeline(
        str(args.model_root.resolve()),
        str(args.qwen3vl_root.resolve()),
        device=str(device),
    )
    model = pipeline.policy_stage.policy
    config = model.config
    return (pipeline, model), {
        "implementation": "telefuser",
        "implementation_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "attention_backend": str(config.attention_implementation),
        "moe_backend": "robby_triton" if bool(config.use_robby_moe_kernel) else "fused_fallback",
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run synchronized core-model and runtime-boundary benchmarks."""
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LingBot-VLA v2 runtime benchmarking requires CUDA")
    torch.cuda.set_device(device)
    torch.empty(1, device=device)
    cpu_inputs, cpu_noise = _load_cpu_inputs(args.input_artifact)

    torch.cuda.reset_peak_memory_stats(device)
    load_started_at = time.perf_counter()
    loaded, identity = (
        _load_upstream(args, device) if args.implementation == "upstream" else _load_telefuser(args, device)
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started_at
    pipeline = loaded[0] if args.implementation == "telefuser" else None
    model = loaded[1] if args.implementation == "telefuser" else loaded

    device_inputs = _to_device(cpu_inputs, device)
    device_noise = cpu_noise.to(device=device)

    @torch.inference_mode()
    def core_model() -> torch.Tensor:
        return model.sample_actions(**device_inputs, noise=device_noise)

    @torch.inference_mode()
    def runtime_request() -> torch.Tensor:
        if pipeline is not None:
            from telefuser.pipelines.lingbot_vla_v2.data import LingBotVlaV2Inputs

            prepared = LingBotVlaV2Inputs(**cpu_inputs)
            return pipeline.predict(prepared, seed=args.seed).canonical_normalized_actions.unsqueeze(0)
        request_inputs = _to_device(cpu_inputs, device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(
            1,
            int(model.config.n_action_steps),
            int(model.config.max_action_dim),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        return model.sample_actions(**request_inputs, noise=noise).detach().to(device="cpu", dtype=torch.float32)

    try:
        core_latency, core_output = _run_samples(
            core_model,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
        )
        runtime_latency, runtime_output = _run_samples(
            runtime_request,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
        )
        return {
            "schema_version": 1,
            "benchmark": "lingbot_vla_v2_upstream_telefuser_runtime",
            **identity,
            "model_root": str(args.model_root.resolve()),
            "qwen3vl_root": str(args.qwen3vl_root.resolve()),
            "input_artifact": str(args.input_artifact.resolve()),
            "seed": args.seed,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "environment": {
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "transformers_version": transformers.__version__,
                "platform": platform.platform(),
            },
            "load_seconds": load_seconds,
            "core_model_latency": core_latency,
            "runtime_request_latency": runtime_latency,
            "core_model_output": _output_summary(core_output),
            "runtime_request_output": _output_summary(runtime_output),
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "measurement_notes": [
                "No parity capture hooks are installed.",
                "core_model reuses device-resident parity inputs and fixed initial noise.",
                (
                    "runtime_request includes CPU-to-GPU input transfer, seeded noise creation, validation, "
                    "and CPU output transfer."
                ),
                (
                    "Image decoding and preprocessing are excluded because both sides consume the same frozen "
                    "parity tensors."
                ),
            ],
        }
    finally:
        if pipeline is not None:
            pipeline.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("upstream", "telefuser"), required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("work_dirs/lingbot-vla-v2-upstream"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--qwen3vl-root", type=Path, required=True)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"implementation": args.implementation, "output": str(args.output), "passed": True}))


if __name__ == "__main__":
    main()
