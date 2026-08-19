"""Benchmark one in-process LingBot-VLA v2 service replica on a single GPU."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psutil
import torch
from PIL import Image

from telefuser.metrics.runtime import collect_runtime_environment
from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline
from telefuser.pipelines.lingbot_vla_v2.service import (
    LingBotVlaV2ActionRequest,
    predict_lingbot_vla_v2_action,
)

_MIB = 1024**2


class PeakRssSampler:
    """Sample process RSS while one benchmark phase is active."""

    def __init__(self, process: psutil.Process, interval_s: float = 0.01) -> None:
        self.process = process
        self.interval_s = interval_s
        self.peak_bytes = process.memory_info().rss
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRssSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s * 4, 0.1))
        self._record()

    def _record(self) -> None:
        try:
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        except psutil.Error:
            pass

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._record()


def parse_image_sizes(value: str) -> tuple[tuple[int, int], ...]:
    """Parse a comma-separated WIDTHxHEIGHT list."""
    sizes: list[tuple[int, int]] = []
    for item in value.split(","):
        parts = item.strip().lower().split("x", maxsplit=1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid image size {item!r}; expected WIDTHxHEIGHT")
        try:
            width, height = (int(part) for part in parts)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid image size {item!r}; expected integers") from error
        if width <= 0 or height <= 0:
            raise argparse.ArgumentTypeError("image dimensions must be positive")
        size = (width, height)
        if size not in sizes:
            sizes.append(size)
    if not sizes:
        raise argparse.ArgumentTypeError("at least one image size is required")
    return tuple(sizes)


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
    """Summarize a latency sample in seconds."""
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
        "max_seconds": max(values),
        "throughput_requests_per_second": len(values) / total,
    }


def encode_image(source: Image.Image, size: tuple[int, int], *, quality: int) -> tuple[str, int]:
    """Resize and JPEG-encode one service input outside measured request time."""
    image = source.resize(size, Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=False)
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode("ascii"), len(payload)


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_snapshot(device: torch.device, process: psutil.Process) -> dict[str, float | None]:
    result: dict[str, float | None] = {"cpu_rss_mib": process.memory_info().rss / _MIB}
    if device.type != "cuda":
        result.update(
            gpu_allocated_mib=None,
            gpu_reserved_mib=None,
            gpu_peak_allocated_mib=None,
            gpu_peak_reserved_mib=None,
        )
        return result
    result.update(
        gpu_allocated_mib=torch.cuda.memory_allocated(device) / _MIB,
        gpu_reserved_mib=torch.cuda.memory_reserved(device) / _MIB,
        gpu_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / _MIB,
        gpu_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / _MIB,
    )
    return result


def measure(
    operation: Callable[[], Any],
    *,
    device: torch.device,
    process: psutil.Process,
    synchronize_cuda: bool,
) -> tuple[Any, dict[str, Any]]:
    """Measure wall time and process/device memory for one operation."""
    if synchronize_cuda:
        _cuda_synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    before = _memory_snapshot(device, process)
    with PeakRssSampler(process) as rss_sampler:
        started_at = time.perf_counter()
        result = operation()
        if synchronize_cuda:
            _cuda_synchronize(device)
        elapsed = time.perf_counter() - started_at
    after = _memory_snapshot(device, process)
    return result, {
        "seconds": elapsed,
        "cpu_rss_before_mib": before["cpu_rss_mib"],
        "cpu_rss_after_mib": after["cpu_rss_mib"],
        "cpu_rss_peak_mib": rss_sampler.peak_bytes / _MIB,
        "gpu_allocated_after_mib": after["gpu_allocated_mib"],
        "gpu_reserved_after_mib": after["gpu_reserved_mib"],
        "gpu_peak_allocated_mib": after["gpu_peak_allocated_mib"],
        "gpu_peak_reserved_mib": after["gpu_peak_reserved_mib"],
    }


def _load_source_image(path: Path | None) -> Image.Image:
    if path is not None:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    return Image.new("RGB", (640, 480), color=(32, 96, 160))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Load one replica and return a JSON-serializable benchmark report."""
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be non-negative and --runs must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LingBot VLA v2 service benchmarking requires one visible CUDA GPU")
    process = psutil.Process()
    source = _load_source_image(args.image)
    encoded_by_size = {size: encode_image(source, size, quality=args.jpeg_quality) for size in args.image_sizes}

    pipeline, load_metrics = measure(
        lambda: get_lingbot_vla_v2_pipeline(
            str(args.model_root),
            str(args.qwen3vl_root),
            device=str(device),
            quantization=args.quantization,
        ),
        device=device,
        process=process,
        synchronize_cuda=False,
    )
    startup_warmup = None
    if args.startup_warmup:
        _, startup_warmup = measure(
            pipeline.warmup,
            device=device,
            process=process,
            synchronize_cuda=True,
        )

    executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot-vla-v2-benchmark")
        if args.execution_mode == "service-thread"
        else None
    )
    active_phases: dict[str, float] = {}
    original_prepare = pipeline.input_processor.prepare
    original_predict = pipeline.predict

    def measured_prepare(observation: Any) -> Any:
        started_at = time.perf_counter()
        result = original_prepare(observation)
        active_phases["preprocess_seconds"] = time.perf_counter() - started_at
        return result

    def measured_predict(inputs: Any, seed: int | None = None) -> Any:
        _cuda_synchronize(device)
        started_at = time.perf_counter()
        result = original_predict(inputs, seed=seed)
        _cuda_synchronize(device)
        active_phases["model_seconds"] = time.perf_counter() - started_at
        return result

    pipeline.input_processor.prepare = measured_prepare
    pipeline.predict = measured_predict

    def request_once(size: tuple[int, int]) -> tuple[dict[str, Any], dict[str, float]]:
        active_phases.clear()
        encoded, _ = encoded_by_size[size]
        payload = {
            "task": args.instruction,
            "state": [0.0] * 14,
            "camera_high": encoded,
            "camera_left_wrist": encoded,
            "camera_right_wrist": encoded,
            "seed": args.seed,
        }
        request = LingBotVlaV2ActionRequest.model_validate(payload)

        def invoke_request() -> Any:
            return predict_lingbot_vla_v2_action(
                pipeline,
                request,
                max_image_bytes=args.max_image_bytes,
            )

        def invoke_service_thread() -> Any:
            assert executor is not None
            return executor.submit(invoke_request).result()

        operation: Callable[[], Any] = invoke_service_thread if executor is not None else invoke_request
        response, metrics = measure(
            operation,
            device=device,
            process=process,
            synchronize_cuda=True,
        )
        if response.horizon != 50 or response.action_dim != 55:
            raise RuntimeError(f"unexpected action shape: {response.horizon}x{response.action_dim}")
        if not all(math.isfinite(value) for row in response.canonical_normalized_actions for value in row):
            raise RuntimeError("benchmark received non-finite actions")
        phases = dict(active_phases)
        phases["boundary_seconds"] = max(
            metrics["seconds"] - phases.get("preprocess_seconds", 0.0) - phases.get("model_seconds", 0.0),
            0.0,
        )
        return metrics, phases

    first_size = args.image_sizes[0]
    try:
        first_request, first_phases = request_once(first_size)
        sizes_report: dict[str, Any] = {}
        for size in args.image_sizes:
            for _ in range(args.warmup):
                request_once(size)
            samples: list[dict[str, Any]] = []
            phases: list[dict[str, float]] = []
            for _ in range(args.runs):
                sample, phase = request_once(size)
                samples.append(sample)
                phases.append(phase)
            encoded_bytes = encoded_by_size[size][1]
            sizes_report[f"{size[0]}x{size[1]}"] = {
                "source_image_size": list(size),
                "encoded_bytes_per_camera": encoded_bytes,
                "total_latency": summarize([sample["seconds"] for sample in samples]),
                "preprocess_latency": summarize([phase["preprocess_seconds"] for phase in phases]),
                "model_latency": summarize([phase["model_seconds"] for phase in phases]),
                "boundary_latency": summarize([phase["boundary_seconds"] for phase in phases]),
                "cpu_rss_peak_mib": max(sample["cpu_rss_peak_mib"] for sample in samples),
                "gpu_peak_allocated_mib": max(sample["gpu_peak_allocated_mib"] for sample in samples),
                "gpu_peak_reserved_mib": max(sample["gpu_peak_reserved_mib"] for sample in samples),
            }
        report = {
            "schema_version": 1,
            "benchmark": "lingbot_vla_v2_single_gpu_service",
            "model_root": str(args.model_root.resolve()),
            "qwen3vl_root": str(args.qwen3vl_root.resolve()),
            "device": str(device),
            "quantization": args.quantization or "bf16",
            "seed": args.seed,
            "instruction": args.instruction,
            "internal_model_image_size": [pipeline.input_processor.image_size] * 2,
            "warmup_runs_per_size": args.warmup,
            "execution_mode": args.execution_mode,
            "measured_runs_per_size": args.runs,
            "environment": collect_runtime_environment([device], repo_root=Path(__file__).resolve().parents[2]),
            "load": load_metrics,
            "startup_warmup": startup_warmup,
            "first_request": {**first_request, "source_image_size": list(first_size), "phases": first_phases},
            "steady_state_by_source_size": sizes_report,
            "memory_after_benchmark": _memory_snapshot(device, process),
        }
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        pipeline.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--qwen3vl-root", required=True, type=Path)
    parser.add_argument("--image", type=Path, help="Optional source image reused for all three camera inputs.")
    parser.add_argument("--image-sizes", type=parse_image_sizes, default=parse_image_sizes("256x256,640x480,1280x720"))
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantization", choices=("torchao-fp8", "tf-kernel-fp8", "bnb-nf4"))
    parser.add_argument("--execution-mode", choices=("service-thread", "direct"), default="service-thread")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--startup-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jpeg-quality", type=int, choices=range(1, 101), default=95)
    parser.add_argument("--max-image-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
