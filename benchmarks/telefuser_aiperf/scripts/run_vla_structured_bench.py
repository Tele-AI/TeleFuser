"""Run the VLA AIPerf workload and sample target process resources."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import psutil

_MIB = 1024.0 * 1024.0


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _process_tree(root_pid: int) -> tuple[set[int], float]:
    try:
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        return set(), 0.0
    process_ids: set[int] = set()
    rss_bytes = 0
    for process in processes:
        try:
            process_ids.add(process.pid)
            rss_bytes += process.memory_info().rss
        except psutil.Error:
            continue
    return process_ids, rss_bytes / _MIB


def _gpu_memory(process_ids: set[int]) -> dict[str, float]:
    if not process_ids:
        return {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    memory: dict[str, float] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            value = float(fields[2])
        except ValueError:
            continue
        if pid in process_ids and value >= 0:
            memory[fields[1]] = memory.get(fields[1], 0.0) + value
    return memory


def _sample(root_pid: int, started_at: float) -> dict[str, Any]:
    process_ids, rss = _process_tree(root_pid)
    return {
        "offset_seconds": time.perf_counter() - started_at,
        "process_ids": sorted(process_ids),
        "cpu_rss_mib": rss,
        "gpu_memory_mib": _gpu_memory(process_ids),
    }


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[3]
    python = os.environ.get("TELEFUSER_AIPERF_PYTHON", str(root / ".venv-aiperf/bin/python"))
    command = [python, "-m", "telefuser_aiperf.cli", "profile", "--config", str(args.config)]
    environment = os.environ.copy()
    adapter_root = root / "benchmarks/telefuser_aiperf"
    environment["PYTHONPATH"] = f"{adapter_root}{os.pathsep}{environment.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    started_at = time.perf_counter()
    process = subprocess.Popen(command, cwd=root, env=environment)
    samples: list[dict[str, Any]] = []
    stop = threading.Event()

    def sample_loop() -> None:
        while not stop.is_set():
            samples.append(_sample(args.service_pid, started_at))
            stop.wait(args.sample_interval)

    sampler = threading.Thread(target=sample_loop, name="vla-aiperf-resource-sampler", daemon=True)
    sampler.start()
    return_code = process.wait()
    stop.set()
    sampler.join(timeout=max(args.sample_interval * 2.0, 1.0))
    samples.append(_sample(args.service_pid, started_at))

    rss_values = [float(sample["cpu_rss_mib"]) for sample in samples]
    gpu_values: dict[str, list[float]] = {}
    for sample in samples:
        for gpu, value in sample["gpu_memory_mib"].items():
            gpu_values.setdefault(gpu, []).append(float(value))
    report = {
        "schema_version": 1,
        "benchmark": "lingbot_vla_v2_aiperf_structured",
        "aiperf_command": command,
        "service_pid": args.service_pid,
        "return_code": return_code,
        "elapsed_seconds": time.perf_counter() - started_at,
        "cpu_rss_mib": _summary(rss_values),
        "gpu_process_memory_mib": {gpu: _summary(values) for gpu, values in sorted(gpu_values.items())},
        "samples": samples[-args.max_samples :],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return return_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("benchmarks/telefuser_aiperf/configs/vla_structured_e2e.yaml"),
    )
    parser.add_argument("--service-pid", type=int, required=True)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/telefuser_aiperf/vla_structured/resource_summary.json"),
    )
    args = parser.parse_args()
    if args.service_pid < 1 or args.sample_interval <= 0 or args.max_samples < 2:
        parser.error("service PID must be positive, interval must be positive, and max-samples must be at least 2")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
