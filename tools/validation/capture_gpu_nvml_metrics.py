#!/usr/bin/env python3
"""Capture GPU metrics through NVML without Docker, DCGM, or ``nvidia-smi``.

This is a deliberately small fallback for experiment nodes where the full
Prometheus/DCGM stack cannot run. Pair it with
``capture_abot_serving_metrics.py``: both artifacts use UTC and monotonic
offset timestamps, so serving, scheduling, and physical-GPU time series can be
correlated without parsing logs by hand.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_NVML_SUCCESS = 0
_NVML_TEMPERATURE_GPU = 0


class NvmlError(RuntimeError):
    """Raised for an NVML initialization or device-query failure."""


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlUtilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


@dataclass(frozen=True)
class CaptureConfig:
    gpu_indices: tuple[int, ...]
    duration_seconds: float
    interval_seconds: float
    output_dir: Path


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _gpu_indices(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("must be a comma-separated non-empty list of GPU indices")
    try:
        indices = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must contain only integer GPU indices") from exc
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("GPU indices must be unique non-negative integers")
    return indices


def parse_args(argv: list[str] | None = None) -> CaptureConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-indices", type=_gpu_indices, default=(0, 1, 2, 3))
    parser.add_argument("--duration", required=True, type=_positive_seconds)
    parser.add_argument("--interval", type=_positive_seconds, default=1.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return CaptureConfig(
        gpu_indices=args.gpu_indices,
        duration_seconds=args.duration,
        interval_seconds=args.interval,
        output_dir=args.output_dir.expanduser().resolve(),
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class NvmlSampler:
    """Minimal ctypes wrapper for the NVML fields used in serving experiments."""

    def __init__(self, gpu_indices: tuple[int, ...]) -> None:
        try:
            library = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as exc:
            raise NvmlError(f"could not load libnvidia-ml.so.1: {exc}") from exc
        self._library = library
        self._init = library.nvmlInit_v2
        self._init.restype = ctypes.c_int
        self._shutdown = library.nvmlShutdown
        self._shutdown.restype = ctypes.c_int
        self._get_count = library.nvmlDeviceGetCount_v2
        self._get_count.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        self._get_count.restype = ctypes.c_int
        self._get_handle = library.nvmlDeviceGetHandleByIndex_v2
        self._get_handle.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self._get_handle.restype = ctypes.c_int
        self._get_name = library.nvmlDeviceGetName
        self._get_name.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
        self._get_name.restype = ctypes.c_int
        self._get_utilization = library.nvmlDeviceGetUtilizationRates
        self._get_utilization.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlUtilization)]
        self._get_utilization.restype = ctypes.c_int
        self._get_memory = library.nvmlDeviceGetMemoryInfo
        self._get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
        self._get_memory.restype = ctypes.c_int
        self._get_power = library.nvmlDeviceGetPowerUsage
        self._get_power.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self._get_power.restype = ctypes.c_int
        self._get_temperature = library.nvmlDeviceGetTemperature
        self._get_temperature.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        self._get_temperature.restype = ctypes.c_int

        self._check(self._init(), "nvmlInit_v2")
        try:
            count = ctypes.c_uint()
            self._check(self._get_count(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
            if any(index >= count.value for index in gpu_indices):
                raise NvmlError(f"requested GPU indices {gpu_indices} exceed detected GPU count {count.value}")
            self._handles = {index: self._handle(index) for index in gpu_indices}
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != _NVML_SUCCESS:
            raise NvmlError(f"{operation} returned NVML status {status}")

    def _handle(self, index: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(self._get_handle(index, ctypes.byref(handle)), f"nvmlDeviceGetHandleByIndex_v2({index})")
        return handle

    def _name(self, handle: ctypes.c_void_p) -> str:
        buffer = ctypes.create_string_buffer(96)
        self._check(self._get_name(handle, buffer, len(buffer)), "nvmlDeviceGetName")
        return buffer.value.decode("utf-8", errors="replace")

    def sample(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, handle in self._handles.items():
            utilization = _NvmlUtilization()
            memory = _NvmlMemory()
            power_milliwatts = ctypes.c_uint()
            temperature = ctypes.c_uint()
            self._check(self._get_utilization(handle, ctypes.byref(utilization)), "nvmlDeviceGetUtilizationRates")
            self._check(self._get_memory(handle, ctypes.byref(memory)), "nvmlDeviceGetMemoryInfo")
            self._check(self._get_power(handle, ctypes.byref(power_milliwatts)), "nvmlDeviceGetPowerUsage")
            self._check(
                self._get_temperature(handle, _NVML_TEMPERATURE_GPU, ctypes.byref(temperature)),
                "nvmlDeviceGetTemperature",
            )
            rows.append(
                {
                    "gpu_index": index,
                    "gpu_name": self._name(handle),
                    "gpu_utilization_percent": int(utilization.gpu),
                    "memory_utilization_percent": int(utilization.memory),
                    "memory_total_bytes": int(memory.total),
                    "memory_used_bytes": int(memory.used),
                    "memory_free_bytes": int(memory.free),
                    "power_watts": round(float(power_milliwatts.value) / 1000.0, 3),
                    "temperature_celsius": int(temperature.value),
                }
            )
        return rows

    def close(self) -> None:
        if getattr(self, "_library", None) is not None:
            self._shutdown()
            self._library = None


def capture(config: CaptureConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = config.output_dir / "gpu-metrics.jsonl"
    manifest_path = config.output_dir / "manifest.json"
    if jsonl_path.exists() or manifest_path.exists():
        raise ValueError(f"refusing to overwrite an existing capture artifact in {config.output_dir}")

    sampler = NvmlSampler(config.gpu_indices)
    started = time.monotonic()
    started_at_utc = _utc_timestamp()
    attempted = 0
    complete = 0
    status = "completed"
    try:
        with jsonl_path.open("x", encoding="utf-8") as output:
            while True:
                if attempted and time.monotonic() >= started + config.duration_seconds:
                    break
                attempted += 1
                record: dict[str, Any] = {
                    "schema_version": _SCHEMA_VERSION,
                    "source": "nvml",
                    "sequence": attempted,
                    "observed_at_utc": _utc_timestamp(),
                    "offset_seconds": round(time.monotonic() - started, 6),
                    "gpu_indices": list(config.gpu_indices),
                    "gpus": [],
                    "error": None,
                }
                try:
                    record["gpus"] = sampler.sample()
                    complete += 1
                except NvmlError as exc:
                    record["error"] = str(exc)
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                remaining = started + config.duration_seconds - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(config.interval_seconds, remaining))
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        sampler.close()

    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "source": "nvml",
        "started_at_utc": started_at_utc,
        "completed_at_utc": _utc_timestamp(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "configuration": {
            "gpu_indices": list(config.gpu_indices),
            "duration_seconds": config.duration_seconds,
            "interval_seconds": config.interval_seconds,
        },
        "samples": {"attempted": attempted, "complete": complete, "errors": attempted - complete},
        "artifact": jsonl_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        manifest = capture(config)
    except (NvmlError, OSError, ValueError) as exc:
        print(f"GPU metrics capture failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"GPU metrics capture {manifest['status']}: {manifest['samples']['complete']}/"
        f"{manifest['samples']['attempted']} complete samples; manifest: {config.output_dir / 'manifest.json'}"
    )
    return 0 if manifest["samples"]["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
