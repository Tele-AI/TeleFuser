"""Validate a real LingBot-VLA v2 native structured API service."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import platform
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil
import requests

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_MIB = 1024**2
_REQUEST_PARAMETER_CONTRACT = {
    "instruction": ("string", True),
    "state": ("array", True),
    "camera_high": ("string", True),
    "camera_left_wrist": ("string", True),
    "camera_right_wrist": ("string", True),
    "seed": ("integer", False),
}
_RESULT_FIELDS = frozenset(
    {
        "canonical_normalized_actions",
        "horizon",
        "action_dim",
        "checkpoint_variant",
        "policy_verified",
        "verification_status",
    }
)
_SENSITIVE_REQUEST_FIELDS = frozenset({"camera_high", "camera_left_wrist", "camera_right_wrist"})


class ValidationFailure(RuntimeError):
    """Raised when the target violates the VLA structured API contract."""


@dataclass(frozen=True)
class RequestConfig:
    """Immutable settings shared by validation workers."""

    base_url: str
    payload: dict[str, Any]
    http_timeout_seconds: float
    task_timeout_seconds: float
    poll_interval_seconds: float
    expected_horizon: int
    expected_action_dim: int


def parse_state_json(value: str) -> list[float]:
    """Parse and validate a finite 14-dimensional RobotWin state."""
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("state must be valid JSON") from error
    if not isinstance(raw, list) or len(raw) != 14:
        raise argparse.ArgumentTypeError("state must be a JSON array containing exactly 14 values")
    state: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(float(item)):
            raise argparse.ArgumentTypeError("state values must be finite numbers")
        state.append(float(item))
    return state


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


def summarize(values: Sequence[float]) -> dict[str, float | int] | None:
    """Summarize a possibly empty sample in seconds."""
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def compare_windows(values: Sequence[float], fraction: float = 0.1) -> dict[str, float | int] | None:
    """Compare the first and last windows of an ordered measurement series."""
    if not values:
        return None
    window_count = max(1, math.ceil(len(values) * fraction))
    first_mean = statistics.fmean(values[:window_count])
    last_mean = statistics.fmean(values[-window_count:])
    delta = last_mean - first_mean
    return {
        "sample_count": len(values),
        "window_count": window_count,
        "first_mean": first_mean,
        "last_mean": last_mean,
        "delta": delta,
        "change_percent": delta / first_mean * 100.0 if first_mean else 0.0,
    }


def validate_service_metadata(metadata: Any) -> None:
    """Validate that the target exposes the native VLA structured contract."""
    if not isinstance(metadata, dict):
        raise ValidationFailure("service metadata must be a JSON object")
    if metadata.get("declared_pipeline_contract") is not True:
        raise ValidationFailure("service does not expose a declared pipeline contract")
    if "vla_action" not in metadata.get("supported_tasks", []):
        raise ValidationFailure("service metadata does not declare the vla_action task")
    if "structured" not in metadata.get("supported_media_types", []):
        raise ValidationFailure("service metadata does not declare structured output")
    task_contract = metadata.get("task_contracts", {}).get("vla_action")
    if not isinstance(task_contract, dict) or task_contract.get("media_type") != "structured":
        raise ValidationFailure("vla_action does not have a structured task contract")
    parameters = task_contract.get("parameters")
    if not isinstance(parameters, dict):
        raise ValidationFailure("vla_action parameters are missing from service metadata")
    if set(parameters) != set(_REQUEST_PARAMETER_CONTRACT):
        raise ValidationFailure(
            "vla_action parameter fields changed: "
            f"expected {sorted(_REQUEST_PARAMETER_CONTRACT)}, observed {sorted(parameters)}"
        )
    for name, (expected_type, expected_required) in _REQUEST_PARAMETER_CONTRACT.items():
        parameter = parameters[name]
        if not isinstance(parameter, dict):
            raise ValidationFailure(f"vla_action parameter contract is invalid: {name}")
        if parameter.get("type") != expected_type or parameter.get("required") is not expected_required:
            raise ValidationFailure(
                f"vla_action parameter {name} changed: expected type={expected_type}, required={expected_required}"
            )
    if task_contract.get("required_inputs") != ["camera_high", "camera_left_wrist", "camera_right_wrist"]:
        raise ValidationFailure("vla_action required_inputs changed")
    if task_contract.get("optional_inputs") != []:
        raise ValidationFailure("vla_action optional_inputs changed")


def validate_action_result(result: Any, *, expected_horizon: int, expected_action_dim: int) -> dict[str, Any]:
    """Validate and summarize one canonical normalized action chunk."""
    if expected_horizon < 1 or expected_action_dim < 1:
        raise ValueError("expected action dimensions must be positive")
    if not isinstance(result, dict):
        raise ValidationFailure("completed task result must be a JSON object")
    if set(result) != set(_RESULT_FIELDS):
        raise ValidationFailure(f"result fields changed: expected {sorted(_RESULT_FIELDS)}, observed {sorted(result)}")
    actions = result.get("canonical_normalized_actions")
    if not isinstance(actions, list) or len(actions) != expected_horizon:
        observed = len(actions) if isinstance(actions, list) else type(actions).__name__
        raise ValidationFailure(f"expected action horizon {expected_horizon}, observed {observed}")
    if result.get("horizon") != expected_horizon:
        raise ValidationFailure(f"result horizon field is not {expected_horizon}")
    if result.get("action_dim") != expected_action_dim:
        raise ValidationFailure(f"result action_dim field is not {expected_action_dim}")

    flat: list[float] = []
    digest = hashlib.sha256()
    for row_index, row in enumerate(actions):
        if not isinstance(row, list) or len(row) != expected_action_dim:
            observed = len(row) if isinstance(row, list) else type(row).__name__
            raise ValidationFailure(f"action row {row_index} has dimension {observed}, expected {expected_action_dim}")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise ValidationFailure("action chunk contains a non-finite or non-numeric value")
            number = float(value)
            flat.append(number)
            digest.update(struct.pack("<d", number))

    policy_verified = result.get("policy_verified")
    verification_status = result.get("verification_status")
    if not isinstance(policy_verified, bool):
        raise ValidationFailure("result policy_verified field must be boolean")
    if not isinstance(verification_status, str) or not verification_status:
        raise ValidationFailure("result verification_status field must be a non-empty string")
    checkpoint_variant = result.get("checkpoint_variant")
    if not isinstance(checkpoint_variant, str) or not checkpoint_variant:
        raise ValidationFailure("result checkpoint_variant field must be a non-empty string")

    return {
        "shape": [expected_horizon, expected_action_dim],
        "value_count": len(flat),
        "minimum": min(flat),
        "maximum": max(flat),
        "mean": statistics.fmean(flat),
        "l2_norm": math.sqrt(sum(value * value for value in flat)),
        "sha256_float64_le": digest.hexdigest(),
        "checkpoint_variant": checkpoint_variant,
        "policy_verified": policy_verified,
        "verification_status": verification_status,
    }


def validate_task_status(status: Any, *, task_id: str) -> None:
    """Validate stable terminal task fields without rejecting safe additive metadata."""
    if not isinstance(status, dict):
        raise ValidationFailure("task status must be a JSON object")
    if status.get("task_id") != task_id:
        raise ValidationFailure("task status returned a different task_id")
    required = {"status", "inference_time_s", "peak_memory_mb", "result"}
    missing = sorted(required.difference(status))
    if missing:
        raise ValidationFailure(f"task status is missing fields: {', '.join(missing)}")
    leaked = sorted(_SENSITIVE_REQUEST_FIELDS.intersection(status))
    if leaked:
        raise ValidationFailure(f"task status echoed sensitive image fields: {', '.join(leaked)}")


def _request_json(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = session.request(method, url, json=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        body = response.text[:1000]
        raise ValidationFailure(f"{method} {url} returned HTTP {response.status_code}: {body}") from error
    try:
        body = response.json()
    except ValueError as error:
        raise ValidationFailure(f"{method} {url} did not return JSON") from error
    if not isinstance(body, dict):
        raise ValidationFailure(f"{method} {url} did not return a JSON object")
    return body


def _new_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def inspect_service(base_url: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Read and validate native service readiness and metadata."""
    with _new_session() as session:
        ready = _request_json(session, "GET", f"{base_url}/v1/service/ready", timeout=timeout_seconds)
        if ready.get("ready") is not True:
            raise ValidationFailure("service readiness endpoint reports not ready")
        metadata = _request_json(session, "GET", f"{base_url}/v1/service/metadata", timeout=timeout_seconds)
        validate_service_metadata(metadata)
        status = _request_json(session, "GET", f"{base_url}/v1/service/status", timeout=timeout_seconds)
        metrics = _request_json(session, "GET", f"{base_url}/v1/service/metrics/json", timeout=timeout_seconds)
    return {"ready": ready, "metadata": metadata, "status": status, "metrics": metrics}


def execute_request(
    session: requests.Session,
    config: RequestConfig,
    *,
    request_index: int,
    worker_index: int,
    run_started_at: float,
) -> dict[str, Any]:
    """Submit, poll, validate, and summarize one real structured request."""
    record: dict[str, Any] = {
        "request_index": request_index,
        "worker_index": worker_index,
        "start_offset_seconds": time.perf_counter() - run_started_at,
    }
    request_started_at = time.perf_counter()
    try:
        submit_started_at = time.perf_counter()
        created = _request_json(
            session,
            "POST",
            f"{config.base_url}/v1/tasks/structured",
            timeout=config.http_timeout_seconds,
            payload=config.payload,
        )
        accepted_at = time.perf_counter()
        record["submit_seconds"] = accepted_at - submit_started_at
        task_id = created.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValidationFailure("structured task creation response has no task_id")
        record["task_id"] = task_id

        deadline = accepted_at + config.task_timeout_seconds
        transitions: list[str] = []
        poll_count = 0
        while True:
            if time.perf_counter() >= deadline:
                raise ValidationFailure(f"task {task_id} exceeded {config.task_timeout_seconds:g}s timeout")
            status = _request_json(
                session,
                "GET",
                f"{config.base_url}/v1/tasks/{task_id}/status",
                timeout=config.http_timeout_seconds,
            )
            poll_count += 1
            task_status = status.get("status") or status.get("task_status")
            if not isinstance(task_status, str):
                raise ValidationFailure(f"task {task_id} status response has no status")
            if not transitions or transitions[-1] != task_status:
                transitions.append(task_status)
            if task_status in _TERMINAL_STATUSES:
                break
            time.sleep(config.poll_interval_seconds)

        completed_at = time.perf_counter()
        validate_task_status(status, task_id=task_id)
        record.update(
            end_to_end_seconds=completed_at - request_started_at,
            accepted_to_terminal_seconds=completed_at - accepted_at,
            poll_count=poll_count,
            status_transitions=transitions,
            terminal_status=task_status,
        )
        inference_time = status.get("inference_time_s")
        if inference_time is not None:
            if isinstance(inference_time, bool) or not isinstance(inference_time, int | float):
                raise ValidationFailure("inference_time_s must be numeric or null")
            inference_time = float(inference_time)
            if not math.isfinite(inference_time) or inference_time < 0:
                raise ValidationFailure("inference_time_s must be finite and non-negative")
        record["inference_time_seconds"] = inference_time
        peak_memory = status.get("peak_memory_mb")
        if peak_memory is not None:
            if isinstance(peak_memory, bool) or not isinstance(peak_memory, int | float):
                raise ValidationFailure("peak_memory_mb must be numeric or null")
            peak_memory = float(peak_memory)
            if not math.isfinite(peak_memory) or peak_memory < 0:
                raise ValidationFailure("peak_memory_mb must be finite and non-negative")
        record["peak_memory_mb"] = peak_memory

        if task_status != "completed":
            raise ValidationFailure(f"task {task_id} reached terminal status {task_status}: {status.get('error')}")
        record["action"] = validate_action_result(
            status.get("result"),
            expected_horizon=config.expected_horizon,
            expected_action_dim=config.expected_action_dim,
        )
        record["outcome"] = "succeeded"
    except (requests.RequestException, ValidationFailure, ValueError) as error:
        record["outcome"] = "failed"
        record["error"] = str(error)
        record.setdefault("end_to_end_seconds", time.perf_counter() - request_started_at)
    return record


def _parse_gpu_process_memory(
    output: str,
    *,
    process_ids: set[int],
    uuid_to_index: dict[str, str],
    gpu_indexes: set[str] | None,
) -> dict[str, float]:
    """Parse nvidia-smi process memory rows for one service process tree."""
    memory_by_gpu: dict[str, float] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            memory_mib = float(fields[2])
        except ValueError:
            continue
        gpu_index = uuid_to_index.get(fields[1], fields[1])
        if pid not in process_ids or (gpu_indexes is not None and gpu_index not in gpu_indexes):
            continue
        memory_by_gpu[gpu_index] = memory_by_gpu.get(gpu_index, 0.0) + memory_mib
    return memory_by_gpu


def _query_gpu_index_map() -> dict[str, str]:
    """Resolve stable GPU UUIDs to physical indexes once per validation run."""
    uuid_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return {
        fields[1].strip(): fields[0].strip()
        for line in uuid_result.stdout.splitlines()
        if len(fields := [field.strip() for field in line.split(",")]) == 2
    }


def _query_gpu_process_memory(
    process_ids: set[int],
    *,
    uuid_to_index: dict[str, str],
    gpu_indexes: set[str] | None,
) -> dict[str, float]:
    """Read GPU memory used by the service process tree through nvidia-smi."""
    process_result = subprocess.run(
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
    return _parse_gpu_process_memory(
        process_result.stdout,
        process_ids=process_ids,
        uuid_to_index=uuid_to_index,
        gpu_indexes=gpu_indexes,
    )


def _sample_local_resources(
    root_pid: int,
    *,
    uuid_to_index: dict[str, str],
    gpu_indexes: set[str] | None,
) -> dict[str, Any]:
    """Sample process-tree RSS and GPU memory without touching model execution."""
    root = psutil.Process(root_pid)
    processes = [root, *root.children(recursive=True)]
    process_ids: set[int] = set()
    rss_bytes = 0
    for process in processes:
        try:
            process_ids.add(process.pid)
            rss_bytes += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {
        "process_ids": sorted(process_ids),
        "cpu_rss_mib": rss_bytes / _MIB,
        "gpu_memory_mib": _query_gpu_process_memory(
            process_ids,
            uuid_to_index=uuid_to_index,
            gpu_indexes=gpu_indexes,
        ),
    }


class ResourceSampler:
    """Periodically sample local service resources in a background thread."""

    def __init__(
        self,
        root_pid: int,
        *,
        interval_seconds: float,
        max_samples: int,
        gpu_indexes: set[str] | None = None,
        sample_function: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if root_pid < 1 or interval_seconds <= 0 or max_samples < 2:
            raise ValueError("invalid resource sampler configuration")
        self.root_pid = root_pid
        self.interval_seconds = interval_seconds
        self.max_samples = max_samples
        self.gpu_indexes = gpu_indexes
        if sample_function is None:
            uuid_to_index = _query_gpu_index_map()

            def sample_function() -> dict[str, Any]:
                return _sample_local_resources(
                    root_pid,
                    uuid_to_index=uuid_to_index,
                    gpu_indexes=gpu_indexes,
                )

        self.sample_function = sample_function
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = time.perf_counter()
        self._sample_count = 0
        self._cpu_rss: list[float] = []
        self._gpu_memory: dict[str, list[float]] = {}
        self._process_ids: set[int] = set()
        self._errors: deque[str] = deque(maxlen=100)
        self._first_samples: list[dict[str, Any]] = []
        recent_capacity = max(1, max_samples // 2)
        self._recent_samples: deque[dict[str, Any]] = deque(maxlen=recent_capacity)

    def start(self) -> None:
        """Start sampling and take an initial sample."""
        self._started_at = time.perf_counter()
        self._record_once()
        self._thread = threading.Thread(target=self._sample_loop, name="vla-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling, take a final sample, and return the report."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_seconds * 2, 1.0))
        self._record_once()
        return self.report()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._record_once()

    def _record_once(self) -> None:
        offset = time.perf_counter() - self._started_at
        try:
            sample = self.sample_function()
            cpu_rss = float(sample["cpu_rss_mib"])
            if not math.isfinite(cpu_rss) or cpu_rss < 0:
                raise ValueError("invalid cpu_rss_mib sample")
            gpu_memory = sample.get("gpu_memory_mib", {})
            if not isinstance(gpu_memory, dict):
                raise ValueError("invalid gpu_memory_mib sample")
            normalized_gpu = {
                str(index): float(value)
                for index, value in gpu_memory.items()
                if math.isfinite(float(value)) and float(value) >= 0
            }
            process_ids = {int(pid) for pid in sample.get("process_ids", [])}
            self._sample_count += 1
            self._cpu_rss.append(cpu_rss)
            self._process_ids.update(process_ids)
            for index, value in normalized_gpu.items():
                self._gpu_memory.setdefault(index, []).append(value)
            retained = {
                "offset_seconds": offset,
                "process_ids": sorted(process_ids),
                "cpu_rss_mib": cpu_rss,
                "gpu_memory_mib": normalized_gpu,
            }
            first_capacity = self.max_samples - self._recent_samples.maxlen
            if len(self._first_samples) < first_capacity:
                self._first_samples.append(retained)
            else:
                self._recent_samples.append(retained)
        except Exception as error:  # pragma: no cover - hardware errors are environment-dependent
            self._errors.append(str(error))

    def report(self) -> dict[str, Any]:
        """Return bounded samples, resource distributions, and first/last trends."""
        retained = self._first_samples + list(self._recent_samples)
        return {
            "enabled": True,
            "root_pid": self.root_pid,
            "interval_seconds": self.interval_seconds,
            "sample_count": self._sample_count,
            "gpu_sample_count": sum(1 for sample in retained if sample["gpu_memory_mib"]),
            "observed_process_ids": sorted(self._process_ids),
            "errors": list(self._errors),
            "cpu_rss_mib": {
                "distribution": summarize(self._cpu_rss),
                "trend": compare_windows(self._cpu_rss),
            },
            "gpu_memory_mib": {
                index: {
                    "distribution": summarize(values),
                    "trend": compare_windows(values),
                }
                for index, values in sorted(self._gpu_memory.items())
            },
            "retained_samples": retained,
        }


class RunAccumulator:
    """Collect aggregate measurements while bounding retained request records."""

    def __init__(self, max_records: int) -> None:
        self._lock = threading.Lock()
        self.max_records = max_records
        self.total = 0
        self.succeeded = 0
        self.failed = 0
        self.task_ids: set[str] = set()
        self.duplicate_task_ids: set[str] = set()
        self.end_to_end: list[float] = []
        self.submit: list[float] = []
        self.accepted_to_terminal: list[float] = []
        self.inference: list[float] = []
        self.peak_memory: list[float] = []
        self.poll_counts: list[float] = []
        self.terminal_statuses: Counter[str] = Counter()
        self.policy_statuses: Counter[str] = Counter()
        self.failures: deque[dict[str, Any]] = deque(maxlen=max_records)
        self.first_successes: list[dict[str, Any]] = []
        self.recent_successes: deque[dict[str, Any]] = deque(maxlen=max_records // 2)

    def add(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.total += 1
            task_id = record.get("task_id")
            if isinstance(task_id, str):
                if task_id in self.task_ids:
                    self.duplicate_task_ids.add(task_id)
                self.task_ids.add(task_id)
            terminal_status = record.get("terminal_status")
            if isinstance(terminal_status, str):
                self.terminal_statuses[terminal_status] += 1
            if record["outcome"] == "failed":
                self.failed += 1
                self.failures.append(record)
                return

            self.succeeded += 1
            self.end_to_end.append(float(record["end_to_end_seconds"]))
            self.submit.append(float(record["submit_seconds"]))
            self.accepted_to_terminal.append(float(record["accepted_to_terminal_seconds"]))
            self.poll_counts.append(float(record["poll_count"]))
            if record.get("inference_time_seconds") is not None:
                self.inference.append(float(record["inference_time_seconds"]))
            if record.get("peak_memory_mb") is not None:
                self.peak_memory.append(float(record["peak_memory_mb"]))
            self.policy_statuses[str(record["action"]["verification_status"])] += 1
            first_capacity = self.max_records - self.recent_successes.maxlen
            if len(self.first_successes) < first_capacity:
                self.first_successes.append(record)
            else:
                self.recent_successes.append(record)

    def report(self, elapsed_seconds: float) -> dict[str, Any]:
        retained_successes = self.first_successes + list(self.recent_successes)
        retained_successes.sort(key=lambda record: int(record["request_index"]))
        failures = sorted(self.failures, key=lambda record: int(record["request_index"]))
        return {
            "requests": {
                "total": self.total,
                "succeeded": self.succeeded,
                "failed": self.failed,
                "success_rate": self.succeeded / self.total if self.total else 0.0,
                "unique_task_ids": len(self.task_ids),
                "duplicate_task_ids": sorted(self.duplicate_task_ids),
                "terminal_statuses": dict(sorted(self.terminal_statuses.items())),
                "policy_statuses": dict(sorted(self.policy_statuses.items())),
            },
            "elapsed_seconds": elapsed_seconds,
            "throughput_requests_per_second": self.succeeded / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "latency_seconds": {
                "end_to_end": summarize(self.end_to_end),
                "submission": summarize(self.submit),
                "accepted_to_terminal": summarize(self.accepted_to_terminal),
                "target_inference": summarize(self.inference),
            },
            "latency_trend": {
                "end_to_end": compare_windows(self.end_to_end),
                "target_inference": compare_windows(self.inference),
            },
            "poll_count": summarize(self.poll_counts),
            "peak_memory_mb": summarize(self.peak_memory),
            "retained_records": {
                "limit_per_outcome": self.max_records,
                "successful": retained_successes,
                "failed": failures,
            },
        }


def run_workload(
    config: RequestConfig,
    *,
    request_count: int | None,
    duration_seconds: float | None,
    concurrency: int,
    max_records: int,
) -> dict[str, Any]:
    """Run a closed-loop fixed-count or duration workload."""
    accumulator = RunAccumulator(max_records)
    counter = 0
    counter_lock = threading.Lock()
    run_started_at = time.perf_counter()
    stop_claiming_at = None if duration_seconds is None else run_started_at + duration_seconds
    workers = concurrency if request_count is None else min(concurrency, request_count)
    barrier = threading.Barrier(workers)

    def claim_request() -> int | None:
        nonlocal counter
        with counter_lock:
            if request_count is not None and counter >= request_count:
                return None
            if stop_claiming_at is not None and time.perf_counter() >= stop_claiming_at:
                return None
            index = counter
            counter += 1
            return index

    def worker(worker_index: int) -> None:
        with _new_session() as session:
            barrier.wait()
            while (request_index := claim_request()) is not None:
                accumulator.add(
                    execute_request(
                        session,
                        config,
                        request_index=request_index,
                        worker_index=worker_index,
                        run_started_at=run_started_at,
                    )
                )

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vla-structured-validator") as executor:
        futures = [executor.submit(worker, worker_index) for worker_index in range(workers)]
        for future in futures:
            future.result()
    return accumulator.report(time.perf_counter() - run_started_at)


def _encode_image(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"camera image does not exist: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _resolve_camera_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    fallback = args.image
    paths = tuple(path or fallback for path in (args.camera_high, args.camera_left_wrist, args.camera_right_wrist))
    if any(path is None for path in paths):
        raise ValueError("provide --image or all three --camera-* paths")
    return paths  # type: ignore[return-value]


def parse_gpu_indexes(value: str) -> set[str]:
    """Parse a comma-separated set of physical GPU indexes."""
    indexes = {item.strip() for item in value.split(",") if item.strip()}
    if not indexes or any(not item.isdigit() for item in indexes):
        raise argparse.ArgumentTypeError("GPU indexes must be a comma-separated list of integers")
    return indexes


def _package_version() -> str:
    try:
        return version("telefuser")
    except PackageNotFoundError:
        return "source"


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for section in ("tasks",):
        before_section = before.get(section, {})
        after_section = after.get(section, {})
        if isinstance(before_section, dict) and isinstance(after_section, dict):
            delta[section] = {
                key: after_section[key] - before_section.get(key, 0)
                for key in after_section
                if isinstance(after_section[key], int | float) and not isinstance(after_section[key], bool)
            }
    return delta


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the service and return a reproducible JSON report."""
    if args.concurrency < 1 or args.warmup < 0 or args.max_records < 2:
        raise ValueError("concurrency must be positive, warmup non-negative, and max-records at least 2")
    if args.requests is not None and args.requests < 1:
        raise ValueError("requests must be positive")
    if args.duration_seconds is not None and args.duration_seconds <= 0:
        raise ValueError("duration-seconds must be positive")
    if args.poll_interval_seconds <= 0 or args.http_timeout_seconds <= 0 or args.task_timeout_seconds <= 0:
        raise ValueError("poll interval and HTTP/task timeouts must be positive")
    if args.expected_horizon < 1 or args.expected_action_dim < 1:
        raise ValueError("expected action dimensions must be positive")
    if args.resource_interval_seconds <= 0 or args.max_resource_samples < 2:
        raise ValueError("resource interval must be positive and max-resource-samples at least 2")
    if args.gpu_indexes is not None and args.service_pid is None:
        raise ValueError("--gpu-indexes requires --service-pid")
    base_url = args.base_url.rstrip("/")
    camera_high, camera_left, camera_right = _resolve_camera_paths(args)
    payload = {
        "task": "vla_action",
        "instruction": args.instruction,
        "state": args.state_json,
        "camera_high": _encode_image(camera_high),
        "camera_left_wrist": _encode_image(camera_left),
        "camera_right_wrist": _encode_image(camera_right),
        "seed": args.seed,
    }
    config = RequestConfig(
        base_url=base_url,
        payload=payload,
        http_timeout_seconds=args.http_timeout_seconds,
        task_timeout_seconds=args.task_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        expected_horizon=args.expected_horizon,
        expected_action_dim=args.expected_action_dim,
    )
    before = inspect_service(base_url, timeout_seconds=args.http_timeout_seconds)
    warmup_records: list[dict[str, Any]] = []
    warmup_started_at = time.perf_counter()
    with _new_session() as session:
        for index in range(args.warmup):
            warmup_records.append(
                execute_request(
                    session,
                    config,
                    request_index=index,
                    worker_index=0,
                    run_started_at=warmup_started_at,
                )
            )
    if any(record["outcome"] != "succeeded" for record in warmup_records):
        raise ValidationFailure("at least one warmup request failed")

    request_count = args.requests
    if request_count is None and args.duration_seconds is None:
        request_count = 1
    resource_sampler: ResourceSampler | None = None
    resource_report: dict[str, Any] = {"enabled": False}
    if args.service_pid is not None:
        if not psutil.pid_exists(args.service_pid):
            raise ValueError(f"service PID does not exist: {args.service_pid}")
        resource_sampler = ResourceSampler(
            args.service_pid,
            interval_seconds=args.resource_interval_seconds,
            max_samples=args.max_resource_samples,
            gpu_indexes=args.gpu_indexes,
        )
        resource_sampler.start()
    try:
        workload = run_workload(
            config,
            request_count=request_count,
            duration_seconds=args.duration_seconds,
            concurrency=args.concurrency,
            max_records=args.max_records,
        )
    finally:
        if resource_sampler is not None:
            resource_report = resource_sampler.stop()
    after = inspect_service(base_url, timeout_seconds=args.http_timeout_seconds)
    requests_report = workload["requests"]
    checks = {
        "service_ready_before": before["ready"].get("ready") is True,
        "service_ready_after": after["ready"].get("ready") is True,
        "warmup_succeeded": all(record["outcome"] == "succeeded" for record in warmup_records),
        "all_measured_requests_succeeded": requests_report["failed"] == 0 and requests_report["total"] > 0,
        "task_ids_unique": not requests_report["duplicate_task_ids"],
        "resource_samples_collected": (not resource_report["enabled"] or resource_report["sample_count"] > 0),
        "gpu_resource_samples_collected": (not resource_report["enabled"] or resource_report["gpu_sample_count"] > 0),
        "queue_drained": (
            after["metrics"].get("queue", {}).get("pending") == 0
            and after["metrics"].get("queue", {}).get("processing") == 0
        ),
    }
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "validation": "lingbot_vla_v2_native_structured_api",
        "passed": all(checks.values()),
        "checks": checks,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "telefuser_version": _package_version(),
            "telefuser_commit": _git_commit(repo_root),
        },
        "target": {
            "base_url": base_url,
            "transport": "HTTP native TeleFuser asynchronous structured task API",
            "declared_quantization_profile": getattr(args, "quantization_profile", "bf16"),
            "metadata": before["metadata"],
            "status_before": before["status"],
            "status_after": after["status"],
            "health_before": before["ready"],
            "health_after": after["ready"],
            "metrics_before": before["metrics"],
            "metrics_after": after["metrics"],
            "metrics_delta": _metric_delta(before["metrics"], after["metrics"]),
        },
        "workload": {
            "mode": "duration" if args.duration_seconds is not None else "fixed_requests",
            "requested_requests": request_count,
            "requested_duration_seconds": args.duration_seconds,
            "concurrency": args.concurrency,
            "warmup_requests": args.warmup,
            "instruction": args.instruction,
            "state_dimension": len(args.state_json),
            "seed": args.seed,
            "camera_files": {
                "high": str(camera_high.resolve()),
                "left_wrist": str(camera_left.resolve()),
                "right_wrist": str(camera_right.resolve()),
            },
            "expected_action_shape": [args.expected_horizon, args.expected_action_dim],
            "poll_interval_seconds": args.poll_interval_seconds,
            "task_timeout_seconds": args.task_timeout_seconds,
        },
        "warmup_records": warmup_records,
        "result": workload,
        "resources": resource_report,
        "interpretation": (
            "This validates service transport, scheduling, and normalized canonical action structure. "
            "It does not establish embodiment-specific robot control semantics."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    camera_group = parser.add_argument_group("camera inputs")
    camera_group.add_argument("--image", type=Path, help="Fallback image reused for camera inputs not set explicitly.")
    camera_group.add_argument("--camera-high", type=Path)
    camera_group.add_argument("--camera-left-wrist", type=Path)
    camera_group.add_argument("--camera-right-wrist", type=Path)
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument(
        "--state-json", type=parse_state_json, default=parse_state_json("[0,0,0,0,0,0,0,0,0,0,0,0,0,0]")
    )
    parser.add_argument("--seed", type=int, default=7)
    workload_group = parser.add_mutually_exclusive_group()
    workload_group.add_argument("--requests", type=int)
    workload_group.add_argument("--duration-seconds", type=float)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=float, default=0.1)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--task-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--expected-horizon", type=int, default=50)
    parser.add_argument("--expected-action-dim", type=int, default=55)
    parser.add_argument(
        "--quantization-profile",
        choices=("bf16", "torchao-fp8", "tf-kernel-fp8", "bnb-nf4"),
        default="bf16",
        help="Operator-declared profile recorded in the report; it does not change the running service.",
    )
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument(
        "--service-pid",
        type=int,
        help="Optional local TeleFuser parent PID; enables process-tree RSS and GPU memory sampling.",
    )
    parser.add_argument("--gpu-indexes", type=parse_gpu_indexes, help="Optional physical GPU indexes to include.")
    parser.add_argument("--resource-interval-seconds", type=float, default=1.0)
    parser.add_argument("--max-resource-samples", type=int, default=10000)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report: dict[str, Any]
    exit_code = 0
    try:
        report = run_validation(args)
        if not report["passed"]:
            exit_code = 1
    except Exception as error:
        report = {
            "schema_version": 1,
            "validation": "lingbot_vla_v2_native_structured_api",
            "passed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fatal_error": str(error),
        }
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "passed": report["passed"],
        "checks": report.get("checks"),
        "requests": report.get("result", {}).get("requests"),
        "latency_seconds": report.get("result", {}).get("latency_seconds"),
        "fatal_error": report.get("fatal_error"),
        "artifact": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
