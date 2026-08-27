"""Run the real-GPU LingBot-VLA v2 release validation suite."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import requests

try:
    from tools.validation import validate_lingbot_vla_v2_service_faults as fault_validator
    from tools.validation import validate_lingbot_vla_v2_structured_service as structured_validator
    from tools.validation.lingbot_vla_v2_validation_common import action_error_metrics
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    import validate_lingbot_vla_v2_service_faults as fault_validator
    import validate_lingbot_vla_v2_structured_service as structured_validator
    from lingbot_vla_v2_validation_common import action_error_metrics


@dataclass(frozen=True)
class ReleaseProfile:
    """One supported LingBot-VLA v2 runtime profile."""

    name: str
    quantization: str | None
    cuda_graph: bool


RELEASE_PROFILES = {
    profile.name: profile
    for profile in (
        ReleaseProfile("bf16-eager", None, False),
        ReleaseProfile("bf16-graph", None, True),
        ReleaseProfile("fused-fp8-graph", "fused-fp8-graph", True),
        ReleaseProfile("torchao-fp8", "torchao-fp8", False),
        ReleaseProfile("tf-kernel-fp8", "tf-kernel-fp8", False),
        ReleaseProfile("bnb-nf4", "bnb-nf4", False),
    )
}
_PACKAGES = (
    "telefuser",
    "torch",
    "transformers",
    "accelerate",
    "torchao",
    "bitsandbytes",
    "tf-kernel",
    "pillow",
    "pydantic",
    "fastapi",
    "uvicorn",
    "requests",
    "psutil",
    "aiperf",
)
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def parse_profiles(value: str) -> tuple[ReleaseProfile, ...]:
    """Parse a comma-separated release profile list."""
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one release profile is required")
    unknown = sorted(set(names) - RELEASE_PROFILES.keys())
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown release profiles: {', '.join(unknown)}")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("release profiles must be unique")
    return tuple(RELEASE_PROFILES[name] for name in names)


def render_service_module(
    profile: ReleaseProfile,
    *,
    model_root: Path,
    qwen3vl_root: Path,
    max_image_bytes: int,
    max_image_pixels: int,
) -> str:
    """Render an auditable pipeline module for one release profile."""
    config = {
        "model_root": str(model_root.resolve()),
        "qwen3vl_root": str(qwen3vl_root.resolve()),
        "device": "cuda:0",
        "quantization": profile.quantization,
        "cuda_graph": profile.cuda_graph,
        "max_image_bytes": max_image_bytes,
        "max_image_pixels": max_image_pixels,
    }
    template = Path(__file__).resolve().parents[2] / "examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py"
    module = ast.parse(template.read_text(encoding="utf-8"), filename=str(template))
    replaced_config = False
    replaced_name = False
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        target = node.targets[0].id
        if target == "PPL_CONFIG":
            node.value = ast.parse(repr(config), mode="eval").body
            replaced_config = True
        elif target == "PIPELINE_CONTRACT" and isinstance(node.value, ast.Dict):
            for index, key in enumerate(node.value.keys):
                if isinstance(key, ast.Constant) and key.value == "pipeline_name":
                    node.value.values[index] = ast.Constant(
                        value=f"lingbot_vla_v2_release_{profile.name.replace('-', '_')}"
                    )
                    replaced_name = True
                    break
    if not replaced_config or not replaced_name:
        raise RuntimeError("native service template no longer exposes PPL_CONFIG and pipeline_name assignments")
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def _sha256_file(path: Path, digest: Any | None = None) -> str:
    result = hashlib.sha256() if digest is None else digest
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def tree_manifest(root: Path, *, include_contents: bool) -> dict[str, Any]:
    """Return one stable fingerprint for a checkpoint or processor tree."""
    if not root.is_dir():
        raise ValueError(f"model tree does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        if include_contents:
            _sha256_file(path, digest)
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "hash_mode": "full_sha256" if include_contents else "filename_and_size",
        "sha256": digest.hexdigest(),
    }


def compare_actions(
    reference: list[list[float]],
    candidate: list[list[float]],
    *,
    min_cosine: float,
    max_relative_l2: float,
    max_absolute_error: float,
    require_exact: bool = False,
) -> dict[str, Any]:
    """Compare two canonical action chunks with the release quality gates."""
    if len(reference) != 50 or len(candidate) != 50:
        raise ValueError("action parity requires two 50x55 chunks")
    if any(len(row) != 55 for row in reference) or any(len(row) != 55 for row in candidate):
        raise ValueError("action parity requires two 50x55 chunks")
    left = np.asarray(reference, dtype=np.float64)
    right = np.asarray(candidate, dtype=np.float64)
    metrics = action_error_metrics(left, right)
    if not metrics["reference_finite"] or not metrics["candidate_finite"]:
        raise ValueError("action parity inputs must be finite")
    reference_norm = float(np.linalg.norm(left.reshape(-1)))
    difference_norm = float(np.linalg.norm((right - left).reshape(-1)))
    relative_l2 = float(metrics["relative_l2"]) if reference_norm else float(difference_norm != 0)
    checks = {
        "finite": True,
        "cosine": float(metrics["cosine"]) >= min_cosine,
        "relative_l2": relative_l2 <= max_relative_l2,
        "max_absolute_error": float(metrics["max_abs"]) <= max_absolute_error,
    }
    exact = bool(metrics["exact"])
    if require_exact:
        checks["exact_replay"] = exact
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cosine_similarity": metrics["cosine"],
        "relative_l2": relative_l2,
        "max_absolute_error": metrics["max_abs"],
        "exact": exact,
        "exact_required": require_exact,
        "thresholds": {
            "min_cosine": min_cosine,
            "max_relative_l2": max_relative_l2,
            "max_absolute_error": max_absolute_error,
        },
    }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _nvidia_environment() -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    gpus: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 5:
            gpus.append(dict(zip(("index", "uuid", "name", "driver_version", "memory_total_mib"), fields, strict=True)))
    return gpus


def _git_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.splitlines()
    return {"commit": commit, "clean": not status, "changed_paths": status}


def _image_payload(image: Path, instruction: str, seed: int) -> dict[str, Any]:
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    return {
        "task": "vla_action",
        "instruction": instruction,
        "state": [0.0] * 14,
        "camera_high": encoded,
        "camera_left_wrist": encoded,
        "camera_right_wrist": encoded,
        "seed": seed,
    }


def _direct_command(args: argparse.Namespace, profile: ReleaseProfile, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "direct",
        "--model-root",
        str(args.model_root),
        "--qwen3vl-root",
        str(args.qwen3vl_root),
        "--image",
        str(args.image),
        "--instruction",
        args.instruction,
        "--seed",
        str(args.seed),
        "--profile",
        profile.name,
        "--max-image-bytes",
        str(args.max_image_bytes),
        "--max-image-pixels",
        str(args.max_image_pixels),
        "--output",
        str(output),
    ]
    return command


def run_direct_subprocess(
    args: argparse.Namespace,
    profile: ReleaseProfile,
    *,
    output: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Run direct inference in an isolated process so GPU state is released."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    started_at = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            _direct_command(args, profile, output),
            cwd=args.repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.profile_timeout_seconds,
            check=False,
        )
    if not output.is_file():
        raise RuntimeError(f"direct profile {profile.name} produced no artifact; see {log_path}")
    report = json.loads(output.read_text(encoding="utf-8"))
    report["subprocess"] = {
        "return_code": result.returncode,
        "elapsed_seconds": time.perf_counter() - started_at,
        "log": str(log_path),
    }
    if result.returncode != 0 or not report.get("passed"):
        raise RuntimeError(f"direct profile {profile.name} failed; see {output} and {log_path}")
    return report


def _service_command(args: argparse.Namespace, module_path: Path, cache_dir: Path) -> list[str]:
    executable = args.telefuser_bin
    return [
        str(executable),
        "serve",
        str(module_path),
        "--task",
        "vla_action",
        "--parallelism",
        "1",
        "--num-replicas",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--cache-dir",
        str(cache_dir),
    ]


def start_service(
    args: argparse.Namespace,
    *,
    module_path: Path,
    cache_dir: Path,
    log_path: Path,
) -> tuple[subprocess.Popen[str], Any]:
    """Start one real service process and wait for native readiness."""
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        _service_command(args, module_path, cache_dir),
        cwd=args.repo_root,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{args.port}"
    deadline = time.monotonic() + args.startup_timeout_seconds
    last_error = "service did not respond"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(f"service exited with code {process.returncode}; see {log_path}")
        try:
            inspection = structured_validator.inspect_service(base_url, timeout_seconds=2.0)
            if inspection["ready"].get("ready") is True:
                return process, log
        except (requests.RequestException, structured_validator.ValidationFailure) as error:
            last_error = str(error)
        time.sleep(0.5)
    process.terminate()
    process.wait(timeout=30)
    log.close()
    raise RuntimeError(f"service did not become ready: {last_error}; see {log_path}")


def _process_tree_ids(process: subprocess.Popen[str]) -> set[int]:
    try:
        root = psutil.Process(process.pid)
        return {process.pid, *(child.pid for child in root.children(recursive=True))}
    except psutil.Error:
        return {process.pid}


def _gpu_compute_process_ids() -> set[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    process_ids: set[int] = set()
    for line in result.stdout.splitlines():
        try:
            process_ids.add(int(line.strip()))
        except ValueError:
            continue
    return process_ids


def stop_service(process: subprocess.Popen[str], log: Any, *, timeout_seconds: float = 60.0) -> dict[str, Any]:
    """Stop one owned service process and verify its GPU contexts disappear."""
    tracked_pids = _process_tree_ids(process)
    started_at = time.perf_counter()
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            for pid in sorted(tracked_pids - {process.pid}, reverse=True):
                try:
                    psutil.Process(pid).terminate()
                except psutil.Error:
                    pass
            process.kill()
            process.wait(timeout=10)
    log.close()
    deadline = time.monotonic() + timeout_seconds
    remaining = tracked_pids & _gpu_compute_process_ids()
    while remaining and time.monotonic() < deadline:
        time.sleep(0.25)
        remaining = tracked_pids & _gpu_compute_process_ids()
    return {
        "passed": process.returncode is not None and not remaining,
        "return_code": process.returncode,
        "tracked_process_ids": sorted(tracked_pids),
        "remaining_gpu_process_ids": sorted(remaining),
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def execute_http_action(
    base_url: str,
    payload: dict[str, Any],
    *,
    http_timeout_seconds: float,
    task_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Submit one structured action and return its full terminal result."""
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            f"{base_url}/v1/tasks/structured",
            json=payload,
            timeout=http_timeout_seconds,
        )
        response.raise_for_status()
        created = response.json()
        task_id = created.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("structured service did not return a task_id")
        deadline = time.monotonic() + task_timeout_seconds
        while time.monotonic() < deadline:
            status_response = session.get(
                f"{base_url}/v1/tasks/{task_id}/status",
                timeout=http_timeout_seconds,
            )
            status_response.raise_for_status()
            status = status_response.json()
            terminal = status.get("status") or status.get("task_status")
            if terminal in _TERMINAL_STATUSES:
                structured_validator.validate_task_status(status, task_id=task_id)
                if terminal != "completed":
                    raise RuntimeError(f"structured task {task_id} ended as {terminal}: {status.get('error')}")
                structured_validator.validate_action_result(
                    status.get("result"), expected_horizon=50, expected_action_dim=55
                )
                return status["result"]
            time.sleep(poll_interval_seconds)
    raise RuntimeError(f"structured task {task_id} exceeded {task_timeout_seconds:g}s")


def validate_client_timeout(
    base_url: str,
    payload: dict[str, Any],
    *,
    http_timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Verify bounded client waiting and cleanup of the timed-out task."""
    config = structured_validator.RequestConfig(
        base_url=base_url,
        payload=payload,
        http_timeout_seconds=http_timeout_seconds,
        task_timeout_seconds=1e-9,
        poll_interval_seconds=poll_interval_seconds,
        expected_horizon=50,
        expected_action_dim=55,
    )
    with requests.Session() as session:
        session.trust_env = False
        record = structured_validator.execute_request(
            session,
            config,
            request_index=0,
            worker_index=0,
            run_started_at=time.perf_counter(),
        )
        task_id = record.get("task_id")
        cancelled = False
        if isinstance(task_id, str):
            response = session.delete(f"{base_url}/v1/tasks/{task_id}", timeout=http_timeout_seconds)
            cancelled = response.status_code == 200
            if cancelled:
                terminal = fault_validator._wait_terminal(
                    session,
                    base_url,
                    task_id,
                    http_timeout_seconds=http_timeout_seconds,
                    task_timeout_seconds=config.http_timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                )
                cancelled = terminal.get("status") in _TERMINAL_STATUSES
    expected_timeout = record.get("outcome") == "failed" and "exceeded" in str(record.get("error"))
    return {"passed": expected_timeout and cancelled, "record": record, "cancel_requested": cancelled}


def run_fault_checks(args: argparse.Namespace, base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run invalid-input, failure, cancellation, and bounded-timeout checks."""
    with requests.Session() as session:
        session.trust_env = False
        cases = fault_validator.validate_request_faults(
            session,
            base_url,
            payload,
            http_timeout_seconds=args.http_timeout_seconds,
            task_timeout_seconds=args.task_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    timeout = validate_client_timeout(
        base_url,
        payload,
        http_timeout_seconds=args.http_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    return {"passed": all(case["passed"] for case in cases) and timeout["passed"], "cases": cases, "timeout": timeout}


def run_structured_workload(
    args: argparse.Namespace,
    profile: ReleaseProfile,
    *,
    base_url: str,
    service_pid: int,
    output: Path,
) -> dict[str, Any]:
    """Run the existing bounded structured-service validator."""
    validation_args = argparse.Namespace(
        base_url=base_url,
        image=args.image,
        camera_high=None,
        camera_left_wrist=None,
        camera_right_wrist=None,
        instruction=args.instruction,
        state_json=[0.0] * 14,
        seed=args.seed,
        requests=args.requests,
        duration_seconds=None,
        concurrency=1,
        warmup=args.warmup,
        poll_interval_seconds=args.poll_interval_seconds,
        http_timeout_seconds=args.http_timeout_seconds,
        task_timeout_seconds=args.task_timeout_seconds,
        expected_horizon=50,
        expected_action_dim=55,
        quantization_profile=profile.quantization or "bf16",
        max_records=max(args.requests, 2),
        service_pid=service_pid,
        gpu_indexes=None,
        resource_interval_seconds=args.resource_interval_seconds,
        max_resource_samples=max(args.requests * 20, 100),
    )
    report = structured_validator.run_validation(validation_args)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_aiperf(args: argparse.Namespace, *, base_url: str, service_pid: int, profile_dir: Path) -> dict[str, Any]:
    """Run the repository-owned AIPerf VLA workload against the active service."""
    if not args.aiperf:
        return {"passed": False, "skipped": True, "reason": "disabled by --no-aiperf"}
    environment = os.environ.copy()
    environment.update(
        TELEFUSER_AIPERF_URL=base_url,
        TELEFUSER_AIPERF_HEALTH_URL=f"{base_url}/v1/service/ready",
        TELEFUSER_AIPERF_METRICS_URL=f"{base_url}/v1/service/metrics",
        TELEFUSER_AIPERF_SERVICE_PID=str(service_pid),
        TELEFUSER_AIPERF_RESOURCE_OUTPUT=str(profile_dir / "aiperf_resources.json"),
        TELEFUSER_AIPERF_REQUESTS=str(args.aiperf_requests),
        TELEFUSER_AIPERF_WARMUP_REQUESTS=str(args.aiperf_warmup),
        TELEFUSER_AIPERF_CONCURRENCY="1",
        TELEFUSER_VLA_PYTHON=sys.executable,
    )
    artifacts_root = args.repo_root / "artifacts/telefuser_aiperf/vla_structured"
    before = {path.resolve() for path in artifacts_root.iterdir()} if artifacts_root.is_dir() else set()
    log_path = profile_dir / "aiperf.log"
    started_at = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            ["bash", "benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh"],
            cwd=args.repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.profile_timeout_seconds,
            check=False,
        )
    after = {path.resolve() for path in artifacts_root.iterdir()} if artifacts_root.is_dir() else set()
    return {
        "passed": result.returncode == 0,
        "return_code": result.returncode,
        "elapsed_seconds": time.perf_counter() - started_at,
        "log": str(log_path),
        "resource_artifact": str(profile_dir / "aiperf_resources.json"),
        "new_artifacts": sorted(str(path) for path in after - before),
    }


def _profile_checks(profile_report: dict[str, Any]) -> dict[str, bool]:
    return {
        "direct_inference": profile_report.get("direct", {}).get("passed") is True,
        "http_inference": profile_report.get("http_action") is not None,
        "direct_http_parity": profile_report.get("direct_http_parity", {}).get("passed") is True,
        "seeded_http_replay": profile_report.get("seeded_http_replay", {}).get("passed") is True,
        "continuous_workload": profile_report.get("structured_workload", {}).get("passed") is True,
        "fault_handling": profile_report.get("faults", {}).get("passed") is True,
        "dynamic_instruction": profile_report.get("dynamic_instruction", {}).get("passed") is True,
        "aiperf": profile_report.get("aiperf", {}).get("passed") is True,
        "shutdown_and_gpu_release": profile_report.get("shutdown", {}).get("passed") is True,
    }


def run_profile(
    args: argparse.Namespace,
    profile: ReleaseProfile,
    *,
    root: Path,
) -> dict[str, Any]:
    """Run all release checks for one runtime profile."""
    profile_dir = root / profile.name
    profile_dir.mkdir(parents=True, exist_ok=True)
    module_path = profile_dir / "service_profile.py"
    module_path.write_text(
        render_service_module(
            profile,
            model_root=args.model_root,
            qwen3vl_root=args.qwen3vl_root,
            max_image_bytes=args.max_image_bytes,
            max_image_pixels=args.max_image_pixels,
        ),
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "profile": profile.name,
        "quantization": profile.quantization or "bf16",
        "cuda_graph": profile.cuda_graph,
        "service_module": str(module_path),
        "passed": False,
    }
    process: subprocess.Popen[str] | None = None
    service_log: Any | None = None
    try:
        direct_output = profile_dir / "direct.json"
        report["direct"] = run_direct_subprocess(
            args,
            profile,
            output=direct_output,
            log_path=profile_dir / "direct.log",
        )
        process, service_log = start_service(
            args,
            module_path=module_path,
            cache_dir=profile_dir / "service_cache",
            log_path=profile_dir / "service.log",
        )
        base_url = f"http://127.0.0.1:{args.port}"
        payload = _image_payload(args.image, args.instruction, args.seed)
        report["http_action"] = execute_http_action(
            base_url,
            payload,
            http_timeout_seconds=args.http_timeout_seconds,
            task_timeout_seconds=args.task_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        report["direct_http_parity"] = compare_actions(
            report["direct"]["result"]["canonical_normalized_actions"],
            report["http_action"]["canonical_normalized_actions"],
            min_cosine=args.min_cosine,
            max_relative_l2=args.max_relative_l2,
            max_absolute_error=args.max_absolute_error,
        )
        replay_action = execute_http_action(
            base_url,
            payload,
            http_timeout_seconds=args.http_timeout_seconds,
            task_timeout_seconds=args.task_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        report["seeded_http_replay"] = compare_actions(
            report["http_action"]["canonical_normalized_actions"],
            replay_action["canonical_normalized_actions"],
            min_cosine=args.min_cosine,
            max_relative_l2=args.max_relative_l2,
            max_absolute_error=args.max_absolute_error,
            require_exact=profile.quantization is not None,
        )
        dynamic_payload = dict(payload, instruction=args.dynamic_instruction)
        dynamic_action = execute_http_action(
            base_url,
            dynamic_payload,
            http_timeout_seconds=args.http_timeout_seconds,
            task_timeout_seconds=args.task_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        report["dynamic_instruction"] = {
            "passed": True,
            "instruction": args.dynamic_instruction,
            "action": structured_validator.validate_action_result(
                dynamic_action,
                expected_horizon=50,
                expected_action_dim=55,
            ),
            "execution": "eager_prefix_with_denoising_cuda_graph" if profile.cuda_graph else "eager",
        }
        report["direct"]["result"] = structured_validator.validate_action_result(
            report["direct"]["result"],
            expected_horizon=50,
            expected_action_dim=55,
        )
        report["http_action"] = structured_validator.validate_action_result(
            report["http_action"],
            expected_horizon=50,
            expected_action_dim=55,
        )
        report["structured_workload"] = run_structured_workload(
            args,
            profile,
            base_url=base_url,
            service_pid=process.pid,
            output=profile_dir / "structured_workload.json",
        )
        report["faults"] = run_fault_checks(args, base_url, payload)
        report["aiperf"] = run_aiperf(args, base_url=base_url, service_pid=process.pid, profile_dir=profile_dir)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None and service_log is not None:
            report["shutdown"] = stop_service(process, service_log)
    report["checks"] = _profile_checks(report)
    report["passed"] = all(report["checks"].values())
    (profile_dir / "profile_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_restart_check(
    args: argparse.Namespace,
    profile: ReleaseProfile,
    *,
    root: Path,
) -> dict[str, Any]:
    """Restart one previously generated profile and execute a fresh request."""
    profile_dir = root / profile.name
    process: subprocess.Popen[str] | None = None
    log: Any | None = None
    report: dict[str, Any] = {"profile": profile.name, "passed": False}
    try:
        process, log = start_service(
            args,
            module_path=profile_dir / "service_profile.py",
            cache_dir=profile_dir / "restart_cache",
            log_path=profile_dir / "restart_service.log",
        )
        result = execute_http_action(
            f"http://127.0.0.1:{args.port}",
            _image_payload(args.image, args.instruction, args.seed),
            http_timeout_seconds=args.http_timeout_seconds,
            task_timeout_seconds=args.task_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        report["action"] = structured_validator.validate_action_result(
            result,
            expected_horizon=50,
            expected_action_dim=55,
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None and log is not None:
            report["shutdown"] = stop_service(process, log)
    report["passed"] = "action" in report and report.get("shutdown", {}).get("passed") is True
    return report


def run_direct(args: argparse.Namespace) -> int:
    """Execute one profile directly in the current isolated child process."""
    profile = RELEASE_PROFILES[args.profile]
    report: dict[str, Any] = {
        "schema_version": 1,
        "validation": "lingbot_vla_v2_release_direct",
        "profile": profile.name,
        "passed": False,
    }
    pipeline = None
    try:
        from telefuser.metrics.runtime import collect_runtime_environment
        from telefuser.models.lingbot_vla_v2_quantization import lingbot_vla_v2_quantization_identity
        from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline
        from telefuser.pipelines.lingbot_vla_v2.service import (
            LingBotVlaV2ActionRequest,
            predict_lingbot_vla_v2_action,
        )

        pipeline = get_lingbot_vla_v2_pipeline(
            str(args.model_root),
            str(args.qwen3vl_root),
            device="cuda:0",
            warmup=True,
            quantization=profile.quantization,
            cuda_graph=profile.cuda_graph,
        )
        encoded = base64.b64encode(args.image.read_bytes()).decode("ascii")
        request = LingBotVlaV2ActionRequest(
            task=args.instruction,
            state=[0.0] * 14,
            camera_high=encoded,
            camera_left_wrist=encoded,
            camera_right_wrist=encoded,
            seed=args.seed,
        )
        result = predict_lingbot_vla_v2_action(
            pipeline,
            request,
            max_image_bytes=args.max_image_bytes,
            max_image_pixels=args.max_image_pixels,
        )
        graph_ready = bool(getattr(pipeline.policy_stage.policy, "cuda_graph_ready", False))
        if profile.cuda_graph and not graph_ready:
            raise RuntimeError("CUDA Graph profile completed warmup without a ready denoising graph")
        report.update(
            passed=True,
            result=result.model_dump(mode="json"),
            quantization_runtime=lingbot_vla_v2_quantization_identity(pipeline.policy_stage.policy),
            cuda_graph_ready=graph_ready,
            environment=collect_runtime_environment(["cuda:0"], repo_root=Path(__file__).resolve().parents[2]),
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if pipeline is not None:
            pipeline.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


def run_suite(args: argparse.Namespace) -> int:
    """Run every selected profile and write an incremental release report."""
    args.repo_root = Path(__file__).resolve().parents[2]
    for path in (args.model_root, args.qwen3vl_root, args.image, args.telefuser_bin):
        if not path.exists():
            raise ValueError(f"required path does not exist: {path}")
    repository_state = _git_state(args.repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "release_summary.json"
    include_contents = args.manifest_mode == "full_sha256"
    all_profiles_selected = {profile.name for profile in args.profiles} == set(RELEASE_PROFILES)
    report: dict[str, Any] = {
        "schema_version": 1,
        "validation": "lingbot_vla_v2_real_gpu_release_suite",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "release_complete": args.aiperf and include_contents and all_profiles_selected,
        "configuration": {
            "profiles": [profile.name for profile in args.profiles],
            "gpu_index": args.gpu_index,
            "requests": args.requests,
            "warmup": args.warmup,
            "aiperf_enabled": args.aiperf,
            "aiperf_requests": args.aiperf_requests,
            "restart_profile": args.restart_profile,
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "packages": _package_versions(),
            "nvidia_smi": _nvidia_environment(),
            "repository": repository_state,
        },
        "artifacts": {
            "checkpoint": tree_manifest(args.model_root, include_contents=include_contents),
            "processor": tree_manifest(args.qwen3vl_root, include_contents=include_contents),
            "input_image": {"path": str(args.image.resolve()), "sha256": _sha256_file(args.image)},
        },
        "profiles": [],
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for profile in args.profiles:
        report["profiles"].append(run_profile(args, profile, root=args.output_dir))
        summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    restart_profile = RELEASE_PROFILES[args.restart_profile]
    if restart_profile.name not in {profile.name for profile in args.profiles}:
        report["restart"] = {"profile": restart_profile.name, "passed": False, "error": "profile was not selected"}
    else:
        report["restart"] = run_restart_check(args, restart_profile, root=args.output_dir)
    report["checks"] = {
        "all_profiles_passed": all(profile["passed"] for profile in report["profiles"]),
        "restart_passed": report["restart"]["passed"],
        "all_profiles_selected": all_profiles_selected,
        "full_manifests": include_contents,
        "aiperf_included": args.aiperf,
    }
    report["passed"] = all(report["checks"].values())
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "artifact": str(summary_path)}))
    return 0 if report["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    suite = subparsers.add_parser("suite", help="Run the complete real-service release suite.")
    suite.add_argument("--model-root", type=Path, required=True)
    suite.add_argument("--qwen3vl-root", type=Path, required=True)
    suite.add_argument("--image", type=Path, required=True)
    suite.add_argument(
        "--profiles",
        type=parse_profiles,
        default=tuple(RELEASE_PROFILES.values()),
        help=f"Comma-separated profiles; defaults to all: {','.join(RELEASE_PROFILES)}",
    )
    suite.add_argument("--restart-profile", choices=tuple(RELEASE_PROFILES), default="bf16-eager")
    suite.add_argument("--gpu-index", default="0", help="Physical GPU selector passed through CUDA_VISIBLE_DEVICES.")
    suite.add_argument("--port", type=int, default=18080)
    default_bin = Path(sys.executable).with_name("telefuser")
    suite.add_argument("--telefuser-bin", type=Path, default=default_bin)
    suite.add_argument("--requests", type=int, default=20)
    suite.add_argument("--warmup", type=int, default=1)
    suite.add_argument("--aiperf", action=argparse.BooleanOptionalAction, default=True)
    suite.add_argument("--aiperf-requests", type=int, default=20)
    suite.add_argument("--aiperf-warmup", type=int, default=2)
    suite.add_argument("--instruction", default="pick up the object")
    suite.add_argument("--dynamic-instruction", default="pick up the red block and place it on the left")
    suite.add_argument("--seed", type=int, default=7)
    suite.add_argument("--http-timeout-seconds", type=float, default=30.0)
    suite.add_argument("--task-timeout-seconds", type=float, default=300.0)
    suite.add_argument("--startup-timeout-seconds", type=float, default=600.0)
    suite.add_argument("--profile-timeout-seconds", type=float, default=3600.0)
    suite.add_argument("--poll-interval-seconds", type=float, default=0.1)
    suite.add_argument("--resource-interval-seconds", type=float, default=0.5)
    suite.add_argument("--max-image-bytes", type=int, default=10 * 1024 * 1024)
    suite.add_argument("--max-image-pixels", type=int, default=16 * 1024 * 1024)
    suite.add_argument("--min-cosine", type=float, default=0.995)
    suite.add_argument("--max-relative-l2", type=float, default=0.10)
    suite.add_argument("--max-absolute-error", type=float, default=0.5)
    suite.add_argument("--manifest-mode", choices=("full_sha256", "filename_and_size"), default="full_sha256")
    suite.add_argument("--output-dir", type=Path, required=True)
    suite.set_defaults(handler=run_suite)

    direct = subparsers.add_parser("direct", help=argparse.SUPPRESS)
    direct.add_argument("--model-root", type=Path, required=True)
    direct.add_argument("--qwen3vl-root", type=Path, required=True)
    direct.add_argument("--image", type=Path, required=True)
    direct.add_argument("--instruction", required=True)
    direct.add_argument("--seed", type=int, required=True)
    direct.add_argument("--profile", choices=tuple(RELEASE_PROFILES), required=True)
    direct.add_argument("--max-image-bytes", type=int, required=True)
    direct.add_argument("--max-image-pixels", type=int, required=True)
    direct.add_argument("--output", type=Path, required=True)
    direct.set_defaults(handler=run_direct)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "suite":
        if args.requests < 1 or args.warmup < 0 or args.aiperf_requests < 1 or args.aiperf_warmup < 0:
            raise SystemExit("request counts must be positive and warmup counts must be non-negative")
        if (
            min(
                args.http_timeout_seconds,
                args.task_timeout_seconds,
                args.startup_timeout_seconds,
                args.profile_timeout_seconds,
                args.poll_interval_seconds,
                args.resource_interval_seconds,
            )
            <= 0
        ):
            raise SystemExit("timeouts, polling, and sampling intervals must be positive")
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
