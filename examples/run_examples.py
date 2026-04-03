"""TeleFuser example runner.

Runs configured pipelines in isolated subprocesses, compares outputs against
baselines (PSNR/SSIM for video, pixel diff for image), and prints a results table.

Usage:
    python examples/run_examples.py --list
    python examples/run_examples.py --pipeline wan21_1_3b_t2v
    python examples/run_examples.py --all
    python examples/run_examples.py --all --update-baseline
    python examples/run_examples.py --all --gpus 0,1,2,3  # Parallel execution
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_CONFIG_PATH = Path(__file__).resolve().parent / "example_config.yaml"
_RESULT_MARKER = "###RESULT###"


def _pipeline_slug(pipeline_key: str) -> str:
    """Convert pipeline key to filesystem-safe slug."""
    return pipeline_key.replace("/", "__")


def _parse_script_path(script: str) -> tuple[str, str]:
    """Parse script path into (example_dir, example_name).

    Args:
        script: Relative path from examples/, e.g. "wan_video/wan21_1_3b_text_to_video_h100.py"

    Returns:
        (example_dir, example_name), e.g. ("wan_video", "wan21_1_3b_text_to_video_h100")
    """
    parts = script.replace("/", "__").replace(".py", "").split("__")
    if len(parts) >= 2:
        return parts[0], "__".join(parts[1:])
    return "unknown", parts[0] if parts else "unknown"


def _generate_output_filename(script: str, gpu_count: int, resolution: str | None, output_type: str) -> str:
    """Generate standardized output filename.

    Format: {example_dir}__{example_name}_{gpu_count}gpu_{resolution}.{ext}
    """
    example_dir, example_name = _parse_script_path(script)
    res = resolution or "unknown"
    ext = "mp4" if output_type == "video" else "png"
    return f"{example_dir}__{example_name}_{gpu_count}gpu_{res}.{ext}"


def _generate_log_filename(script: str, gpu_count: int, timestamp: str) -> str:
    """Generate standardized log filename.

    Format: {timestamp}_{example_dir}__{example_name}_{gpu_count}gpu.log
    """
    example_dir, example_name = _parse_script_path(script)
    return f"{timestamp}_{example_dir}__{example_name}_{gpu_count}gpu.log"


# Cached git and system info (computed once per script run)
_git_info_cache: tuple[str, str] | None = None
_system_info_cache: dict[str, str] | None = None


def _get_git_info() -> tuple[str, str]:
    """Get current git branch name and short commit hash (first 5 chars).

    Result is cached to avoid repeated subprocess calls.

    Returns:
        (branch_name, short_hash) - falls back to ("unknown", "unknown") if not in git repo.
    """
    global _git_info_cache
    if _git_info_cache is not None:
        return _git_info_cache

    try:
        # Get branch name
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=5,
        )
        branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

        # Get short commit hash (first 5 chars)
        hash_result = subprocess.run(
            ["git", "rev-parse", "--short=5", "HEAD"],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=5,
        )
        short_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else "unknown"

        _git_info_cache = (branch, short_hash)
        return _git_info_cache
    except (subprocess.TimeoutExpired, FileNotFoundError):
        _git_info_cache = ("unknown", "unknown")
        return _git_info_cache


def _get_system_info() -> dict[str, str]:
    """Get system environment information.

    Result is cached to avoid repeated computation.

    Returns:
        Dict with python_version, pytorch_version, cuda_version, system_info, gpu_info.
    """
    global _system_info_cache
    if _system_info_cache is not None:
        return _system_info_cache

    info = {}

    # Python version
    info["python_version"] = platform.python_version()

    # System info
    info["system"] = f"{platform.system()} {platform.release()}"
    info["architecture"] = platform.machine()

    # PyTorch and CUDA info
    try:
        info["pytorch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"

        # GPU info
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            info["gpu_count"] = str(gpu_count)
            info["gpu_devices"] = ", ".join(gpu_names)
        else:
            info["gpu_count"] = "0"
            info["gpu_devices"] = "N/A"
    except Exception:
        info["pytorch_version"] = "N/A"
        info["cuda_version"] = "N/A"
        info["cudnn_version"] = "N/A"
        info["gpu_count"] = "0"
        info["gpu_devices"] = "N/A"

    _system_info_cache = info
    return info


def _get_date_dir(output_root: str) -> str:
    """Get date-based output directory with git info and minute precision.

    Format: {branch}_{short_hash}_{YYYY-MM-DD_HHMM}
    """
    branch, short_hash = _get_git_info()
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    dir_name = f"{branch}_{short_hash}_{date_str}"
    return os.path.join(output_root, dir_name)


def _resolve_output_root(output_root: str) -> str:
    """Resolve output_root to absolute path if relative."""
    if not os.path.isabs(output_root):
        return os.path.join(_PROJECT_ROOT, output_root)
    return output_root


def _is_oom(exc_or_text: Exception | str) -> bool:
    """Check if an exception or text indicates an out-of-memory error."""
    text = str(exc_or_text).lower()
    return "out of memory" in text or "cuda out of memory" in text


# =============================================================================
# GPU Resource Pool
# =============================================================================


def _build_reproduce_cmd(pipeline_key: str, config_path: str | None) -> str:
    """Build reproduce command string for logging and error reporting.

    Args:
        pipeline_key: Pipeline name from config.
        config_path: Optional config YAML path.

    Returns:
        Reproduce command string.
    """
    cmd = f"python examples/run_examples.py --pipeline {pipeline_key}"
    if config_path:
        cmd += f" --config {config_path}"
    return cmd


class GPUPool:
    """Manages GPU device allocation for parallel pipeline execution."""

    def __init__(self, gpu_ids: list[int]):
        """Initialize with available GPU device IDs.

        Args:
            gpu_ids: List of GPU device IDs available for scheduling.
        """
        self.available = sorted(gpu_ids)
        self.total = len(gpu_ids)  # Total GPUs in pool (unchanged)
        self.allocated: dict[int, list[int]] = {}  # job_id -> list of GPU IDs
        self._next_job_id = 0

    def total_count(self) -> int:
        """Get total GPU count in the pool.

        Returns:
            Total number of GPUs (regardless of current availability).
        """
        return self.total

    def allocate(self, gpu_count: int) -> tuple[int, list[int]] | None:
        """Allocate GPUs for a new job.

        Args:
            gpu_count: Number of GPUs required.

        Returns:
            (job_id, gpu_ids) if allocation successful, None if insufficient GPUs.
        """
        if len(self.available) < gpu_count:
            return None

        # Take first N available GPUs
        allocated_ids = self.available[:gpu_count]
        self.available = self.available[gpu_count:]

        job_id = self._next_job_id
        self._next_job_id += 1
        self.allocated[job_id] = allocated_ids

        return job_id, allocated_ids

    def release(self, job_id: int) -> list[int]:
        """Release GPUs from a completed job.

        Args:
            job_id: ID of the job to release.

        Returns:
            List of GPU IDs that were released.
        """
        if job_id not in self.allocated:
            return []

        gpu_ids = self.allocated.pop(job_id)
        self.available.extend(gpu_ids)
        self.available.sort()
        return gpu_ids

    def available_count(self) -> int:
        """Return number of available GPUs."""
        return len(self.available)

    def total_count(self) -> int:
        """Return total GPU count (available + allocated)."""
        return len(self.available) + sum(len(v) for v in self.allocated.values())


# =============================================================================
# Pipeline Scheduler
# =============================================================================


@dataclass
class RunningJob:
    """Tracks a running pipeline subprocess."""

    job_id: int
    pipeline_key: str
    ppl_cfg: PipelineConfig
    gpu_ids: list[int]
    process: subprocess.Popen
    start_time: float
    log_path: str


class PipelineScheduler:
    """Schedules pipelines across a GPU pool for parallel execution."""

    def __init__(
        self,
        gpu_pool: GPUPool,
        pipelines: dict[str, PipelineConfig],
        output_root: str,
        output_dir: str | None = None,
        config_path: str | None = None,
        update_baseline: bool = False,
        verbose: bool = False,
    ):
        """Initialize scheduler.

        Args:
            gpu_pool: GPU resource pool for allocation.
            pipelines: Pipeline configurations to run.
            output_root: Root directory for baseline comparison.
            output_dir: Directory for saving outputs (defaults to output_root).
            config_path: Optional config YAML path.
            update_baseline: Whether to update baseline outputs.
            verbose: Whether to show verbose output.
        """
        self.gpu_pool = gpu_pool
        self.output_root = output_root
        self.output_dir = output_dir or output_root
        self.config_path = config_path
        self.update_baseline = update_baseline
        self.verbose = verbose

        # Filter pipelines that require more GPUs than available in pool
        total_gpus = gpu_pool.total_count()
        skipped_pipelines: list[tuple[str, int]] = []  # (name, required_gpu_count)
        valid_pipelines: dict[str, PipelineConfig] = {}

        for name, ppl_cfg in pipelines.items():
            if ppl_cfg.gpu_count > total_gpus:
                skipped_pipelines.append((name, ppl_cfg.gpu_count))
            else:
                valid_pipelines[name] = ppl_cfg

        # Log skipped pipelines
        if skipped_pipelines:
            for name, required in skipped_pipelines:
                print(f"  [SKIP] {name}: requires {required} GPUs, but pool has {total_gpus}")

        # Sort pipelines by gpu_count descending for greedy scheduling
        sorted_pipelines = sorted(
            valid_pipelines.items(),
            key=lambda x: x[1].gpu_count,
            reverse=True,
        )
        self.pending: list[tuple[str, PipelineConfig]] = list(sorted_pipelines)

        self.running: dict[int, RunningJob] = {}  # job_id -> RunningJob
        self.results: list[Result] = []

    def has_pending(self) -> bool:
        """Check if there are pending pipelines."""
        return len(self.pending) > 0

    def has_running(self) -> bool:
        """Check if there are running jobs."""
        return len(self.running) > 0

    def schedule_next(self) -> tuple[str, list[int]] | None:
        """Try to schedule a pending pipeline.

        Returns:
            (pipeline_key, gpu_ids) if scheduled, None if no GPU available or no pending.
        """
        if not self.pending:
            return None

        # Find first pending pipeline that fits available GPUs
        for idx, (name, ppl_cfg) in enumerate(self.pending):
            if ppl_cfg.gpu_count <= self.gpu_pool.available_count():
                allocation = self.gpu_pool.allocate(ppl_cfg.gpu_count)
                if allocation:
                    job_id, gpu_ids = allocation

                    # Remove from pending
                    self.pending.pop(idx)

                    # Start subprocess
                    env = self._build_env(gpu_ids)
                    cmd = self._build_cmd(name)

                    # Prepare log file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    logs_dir = os.path.join(self.output_dir, "logs")
                    os.makedirs(logs_dir, exist_ok=True)
                    log_filename = _generate_log_filename(ppl_cfg.script, ppl_cfg.gpu_count, timestamp)
                    log_path = os.path.join(logs_dir, log_filename)

                    # Spawn subprocess (capture mode, log to file)
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=_PROJECT_ROOT,
                        env=env,
                    )

                    self.running[job_id] = RunningJob(
                        job_id=job_id,
                        pipeline_key=name,
                        ppl_cfg=ppl_cfg,
                        gpu_ids=gpu_ids,
                        process=process,
                        start_time=time.time(),
                        log_path=log_path,
                    )

                    return name, gpu_ids

        return None

    def collect_finished(self, timeout_grace: float = 5.0) -> list[Result]:
        """Check running jobs and collect completed results.

        Args:
            timeout_grace: Extra time to wait after timeout before force kill.

        Returns:
            List of newly completed results.
        """
        new_results: list[Result] = []
        finished_job_ids: list[int] = []

        for job_id, job in self.running.items():
            elapsed = time.time() - job.start_time
            timeout = job.ppl_cfg.timeout_seconds

            # Check if process finished
            if job.process.poll() is not None:
                result = self._process_completed_job(job)
                new_results.append(result)
                finished_job_ids.append(job_id)
            # Check if timeout exceeded
            elif elapsed > timeout:
                # Grace period then force kill
                if elapsed > timeout + timeout_grace:
                    job.process.kill()
                    job.process.wait()
                    result = self._process_timeout_job(job)
                    new_results.append(result)
                    finished_job_ids.append(job_id)

        # Release GPUs and remove from running
        for job_id in finished_job_ids:
            self.gpu_pool.release(job_id)
            self.running.pop(job_id)

        self.results.extend(new_results)
        return new_results

    def wait_all(self) -> list[Result]:
        """Wait for all remaining jobs to complete.

        Returns:
            All results collected.
        """
        while self.has_running() or self.has_pending():
            self.schedule_next()
            self.collect_finished()
            time.sleep(0.5)

        return self.results

    def _build_env(self, gpu_ids: list[int]) -> dict:
        """Build environment with CUDA_VISIBLE_DEVICES set."""
        env = os.environ.copy()
        existing_pypath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{_PROJECT_ROOT}{os.pathsep}{existing_pypath}" if existing_pypath else _PROJECT_ROOT
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(id) for id in gpu_ids)
        return env

    def _build_cmd(self, pipeline_key: str) -> list[str]:
        """Build subprocess command."""
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-single",
            pipeline_key,
        ]
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        cmd.extend(["--output-dir", self.output_dir])
        return cmd

    def _build_reproduce_cmd(self, pipeline_key: str) -> str:
        """Build reproduce command string for logging."""
        return _build_reproduce_cmd(pipeline_key, self.config_path)

    def _process_completed_job(self, job: RunningJob) -> Result:
        """Process a completed job and return Result."""
        stdout_text, stderr_text = job.process.communicate()
        elapsed = time.time() - job.start_time

        # Parse result
        data = _parse_runner_output(stdout_text)
        status = data.get("status", "ERROR")
        error_msg = data.get("error", "")
        error_cat = data.get("error_category", "")
        peak_mem = data.get("peak_gpu_memory_mb", 0.0)
        script = data.get("script", job.ppl_cfg.script)

        # OOM detection fallback
        if not error_cat and stderr_text and _is_oom(stderr_text):
            error_cat = "OOM_ERROR"

        if not data and job.process.returncode != 0:
            error_msg = f"Process exited with code {job.process.returncode}"

        # Save log
        reproduce_cmd = self._build_reproduce_cmd(job.pipeline_key)
        branch, short_hash = _get_git_info()
        sys_info = _get_system_info()

        with open(job.log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Pipeline: {job.pipeline_key} ===\n")
            f.write(f"=== Branch: {branch} ===\n")
            f.write(f"=== Commit: {short_hash} ===\n")
            f.write(f"=== Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')} ===\n")
            f.write(f"=== GPUs: {','.join(str(id) for id in job.gpu_ids)} ===\n")
            f.write(f"=== Command: {reproduce_cmd} ===\n")
            f.write(f"=== System Info ===\n")
            f.write(f"  Python: {sys_info['python_version']}\n")
            f.write(f"  PyTorch: {sys_info['pytorch_version']}\n")
            f.write(f"  CUDA: {sys_info['cuda_version']}\n")
            f.write(f"  cuDNN: {sys_info['cudnn_version']}\n")
            f.write(f"  System: {sys_info['system']}\n")
            f.write(f"  Arch: {sys_info['architecture']}\n")
            f.write(f"  GPU Count: {sys_info['gpu_count']}\n")
            f.write(f"  GPU Devices: {sys_info['gpu_devices']}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout_text or "(empty)\n")
            f.write("\n=== STDERR ===\n")
            f.write(stderr_text or "(empty)\n")

        result = Result(
            name=job.pipeline_key,
            status=status,
            elapsed_seconds=round(elapsed, 2),
            peak_gpu_memory_mb=round(peak_mem, 2),
            num_frames=data.get("num_frames"),
            resolution=data.get("resolution"),
            num_steps=data.get("num_steps"),
            error_category=error_cat,
            error_message=error_msg,
            script=script,
            reproduce_command=reproduce_cmd,
            log_path=job.log_path,
        )

        # Compare against baseline
        if status == "PASS":
            output_path = data.get("output_path")
            cmp = compare_against_baseline(
                self.output_root,
                job.ppl_cfg.script,
                job.ppl_cfg.gpu_count,
                output_path,
                job.ppl_cfg.output_type,
                job.ppl_cfg.psnr_min,
                job.ppl_cfg.ssim_min,
                job.ppl_cfg.pixel_diff_max,
            )
            result.regression_metrics = cmp.get("metrics", {})
            result.note = cmp["message"]
            if cmp["baseline_exists"] and not cmp["passed"]:
                result.status = "FAIL"

            if self.update_baseline and output_path and os.path.exists(output_path):
                _update_baseline(self.output_root, output_path)
                result.note += " [baseline updated]"

        # Performance/memory threshold checks
        if (
            result.status == "PASS"
            and job.ppl_cfg.max_elapsed_seconds
            and result.elapsed_seconds > job.ppl_cfg.max_elapsed_seconds
        ):
            result.status = "FAIL"
            result.note += f" [PERF: {result.elapsed_seconds:.1f}s > {job.ppl_cfg.max_elapsed_seconds:.1f}s]"

        if (
            result.status == "PASS"
            and job.ppl_cfg.max_gpu_memory_mb
            and result.peak_gpu_memory_mb > job.ppl_cfg.max_gpu_memory_mb
        ):
            result.status = "FAIL"
            result.note += f" [MEM: {result.peak_gpu_memory_mb:.0f}MB > {job.ppl_cfg.max_gpu_memory_mb:.0f}MB]"

        if result.status != "PASS" and not result.note:
            result.note = error_msg[:60] if error_msg else ""
            if error_cat:
                result.note = f"[{error_cat}] {result.note}"

        return result

    def _process_timeout_job(self, job: RunningJob) -> Result:
        """Process a timeout job and return Result."""
        elapsed = time.time() - job.start_time
        stdout_text, stderr_text = job.process.communicate()

        reproduce_cmd = self._build_reproduce_cmd(job.pipeline_key)
        branch, short_hash = _get_git_info()
        sys_info = _get_system_info()

        # Save log
        with open(job.log_path, "w", encoding="utf-8") as f:
            f.write(f"=== Pipeline: {job.pipeline_key} ===\n")
            f.write(f"=== Branch: {branch} ===\n")
            f.write(f"=== Commit: {short_hash} ===\n")
            f.write(f"=== Timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')} ===\n")
            f.write(f"=== GPUs: {','.join(str(id) for id in job.gpu_ids)} ===\n")
            f.write(f"=== Command: {reproduce_cmd} ===\n")
            f.write(f"=== System Info ===\n")
            f.write(f"  Python: {sys_info['python_version']}\n")
            f.write(f"  PyTorch: {sys_info['pytorch_version']}\n")
            f.write(f"  CUDA: {sys_info['cuda_version']}\n")
            f.write(f"  cuDNN: {sys_info['cudnn_version']}\n")
            f.write(f"  System: {sys_info['system']}\n")
            f.write(f"  Arch: {sys_info['architecture']}\n")
            f.write(f"  GPU Count: {sys_info['gpu_count']}\n")
            f.write(f"  GPU Devices: {sys_info['gpu_devices']}\n\n")
            f.write("=== TIMEOUT ===\n")
            f.write(f"Timeout after {job.ppl_cfg.timeout_seconds}s\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout_text or "(empty)\n")
            f.write("\n=== STDERR ===\n")
            f.write(stderr_text or "(empty)\n")

        return Result(
            name=job.pipeline_key,
            status="TIMEOUT",
            elapsed_seconds=round(elapsed, 2),
            script=job.ppl_cfg.script,
            reproduce_command=reproduce_cmd,
            log_path=job.log_path,
            note=f"Timeout after {job.ppl_cfg.timeout_seconds}s",
        )

    def get_running_status(self) -> list[tuple[str, list[int], float]]:
        """Get status of currently running jobs.

        Returns:
            List of (pipeline_key, gpu_ids, elapsed_seconds).
        """
        return [
            (job.pipeline_key, job.gpu_ids, round(time.time() - job.start_time, 1)) for job in self.running.values()
        ]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline test."""

    script: str  # Relative path from examples/, e.g. "wan_video/wan21_1_3b_text_to_video_h100.py"
    enabled: bool = True
    gpu_count: int = 1
    timeout_seconds: int = 1800
    output_type: str = "video"  # "video" | "image"
    seed: int = 42
    model_root: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    resolution: str | None = None
    aspect_ratio: str | None = None
    input_image_path: str | None = None
    first_image_path: str | None = None
    last_image_path: str | None = None
    input_video_path: str | None = None
    ppl_config_overrides: dict = field(default_factory=dict)
    # Regression thresholds
    psnr_min: float = 25.0
    ssim_min: float = 0.85
    pixel_diff_max: float = 0.02
    # Performance thresholds (None = disabled)
    max_elapsed_seconds: float | None = None
    max_gpu_memory_mb: float | None = None


@dataclass
class Config:
    """Top-level example config."""

    output_root: str = "work_dirs/example_outputs"
    pipelines: dict[str, PipelineConfig] = field(default_factory=dict)


def load_config(config_path: str | None = None) -> Config:
    """Load example config from YAML."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        return Config()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.get("defaults", {})
    output_root = raw.get("output_root", "work_dirs/example_outputs")

    pipelines: dict[str, PipelineConfig] = {}
    valid_fields = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    for name, ppl_data in raw.get("pipelines", {}).items():
        if ppl_data is None:
            ppl_data = {}
        # Merge defaults for missing fields
        merged = {**defaults, **ppl_data}
        # Only pass fields that PipelineConfig accepts
        filtered = {k: v for k, v in merged.items() if k in valid_fields}
        pipelines[name] = PipelineConfig(**filtered)

    return Config(output_root=output_root, pipelines=pipelines)


# =============================================================================
# Subprocess Worker (--run-single mode)
# =============================================================================


def _import_example_module(example_path: str) -> ModuleType:
    """Dynamically import an example script as a module."""
    path = Path(example_path).resolve()
    module_name = f"_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {example_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _patch_ppl_config(module: ModuleType, overrides: dict) -> None:
    """Patch PPL_CONFIG or DEFAULT_CONFIG in the module with overrides."""
    from telefuser.core.config import AttnImplType

    config_attr = None
    for attr_name in ("PPL_CONFIG", "DEFAULT_CONFIG"):
        if hasattr(module, attr_name):
            config_attr = attr_name
            break

    if config_attr is None or not overrides:
        return

    config = getattr(module, config_attr)
    for key, value in overrides.items():
        if value is not None:
            if key == "attn_impl" and isinstance(value, str):
                value = AttnImplType[value]
            config[key] = value


def _extract_click_default(module: ModuleType, param_name: str) -> str | None:
    """Extract a click option's default value from the module's main() command."""
    import click

    main_func = getattr(module, "main", None)
    if main_func is None or not isinstance(main_func, click.BaseCommand):
        return None
    for param in main_func.params:
        if isinstance(param, click.core.Option):
            # Match by param.name (e.g., 'prompt') or by opts (e.g., '--prompt')
            if param.name == param_name or f"--{param_name}" in param.opts:
                if param.default is not None:
                    return str(param.default)
    return None


def _extract_ppl_config_default(module: ModuleType, key: str) -> str | None:
    """Extract a default value from the module's PPL_CONFIG or DEFAULT_CONFIG."""
    config_attr = None
    for attr_name in ("PPL_CONFIG", "DEFAULT_CONFIG"):
        if hasattr(module, attr_name):
            config_attr = attr_name
            break
    if config_attr is None:
        return None
    config = getattr(module, config_attr)
    return config.get(key)


def _call_get_pipeline(module: ModuleType, config: dict) -> object:
    """Call get_pipeline() with arguments matched by signature inspection."""
    func = module.get_pipeline
    params = list(inspect.signature(func).parameters.keys())
    kwargs: dict = {}

    if "parallelism" in params:
        kwargs["parallelism"] = config.get("gpu_count", 1)
    elif "num_gpus" in params:
        kwargs["num_gpus"] = config.get("gpu_count", 1)

    if "model_root" in params and config.get("model_root"):
        kwargs["model_root"] = config["model_root"]

    # FlashVSR: dit_path/vae_path built from model_root + module constants
    model_root = config.get("model_root", "")
    # If model_root not in config, try to extract default from module's click command
    if not model_root:
        model_root = _extract_click_default(module, "model_root") or ""
    if "dit_path" in params and "dit_path" not in kwargs:
        dit_filename = getattr(module, "DIT_FILENAME", None)
        if dit_filename and model_root:
            kwargs["dit_path"] = os.path.join(model_root, dit_filename)
    if "vae_path" in params and "vae_path" not in kwargs:
        vae_filename = getattr(module, "VAE_FILENAME", None)
        if vae_filename and model_root:
            kwargs["vae_path"] = os.path.join(model_root, vae_filename)

    return func(**kwargs)


def _call_run(module: ModuleType, pipeline: object, config: dict) -> object:
    """Call run() with arguments matched by signature inspection."""
    func = module.run
    params = list(inspect.signature(func).parameters.keys())
    kwargs: dict = {"pipeline": pipeline}

    # Helper to get param with multiple fallback sources
    def get_param(param_name: str, default: str | None = None) -> str | None:
        # Priority: YAML config > Click default > PPL_CONFIG
        val = config.get(param_name)
        if val:
            return val
        val = _extract_click_default(module, param_name)
        if val:
            return val
        val = _extract_ppl_config_default(module, param_name)
        if val:
            return val
        return default

    # Prompt (required for most pipelines)
    if "prompt" in params:
        prompt_val = get_param("prompt")
        if prompt_val:
            kwargs["prompt"] = prompt_val

    # Negative prompt
    if "negative_prompt" in params:
        kwargs["negative_prompt"] = get_param("negative_prompt", "")

    if "seed" in params:
        kwargs["seed"] = config.get("seed", 42)

    # Resolution (e.g., "480p", "720p")
    if "resolution" in params:
        kwargs["resolution"] = get_param("resolution")

    # Aspect ratio
    if "aspect_ratio" in params:
        kwargs["aspect_ratio"] = get_param("aspect_ratio")

    if "height" in params:
        kwargs["height"] = config.get("height", 480)
    if "width" in params:
        kwargs["width"] = config.get("width", 832)

    # Image input (for I2V pipelines)
    if "image" in params and config.get("input_image_path"):
        from PIL import Image

        kwargs["image"] = Image.open(config["input_image_path"]).convert("RGB")

    # First/Last image input (for FL2V pipelines)
    if "first_image" in params:
        from PIL import Image

        img_path = config.get("first_image_path") or config.get("input_image_path")
        if img_path:
            kwargs["first_image"] = Image.open(img_path).convert("RGB")

    if "last_image" in params and config.get("last_image_path"):
        from PIL import Image

        kwargs["last_image"] = Image.open(config["last_image_path"]).convert("RGB")

    # Video input
    if "input_video" in params and config.get("input_video_path"):
        from telefuser.utils.video import VideoData

        video_data = VideoData(video_file=config["input_video_path"], height=360, width=640)
        kwargs["input_video"] = video_data

    if "LQ_video" in params and config.get("input_video_path"):
        from telefuser.utils.video import VideoData

        kwargs["LQ_video"] = VideoData(video_file=config["input_video_path"], height=360, width=640).raw_data()

    if "scale" in params:
        kwargs["scale"] = 2

    # FlashVSR chunked calling pattern
    if "input_video" in kwargs and "scale" in kwargs:
        return _call_run_flashvsr_chunked(func, kwargs)

    return func(**kwargs)


def _call_run_flashvsr_chunked(func: callable, kwargs: dict) -> object:
    """Call FlashVSR run() in chunks matching the example's main() pattern."""
    from telefuser.utils.video import VideoData

    video_data = kwargs.pop("input_video")
    pipeline = kwargs["pipeline"]

    frames = video_data.raw_data() if isinstance(video_data, VideoData) else list(video_data)
    final_video = []

    # First chunk: 25 frames
    video = func(**{**kwargs, "input_video": frames[:25]})
    if isinstance(video, list):
        final_video.extend(video)

    # Remaining: 16-frame chunks
    offset = 25
    while offset < len(frames):
        end = min(offset + 16, len(frames))
        video = func(**{**kwargs, "input_video": frames[offset:end]})
        if isinstance(video, list):
            final_video.extend(video)
        offset += 16

    if hasattr(pipeline, "clean_cache"):
        pipeline.clean_cache()

    return final_video


def _save_output(
    output: object,
    output_dir: str,
    output_type: str,
    fps: int = 15,
    filename: str | None = None,
) -> tuple[str | None, int | None, str | None]:
    """Save pipeline output to file. Returns (path, num_frames, resolution)."""
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    if output is None:
        return None, None, None

    # Use custom filename or default
    default_name = f"output.{('mp4' if output_type == 'video' else 'png')}"
    output_filename = filename or default_name

    if output_type == "video":
        if isinstance(output, (list, tuple)) and len(output) > 0:
            # Unpack tuple from longcat pipelines: (frames, latents)
            frames = output
            if isinstance(output, tuple):
                frames = output[0] if isinstance(output[0], list) else output
            output_path = os.path.join(output_dir, output_filename)
            from telefuser.utils.video import save_video

            save_video(frames, output_path, fps=fps, quality=6)
            first = frames[0]
            resolution = f"{first.width}x{first.height}" if isinstance(first, Image.Image) else None
            return output_path, len(frames), resolution

    if output_type == "image":
        output_path = os.path.join(output_dir, output_filename)
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], Image.Image):
            output[0].save(output_path)
            return output_path, 1, f"{output[0].width}x{output[0].height}"
        if isinstance(output, Image.Image):
            output.save(output_path)
            return output_path, 1, f"{output.width}x{output.height}"

    return None, None, None


def _validate_output(output: object) -> list[str]:
    """Check pipeline output for NaN/Inf/None. Returns warning strings."""
    warnings: list[str] = []
    if output is None:
        warnings.append("Output is None")
        return warnings

    if isinstance(output, list) and len(output) == 0:
        warnings.append("Output is an empty list")

    tensor = None
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, list) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        tensor = output[0]

    if tensor is not None:
        if torch.isnan(tensor).any():
            warnings.append("Output contains NaN")
        if torch.isinf(tensor).any():
            warnings.append("Output contains Inf")

    return warnings


def _emit_result(data: dict) -> None:
    """Print JSON result for parent process to parse."""
    print(f"{_RESULT_MARKER}{json.dumps(data)}", flush=True)


def _run_single(pipeline_key: str, config_path: str | None, output_dir: str | None) -> None:
    """Subprocess entry point: load, run, save one pipeline.

    Args:
        pipeline_key: Pipeline name from config.
        config_path: Optional config YAML path.
        output_dir: Target output directory (date-based folder for results).
                   If None, creates a new date-based directory under output_root.
    """
    cfg = load_config(config_path)
    ppl_cfg = cfg.pipelines.get(pipeline_key)
    if ppl_cfg is None:
        _emit_result({"status": "ERROR", "error": f"Pipeline '{pipeline_key}' not found in config"})
        sys.exit(1)

    examples_root = os.path.join(_PROJECT_ROOT, "examples")
    script_path = os.path.join(examples_root, ppl_cfg.script)

    output_root = _resolve_output_root(cfg.output_root)
    if output_dir:
        target_dir = output_dir
    else:
        target_dir = _get_date_dir(output_root)
        os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    runner_config = {
        "gpu_count": ppl_cfg.gpu_count,
        "seed": ppl_cfg.seed,
        "model_root": ppl_cfg.model_root,
        "prompt": ppl_cfg.prompt,
        "negative_prompt": ppl_cfg.negative_prompt,
        "resolution": ppl_cfg.resolution,
        "aspect_ratio": ppl_cfg.aspect_ratio,
        "input_image_path": ppl_cfg.input_image_path,
        "first_image_path": ppl_cfg.first_image_path,
        "last_image_path": ppl_cfg.last_image_path,
        "input_video_path": ppl_cfg.input_video_path,
        "ppl_config_overrides": ppl_cfg.ppl_config_overrides,
        "script": ppl_cfg.script,  # Pass script for filename generation
    }
    # Merge ppl_config_overrides into runner_config for height/width access
    runner_config.update(ppl_cfg.ppl_config_overrides)

    pipeline = None
    gpu_mem_peak = 0.0
    start = time.time()

    # Phase 1: Model Loading
    try:
        module = _import_example_module(script_path)
        _patch_ppl_config(module, ppl_cfg.ppl_config_overrides)
        pipeline = _call_get_pipeline(module, runner_config)
    except Exception as e:
        tb = traceback.format_exc()
        category = "OOM_ERROR" if _is_oom(e) else "MODEL_LOAD_ERROR"
        _emit_result(
            {
                "status": "ERROR",
                "error": f"{e}\n{tb}",
                "error_category": category,
                "elapsed": round(time.time() - start, 2),
            }
        )
        sys.exit(1)

    # Phase 2: Inference
    output = None
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        output = _call_run(module, pipeline, runner_config)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        tb = traceback.format_exc()
        category = "OOM_ERROR" if _is_oom(e) else "INFERENCE_ERROR"
        _emit_result(
            {
                "status": "ERROR",
                "error": f"{e}\n{tb}",
                "error_category": category,
                "elapsed": round(time.time() - start, 2),
                "peak_gpu_memory_mb": round(gpu_mem_peak, 2),
            }
        )
        del pipeline
        gc.collect()
        sys.exit(1)

    # Phase 3: Validation & Save
    warnings = _validate_output(output)
    ppl_config = getattr(module, "PPL_CONFIG", {})
    num_steps = ppl_config.get("num_inference_steps")
    if isinstance(num_steps, list):
        num_steps = sum(num_steps)
    output_fps = ppl_config.get("target_fps", 15)

    # First save to temp location to get resolution
    temp_dir = os.path.join(output_root, "temp", timestamp)
    try:
        temp_path, num_frames, resolution = _save_output(output, temp_dir, ppl_cfg.output_type, fps=output_fps)
    except Exception as e:
        tb = traceback.format_exc()
        _emit_result(
            {
                "status": "ERROR",
                "error": f"{e}\n{tb}",
                "error_category": "OUTPUT_ERROR",
                "elapsed": round(time.time() - start, 2),
                "peak_gpu_memory_mb": round(gpu_mem_peak, 2),
            }
        )
        pipeline = None  # noqa: F841
        gc.collect()
        sys.exit(1)

    # Move to final location with correct filename
    final_filename = _generate_output_filename(ppl_cfg.script, ppl_cfg.gpu_count, resolution, ppl_cfg.output_type)
    final_path = os.path.join(target_dir, final_filename)

    if temp_path and os.path.exists(temp_path):
        shutil.move(temp_path, final_path)
        # Clean up temp directory
        try:
            os.rmdir(temp_dir)
            parent_temp = os.path.dirname(temp_dir)
            if not os.listdir(parent_temp):
                os.rmdir(parent_temp)
        except OSError:
            pass  # Directory not empty or other error

    elapsed = time.time() - start
    status = "PASS"
    error_msg = ""
    if warnings:
        severe = [w for w in warnings if "NaN" in w or "Inf" in w or "is None" in w]
        if severe:
            status = "ERROR"
            error_msg = "; ".join(warnings)

    _emit_result(
        {
            "status": status,
            "output_path": final_path,
            "error": error_msg,
            "elapsed": round(elapsed, 2),
            "peak_gpu_memory_mb": round(gpu_mem_peak, 2),
            "num_frames": num_frames,
            "resolution": resolution,
            "num_steps": num_steps,
            "script": ppl_cfg.script,
        }
    )

    del pipeline  # noqa: F821
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Metrics
# =============================================================================


def compute_video_metrics(baseline_path: str, current_path: str) -> dict[str, float]:
    """Compute PSNR and SSIM between two videos using streaming frame comparison."""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    from telefuser.utils.video import VideoData

    try:
        video_true = VideoData(video_file=baseline_path)
        video_test = VideoData(video_file=current_path)
    except Exception:
        return {}

    # Compare up to the shorter video length
    n_frames = min(len(video_true), len(video_test))
    if n_frames == 0:
        return {}

    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    for i in range(n_frames):
        frame_true = np.array(video_true[i])
        frame_test = np.array(video_test[i])
        psnr_sum += peak_signal_noise_ratio(frame_true, frame_test)
        ssim_sum += structural_similarity(frame_true, frame_test, channel_axis=2)
        n += 1

    del video_true, video_test

    return {"psnr": psnr_sum / n, "ssim": ssim_sum / n}


def compute_image_diff(baseline_path: str, current_path: str) -> float | None:
    """Compute mean absolute pixel difference (0-1) between two images."""
    from PIL import Image

    img_a = np.array(Image.open(baseline_path).convert("RGB")).astype(np.float32) / 255.0
    img_b = np.array(Image.open(current_path).convert("RGB")).astype(np.float32) / 255.0
    if img_a.shape != img_b.shape:
        return None
    return float(np.mean(np.abs(img_a - img_b)))


# =============================================================================
# Resume Support
# =============================================================================


def _check_output_exists(output_dir: str, script: str, gpu_count: int, output_type: str) -> bool:
    """Check if output file already exists in the specified output directory.

    Args:
        output_dir: Target output directory (date-based folder) to check.
        script: Pipeline script path.
        gpu_count: Number of GPUs used.
        output_type: Output type (video/image).

    Returns:
        True if output file exists in the directory.
    """
    example_dir, example_name = _parse_script_path(script)
    ext = "mp4" if output_type == "video" else "png"
    pattern = f"{example_dir}__{example_name}_{gpu_count}gpu_"

    if not os.path.isdir(output_dir):
        return False

    for f in os.listdir(output_dir):
        if f.startswith(pattern) and f.endswith(f".{ext}"):
            return True

    return False


def _get_completed_pipelines(output_dir: str, pipelines: dict[str, PipelineConfig]) -> set[str]:
    """Get set of pipeline names that already have output files in the specified directory.

    Args:
        output_dir: Target output directory (date-based folder) to check.
        pipelines: Dict of pipeline name to config.

    Returns:
        Set of pipeline names with existing output files.
    """
    completed = set()
    for name, cfg in pipelines.items():
        if _check_output_exists(output_dir, cfg.script, cfg.gpu_count, cfg.output_type):
            completed.add(name)
    return completed


def _load_existing_results(output_root: str) -> dict:
    """Load existing results dict from report file if it exists."""
    report_path = os.path.join(output_root, "example_report.json")
    if not os.path.exists(report_path):
        return {}

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        return report.get("results", {})
    except (json.JSONDecodeError, KeyError):
        return {}


# =============================================================================
# Baseline Comparison
# =============================================================================


def _baseline_dir(output_root: str) -> str:
    """Get baseline directory (independent of pipeline)."""
    return os.path.join(output_root, "baseline")


def _find_baseline_file(baseline_dir: str, script: str, gpu_count: int, output_type: str) -> str | None:
    """Find existing baseline file with any resolution."""
    example_dir, example_name = _parse_script_path(script)
    ext = "mp4" if output_type == "video" else "png"
    pattern = f"{example_dir}__{example_name}_{gpu_count}gpu_"
    if not os.path.isdir(baseline_dir):
        return None
    for f in os.listdir(baseline_dir):
        if f.startswith(pattern) and f.endswith(f".{ext}"):
            return os.path.join(baseline_dir, f)
    return None


def _get_baseline_path(output_root: str, script: str, gpu_count: int, output_type: str) -> str | None:
    """Get baseline file path."""
    bdir = _baseline_dir(output_root)
    return _find_baseline_file(bdir, script, gpu_count, output_type)


def _update_baseline(output_root: str, current_path: str) -> str:
    """Copy current output to baseline directory."""
    bdir = _baseline_dir(output_root)
    os.makedirs(bdir, exist_ok=True)
    dst = os.path.join(bdir, os.path.basename(current_path))
    # Remove old baseline if exists (different resolution)
    if os.path.exists(dst):
        os.remove(dst)
    shutil.copy2(current_path, dst)
    return dst


def compare_against_baseline(
    output_root: str,
    script: str,
    gpu_count: int,
    current_path: str | None,
    output_type: str,
    psnr_min: float,
    ssim_min: float,
    pixel_diff_max: float,
) -> dict:
    """Compare current output against baseline. Returns dict with passed, metrics, message."""
    baseline_path = _get_baseline_path(output_root, script, gpu_count, output_type)

    if baseline_path is None:
        if current_path and os.path.exists(current_path):
            saved = _update_baseline(output_root, current_path)
            return {"passed": True, "baseline_exists": False, "metrics": {}, "message": f"Saved as baseline: {saved}"}
        return {"passed": True, "baseline_exists": False, "metrics": {}, "message": "No baseline (first run)"}

    if not current_path or not os.path.exists(current_path):
        return {"passed": False, "baseline_exists": True, "metrics": {}, "message": "No output file produced"}

    try:
        if output_type == "video":
            m = compute_video_metrics(baseline_path, current_path)
            psnr, ssim = m.get("psnr"), m.get("ssim")
            passed = True
            msgs = []
            if psnr is not None and psnr < psnr_min:
                passed = False
                msgs.append(f"PSNR {psnr:.2f} < {psnr_min}")
            if ssim is not None and ssim < ssim_min:
                passed = False
                msgs.append(f"SSIM {ssim:.4f} < {ssim_min}")
            msg = "; ".join(msgs) if msgs else f"PSNR={psnr:.2f}, SSIM={ssim:.4f}"
            return {"passed": passed, "baseline_exists": True, "metrics": m, "message": msg}
        else:
            diff = compute_image_diff(baseline_path, current_path)
            if diff is None:
                return {"passed": False, "baseline_exists": True, "metrics": {}, "message": "Image size mismatch"}
            passed = diff <= pixel_diff_max
            msg = f"pixel_diff={diff:.6f}" + ("" if passed else f" > {pixel_diff_max}")
            return {"passed": passed, "baseline_exists": True, "metrics": {"pixel_diff": diff}, "message": msg}
    except (ImportError, ModuleNotFoundError) as e:
        return {"passed": True, "baseline_exists": True, "metrics": {}, "message": f"Comparison skipped ({e})"}
    except Exception as e:
        return {"passed": False, "baseline_exists": True, "metrics": {}, "message": f"Comparison error: {e}"}


# =============================================================================
# Orchestration
# =============================================================================


@dataclass
class Result:
    """Result for a single pipeline run."""

    name: str
    status: str  # PASS | FAIL | ERROR | TIMEOUT | SKIP
    elapsed_seconds: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    num_frames: int | None = None
    resolution: str | None = None
    num_steps: int | None = None
    error_category: str = ""
    error_message: str = ""
    note: str = ""
    regression_metrics: dict = field(default_factory=dict)
    script: str = ""  # Script path for filename generation
    reproduce_command: str = ""  # Command to reproduce this test
    log_path: str | None = None  # Path to log file


def _parse_runner_output(stdout: str) -> dict:
    """Extract JSON result from subprocess stdout."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith(_RESULT_MARKER):
            try:
                return json.loads(line[len(_RESULT_MARKER) :])
            except json.JSONDecodeError:
                pass
    return {}


def _run_subprocess_with_tee(
    cmd: list[str],
    timeout: float,
    cwd: str,
    env: dict,
    verbose: bool,
    log_path: str,
) -> tuple[int, str, str]:
    """Run subprocess with optional real-time output tee to terminal and log file.

    Returns:
        (return_code, stdout_text, stderr_text)
    """
    if verbose:
        # Real-time tee mode: output to terminal AND capture for log file
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def read_stream(stream, output_list, stream_name):
            for line in iter(stream.readline, ""):
                if stream_name == "stdout":
                    print(line, end="", flush=True)
                else:
                    print(f"[stderr] {line}", end="", flush=True)
                output_list.append(line)
            stream.close()

        t_out = threading.Thread(target=read_stream, args=(proc.stdout, stdout_lines, "stdout"))
        t_err = threading.Thread(target=read_stream, args=(proc.stderr, stderr_lines, "stderr"))
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        t_out.join(timeout=5)
        t_err.join(timeout=5)

        stdout_text = "".join(stdout_lines)
        stderr_text = "".join(stderr_lines)
        return_code = proc.returncode
    else:
        # Capture mode: silent execution, output saved to log file
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=env,
            )
            return_code = proc.returncode
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
        except subprocess.TimeoutExpired as e:
            return_code = -1
            stdout_text = e.stdout or ""
            stderr_text = e.stderr or ""

    return return_code, stdout_text, stderr_text


def run_pipeline(
    pipeline_key: str,
    ppl_cfg: PipelineConfig,
    output_root: str,
    output_dir: str | None = None,
    config_path: str | None = None,
    update_baseline: bool = False,
    verbose: bool = False,
    gpu_ids: list[int] | None = None,
) -> Result:
    """Run a single pipeline in a subprocess and evaluate results.

    Args:
        pipeline_key: Pipeline name from config.
        ppl_cfg: Pipeline configuration.
        output_root: Root directory for baseline comparison.
        output_dir: Directory for saving outputs (defaults to output_root).
        config_path: Optional config YAML path.
        update_baseline: Whether to update baseline outputs.
        verbose: Whether to show verbose output.
        gpu_ids: Optional explicit GPU IDs to use. If None, auto-assigns GPUs.

    Returns:
        Result object with status, metrics, and details.
    """
    gpu_count = ppl_cfg.gpu_count
    output_dir = output_dir or output_root

    # Generate reproduce command
    reproduce_cmd = _build_reproduce_cmd(pipeline_key, config_path)

    # Assign GPUs
    env = os.environ.copy()
    existing_pypath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_PROJECT_ROOT}{os.pathsep}{existing_pypath}" if existing_pypath else _PROJECT_ROOT

    # Set CUDA_VISIBLE_DEVICES: use provided gpu_ids or auto-assign
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(id) for id in gpu_ids)
    elif not env.get("CUDA_VISIBLE_DEVICES"):
        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available > 0:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(min(gpu_count, available)))

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-single",
        pipeline_key,
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    cmd.extend(["--output-dir", output_dir])

    start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare log file path (needed for tee mode)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_filename = _generate_log_filename(ppl_cfg.script, gpu_count, timestamp)
    log_path = os.path.join(logs_dir, log_filename)

    return_code, stdout_text, stderr_text = _run_subprocess_with_tee(
        cmd=cmd,
        timeout=ppl_cfg.timeout_seconds,
        cwd=_PROJECT_ROOT,
        env=env,
        verbose=verbose,
        log_path=log_path,
    )
    elapsed = time.time() - start

    # Handle timeout
    if return_code == -1:
        return Result(
            name=pipeline_key,
            status="TIMEOUT",
            elapsed_seconds=round(elapsed, 2),
            script=ppl_cfg.script,
            reproduce_command=reproduce_cmd,
            log_path=log_path,
            note=f"Timeout after {ppl_cfg.timeout_seconds}s",
        )

    # Parse result
    data = _parse_runner_output(stdout_text)
    status = data.get("status", "ERROR")
    error_msg = data.get("error", "")
    error_cat = data.get("error_category", "")
    peak_mem = data.get("peak_gpu_memory_mb", 0.0)
    script = data.get("script", ppl_cfg.script)

    # OOM detection fallback
    if not error_cat and stderr_text and _is_oom(stderr_text):
        error_cat = "OOM_ERROR"

    if not data and return_code != 0:
        error_msg = f"Process exited with code {return_code}"

    # Save log with new naming convention
    branch, short_hash = _get_git_info()
    sys_info = _get_system_info()
    gpu_ids_str = ",".join(str(id) for id in (gpu_ids or []))

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Pipeline: {pipeline_key} ===\n")
        f.write(f"=== Branch: {branch} ===\n")
        f.write(f"=== Commit: {short_hash} ===\n")
        f.write(f"=== Timestamp: {timestamp} ===\n")
        f.write(f"=== GPUs: {gpu_ids_str} ===\n")
        f.write(f"=== Command: {reproduce_cmd} ===\n")
        f.write(f"=== System Info ===\n")
        f.write(f"  Python: {sys_info['python_version']}\n")
        f.write(f"  PyTorch: {sys_info['pytorch_version']}\n")
        f.write(f"  CUDA: {sys_info['cuda_version']}\n")
        f.write(f"  cuDNN: {sys_info['cudnn_version']}\n")
        f.write(f"  System: {sys_info['system']}\n")
        f.write(f"  Arch: {sys_info['architecture']}\n")
        f.write(f"  GPU Count: {sys_info['gpu_count']}\n")
        f.write(f"  GPU Devices: {sys_info['gpu_devices']}\n\n")
        f.write("=== STDOUT ===\n")
        f.write(stdout_text or "(empty)\n")
        f.write("\n=== STDERR ===\n")
        f.write(stderr_text or "(empty)\n")

    result = Result(
        name=pipeline_key,
        status=status,
        elapsed_seconds=round(elapsed, 2),
        peak_gpu_memory_mb=round(peak_mem, 2),
        num_frames=data.get("num_frames"),
        resolution=data.get("resolution"),
        num_steps=data.get("num_steps"),
        error_category=error_cat,
        error_message=error_msg,
        script=script,
        reproduce_command=reproduce_cmd,
        log_path=log_path,
    )

    # Compare against baseline
    if status == "PASS":
        output_path = data.get("output_path")
        cmp = compare_against_baseline(
            output_root,
            ppl_cfg.script,
            gpu_count,
            output_path,
            ppl_cfg.output_type,
            ppl_cfg.psnr_min,
            ppl_cfg.ssim_min,
            ppl_cfg.pixel_diff_max,
        )
        result.regression_metrics = cmp.get("metrics", {})
        result.note = cmp["message"]
        if cmp["baseline_exists"] and not cmp["passed"]:
            result.status = "FAIL"

        if update_baseline and output_path and os.path.exists(output_path):
            _update_baseline(output_root, output_path)
            result.note += " [baseline updated]"

    # Performance/memory threshold checks
    if result.status == "PASS" and ppl_cfg.max_elapsed_seconds and result.elapsed_seconds > ppl_cfg.max_elapsed_seconds:
        result.status = "FAIL"
        result.note += f" [PERF: {result.elapsed_seconds:.1f}s > {ppl_cfg.max_elapsed_seconds:.1f}s]"

    if result.status == "PASS" and ppl_cfg.max_gpu_memory_mb and result.peak_gpu_memory_mb > ppl_cfg.max_gpu_memory_mb:
        result.status = "FAIL"
        result.note += f" [MEM: {result.peak_gpu_memory_mb:.0f}MB > {ppl_cfg.max_gpu_memory_mb:.0f}MB]"

    if result.status != "PASS" and not result.note:
        result.note = error_msg[:60] if error_msg else ""
        if error_cat:
            result.note = f"[{error_cat}] {result.note}"

    return result


# =============================================================================
# Reporting
# =============================================================================


def _fmt(val: object, fmt: str) -> str:
    if val is None:
        return "-"
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


def print_results_table(results: list[Result]) -> None:
    """Print a summary table to console."""
    print()
    print("=" * 130)
    print("REGRESSION TEST RESULTS")
    print("=" * 130)
    header = (
        f"  {'Pipeline':<45} {'Status':<8} {'Steps':>6} {'Frames':>7} {'Resolution':>12}"
        f" {'Time(s)':>8} {'VRAM(GB)':>9}  {'PSNR':>7} {'SSIM':>7}  Note"
    )
    print(header)
    print("-" * 130)

    pass_count = fail_count = skip_count = 0
    for r in results:
        if r.status == "PASS":
            pass_count += 1
        elif r.status == "SKIP":
            skip_count += 1
        else:
            fail_count += 1

        steps = _fmt(r.num_steps, "d")
        frames = _fmt(r.num_frames, "d")
        res = r.resolution or "-"
        t = f"{r.elapsed_seconds:.1f}" if r.elapsed_seconds > 0 else "-"
        vram = f"{r.peak_gpu_memory_mb / 1024:.2f}" if r.peak_gpu_memory_mb > 0 else "-"
        psnr = _fmt(r.regression_metrics.get("psnr"), ".1f")
        ssim = _fmt(r.regression_metrics.get("ssim"), ".4f")
        note = (r.note or "")[:50]
        print(
            f"  {r.name:<45} {r.status:<8} {steps:>6} {frames:>7} {res:>12} {t:>8} {vram:>9}  {psnr:>7} {ssim:>7}  {note}"
        )

    print("-" * 130)
    print(f"Total: {len(results)} | PASS: {pass_count} | FAIL: {fail_count} | SKIP: {skip_count}")
    print("=" * 130)


def _generate_analysis_hint(error_category: str, error_message: str) -> str:
    """Generate analysis hint based on error category."""
    hints = {
        "MODEL_LOAD_ERROR": "模型加载失败，检查 model_root 路径和模型文件完整性",
        "INFERENCE_ERROR": "推理过程出错，查看 log_path 中的 traceback 定位具体模块",
        "OUTPUT_ERROR": "输出保存失败，检查输出目录权限和磁盘空间",
        "OOM_ERROR": "GPU内存不足，考虑减少 batch_size 或使用 lower resolution",
        "TIMEOUT": "执行超时，考虑增加 timeout_seconds 或检查是否有死循环",
        "FAIL": "输出质量不达标，检查 PSNR/SSIM 值和 baseline 对比",
    }
    return hints.get(error_category, "请查看 log_path 中的详细日志进行排查")


def _get_last_n_lines_from_log(log_path: str | None, n: int = 50) -> str:
    """Get last N lines from log file."""
    if log_path is None or not os.path.exists(log_path):
        return ""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:
        return ""


def save_report_json(output_root: str, results: list[Result], existing_results: dict | None = None) -> str:
    """Save results to JSON report with enhanced failure details.

    Args:
        output_root: Output directory for the report.
        results: List of Result objects from current run.
        existing_results: Optional dict of existing results to merge (for resume mode).
    """
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "example_report.json")

    env_info = {}
    try:
        env_info = {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "N/A",
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception:
        pass

    # Merge with existing results if provided (for resume mode)
    all_results_dict = {}
    if existing_results:
        all_results_dict.update(existing_results)
    for r in results:
        all_results_dict[r.name] = asdict(r)

    # Build counts from all results
    counts = {"pass": 0, "fail": 0, "skip": 0, "error": 0, "timeout": 0}
    for data in all_results_dict.values():
        key = data.get("status", "error").lower()
        counts[key] = counts.get(key, 0) + 1

    # Build failed details list
    failed_details = []
    failed_commands = []
    for data in all_results_dict.values():
        if data.get("status") not in ("PASS", "SKIP"):
            detail = {
                "name": data.get("name", ""),
                "status": data.get("status", ""),
                "error_category": data.get("error_category", ""),
                "error_message": (data.get("error_message", "") or "")[:500],
                "reproduce_command": data.get("reproduce_command", ""),
                "log_path": data.get("log_path"),
                "last_50_lines_log": _get_last_n_lines_from_log(data.get("log_path"), 50),
                "analysis_hint": _generate_analysis_hint(data.get("error_category", ""), data.get("error_message", "")),
            }
            failed_details.append(detail)
            if data.get("reproduce_command"):
                failed_commands.append(data["reproduce_command"])

    report = {
        "generated_at": datetime.now().isoformat(),
        "environment": env_info,
        "summary": {"total": len(all_results_dict), **counts},
        "results": all_results_dict,
        "failed_details": failed_details,
        "reproduce_all_failed": " && ".join(failed_commands) if failed_commands else "",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleFuser Pipeline Regression Test")
    parser.add_argument("--list", action="store_true", dest="list_pipelines", help="List configured pipelines")
    parser.add_argument("--pipeline", type=str, help="Run a specific pipeline by name")
    parser.add_argument("--all", action="store_true", help="Run all enabled pipelines")
    parser.add_argument("--update-baseline", action="store_true", help="Update baseline outputs after successful runs")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Resume from existing output directory. Skips pipelines with PASS status in the report.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Available GPU devices for parallel execution (e.g., '0,1,2,3'). "
        "When specified, enables parallel scheduling across these GPUs. "
        "Default: use all visible GPUs (sequential execution).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show real-time log output from each pipeline")

    # Internal: subprocess self-invocation
    parser.add_argument("--run-single", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=str, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Subprocess mode: run a single pipeline and exit
    if args.run_single:
        _run_single(args.run_single, args.config, args.output_dir)
        return

    # Load config
    cfg = load_config(args.config)

    if args.list_pipelines:
        print(f"\nConfigured pipelines ({len(cfg.pipelines)}):\n")
        print(f"  {'Name':<40} {'Enabled':<8} {'GPUs':>4} {'Type':<6} {'Script'}")
        print("  " + "-" * 110)
        for name, ppl in cfg.pipelines.items():
            enabled = "ON" if ppl.enabled else "OFF"
            print(f"  {name:<40} {enabled:<8} {ppl.gpu_count:>4} {ppl.output_type:<6} {ppl.script}")
        return

    # Determine which pipelines to run
    if args.pipeline:
        if args.pipeline not in cfg.pipelines:
            print(f"Error: pipeline '{args.pipeline}' not found in config. Use --list to see available pipelines.")
            sys.exit(1)
        to_run = {args.pipeline: cfg.pipelines[args.pipeline]}
    elif args.all:
        to_run = {k: v for k, v in cfg.pipelines.items() if v.enabled}
    else:
        parser.print_help()
        return

    if not to_run:
        print("No pipelines to run.")
        return

    output_root = _resolve_output_root(cfg.output_root)
    if args.resume:
        output_dir = os.path.abspath(args.resume)
        if not os.path.isdir(output_dir):
            print(f"Error: Resume directory does not exist: {output_dir}")
            sys.exit(1)
    else:
        output_dir = _get_date_dir(output_root)
        os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Baseline directory: {os.path.join(output_root, 'baseline')}")

    # Load completed pipelines for resume (check output file existence)
    completed_pipelines = set()
    if args.resume:
        completed_pipelines = _get_completed_pipelines(output_dir, to_run)
        if completed_pipelines:
            print(f"Resuming from: {output_dir}")
            print(f"Skipping {len(completed_pipelines)} pipelines with existing output: {sorted(completed_pipelines)}")
            print("-" * 60)

            # Filter out completed pipelines
            to_run = {k: v for k, v in to_run.items() if k not in completed_pipelines}

            if not to_run:
                print("All pipelines already have output files. Nothing to run.")
                return

    # Determine GPU pool
    gpu_ids: list[int] = []
    if args.gpus:
        # Parse user-specified GPU IDs
        try:
            gpu_ids = [int(id.strip()) for id in args.gpus.split(",")]
        except ValueError:
            print(f"Error: Invalid GPU IDs format '{args.gpus}'. Use comma-separated integers (e.g., '0,1,2,3')")
            sys.exit(1)
    else:
        # Use all visible GPUs (sequential execution fallback)
        if torch.cuda.is_available():
            gpu_ids = list(range(torch.cuda.device_count()))

    # Check if any GPU is available
    if not gpu_ids:
        print("Warning: No GPUs available. Running in CPU mode (very slow).")
        gpu_ids = []  # Empty pool, pipelines will run without CUDA_VISIBLE_DEVICES

    # Run pipelines
    results: list[Result] = []
    total = len(to_run)
    run_start = time.time()

    if args.gpus and len(gpu_ids) > 1:
        # Parallel scheduling mode
        print(f"\nParallel execution with GPUs: {gpu_ids}")
        print(f"Pipelines to run: {total}")
        print("-" * 60)

        gpu_pool = GPUPool(gpu_ids)
        scheduler = PipelineScheduler(
            gpu_pool=gpu_pool,
            pipelines=to_run,
            output_root=output_root,
            output_dir=output_dir,
            config_path=args.config,
            update_baseline=args.update_baseline,
            verbose=args.verbose,
        )

        # Run scheduler loop
        last_status_print = 0.0
        status_print_interval = 10.0  # Print status every 10s when idle

        while scheduler.has_pending() or scheduler.has_running():
            # Try to schedule new pipelines
            scheduled = scheduler.schedule_next()
            if scheduled:
                name, allocated_gpus = scheduled
                print(f"  Started: {name} on GPUs {allocated_gpus}")
                last_status_print = time.time()  # Reset timer on state change

            # Collect finished jobs
            finished = scheduler.collect_finished()
            for r in finished:
                print(f"  Finished: {r.name} -> {r.status} ({r.elapsed_seconds:.1f}s) {r.note[:40]}")
                last_status_print = time.time()  # Reset timer on state change

            # Show running status only periodically or on change
            elapsed_total = time.time() - run_start
            if elapsed_total - last_status_print >= status_print_interval:
                running_status = scheduler.get_running_status()
                if running_status:
                    running_str = ", ".join(f"{name}(GPU{gpus})" for name, gpus, elapsed in running_status)
                    elapsed_str = ", ".join(f"{elapsed:.0f}s" for _, _, elapsed in running_status)
                    print(f"  [{elapsed_total:.0f}s] Running: {running_str} ({elapsed_str})")
                last_status_print = elapsed_total

            time.sleep(0.5)

        results = scheduler.results
    else:
        # Sequential execution mode (original behavior)
        total_gpus = len(gpu_ids) if gpu_ids else 0
        print(f"\nSequential execution ({total_gpus} GPU available)")
        print("-" * 60)

        # Filter pipelines that require more GPUs than available
        filtered_to_run: dict[str, PipelineConfig] = {}
        for name, ppl_cfg in to_run.items():
            if ppl_cfg.gpu_count > total_gpus:
                print(f"  [SKIP] {name}: requires {ppl_cfg.gpu_count} GPUs, but only {total_gpus} available")
            else:
                filtered_to_run[name] = ppl_cfg

        total = len(filtered_to_run)
        for idx, (name, ppl_cfg) in enumerate(filtered_to_run.items(), 1):
            elapsed_total = time.time() - run_start
            print(f"\n[{idx}/{total}] ({elapsed_total:.0f}s) Running: {name} ...")

            result = run_pipeline(
                name,
                ppl_cfg,
                output_root,
                output_dir,
                args.config,
                args.update_baseline,
                verbose=args.verbose,
            )
            results.append(result)

            if not args.verbose:
                print(f"  -> {result.status} ({result.elapsed_seconds:.1f}s) {result.note}")

    # Report
    # Load existing results for resume mode
    existing_results = {}
    if args.resume:
        existing_results = _load_existing_results(output_dir)

    print_results_table(results)
    report_path = save_report_json(output_dir, results, existing_results)
    print(f"\nJSON report: {report_path}")

    fail_count = sum(1 for r in results if r.status in ("FAIL", "ERROR", "TIMEOUT"))
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
