"""ABot-World LiveKit pipeline file for ``telefuser stream-serve``."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_IMAGE_PATH = (
    _PROJECT_ROOT.parent / "ABot-World" / "web_client" / "datasets" / "images" / "84b90ad568b693d2.png"
)
_LOADER_PATH = Path(__file__).with_name("_loader.py")
_LOADER_SPEC = importlib.util.spec_from_file_location("abot_world_example_loader", _LOADER_PATH)
if _LOADER_SPEC is None or _LOADER_SPEC.loader is None:
    raise RuntimeError(f"Could not load ABot example loader: {_LOADER_PATH}")
_LOADER = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(_LOADER)
DEFAULT_PROMPT = _LOADER.DEFAULT_PROMPT
get_pipeline = _LOADER.get_pipeline

_DEFAULT_SCHEDULER_MODE = "batched"
_DEFAULT_MAX_BATCH_SIZE = 2
_DEFAULT_BATCHING_WINDOW_MS = 2.0
_DEFAULT_MAX_DEADLINE_BATCH_WAIT_MS = 0.0
_SCHEDULER_MODE_ENV = "TELEFUSER_ABOT_SCHEDULER_MODE"
_MAX_BATCH_SIZE_ENV = "TELEFUSER_ABOT_MAX_BATCH_SIZE"
_BATCHING_WINDOW_MS_ENV = "TELEFUSER_ABOT_BATCHING_WINDOW_MS"
_MAX_DEADLINE_BATCH_WAIT_MS_ENV = "TELEFUSER_ABOT_MAX_DEADLINE_BATCH_WAIT_MS"
_DEFAULT_PUBLISHER_FRAME_CREDIT_ENABLED = False
_DEFAULT_PUBLISHER_FRAME_CREDIT_TARGET_SECONDS = 3.0
_DEFAULT_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES = 4
_DEFAULT_PUBLISHER_FRAME_CREDIT_GUARD_MS = 50.0
_PUBLISHER_FRAME_CREDIT_ENABLED_ENV = "TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_ENABLED"
_PUBLISHER_FRAME_CREDIT_TARGET_SECONDS_ENV = "TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_TARGET_SECONDS"
_PUBLISHER_FRAME_CREDIT_TARGET_FRAMES_ENV = "TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_TARGET_FRAMES"
_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES_ENV = "TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES"
_PUBLISHER_FRAME_CREDIT_GUARD_MS_ENV = "TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_GUARD_MS"
_DEFAULT_BATCH_COMPUTE_PROFILE = "none"
_BATCH_COMPUTE_PROFILE_ENV = "TELEFUSER_ABOT_BATCH_COMPUTE_PROFILE"
_DEFAULT_BATCH_COMPUTE_SAFETY_FACTOR = 1.10
_BATCH_COMPUTE_SAFETY_FACTOR_ENV = "TELEFUSER_ABOT_BATCH_COMPUTE_SAFETY_FACTOR"

# Raw P95 full-chunk wall times from
# results/experiments/abot_h100_batch_vs_serial_lf3_20260814/batched/results.json.
# They are deliberately opt-in: a timing profile from an H100 must never be
# assumed safe on another GPU, model shape, LF, or execution backend.
_BATCH_COMPUTE_PRIOR_PROFILES_SECONDS: dict[str, dict[int, float]] = {
    "none": {},
    "h100_lf3_eager_full_pipeline_v1": {
        2: 0.7404982000589371,
        3: 1.0691392589360476,
        4: 1.407263021916151,
    },
    "h100_lf3_cuda_graph_v1": {2: 0.6922691259533167, 3: 1.0319155678153038},
}


def _serving_schedule_from_environment() -> tuple[str, int, float, float]:
    """Return the worker-local ABot scheduling settings selected by the operator.

    The retained-session admission limit is intentionally configured by
    ``telefuser stream-serve --max-sessions-per-worker`` instead of here.
    Keeping these knobs separate makes it possible to run a fixed-capacity
    all-active trace with four retained sessions and one DiT batch of four on
    each GPU, without changing the conservative B=2 default profile.
    """
    scheduler_mode = os.getenv(_SCHEDULER_MODE_ENV, _DEFAULT_SCHEDULER_MODE).strip().lower()
    if scheduler_mode not in {"batched", "round_robin"}:
        raise ValueError(f"{_SCHEDULER_MODE_ENV} must be 'batched' or 'round_robin'")

    raw_max_batch_size = os.getenv(_MAX_BATCH_SIZE_ENV)
    if raw_max_batch_size is None:
        max_batch_size = _DEFAULT_MAX_BATCH_SIZE
    else:
        try:
            max_batch_size = int(raw_max_batch_size)
        except ValueError as exc:
            raise ValueError(f"{_MAX_BATCH_SIZE_ENV} must be a positive integer") from exc
        if max_batch_size < 1:
            raise ValueError(f"{_MAX_BATCH_SIZE_ENV} must be a positive integer")

    raw_batching_window_ms = os.getenv(_BATCHING_WINDOW_MS_ENV)
    if raw_batching_window_ms is None:
        batching_window_ms = _DEFAULT_BATCHING_WINDOW_MS
    else:
        try:
            batching_window_ms = float(raw_batching_window_ms)
        except ValueError as exc:
            raise ValueError(f"{_BATCHING_WINDOW_MS_ENV} must be a non-negative finite number") from exc
        if not math.isfinite(batching_window_ms) or batching_window_ms < 0:
            raise ValueError(f"{_BATCHING_WINDOW_MS_ENV} must be a non-negative finite number")

    raw_deadline_batch_wait_ms = os.getenv(_MAX_DEADLINE_BATCH_WAIT_MS_ENV)
    if raw_deadline_batch_wait_ms is None:
        deadline_batch_wait_ms = _DEFAULT_MAX_DEADLINE_BATCH_WAIT_MS
    else:
        try:
            deadline_batch_wait_ms = float(raw_deadline_batch_wait_ms)
        except ValueError as exc:
            raise ValueError(f"{_MAX_DEADLINE_BATCH_WAIT_MS_ENV} must be a non-negative finite number") from exc
        if not math.isfinite(deadline_batch_wait_ms) or deadline_batch_wait_ms < 0:
            raise ValueError(f"{_MAX_DEADLINE_BATCH_WAIT_MS_ENV} must be a non-negative finite number")

    return scheduler_mode, max_batch_size, batching_window_ms, deadline_batch_wait_ms


def _publisher_frame_credit_from_environment() -> tuple[bool, float, int | None, int, float]:
    raw_enabled = (
        os.getenv(_PUBLISHER_FRAME_CREDIT_ENABLED_ENV, str(_DEFAULT_PUBLISHER_FRAME_CREDIT_ENABLED)).strip().lower()
    )
    if raw_enabled in {"1", "true", "yes", "on"}:
        enabled = True
    elif raw_enabled in {"0", "false", "no", "off"}:
        enabled = False
    else:
        raise ValueError(f"{_PUBLISHER_FRAME_CREDIT_ENABLED_ENV} must be a boolean")

    def finite_float(name: str, default: float, *, positive: bool) -> float:
        raw = os.getenv(name, str(default))
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"{name} must be a {'positive' if positive else 'non-negative'} finite number")
        return value

    target_seconds = finite_float(
        _PUBLISHER_FRAME_CREDIT_TARGET_SECONDS_ENV, _DEFAULT_PUBLISHER_FRAME_CREDIT_TARGET_SECONDS, positive=True
    )
    raw_target_frames = os.getenv(_PUBLISHER_FRAME_CREDIT_TARGET_FRAMES_ENV)
    target_frames: int | None = None
    if raw_target_frames is not None:
        try:
            target_frames = int(raw_target_frames)
        except ValueError as exc:
            raise ValueError(f"{_PUBLISHER_FRAME_CREDIT_TARGET_FRAMES_ENV} must be a positive integer") from exc
        if target_frames <= 0:
            raise ValueError(f"{_PUBLISHER_FRAME_CREDIT_TARGET_FRAMES_ENV} must be a positive integer")
    raw_reserve_frames = os.getenv(
        _PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES_ENV, str(_DEFAULT_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES)
    )
    try:
        reserve_frames = int(raw_reserve_frames)
    except ValueError as exc:
        raise ValueError(f"{_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES_ENV} must be a non-negative integer") from exc
    if reserve_frames < 0:
        raise ValueError(f"{_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES_ENV} must be a non-negative integer")
    guard_ms = finite_float(
        _PUBLISHER_FRAME_CREDIT_GUARD_MS_ENV, _DEFAULT_PUBLISHER_FRAME_CREDIT_GUARD_MS, positive=False
    )
    return enabled, target_seconds, target_frames, reserve_frames, guard_ms


def _batch_compute_profile_from_environment() -> tuple[str, dict[int, float]]:
    """Return an explicit, hardware-specific cold-start batch timing profile."""
    name = os.getenv(_BATCH_COMPUTE_PROFILE_ENV, _DEFAULT_BATCH_COMPUTE_PROFILE).strip().lower()
    profile = _BATCH_COMPUTE_PRIOR_PROFILES_SECONDS.get(name)
    if profile is None:
        choices = ", ".join(sorted(_BATCH_COMPUTE_PRIOR_PROFILES_SECONDS))
        raise ValueError(f"{_BATCH_COMPUTE_PROFILE_ENV} must be one of: {choices}")
    return name, dict(profile)


def _batch_compute_safety_factor_from_environment() -> float:
    """Return the conservative multiplier for offline and online batch timings."""
    raw = os.getenv(_BATCH_COMPUTE_SAFETY_FACTOR_ENV, str(_DEFAULT_BATCH_COMPUTE_SAFETY_FACTOR))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{_BATCH_COMPUTE_SAFETY_FACTOR_ENV} must be a finite number greater than or equal to 1"
        ) from exc
    if not math.isfinite(value) or value < 1.0:
        raise ValueError(f"{_BATCH_COMPUTE_SAFETY_FACTOR_ENV} must be a finite number greater than or equal to 1")
    return value


def get_service(gpu_num: int = 1, gpu_ids: list[str] | None = None) -> ABotWorldLiveKitService:
    """Load one ABot replica on the single GPU assigned to this worker."""
    assigned = list(gpu_ids) if gpu_ids else ["0"]
    if gpu_num != 1 or len(assigned) != 1:
        raise ValueError("Each ABot worker owns exactly one GPU; use multiple workers for multiple GPUs")
    try:
        device_id = int(assigned[0])
    except ValueError as exc:
        raise ValueError(f"ABot worker GPU id must be numeric, got {assigned[0]!r}") from exc
    scheduler_mode, max_batch_size, batching_window_ms, deadline_batch_wait_ms = _serving_schedule_from_environment()
    (
        publisher_frame_credit_enabled,
        publisher_frame_credit_target_seconds,
        publisher_frame_credit_target_frames,
        publisher_frame_credit_reserve_frames,
        publisher_frame_credit_guard_ms,
    ) = _publisher_frame_credit_from_environment()
    batch_compute_profile_name, batch_compute_prior_seconds = _batch_compute_profile_from_environment()
    batch_compute_safety_factor = _batch_compute_safety_factor_from_environment()
    pipeline = get_pipeline(device_id=device_id, pipeline_class=ABotWorldInteractivePipeline)
    return ABotWorldLiveKitService(
        pipeline,
        # The default is the previously measured B=2 baseline. The all-active
        # 4-GPU/16-session trace explicitly selects B=4 through
        # TELEFUSER_ABOT_MAX_BATCH_SIZE=4 on every model worker.
        default_fps=12,
        default_session_config={
            "image_path": str(_DEFAULT_IMAGE_PATH),
            "prompt": DEFAULT_PROMPT,
            "fps": 12,
            "control_latent_frames": 3,
            "seed": 42,
        },
        scheduler_mode=scheduler_mode,
        max_batch_size=max_batch_size,
        batching_window_ms=batching_window_ms,
        max_deadline_batch_wait_ms=deadline_batch_wait_ms,
        batch_compute_safety_factor=batch_compute_safety_factor,
        publisher_frame_credit_enabled=publisher_frame_credit_enabled,
        batch_compute_profile_name=batch_compute_profile_name,
        batch_compute_prior_seconds=batch_compute_prior_seconds,
        publisher_frame_credit_target_seconds=publisher_frame_credit_target_seconds,
        publisher_frame_credit_target_frames=publisher_frame_credit_target_frames,
        publisher_frame_credit_reserve_frames=publisher_frame_credit_reserve_frames,
        publisher_frame_credit_guard_ms=publisher_frame_credit_guard_ms,
    )
