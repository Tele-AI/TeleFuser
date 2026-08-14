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
_SCHEDULER_MODE_ENV = "TELEFUSER_ABOT_SCHEDULER_MODE"
_MAX_BATCH_SIZE_ENV = "TELEFUSER_ABOT_MAX_BATCH_SIZE"
_BATCHING_WINDOW_MS_ENV = "TELEFUSER_ABOT_BATCHING_WINDOW_MS"


def _serving_schedule_from_environment() -> tuple[str, int, float]:
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

    return scheduler_mode, max_batch_size, batching_window_ms


def get_service(gpu_num: int = 1, gpu_ids: list[str] | None = None) -> ABotWorldLiveKitService:
    """Load one ABot replica on the single GPU assigned to this worker."""
    assigned = list(gpu_ids) if gpu_ids else ["0"]
    if gpu_num != 1 or len(assigned) != 1:
        raise ValueError("Each ABot worker owns exactly one GPU; use multiple workers for multiple GPUs")
    try:
        device_id = int(assigned[0])
    except ValueError as exc:
        raise ValueError(f"ABot worker GPU id must be numeric, got {assigned[0]!r}") from exc
    scheduler_mode, max_batch_size, batching_window_ms = _serving_schedule_from_environment()
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
    )
