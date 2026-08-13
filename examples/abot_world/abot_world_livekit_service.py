"""ABot-World LiveKit pipeline file for ``telefuser stream-serve``."""

from __future__ import annotations

import importlib.util
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


def get_service(gpu_num: int = 1, gpu_ids: list[str] | None = None) -> ABotWorldLiveKitService:
    """Load one ABot replica on the single GPU assigned to this worker."""
    assigned = list(gpu_ids) if gpu_ids else ["0"]
    if gpu_num != 1 or len(assigned) != 1:
        raise ValueError("Each ABot worker owns exactly one GPU; use multiple workers for multiple GPUs")
    try:
        device_id = int(assigned[0])
    except ValueError as exc:
        raise ValueError(f"ABot worker GPU id must be numeric, got {assigned[0]!r}") from exc
    pipeline = get_pipeline(device_id=device_id, pipeline_class=ABotWorldInteractivePipeline)
    return ABotWorldLiveKitService(
        pipeline,
        default_fps=8,
        default_session_config={
            "image_path": str(_DEFAULT_IMAGE_PATH),
            "prompt": DEFAULT_PROMPT,
            "fps": 8,
            "control_latent_frames": 2,
            "seed": 42,
        },
    )
