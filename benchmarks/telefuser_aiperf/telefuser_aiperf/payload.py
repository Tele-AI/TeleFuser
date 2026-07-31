"""TeleFuser LiveKit request payload construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiperf.streaming.models import StreamSessionPlan


def build_telefuser_livekit_session_body(
    plan: StreamSessionPlan,
) -> dict[str, Any]:
    """Build a TeleFuser LiveKit session request from a normalized plan."""

    options = dict(plan.request_extra)
    reserved = {"identity", "role", "prompt", "image_path", "config", "session_id"}
    conflicts = sorted(reserved.intersection(options))
    if conflicts:
        raise ValueError("LiveKit request_extra cannot override protocol fields: " + ", ".join(conflicts))
    config = {"task": plan.task, "fps": plan.fps, **options}
    body: dict[str, Any] = {
        "identity": f"aiperf-{plan.planned_session_id}",
        "role": "controller",
        "prompt": plan.prompt,
        "config": config,
    }
    if plan.image_path:
        body["image_path"] = plan.image_path
    return body


def build_sglang_realtime_init(plan: StreamSessionPlan) -> dict[str, Any]:
    """Build the MessagePack init request for SGLang realtime video."""

    options = dict(plan.request_extra)
    reserved = {"type", "prompt", "first_frame", "fps"}
    conflicts = sorted(reserved.intersection(options))
    if conflicts:
        raise ValueError("SGLang request_extra cannot override protocol fields: " + ", ".join(conflicts))
    if not plan.image_path:
        raise ValueError("SGLang LingBot-World v2 requires image_path")
    image_path = Path(plan.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"SGLang first frame does not exist: {image_path}")
    return {
        "type": "init",
        "prompt": plan.prompt,
        "first_frame": image_path.read_bytes(),
        "fps": plan.fps,
        **options,
    }
