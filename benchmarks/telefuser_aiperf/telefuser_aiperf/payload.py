"""TeleFuser LiveKit request payload construction."""

from __future__ import annotations

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
