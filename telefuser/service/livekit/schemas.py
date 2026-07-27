"""Pydantic schemas for the LiveKit serving API."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LiveKitClientRole = Literal["controller", "viewer", "admin"]
LiveKitTokenRole = Literal["controller", "viewer", "admin", "worker"]
SessionStatus = Literal[
    "pending",
    "queued",
    "assigned",
    "joining_room",
    "starting_pipeline",
    "running",
    "draining",
    "closed",
    "failed",
    "expired",
]


def new_session_id() -> str:
    """Generate a public TeleFuser LiveKit session id."""
    return str(uuid.uuid4())


def utc_timestamp() -> float:
    """Return current Unix timestamp in seconds."""
    return time.time()


class SessionCreateRequest(BaseModel):
    """Request body for creating a LiveKit-backed stream session."""

    model_config = ConfigDict(extra="allow")

    identity: str = Field(min_length=1)
    role: Literal["controller"] = "controller"
    prompt: str | None = None
    image_path: str | None = None
    config: dict = Field(default_factory=dict)


class SessionTokenRequest(BaseModel):
    """Request body for minting an additional room token."""

    identity: str = Field(min_length=1)
    role: Literal["viewer"] = "viewer"


class SessionCreateResponse(BaseModel):
    """Response body for creating a LiveKit-backed stream session."""

    session_id: str
    room: str
    livekit_url: str
    token: str
    worker_id: str | None
    status: SessionStatus
    expires_at: float | None = None
    queue_position: int | None = None


class SessionTokenResponse(BaseModel):
    """Response body for minting an additional room token."""

    session_id: str
    room: str
    livekit_url: str
    token: str
    role: Literal["viewer"]


class SessionStatusResponse(BaseModel):
    """Public session status."""

    session_id: str
    room: str
    status: SessionStatus
    worker_id: str | None = None
    pipeline_session_id: str | None = None
    created_at: float
    updated_at: float
    expires_at: float | None = None
    participant_count: int = 0
    error: str | None = None


class SessionDeleteResponse(BaseModel):
    """Response body for deleting a session."""

    session_id: str
    status: SessionStatus


class LiveKitHealthResponse(BaseModel):
    """Liveness and scheduler health for the LiveKit service."""

    status: Literal["healthy", "degraded", "unhealthy"]
    livekit_connected: bool
    workers_total: int
    workers_idle: int
    workers_busy: int
    workers_failed: int
    queued_sessions: int
