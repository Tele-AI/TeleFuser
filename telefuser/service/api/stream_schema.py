"""Request / response schemas for stream endpoints."""

from __future__ import annotations

import json
import time
import uuid

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Wire messages  (used over DataChannel and internally by WebRTC)
# ---------------------------------------------------------------------------


class StreamChunkMessage(BaseModel):
    """Single chunk pushed to the client."""

    type: str = "chunk"
    session_id: str = ""
    index: int | None = None
    data: dict | None = None
    error: str | None = None
    timestamp: float = Field(default_factory=time.time)


class StreamDoneMessage(BaseModel):
    type: str = "done"
    session_id: str = ""
    total_chunks: int = 0
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# WebRTC signaling
# ---------------------------------------------------------------------------


class WebRTCOfferRequest(BaseModel):
    """Body for POST /v1/stream/webrtc/offer."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sdp: str = Field(description="SDP offer from browser")
    type: str = Field(default="offer", description="SDP type")
    task: str = Field(description="Task type, e.g. t2v, i2v")
    prompt: str | None = None
    fps: int | None = Field(default=None, description="Target video FPS")
    config: dict = Field(default_factory=dict, description="Session configuration (bidirectional mode)")

    model_config = {"extra": "allow"}


class WebRTCOfferResponse(BaseModel):
    session_id: str
    sdp: str = Field(description="SDP answer from server")
    type: str = Field(default="answer", description="SDP type")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def serialisable_chunk(chunk: dict) -> dict:
    """Strip non-JSON-serialisable values (e.g. tensors) from a chunk dict."""
    out: dict = {}
    for k, v in chunk.items():
        if isinstance(v, (str, int, float, bool, type(None), list, dict)):
            out[k] = v
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out
