"""Request / response schemas for stream endpoints."""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Stream status messages published through the transport adapter
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
