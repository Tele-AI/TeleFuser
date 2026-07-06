"""WebRTC transport for TeleFuser stream server.

The default TeleFuser installation includes aiortc for WebRTC support.
"""

from __future__ import annotations

try:
    from .chunk_router import ChunkRouter
    from .session_manager import WebRTCSessionManager
    from .track import (
        AudioGeneratorTrack,
        FrameGeneratorTrack,
        IncomingAudioRelay,
        IncomingVideoRelay,
    )
except ImportError:
    AudioGeneratorTrack = None  # type: ignore[assignment,misc]
    ChunkRouter = None  # type: ignore[assignment,misc]
    FrameGeneratorTrack = None  # type: ignore[assignment,misc]
    IncomingAudioRelay = None  # type: ignore[assignment,misc]
    IncomingVideoRelay = None  # type: ignore[assignment,misc]
    WebRTCSessionManager = None  # type: ignore[assignment,misc]

__all__ = [
    "AudioGeneratorTrack",
    "ChunkRouter",
    "FrameGeneratorTrack",
    "IncomingAudioRelay",
    "IncomingVideoRelay",
    "WebRTCSessionManager",
]
