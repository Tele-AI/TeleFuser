"""Bidirectional stream pipeline: replay video with arrow key overlay.

Loads a local video at startup.  Each session receives keyboard arrow-key
events via ``push_chunk()`` and yields video frames with a D-pad HUD
overlay via ``pull_chunks()``.

Usage:
    telefuser stream-serve examples/stream_server/stream_arrow_overlay.py -p 8088 --skip-validation
"""

from __future__ import annotations

import asyncio
import base64
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = str(Path(__file__).parent / "data" / "liveact_1.mp4")
OUTPUT_FPS = 24

_ARROW_KEYS = frozenset({"ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"})

# D-pad colors (BGR)
_COLOR_ACTIVE = (0, 220, 0)
_COLOR_INACTIVE = (80, 80, 80)
_COLOR_BG = (30, 30, 30)


@dataclass
class _SessionState:
    config: dict
    active: bool = True
    pressed_keys: set[str] = field(default_factory=set)


def _draw_dpad(frame: np.ndarray, pressed: set[str]) -> np.ndarray:
    """Draw a D-pad HUD overlay on the bottom-right corner of *frame*."""
    h, w = frame.shape[:2]
    size = min(h, w) // 6
    margin = 20
    cx = w - margin - size
    cy = h - margin - size
    half = size // 2
    tri = size // 3

    overlay = frame.copy()
    cv2.rectangle(overlay, (cx - size, cy - size), (cx + size, cy + size), _COLOR_BG, cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    def _tri(direction: str) -> tuple[list[list[int]], tuple[int, int, int]]:
        color = _COLOR_ACTIVE if direction in pressed else _COLOR_INACTIVE
        if direction == "ArrowUp":
            pts = [[cx, cy - half], [cx - tri, cy - tri // 2], [cx + tri, cy - tri // 2]]
        elif direction == "ArrowDown":
            pts = [[cx, cy + half], [cx - tri, cy + tri // 2], [cx + tri, cy + tri // 2]]
        elif direction == "ArrowLeft":
            pts = [[cx - half, cy], [cx - tri // 2, cy - tri], [cx - tri // 2, cy + tri]]
        else:
            pts = [[cx + half, cy], [cx + tri // 2, cy - tri], [cx + tri // 2, cy + tri]]
        return pts, color

    for key in _ARROW_KEYS:
        pts, color = _tri(key)
        cv2.fillPoly(frame, [np.array(pts, dtype=np.int32)], color)

    return frame


class ArrowOverlayService:
    """Bidirectional service: replays video with arrow-key overlay."""

    def __init__(self, video_path: str = VIDEO_PATH, fps: int = OUTPUT_FPS) -> None:
        self._video_path = video_path
        self._fps = fps
        self._frames: list[np.ndarray] = []
        self._sessions: dict[str, _SessionState] = {}

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        import av as _av

        container = _av.open(self._video_path)
        for vframe in container.decode(video=0):
            bgr = vframe.to_ndarray(format="bgr24")
            self._frames.append(bgr)
        container.close()
        print(f"[ArrowOverlayService] Loaded {len(self._frames)} frames from {self._video_path}")

    def stop(self) -> None:
        for state in self._sessions.values():
            state.active = False
        self._sessions.clear()
        self._frames.clear()

    # -- session management ----------------------------------------------------

    def create_session(self, config: dict) -> str:
        session_id = config.get("session_id") or str(uuid.uuid4())
        self._sessions[session_id] = _SessionState(config=config)
        print(f"[ArrowOverlayService] Session created: {session_id}")
        return session_id

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def close_session(self, session_id: str) -> None:
        state = self._sessions.pop(session_id, None)
        if state is not None:
            state.active = False
            print(f"[ArrowOverlayService] Session closed: {session_id}")

    # -- bidirectional I/O -----------------------------------------------------

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            return
        if chunk.get("type") != "control":
            return
        key = chunk.get("key", "")
        action = chunk.get("action", "")
        if key not in _ARROW_KEYS:
            return
        if action == "press":
            state.pressed_keys.add(key)
        elif action == "release":
            state.pressed_keys.discard(key)

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        state = self._sessions.get(session_id)
        if state is None:
            return

        src_len = len(self._frames)
        if src_len == 0:
            return

        fps = self._fps
        frame_interval = 1.0 / fps
        idx = 0
        chunk_idx = 0

        while state.active:
            t0 = time.monotonic()

            bgr = self._frames[idx % src_len].copy()
            bgr = _draw_dpad(bgr, state.pressed_keys)

            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            yield {
                "type": "chunk",
                "index": chunk_idx,
                "frames_b64": [b64],
                "fps": fps,
                "timestamp": time.time(),
            }

            idx += 1
            chunk_idx += 1

            elapsed = time.monotonic() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


def get_service() -> ArrowOverlayService:
    return ArrowOverlayService()
