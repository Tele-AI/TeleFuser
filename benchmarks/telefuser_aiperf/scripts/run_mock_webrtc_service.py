#!/usr/bin/env python3
"""Run a TeleFuser-style mock WebRTC stream server without model inference."""

from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import time
from dataclasses import dataclass
from typing import Any

import av
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from fastapi import FastAPI, HTTPException, Request

_RTP_CLOCK_RATE = 90_000


@dataclass
class MockWebRTCConfig:
    fps: int = 16
    frame_width: int = 320
    frame_height: int = 180
    frame_num: int = 320
    ack_controls: bool = True


class SyntheticVideoTrack(VideoStreamTrack):
    """Synthetic video track paced at a fixed FPS."""

    def __init__(self, *, fps: int, width: int, height: int, frame_num: int) -> None:
        super().__init__()
        self._fps = max(1, int(fps))
        self._width = int(width)
        self._height = int(height)
        self._frame_num = int(frame_num)
        self._frame_interval_s = 1.0 / self._fps
        self._pts_per_frame = _RTP_CLOCK_RATE // self._fps
        self._last_frame_at: float | None = None
        self._frame_index = 0

    async def recv(self) -> av.VideoFrame:
        if self._frame_index >= self._frame_num:
            raise MediaStreamError("Synthetic stream ended")
        if self._last_frame_at is not None:
            delay = self._frame_interval_s - (time.perf_counter() - self._last_frame_at)
            if delay > 0:
                await asyncio.sleep(delay)

        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        image[:, :, 0] = (24 + self._frame_index) % 255
        image[: max(8, self._height // 8), :, 1] = 180
        image[:, : max(8, self._width // 12), 2] = 220
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = self._frame_index * self._pts_per_frame
        frame.time_base = fractions.Fraction(1, _RTP_CLOCK_RATE)
        self._frame_index += 1
        self._last_frame_at = time.perf_counter()
        return frame


def _get_option(payload: dict[str, Any], key: str, default: Any) -> Any:
    nested = payload.get("config")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return payload.get(key, default)


def create_app(default_config: MockWebRTCConfig) -> FastAPI:
    app = FastAPI(title="TeleFuser Mock WebRTC Stream")
    sessions: dict[str, RTCPeerConnection] = {}

    @app.get("/v1/service/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "stream_ready": True,
            "model_loaded": False,
            "active_sessions": len(sessions),
        }

    @app.post("/v1/stream/webrtc/offer")
    async def offer(request: Request) -> dict[str, Any]:
        body = await request.json()
        session_id = str(body.get("session_id") or f"mock-{int(time.time() * 1000)}")
        if session_id in sessions:
            raise HTTPException(status_code=409, detail=f"Session already exists: {session_id}")

        fps = int(_get_option(body, "fps", default_config.fps))
        width = int(_get_option(body, "frame_width", default_config.frame_width))
        height = int(_get_option(body, "frame_height", default_config.frame_height))
        frame_num = int(_get_option(body, "frame_num", default_config.frame_num))
        ack_controls = bool(_get_option(body, "ack_controls", default_config.ack_controls))

        pc = RTCPeerConnection()
        sessions[session_id] = pc
        pc.addTrack(SyntheticVideoTrack(fps=fps, width=width, height=height, frame_num=frame_num))

        @pc.on("datachannel")
        def _on_datachannel(channel) -> None:
            @channel.on("open")
            def _on_open() -> None:
                try:
                    channel.send(
                        json.dumps(
                            {
                                "type": "chunk",
                                "session_id": session_id,
                                "data": {"stage": "session_started", "model_loaded": False},
                                "timestamp": time.time(),
                            }
                        )
                    )
                except Exception:
                    pass

            @channel.on("message")
            def _on_message(message) -> None:
                if not ack_controls:
                    return
                try:
                    payload = json.loads(message) if isinstance(message, str) else {}
                except json.JSONDecodeError:
                    payload = {}
                try:
                    channel.send(
                        json.dumps(
                            {
                                "type": "chunk",
                                "session_id": session_id,
                                "data": {
                                    "stage": "control_state",
                                    "control": payload,
                                    "model_loaded": False,
                                },
                                "timestamp": time.time(),
                            }
                        )
                    )
                except Exception:
                    pass

        @pc.on("connectionstatechange")
        async def _on_connectionstatechange() -> None:
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await _close_session(session_id)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type=body.get("type", "offer")))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception:
            await _close_session(session_id)
            raise

        return {
            "session_id": session_id,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def _close_session(session_id: str) -> bool:
        pc = sessions.pop(session_id, None)
        if pc is None:
            return False
        await pc.close()
        return True

    @app.delete("/v1/stream/webrtc/{session_id}")
    async def delete(session_id: str) -> dict[str, Any]:
        closed = await _close_session(session_id)
        if not closed:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return {"session_id": session_id, "status": "closed"}

    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--frame-width", type=int, default=320)
    parser.add_argument("--frame-height", type=int, default=180)
    parser.add_argument("--frame-num", type=int, default=320)
    parser.add_argument("--no-ack-controls", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = MockWebRTCConfig(
        fps=args.fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        frame_num=args.frame_num,
        ack_controls=not args.no_ack_controls,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
