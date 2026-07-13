#!/usr/bin/env python3
"""Run a SGLang-style mock realtime stream server without model inference."""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any

try:
    import msgspec.msgpack
except ImportError as exc:  # pragma: no cover - runtime dependency check
    msgpack = None
    MSGPACK_IMPORT_ERROR = exc
else:  # pragma: no cover
    msgpack = msgspec.msgpack
    MSGPACK_IMPORT_ERROR = None


@dataclass
class MockServerConfig:
    fps: int = 16
    frames_per_chunk: int = 1
    max_chunks: int = 192
    frame_payload_bytes: int = 32768
    control_ack_delay_ms: float = 0.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--frames-per-chunk", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=192)
    parser.add_argument("--frame-payload-bytes", type=int, default=32768)
    parser.add_argument("--control-ack-delay-ms", type=float, default=0.0)
    return parser


def _int_option(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_option(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def create_app(config: MockServerConfig):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("fastapi is required for the mock stream server.") from exc

    globals()["WebSocket"] = WebSocket
    globals()["WebSocketDisconnect"] = WebSocketDisconnect
    app = FastAPI(title="SGLang Mock Stream Transport")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": False, "server": "mock_stream_transport"}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "mock-sglang-diffusion",
                    "object": "model",
                    "owned_by": "aiperf-mock",
                    "pipeline_class": "MockRealtimePipeline",
                    "model_loaded": False,
                }
            ],
        }

    @app.websocket("/v1/realtime_video/generate")
    async def realtime_video(websocket: WebSocket) -> None:
        if msgpack is None:
            raise RuntimeError("msgspec is required for the mock stream server.") from MSGPACK_IMPORT_ERROR

        await websocket.accept()
        send_lock = asyncio.Lock()
        pending_frame_event_ids: asyncio.Queue[int] = asyncio.Queue()
        disconnected = asyncio.Event()
        raw_init = await websocket.receive_bytes()
        init_payload = msgpack.decode(raw_init)
        if not isinstance(init_payload, dict):
            init_payload = {}

        fps = _int_option(init_payload, "fps", config.fps)
        frames_per_chunk = max(1, _int_option(init_payload, "num_frames", config.frames_per_chunk))
        max_chunks = max(1, _int_option(init_payload, "max_chunks", config.max_chunks))
        payload_bytes = max(0, _int_option(init_payload, "frame_payload_bytes", config.frame_payload_bytes))
        ack_delay_ms = max(0.0, _float_option(init_payload, "control_ack_delay_ms", config.control_ack_delay_ms))
        frame_payload = b"0" * payload_bytes

        async def _safe_send(payload: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_bytes(msgpack.encode(payload))

        async def _receive_controls() -> None:
            while not disconnected.is_set():
                try:
                    raw_message = await websocket.receive_bytes()
                except WebSocketDisconnect:
                    disconnected.set()
                    return
                except Exception:
                    disconnected.set()
                    return
                try:
                    message = msgpack.decode(raw_message)
                except Exception:
                    continue
                if not isinstance(message, dict):
                    continue
                event_id = message.get("event_id")
                if event_id is None:
                    continue
                event_id = int(event_id)
                await pending_frame_event_ids.put(event_id)
                if ack_delay_ms > 0:
                    await asyncio.sleep(ack_delay_ms / 1000.0)
                await _safe_send(
                    {
                        "type": "chunk_stats",
                        "event_id": event_id,
                        "stage": "control_state",
                        "model_loaded": False,
                        "timestamp": time.time(),
                    }
                )

        receiver_task = asyncio.create_task(_receive_controls())
        frame_interval_s = frames_per_chunk / max(fps, 1)
        try:
            for chunk_index in range(max_chunks):
                if disconnected.is_set():
                    break
                await asyncio.sleep(frame_interval_s)
                event_id = None
                try:
                    event_id = pending_frame_event_ids.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                await _safe_send(
                    {
                        "type": "frame_batch",
                        "chunk_index": chunk_index,
                        "event_id": event_id,
                        "num_frames": frames_per_chunk,
                        "is_final_frame_batch": True,
                        "content_type": "application/octet-stream",
                        "total_size": len(frame_payload),
                        "payload": frame_payload,
                        "model_loaded": False,
                        "timestamp": time.time(),
                    }
                )
                await _safe_send(
                    {
                        "type": "chunk_stats",
                        "session_id": "mock-sglang-session",
                        "chunk_index": chunk_index,
                        "event_id": event_id,
                        "request_prepare_ms": 1,
                        "scheduler_forward_ms": 1,
                        "pace_wait_ms": 0,
                        "header_write_ms": 1,
                        "raw_payload_build_ms": 1,
                        "raw_write_ms": 1,
                        "ws_write_ms": 2,
                        "chunk_total_ms": 5,
                        "num_batches": 1,
                        "num_frames": frames_per_chunk,
                        "raw_bytes": len(frame_payload),
                        "ws_payload_bytes": len(frame_payload),
                        "content_type": "application/octet-stream",
                        "memory_device": "cuda:0",
                        "peak_memory_mb": 256,
                        "model_loaded": False,
                        "timestamp": time.time(),
                    }
                )
        finally:
            disconnected.set()
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
            try:
                await websocket.close()
            except Exception:
                pass

    return app


def main() -> None:
    args = _build_parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("uvicorn is required for the mock stream server.") from exc
    config = MockServerConfig(
        fps=args.fps,
        frames_per_chunk=args.frames_per_chunk,
        max_chunks=args.max_chunks,
        frame_payload_bytes=args.frame_payload_bytes,
        control_ack_delay_ms=args.control_ack_delay_ms,
    )
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
