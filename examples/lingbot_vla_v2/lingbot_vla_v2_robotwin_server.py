"""Serve LingBot-VLA v2 through the upstream RoboTwin policy protocol."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from typing import Any, Mapping, Protocol

import click
import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from telefuser.pipelines.lingbot_vla_v2 import (
    ROBOTWIN_CAMERA_KEYS,
    LingBotVlaV2Observation,
    RobotWinProfile,
)
from telefuser.pipelines.lingbot_vla_v2.runtime import (
    LINGBOT_VLA_V2_QUANTIZATION_CHOICES,
    get_lingbot_vla_v2_pipeline,
)
from telefuser.utils.logging import logger


class _Pipeline(Protocol):
    config: Any

    def __call__(self, observation: LingBotVlaV2Observation) -> Any: ...

    def close(self) -> None: ...


def _pack_numpy(value: Any) -> Any:
    """Encode NumPy values using the upstream msgpack_numpy wire format."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot encode value of type {type(value)!r}")


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    """Decode NumPy values produced by the upstream msgpack_numpy helper."""
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack_message(payload: Mapping[str, Any]) -> bytes:
    """Pack one RoboTwin policy protocol message."""
    return msgpack.packb(dict(payload), default=_pack_numpy)


def unpack_message(payload: bytes) -> dict[str, Any]:
    """Unpack and validate one RoboTwin policy protocol message."""
    decoded = msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False)
    if not isinstance(decoded, dict):
        raise ValueError("RoboTwin request must be a MessagePack object")
    return decoded


class RobotWinPolicyAdapter:
    """Translate upstream RoboTwin observations to the TeleFuser VLA SDK."""

    def __init__(
        self,
        pipeline: _Pipeline,
        *,
        profile: RobotWinProfile | None = None,
        use_length: int = 50,
    ) -> None:
        if not 1 <= use_length <= 50:
            raise ValueError(f"use_length must be in [1, 50], got {use_length}")
        self.pipeline = pipeline
        self.profile = profile or pipeline.config.robot_profile
        self.use_length = use_length
        self._lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        """Describe the action contract sent when a client connects."""
        return {
            "robot_profile": self.profile.name,
            "action_horizon": self.use_length,
            "action_dim": self.profile.raw_state_dim,
            "policy_verified": False,
            "verification_status": "unverified_official_6b_base",
        }

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Return one absolute-position RoboTwin action chunk."""
        if request.get("reset", False):
            return self._reset(request)

        missing = [key for key in (*ROBOTWIN_CAMERA_KEYS, "observation.state", "task") if key not in request]
        if missing:
            raise ValueError(f"RoboTwin observation is missing fields: {missing}")

        observation = LingBotVlaV2Observation(
            task=request["task"],
            state=request["observation.state"],
            images={key: request[key] for key in ROBOTWIN_CAMERA_KEYS},
        )
        with self._lock:
            canonical_chunk = self.pipeline(observation)
            action_chunk = self.profile.structure_actions(
                canonical_chunk.canonical_normalized_actions,
            )
        if action_chunk.horizon < self.use_length:
            raise RuntimeError(
                f"policy returned horizon {action_chunk.horizon}, shorter than use_length={self.use_length}"
            )
        actions = action_chunk.raw_actions[: self.use_length].numpy()
        return {
            "action": actions,
            "policy_verified": canonical_chunk.policy_verified,
            "verification_status": canonical_chunk.verification_status,
        }

    def _reset(self, request: Mapping[str, Any]) -> dict[str, Any]:
        robot_name = request.get("robo_name", self.profile.name)
        if robot_name != self.profile.name:
            raise ValueError(f"unsupported robot profile: {robot_name!r}")
        if request.get("path_to_pi_model") not in (None, ""):
            raise ValueError("runtime checkpoint switching is not supported")
        return {"action": None}

    def close(self) -> None:
        """Release resources owned by the resident policy."""
        self.pipeline.close()


def create_robotwin_app(adapter: RobotWinPolicyAdapter) -> FastAPI:
    """Create a standalone app compatible with upstream WebsocketClientPolicy."""
    app = FastAPI(title="LingBot-VLA v2 RoboTwin Policy")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/")
    async def policy_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_bytes(pack_message(adapter.metadata))
        previous_total_ms: float | None = None
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload is None:
                    raise ValueError("RoboTwin requests must use binary MessagePack frames")

                round_started_at = time.monotonic()
                request = unpack_message(payload)
                inference_started_at = time.monotonic()
                response = await asyncio.to_thread(adapter.infer, request)
                inference_ms = (time.monotonic() - inference_started_at) * 1000.0
                response = dict(response)
                response["server_timing"] = {"infer_ms": inference_ms}
                if previous_total_ms is not None:
                    response["server_timing"]["prev_total_ms"] = previous_total_ms
                await websocket.send_bytes(pack_message(response))
                previous_total_ms = (time.monotonic() - round_started_at) * 1000.0
        except WebSocketDisconnect:
            return
        except Exception as error:
            logger.exception("LingBot-VLA v2 RoboTwin request failed")
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await websocket.send_text(f"{type(error).__name__}: {error}")
                await websocket.close(code=1011)

    return app


@click.command()
@click.option("--model-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--qwen3vl-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=9330, show_default=True, type=click.IntRange(1, 65535))
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--use-length", default=50, show_default=True, type=click.IntRange(1, 50))
@click.option("--cuda-graph", is_flag=True, help="Enable fixed-shape CUDA Graph inference")
@click.option(
    "--quantization",
    type=click.Choice(LINGBOT_VLA_V2_QUANTIZATION_CHOICES),
    default=None,
)
def main(
    model_root: str,
    qwen3vl_root: str,
    host: str,
    port: int,
    device: str,
    use_length: int,
    cuda_graph: bool,
    quantization: str | None,
) -> None:
    """Start one resident LingBot-VLA v2 policy for a RoboTwin client."""
    pipeline = get_lingbot_vla_v2_pipeline(
        model_root,
        qwen3vl_root,
        device=device,
        warmup=True,
        quantization=quantization,
        cuda_graph=cuda_graph,
    )
    adapter = RobotWinPolicyAdapter(pipeline, use_length=use_length)
    try:
        uvicorn.run(create_robotwin_app(adapter), host=host, port=port, workers=1)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
