"""Serve LingBot-VLA v2 through the upstream RoboTwin policy protocol."""

from __future__ import annotations

import asyncio
import contextlib
import operator
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Mapping, Protocol

import click
import msgpack
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from telefuser.pipelines.lingbot_vla_v2 import (
    ROBOTWIN_ACTION_ORDER,
    ROBOTWIN_CAMERA_KEYS,
    LingBotVlaV2VLAPolicy,
    RobotWinProfile,
)
from telefuser.pipelines.lingbot_vla_v2.runtime import (
    LINGBOT_VLA_V2_QUANTIZATION_CHOICES,
    get_lingbot_vla_v2_pipeline,
)
from telefuser.utils.logging import logger
from telefuser.vla import RobotObservation, RobotState, VLARegistry, VLASessionManager
from telefuser.vla.runtime import ActionChunkScheduler

ROBOTWIN_PROTOCOL_VERSION = "1.0"
ROBOTWIN_ACTION_TYPE = "absolute_qpos"
ROBOTWIN_ACTION_DTYPE = "float32"
ROBOTWIN_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_TRACE_ID_FIELDS = ("request_id", "episode_id")
_VLA_SESSION_ID_FIELD = "_telefuser_vla_session_id"


class _Pipeline(Protocol):
    config: Any

    def __call__(self, observation: Any, seed: int | None = None) -> Any: ...

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


def _optional_seed(request: Mapping[str, Any]) -> int | None:
    value = request.get("seed")
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("seed must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("seed must be an integer") from error


def _trace_fields(request: Mapping[str, Any]) -> dict[str, str | int]:
    fields: dict[str, str | int] = {}
    for name in _TRACE_ID_FIELDS:
        value = request.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, str | int) or isinstance(value, str) and not value:
            raise ValueError(f"{name} must be a non-empty string or integer")
        fields[name] = value
    return fields


def _optional_nonnegative_int(request: Mapping[str, Any], field: str) -> int | None:
    value = request.get(field)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        parsed = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{field} must be a non-negative integer") from error
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed


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
        self._next_sequence: dict[str, int] = {}
        registry = VLARegistry()
        registry.register_policy("lingbot-vla-v2", LingBotVlaV2VLAPolicy(pipeline))
        registry.register_embodiment(self.profile.embodiment_id, self.profile)
        self._sessions = VLASessionManager(registry)

    @property
    def metadata(self) -> dict[str, Any]:
        """Describe the action contract sent when a client connects."""
        return {
            "protocol_version": ROBOTWIN_PROTOCOL_VERSION,
            "robot_profile": self.profile.name,
            "action_type": ROBOTWIN_ACTION_TYPE,
            "action_horizon": self.use_length,
            "action_dim": self.profile.raw_state_dim,
            "action_dtype": ROBOTWIN_ACTION_DTYPE,
            "action_order": list(ROBOTWIN_ACTION_ORDER),
            "max_request_bytes": ROBOTWIN_MAX_REQUEST_BYTES,
            "policy_verified": False,
            "verification_status": "unverified_official_6b_base",
        }

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Return one absolute-position RoboTwin action chunk."""
        trace_fields = _trace_fields(request)
        if request.get("reset", False):
            return {**self._reset(request), **trace_fields}

        missing = [key for key in (*ROBOTWIN_CAMERA_KEYS, "observation.state", "task") if key not in request]
        if missing:
            raise ValueError(f"RoboTwin observation is missing fields: {missing}")

        seed = _optional_seed(request)
        episode_id = str(trace_fields.get("episode_id", "default"))
        session_id_value = request.get(_VLA_SESSION_ID_FIELD, f"legacy:{episode_id}")
        if not isinstance(session_id_value, str) or not session_id_value:
            raise ValueError("internal VLA session ID must be a non-empty string")
        sequence_id = _optional_nonnegative_int(request, "sequence_id")
        observation_timestamp_ns = _optional_nonnegative_int(request, "observation_timestamp_ns")
        if observation_timestamp_ns is None:
            observation_timestamp_ns = time.time_ns()
        robot_observation = RobotObservation(
            state=RobotState(
                values=torch.tensor(request["observation.state"], dtype=torch.float32, device="cpu"),
                dimension_names=ROBOTWIN_ACTION_ORDER,
                timestamp_ns=observation_timestamp_ns,
            ),
            images={key: request[key] for key in ROBOTWIN_CAMERA_KEYS},
        )
        adapter_started_at = time.monotonic()
        with self._lock:
            lock_wait_ms = (time.monotonic() - adapter_started_at) * 1000.0
            if sequence_id is None:
                sequence_id = self._next_sequence.get(session_id_value, 0)
            self._next_sequence[session_id_value] = sequence_id + 1
            if session_id_value not in self._sessions.session_ids():
                self._sessions.open(
                    session_id_value,
                    model_id="lingbot-vla-v2",
                    embodiment_id=self.profile.embodiment_id,
                    episode_id=episode_id,
                    execute_horizon=self.use_length,
                )
            vla_timings: dict[str, float] = {}
            action_chunk = self._sessions.get(session_id_value).predict(
                robot_observation,
                request["task"],
                sequence_id,
                seed=seed,
                timings=vla_timings,
            )
            pipeline_ms = vla_timings["policy_ms"]
            action_mapping_ms = vla_timings["decode_actions_ms"] + vla_timings["prepare_actions_ms"]
        if action_chunk.valid_length < self.use_length:
            raise RuntimeError(
                f"policy returned horizon {action_chunk.valid_length}, shorter than use_length={self.use_length}"
            )
        actions = np.ascontiguousarray(action_chunk.actions[: self.use_length].numpy(), dtype=np.float32)
        expected_shape = (self.use_length, self.profile.raw_state_dim)
        if actions.shape != expected_shape:
            raise RuntimeError(f"mapped actions must have shape {expected_shape}, got {actions.shape}")
        if not np.isfinite(actions).all():
            raise RuntimeError("mapped actions must contain only finite values")
        response: dict[str, Any] = {
            "action": actions,
            "policy_verified": action_chunk.metadata["policy_verified"],
            "verification_status": action_chunk.metadata["verification_status"],
            "server_timing": {
                "lock_wait_ms": lock_wait_ms,
                "pipeline_ms": pipeline_ms,
                "action_mapping_ms": action_mapping_ms,
                "adapter_total_ms": (time.monotonic() - adapter_started_at) * 1000.0,
            },
            **trace_fields,
        }
        if seed is not None:
            response["seed"] = seed
        return response

    def _reset(self, request: Mapping[str, Any]) -> dict[str, Any]:
        robot_name = request.get("robo_name", self.profile.name)
        if robot_name != self.profile.name:
            raise ValueError(f"unsupported robot profile: {robot_name!r}")
        if request.get("path_to_pi_model") not in (None, ""):
            raise ValueError("runtime checkpoint switching is not supported")
        episode_id = str(request.get("episode_id", "default"))
        session_id = request.get(_VLA_SESSION_ID_FIELD, f"legacy:{episode_id}")
        if isinstance(session_id, str) and session_id in self._sessions.session_ids():
            self._sessions.reset(session_id, episode_id)
            self._next_sequence[session_id] = 0
        return {"action": None}

    def release_session(self, session_id: str) -> None:
        """Release semantic session state after a WebSocket disconnect."""
        with self._lock:
            if session_id in self._sessions.session_ids():
                self._sessions.close(session_id)
            self._next_sequence.pop(session_id, None)

    def close(self) -> None:
        """Release resources owned by the resident policy."""
        for session_id in self._sessions.session_ids():
            self._sessions.close(session_id)
        self._next_sequence.clear()
        self.pipeline.close()


def create_robotwin_app(
    adapter: RobotWinPolicyAdapter,
    *,
    max_pending_sessions: int = 32,
) -> FastAPI:
    """Create a standalone app compatible with upstream WebsocketClientPolicy."""
    scheduler = ActionChunkScheduler(adapter.infer, max_pending_sessions=max_pending_sessions)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await scheduler.start()
        try:
            yield
        finally:
            await scheduler.close()

    app = FastAPI(title="LingBot-VLA v2 RoboTwin Policy", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/")
    async def policy_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_bytes(pack_message({**adapter.metadata, **scheduler.metadata}))
        connection_key = str(id(websocket))
        session_keys: set[str] = set()
        delivery_tasks: set[asyncio.Task[None]] = set()
        send_lock = asyncio.Lock()
        previous_total_ms: float | None = None

        async def deliver(
            response_future: asyncio.Future[dict[str, Any]],
            *,
            decode_ms: float,
            round_started_at: float,
        ) -> None:
            nonlocal previous_total_ms
            try:
                response = dict(await response_future)
                server_timing = dict(response.get("server_timing", {}))
                server_timing["decode_ms"] = decode_ms
                response["server_timing"] = server_timing
                async with send_lock:
                    if previous_total_ms is not None:
                        server_timing["prev_total_ms"] = previous_total_ms
                    await websocket.send_bytes(pack_message(response))
                    previous_total_ms = (time.monotonic() - round_started_at) * 1000.0
            except asyncio.CancelledError:
                raise
            except WebSocketDisconnect:
                return
            except Exception as error:
                logger.exception("LingBot-VLA v2 RoboTwin request failed")
                async with send_lock:
                    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                        await websocket.send_text(f"{type(error).__name__}: {error}")
                        await websocket.close(code=1011)

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                payload = message.get("bytes")
                if payload is None:
                    raise ValueError("RoboTwin requests must use binary MessagePack frames")

                round_started_at = time.monotonic()
                decode_started_at = time.monotonic()
                request = unpack_message(payload)
                decode_ms = (time.monotonic() - decode_started_at) * 1000.0
                trace_fields = _trace_fields(request)
                episode_id = trace_fields.get("episode_id", "default")
                session_key = f"{connection_key}:{episode_id!r}"
                request[_VLA_SESSION_ID_FIELD] = session_key
                if request.get("reset", False):
                    for previous_session_key in session_keys - {session_key}:
                        scheduler.release_session(previous_session_key)
                        await asyncio.to_thread(adapter.release_session, previous_session_key)
                    scheduler.release_session(session_key)
                    session_keys.intersection_update({session_key})
                session_keys.add(session_key)
                response_future = scheduler.submit(request, session_key=session_key)
                task = asyncio.create_task(
                    deliver(
                        response_future,
                        decode_ms=decode_ms,
                        round_started_at=round_started_at,
                    )
                )
                delivery_tasks.add(task)
                task.add_done_callback(delivery_tasks.discard)
        except WebSocketDisconnect:
            return
        except Exception as error:
            logger.exception("LingBot-VLA v2 RoboTwin request failed")
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await websocket.send_text(f"{type(error).__name__}: {error}")
                await websocket.close(code=1011)
        finally:
            for session_key in session_keys:
                scheduler.release_session(session_key)
            for task in delivery_tasks:
                task.cancel()
            if delivery_tasks:
                await asyncio.gather(*delivery_tasks, return_exceptions=True)
            if session_keys:
                await asyncio.gather(
                    *(asyncio.to_thread(adapter.release_session, session_key) for session_key in session_keys)
                )

    return app


def _configure_h100_sdpa_backends(device: str) -> None:
    """Avoid unsupported cuDNN SDPA plans in the isolated H100 policy process."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        return
    if "H100" not in torch.cuda.get_device_name(resolved_device):
        return

    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    logger.info("Disabled cuDNN SDPA for the LingBot-VLA v2 H100 policy process")


@click.command()
@click.option("--model-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--qwen3vl-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=9330, show_default=True, type=click.IntRange(1, 65535))
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--use-length", default=50, show_default=True, type=click.IntRange(1, 50))
@click.option(
    "--max-pending-sessions",
    default=32,
    show_default=True,
    type=click.IntRange(1),
    help="Bound the number of sessions waiting behind the GPU worker",
)
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
    max_pending_sessions: int,
    cuda_graph: bool,
    quantization: str | None,
) -> None:
    """Start one resident LingBot-VLA v2 policy for a RoboTwin client."""
    _configure_h100_sdpa_backends(device)
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
        uvicorn.run(
            create_robotwin_app(adapter, max_pending_sessions=max_pending_sessions),
            host=host,
            port=port,
            workers=1,
            ws_max_size=ROBOTWIN_MAX_REQUEST_BYTES,
        )
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
