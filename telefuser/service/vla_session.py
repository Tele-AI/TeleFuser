"""Generic versioned WebSocket transport for typed VLA sessions."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from telefuser.service.core.replica_worker import ReplicaDeadError
from telefuser.vla import VLASessionManager
from telefuser.vla.serialization import (
    DEFAULT_MAX_TENSOR_BYTES,
    VLA_WIRE_ENCODING,
    action_space_from_wire,
    action_space_to_wire,
    dumps_wire_message,
    loads_wire_message,
    robot_action_chunk_to_wire,
    robot_observation_from_wire,
)

VLA_SESSION_PROTOCOL_VERSION = "1.0"
DEFAULT_MAX_VLA_MESSAGE_BYTES = 64 * 1024 * 1024


class VLAErrorCode(str, Enum):
    """Stable error categories returned by the VLA session protocol."""

    INVALID_MESSAGE = "invalid_message"
    UNSUPPORTED_VERSION = "unsupported_version"
    SESSION_EXISTS = "session_exists"
    UNKNOWN_SESSION = "unknown_session"
    UNKNOWN_COMPONENT = "unknown_component"
    ACTION_SPACE_MISMATCH = "action_space_mismatch"
    OUT_OF_ORDER = "out_of_order"
    EXPIRED = "expired"
    TIMEOUT = "timeout"
    REPLICA_UNAVAILABLE = "replica_unavailable"
    SESSION_UNAVAILABLE = "session_unavailable"
    INTERNAL_ERROR = "internal_error"


class VLAProtocolError(ValueError):
    """Protocol failure carrying a stable machine-readable error code."""

    def __init__(self, code: VLAErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def create_vla_session_app(
    sessions: VLASessionManager,
    *,
    max_message_bytes: int = DEFAULT_MAX_VLA_MESSAGE_BYTES,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> FastAPI:
    """Create an additive WebSocket app for an already-loaded VLA registry."""
    if not isinstance(max_message_bytes, int) or isinstance(max_message_bytes, bool) or max_message_bytes < 1:
        raise ValueError("max_message_bytes must be a positive integer")
    if not isinstance(max_tensor_bytes, int) or isinstance(max_tensor_bytes, bool) or max_tensor_bytes < 1:
        raise ValueError("max_tensor_bytes must be a positive integer")

    app = FastAPI(title="TeleFuser VLA Session")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/v1/vla/session")
    async def session_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        owned_sessions: set[str] = set()
        recovery_tasks: dict[str, asyncio.Task[None]] = {}
        await _send(
            websocket,
            {
                "type": "HELLO",
                "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
                "encoding": VLA_WIRE_ENCODING,
                "operations": ["OPEN", "PREDICT", "RESET", "CLOSE"],
                "model_ids": list(sessions.registry.model_ids()),
                "embodiment_ids": list(sessions.registry.embodiment_ids()),
                "max_message_bytes": max_message_bytes,
                "max_tensor_bytes": max_tensor_bytes,
            },
        )
        try:
            while True:
                request_id: str | int | None = None
                try:
                    message = loads_wire_message(
                        await websocket.receive_text(),
                        max_message_bytes=max_message_bytes,
                    )
                    request_id = _request_id(message)
                    response = await _handle_message(
                        sessions,
                        message,
                        owned_sessions=owned_sessions,
                        recovery_tasks=recovery_tasks,
                        max_tensor_bytes=max_tensor_bytes,
                    )
                except WebSocketDisconnect:
                    break
                except Exception as error:
                    response = _error_response(error, request_id=request_id)
                await _send(websocket, response)
        finally:
            for task in list(recovery_tasks.values()):
                with contextlib.suppress(Exception):
                    await task
            for session_id in tuple(owned_sessions):
                with contextlib.suppress(KeyError):
                    await asyncio.to_thread(sessions.close, session_id)

    return app


async def _handle_message(
    sessions: VLASessionManager,
    message: Mapping[str, Any],
    *,
    owned_sessions: set[str],
    recovery_tasks: dict[str, asyncio.Task[None]],
    max_tensor_bytes: int,
) -> dict[str, Any]:
    operation = message.get("type")
    if not isinstance(operation, str):
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "message type must be a string")
    operation = operation.upper()
    request_id = _request_id(message)
    if operation == "OPEN":
        return _open_session(sessions, message, owned_sessions=owned_sessions, request_id=request_id)

    session_id = _required_string(message, "session_id")
    if session_id not in owned_sessions:
        raise VLAProtocolError(VLAErrorCode.UNKNOWN_SESSION, f"session is not open on this connection: {session_id!r}")
    if operation == "PREDICT":
        recovery = recovery_tasks.get(session_id)
        if recovery is not None and not recovery.done():
            raise VLAProtocolError(VLAErrorCode.SESSION_UNAVAILABLE, "session is recovering from a timed-out request")
        observation_payload = message.get("observation")
        if not isinstance(observation_payload, Mapping):
            raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "PREDICT requires an observation object")
        observation = robot_observation_from_wire(observation_payload, max_tensor_bytes=max_tensor_bytes)
        timestamp_ns = _required_nonnegative_int(message, "observation_timestamp_ns")
        if timestamp_ns != observation.state.timestamp_ns:
            raise VLAProtocolError(
                VLAErrorCode.INVALID_MESSAGE,
                "PREDICT observation_timestamp_ns must match observation.state.timestamp_ns",
            )
        timeout_s = _optional_timeout_s(message)
        predict_task = asyncio.create_task(
            asyncio.to_thread(
                sessions.get(session_id).predict,
                observation,
                _required_string(message, "instruction"),
                _required_nonnegative_int(message, "sequence_id"),
                seed=_optional_integer(message, "seed"),
                now_ns=_optional_nonnegative_int(message, "observation_clock_now_ns"),
            )
        )
        try:
            if timeout_s is None:
                chunk = await predict_task
            else:
                chunk = await asyncio.wait_for(asyncio.shield(predict_task), timeout=timeout_s)
        except asyncio.TimeoutError as error:
            recovery_tasks[session_id] = asyncio.create_task(
                _recover_timed_out_session(sessions, session_id, predict_task, recovery_tasks)
            )
            raise VLAProtocolError(VLAErrorCode.TIMEOUT, "PREDICT exceeded request_ttl_ms") from error
        return {
            "type": "ACTION_CHUNK",
            "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
            "request_id": request_id,
            "session_id": session_id,
            "chunk": robot_action_chunk_to_wire(chunk),
        }
    if operation == "RESET":
        await _await_recovery(session_id, recovery_tasks)
        await asyncio.to_thread(sessions.reset, session_id, _optional_string(message, "episode_id"))
        return _success_response("RESET", session_id, request_id)
    if operation == "CLOSE":
        await _await_recovery(session_id, recovery_tasks)
        await asyncio.to_thread(sessions.close, session_id)
        owned_sessions.remove(session_id)
        return _success_response("CLOSE", session_id, request_id)
    raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, f"unsupported VLA operation: {operation!r}")


def _open_session(
    sessions: VLASessionManager,
    message: Mapping[str, Any],
    *,
    owned_sessions: set[str],
    request_id: str | int | None,
) -> dict[str, Any]:
    version = message.get("protocol_version")
    if version != VLA_SESSION_PROTOCOL_VERSION:
        raise VLAProtocolError(VLAErrorCode.UNSUPPORTED_VERSION, f"unsupported protocol_version: {version!r}")
    session_id = _required_string(message, "session_id")
    if session_id in sessions.session_ids():
        raise VLAProtocolError(VLAErrorCode.SESSION_EXISTS, f"VLA session is already open: {session_id!r}")
    model_id = _required_string(message, "model_id")
    embodiment_id = _required_string(message, "embodiment_id")
    try:
        policy = sessions.registry.get_policy(model_id)
        embodiment = sessions.registry.get_embodiment(embodiment_id)
    except KeyError as error:
        raise VLAProtocolError(VLAErrorCode.UNKNOWN_COMPONENT, str(error)) from error
    expected_payload = message.get("expected_robot_action_space")
    if expected_payload is not None:
        if not isinstance(expected_payload, Mapping):
            raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "expected_robot_action_space must be an object")
        action_space_from_wire(expected_payload).require_compatible(
            embodiment.robot_action_space,
            context="client and embodiment robot action space",
        )
    sessions.open(
        session_id,
        model_id=model_id,
        embodiment_id=embodiment_id,
        episode_id=_required_string(message, "episode_id"),
        execute_horizon=_optional_positive_int(message, "execute_horizon"),
        max_observation_age_ns=_optional_positive_int(message, "max_observation_age_ns"),
    )
    owned_sessions.add(session_id)
    capabilities = policy.capabilities()
    return {
        "type": "OPENED",
        "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
        "request_id": request_id,
        "session_id": session_id,
        "model_id": model_id,
        "embodiment_id": embodiment_id,
        "model_action_space": action_space_to_wire(capabilities.output_action_space),
        "robot_action_space": action_space_to_wire(embodiment.robot_action_space),
        "max_horizon": capabilities.max_horizon,
        "stateful": capabilities.stateful,
        "supports_seed": capabilities.supports_seed,
    }


async def _recover_timed_out_session(
    sessions: VLASessionManager,
    session_id: str,
    predict_task: asyncio.Task[Any],
    recovery_tasks: dict[str, asyncio.Task[None]],
) -> None:
    try:
        with contextlib.suppress(Exception):
            await predict_task
        with contextlib.suppress(KeyError):
            await asyncio.to_thread(sessions.reset, session_id)
    finally:
        recovery_tasks.pop(session_id, None)


async def _await_recovery(session_id: str, recovery_tasks: dict[str, asyncio.Task[None]]) -> None:
    recovery = recovery_tasks.get(session_id)
    if recovery is not None:
        await recovery


async def _send(websocket: WebSocket, payload: Mapping[str, Any]) -> None:
    await websocket.send_text(dumps_wire_message(payload))


def _error_response(error: Exception, *, request_id: str | int | None) -> dict[str, Any]:
    code = _error_code(error)
    message = str(error) if code is not VLAErrorCode.INTERNAL_ERROR else "internal VLA session error"
    return {
        "type": "ERROR",
        "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
        "request_id": request_id,
        "error": {"code": code.value, "message": message},
    }


def _error_code(error: Exception) -> VLAErrorCode:
    if isinstance(error, VLAProtocolError):
        return error.code
    if isinstance(error, ReplicaDeadError):
        return VLAErrorCode.REPLICA_UNAVAILABLE
    if isinstance(error, KeyError):
        return VLAErrorCode.UNKNOWN_SESSION
    if isinstance(error, ValueError):
        message = str(error).lower()
        if "action space" in message or "action-space" in message:
            return VLAErrorCode.ACTION_SPACE_MISMATCH
        if "sequence_id" in message or "must increase" in message:
            return VLAErrorCode.OUT_OF_ORDER
        if "stale" in message or "expired" in message:
            return VLAErrorCode.EXPIRED
        if "already open" in message:
            return VLAErrorCode.SESSION_EXISTS
        return VLAErrorCode.INVALID_MESSAGE
    if isinstance(error, RuntimeError) and "replica" in str(error).lower():
        return VLAErrorCode.REPLICA_UNAVAILABLE
    return VLAErrorCode.INTERNAL_ERROR


def _success_response(operation: str, session_id: str, request_id: str | int | None) -> dict[str, Any]:
    return {
        "type": f"{operation}_ACK",
        "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
        "request_id": request_id,
        "session_id": session_id,
    }


def _request_id(message: Mapping[str, Any]) -> str | int | None:
    value = message.get("request_id")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)) or isinstance(value, str) and not value:
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "request_id must be a non-empty string or integer")
    return value


def _required_string(message: Mapping[str, Any], field: str) -> str:
    value = message.get(field)
    if not isinstance(value, str) or not value:
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, f"{field} must be a non-empty string")
    return value


def _optional_string(message: Mapping[str, Any], field: str) -> str | None:
    value = message.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, f"{field} must be a non-empty string")
    return value


def _required_nonnegative_int(message: Mapping[str, Any], field: str) -> int:
    value = _optional_nonnegative_int(message, field)
    if value is None:
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, f"{field} is required")
    return value


def _optional_nonnegative_int(message: Mapping[str, Any], field: str) -> int | None:
    value = message.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, f"{field} must be a non-negative integer")
    return value


def _optional_positive_int(message: Mapping[str, Any], field: str) -> int | None:
    value = _optional_nonnegative_int(message, field)
    if value is not None and value < 1:
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, f"{field} must be a positive integer")
    return value


def _optional_integer(message: Mapping[str, Any], field: str) -> int | None:
    value = message.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, f"{field} must be an integer")
    return value


def _optional_timeout_s(message: Mapping[str, Any]) -> float | None:
    value = message.get("request_ttl_ms")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "request_ttl_ms must be a positive finite number")
    timeout_ms = float(value)
    if not math.isfinite(timeout_ms) or timeout_ms <= 0:
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "request_ttl_ms must be a positive finite number")
    return timeout_ms / 1000.0
