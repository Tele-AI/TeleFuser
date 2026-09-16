"""Generic versioned WebSocket transport for typed VLA sessions."""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from telefuser.service.core.pipeline_pool import PipelinePool
from telefuser.service.core.replica_worker import ReplicaDeadError, ReplicaVLAError
from telefuser.vla import VLASessionManager
from telefuser.vla.contracts import ActionSpaceSpec, ObservationSpaceSpec, RobotActionChunk
from telefuser.vla.runtime import ActionChunkStateMachine, ChunkStatus
from telefuser.vla.runtime.chunk_state import ChunkTicket
from telefuser.vla.serialization import (
    DEFAULT_MAX_TENSOR_BYTES,
    VLA_SESSION_PROTOCOL_VERSION,
    VLA_WIRE_ENCODING,
    action_space_from_wire,
    action_space_to_wire,
    dumps_wire_message,
    loads_wire_message,
    observation_space_from_wire,
    observation_space_to_wire,
    robot_action_chunk_from_wire,
    robot_action_chunk_to_wire,
    robot_observation_from_wire,
)

DEFAULT_MAX_VLA_MESSAGE_BYTES = 64 * 1024 * 1024


class VLAErrorCode(str, Enum):
    """Stable error categories returned by the VLA session protocol."""

    INVALID_MESSAGE = "invalid_message"
    UNSUPPORTED_VERSION = "unsupported_version"
    SESSION_EXISTS = "session_exists"
    UNKNOWN_SESSION = "unknown_session"
    UNKNOWN_COMPONENT = "unknown_component"
    ACTION_SPACE_MISMATCH = "action_space_mismatch"
    OBSERVATION_SPACE_MISMATCH = "observation_space_mismatch"
    OUT_OF_ORDER = "out_of_order"
    SUPERSEDED = "superseded"
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


class _VLABackend(Protocol):
    def metadata(self) -> Mapping[str, Any]: ...

    async def open(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    async def predict(self, session_id: str, payload: Mapping[str, Any]) -> RobotActionChunk: ...

    async def reset(self, session_id: str, episode_id: str | None) -> None: ...

    async def close(self, session_id: str) -> None: ...


class _LocalVLABackend:
    def __init__(self, sessions: VLASessionManager, max_tensor_bytes: int) -> None:
        self.sessions = sessions
        self.max_tensor_bytes = max_tensor_bytes

    def metadata(self) -> Mapping[str, Any]:
        return {
            "model_ids": list(self.sessions.registry.model_ids()),
            "embodiment_ids": list(self.sessions.registry.embodiment_ids()),
        }

    async def open(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _required_string(payload, "session_id")
        if session_id in self.sessions.session_ids():
            raise VLAProtocolError(VLAErrorCode.SESSION_EXISTS, f"VLA session is already open: {session_id!r}")
        model_id = _required_string(payload, "model_id")
        embodiment_id = _required_string(payload, "embodiment_id")
        try:
            policy = self.sessions.registry.get_policy(model_id)
            embodiment = self.sessions.registry.get_embodiment(embodiment_id)
        except KeyError as error:
            raise VLAProtocolError(VLAErrorCode.UNKNOWN_COMPONENT, str(error)) from error
        _validate_expected_action_space(payload, embodiment.robot_action_space)
        _validate_expected_observation_space(payload, embodiment.observation_space)
        await asyncio.to_thread(
            self.sessions.open,
            session_id,
            model_id=model_id,
            embodiment_id=embodiment_id,
            episode_id=_required_string(payload, "episode_id"),
            execute_horizon=_optional_positive_int(payload, "execute_horizon"),
            max_observation_age_ns=_optional_positive_int(payload, "max_observation_age_ns"),
        )
        capabilities = policy.capabilities()
        return {
            "session_id": session_id,
            "model_id": model_id,
            "embodiment_id": embodiment_id,
            "model_action_space": action_space_to_wire(capabilities.output_action_space),
            "robot_action_space": action_space_to_wire(embodiment.robot_action_space),
            "robot_observation_space": observation_space_to_wire(embodiment.observation_space),
            "max_horizon": capabilities.max_horizon,
            "stateful": capabilities.stateful,
            "supports_seed": capabilities.supports_seed,
        }

    async def predict(self, session_id: str, payload: Mapping[str, Any]) -> RobotActionChunk:
        observation_payload = payload.get("observation")
        if not isinstance(observation_payload, Mapping):
            raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "PREDICT requires an observation object")
        observation = robot_observation_from_wire(observation_payload, max_tensor_bytes=self.max_tensor_bytes)
        timestamp_ns = _required_nonnegative_int(payload, "observation_timestamp_ns")
        if timestamp_ns != observation.state.timestamp_ns:
            raise VLAProtocolError(
                VLAErrorCode.INVALID_MESSAGE,
                "PREDICT observation_timestamp_ns must match observation.state.timestamp_ns",
            )
        return await asyncio.to_thread(
            self.sessions.get(session_id).predict,
            observation,
            _required_string(payload, "instruction"),
            _required_nonnegative_int(payload, "sequence_id"),
            seed=_optional_integer(payload, "seed"),
            now_ns=_optional_nonnegative_int(payload, "observation_clock_now_ns"),
        )

    async def reset(self, session_id: str, episode_id: str | None) -> None:
        await asyncio.to_thread(self.sessions.reset, session_id, episode_id)

    async def close(self, session_id: str) -> None:
        self.sessions.close(session_id)


class _PipelinePoolVLABackend:
    def __init__(self, pool: PipelinePool, max_tensor_bytes: int) -> None:
        self.pool = pool
        self.max_tensor_bytes = max_tensor_bytes

    def metadata(self) -> Mapping[str, Any]:
        return self.pool.vla_metadata()

    async def open(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _required_string(payload, "session_id")
        try:
            await self.pool.open_session(session_id)
            return await self.pool.run_vla_operation(session_id, "OPEN", dict(payload))
        except Exception:
            with contextlib.suppress(KeyError):
                await self.pool.close_session(session_id)
            raise

    async def predict(self, session_id: str, payload: Mapping[str, Any]) -> RobotActionChunk:
        replica_payload = dict(payload)
        replica_payload["_max_tensor_bytes"] = self.max_tensor_bytes
        response = await self.pool.run_vla_operation(session_id, "PREDICT", replica_payload)
        chunk_payload = response.get("chunk")
        if not isinstance(chunk_payload, Mapping):
            raise RuntimeError("VLA replica returned an invalid action chunk")
        return robot_action_chunk_from_wire(chunk_payload, max_tensor_bytes=self.max_tensor_bytes)

    async def reset(self, session_id: str, episode_id: str | None) -> None:
        payload: dict[str, Any] = {"session_id": session_id}
        if episode_id is not None:
            payload["episode_id"] = episode_id
        await self.pool.run_vla_operation(session_id, "RESET", payload)

    async def close(self, session_id: str) -> None:
        try:
            await self.pool.run_vla_operation(session_id, "CLOSE", {"session_id": session_id})
        finally:
            with contextlib.suppress(KeyError):
                await self.pool.close_session(session_id)


@dataclass
class _PredictionJob:
    message: Mapping[str, Any]
    ticket: ChunkTicket
    response: asyncio.Future[dict[str, Any]]


class _SessionRuntime:
    """Run one prediction at a time while retaining only the latest waiting request."""

    def __init__(
        self,
        backend: _VLABackend,
        session_id: str,
        state: ActionChunkStateMachine,
        *,
        stateful: bool,
    ) -> None:
        self.backend = backend
        self.session_id = session_id
        self.state = state
        self.stateful = stateful
        self._waiting: _PredictionJob | None = None
        self._runner: asyncio.Task[None] | None = None
        self._accepting = True
        self._recovering = False
        self._barrier_active = False

    def submit(self, message: Mapping[str, Any]) -> asyncio.Future[dict[str, Any]]:
        """Admit a request immediately without waiting for the replica."""
        loop = asyncio.get_running_loop()
        response: asyncio.Future[dict[str, Any]] = loop.create_future()
        if not self._accepting or self._recovering:
            response.set_exception(
                VLAProtocolError(VLAErrorCode.SESSION_UNAVAILABLE, "session is not accepting predictions")
            )
            return response

        timestamp_ns = _required_nonnegative_int(message, "observation_timestamp_ns")
        timeout_s = _optional_timeout_s(message)
        ticket = self.state.submit(
            _required_nonnegative_int(message, "sequence_id"),
            timestamp_ns,
            request_ttl_ms=None if timeout_s is None else timeout_s * 1000.0,
            observation_clock_now_ns=_optional_nonnegative_int(message, "observation_clock_now_ns"),
        )
        admission = self.state.status(ticket)
        if admission is ChunkStatus.REJECTED:
            response.set_exception(
                VLAProtocolError(VLAErrorCode.OUT_OF_ORDER, self.state.reason(ticket) or "action chunk was rejected")
            )
            return response
        if admission is ChunkStatus.EXPIRED:
            response.set_exception(
                VLAProtocolError(VLAErrorCode.EXPIRED, self.state.reason(ticket) or "observation expired")
            )
            return response

        job = _PredictionJob(message, ticket, response)
        if self._waiting is not None:
            self._fail(
                self._waiting,
                VLAProtocolError(VLAErrorCode.SUPERSEDED, "a newer observation superseded the waiting request"),
            )
        self._waiting = job
        if self._runner is None:
            self._runner = asyncio.create_task(self._run())
        return response

    async def reset(self, episode_id: str | None) -> None:
        """Apply a barrier, discard earlier results, and reset policy history."""
        self._accepting = False
        self._barrier_active = True
        self.state.reset(episode_id)
        if self._waiting is not None:
            self._fail(
                self._waiting,
                VLAProtocolError(VLAErrorCode.SESSION_UNAVAILABLE, "request was discarded by RESET"),
            )
            self._waiting = None
        await self._await_runner()
        await self.backend.reset(self.session_id, episode_id)
        self.state.reset(episode_id)
        self._barrier_active = False
        self._accepting = True

    async def close(self, *, disconnect: bool = False) -> None:
        """Stop admission, isolate late results, and close the backend session."""
        self._accepting = False
        self._barrier_active = True
        self.state.reset()
        if self._waiting is not None:
            self._fail(
                self._waiting,
                VLAProtocolError(VLAErrorCode.SESSION_UNAVAILABLE, "request was discarded by CLOSE"),
            )
            self._waiting = None
        await self._await_runner()
        await self.backend.close(self.session_id)
        if disconnect:
            self.state.disconnect()

    async def _run(self) -> None:
        try:
            while self._waiting is not None:
                job = self._waiting
                self._waiting = None
                if self.state.status(job.ticket) is not ChunkStatus.PENDING:
                    self._fail(
                        job,
                        VLAProtocolError(
                            VLAErrorCode.SUPERSEDED,
                            self.state.reason(job.ticket) or "action chunk was superseded",
                        ),
                    )
                    continue
                await self._predict(job)
        except Exception as error:
            self._accepting = False
            if self._waiting is not None:
                self._fail(self._waiting, error)
                self._waiting = None
        finally:
            self._runner = None

    async def _predict(self, job: _PredictionJob) -> None:
        self.state.mark_inference_started(job.ticket)
        predict_task = asyncio.create_task(self.backend.predict(self.session_id, job.message))
        timeout_s = _optional_timeout_s(job.message)
        try:
            if timeout_s is None:
                chunk = await predict_task
            else:
                chunk = await asyncio.wait_for(asyncio.shield(predict_task), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._recovering = True
            self._fail(job, VLAProtocolError(VLAErrorCode.TIMEOUT, "PREDICT exceeded request_ttl_ms"))
            try:
                try:
                    chunk = await predict_task
                except Exception as error:
                    self.state.reject(job.ticket, str(error) or "timed-out VLA inference failed")
                else:
                    self.state.complete(
                        job.ticket,
                        chunk,
                        observation_clock_now_ns=_optional_nonnegative_int(
                            job.message,
                            "observation_clock_now_ns",
                        ),
                    )
                if not self._barrier_active:
                    await self.backend.reset(self.session_id, None)
                    self.state.reset()
            finally:
                self._recovering = False
            return
        except Exception as error:
            self.state.reject(job.ticket, str(error) or "VLA inference failed")
            self._fail(job, error)
            return

        completion = self.state.complete(
            job.ticket,
            chunk,
            observation_clock_now_ns=_optional_nonnegative_int(job.message, "observation_clock_now_ns"),
        )
        if completion is ChunkStatus.READY:
            self._succeed(
                job,
                {
                    "type": "ACTION_CHUNK",
                    "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
                    "request_id": _request_id(job.message),
                    "session_id": self.session_id,
                    "chunk": robot_action_chunk_to_wire(chunk),
                },
            )
            return
        code = VLAErrorCode.EXPIRED if completion is ChunkStatus.EXPIRED else VLAErrorCode.SUPERSEDED
        self._fail(job, VLAProtocolError(code, self.state.reason(job.ticket) or "action chunk was discarded"))
        if self.stateful and not self._barrier_active:
            await self.backend.reset(self.session_id, None)

    async def _await_runner(self) -> None:
        if self._runner is not None:
            await self._runner

    @staticmethod
    def _succeed(job: _PredictionJob, response: dict[str, Any]) -> None:
        if not job.response.done():
            job.response.set_result(response)

    @staticmethod
    def _fail(job: _PredictionJob, error: Exception) -> None:
        if not job.response.done():
            job.response.set_exception(error)


def create_vla_session_app(
    sessions: VLASessionManager,
    *,
    max_message_bytes: int = DEFAULT_MAX_VLA_MESSAGE_BYTES,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> FastAPI:
    """Create an additive WebSocket app for an already-loaded VLA registry."""
    return _create_vla_session_app(
        _LocalVLABackend(sessions, max_tensor_bytes),
        max_message_bytes=max_message_bytes,
        max_tensor_bytes=max_tensor_bytes,
    )


def create_pipeline_pool_vla_session_app(
    pool: PipelinePool,
    *,
    max_message_bytes: int = DEFAULT_MAX_VLA_MESSAGE_BYTES,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
    close_pool_on_shutdown: bool = True,
) -> FastAPI:
    """Create a VLA WebSocket app backed by session-affine pipeline replicas."""
    app = _create_vla_session_app(
        _PipelinePoolVLABackend(pool, max_tensor_bytes),
        max_message_bytes=max_message_bytes,
        max_tensor_bytes=max_tensor_bytes,
    )
    if close_pool_on_shutdown:

        @app.on_event("shutdown")
        async def close_pool() -> None:
            await pool.aclose()

    return app


def _create_vla_session_app(
    backend: _VLABackend,
    *,
    max_message_bytes: int,
    max_tensor_bytes: int,
) -> FastAPI:
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
        runtimes: dict[str, _SessionRuntime] = {}
        delivery_tasks: set[asyncio.Task[None]] = set()
        send_lock = asyncio.Lock()
        metadata = backend.metadata()
        await _locked_send(
            websocket,
            send_lock,
            {
                "type": "HELLO",
                "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
                "encoding": VLA_WIRE_ENCODING,
                "operations": ["OPEN", "PREDICT", "RESET", "CLOSE"],
                "capabilities": {
                    "observation_contract": True,
                    "concurrent_predict": True,
                    "prediction_queue": "latest-wins",
                    "control_barriers": True,
                },
                "model_ids": list(metadata.get("model_ids", [])),
                "embodiment_ids": list(metadata.get("embodiment_ids", [])),
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
                    operation = message.get("type")
                    if isinstance(operation, str) and operation.upper() == "PREDICT":
                        session_id = _required_string(message, "session_id")
                        if session_id not in owned_sessions:
                            raise VLAProtocolError(
                                VLAErrorCode.UNKNOWN_SESSION,
                                f"session is not open on this connection: {session_id!r}",
                            )
                        response_future = runtimes[session_id].submit(message)
                        task = asyncio.create_task(
                            _deliver_prediction(websocket, send_lock, response_future, request_id=request_id)
                        )
                        delivery_tasks.add(task)
                        task.add_done_callback(delivery_tasks.discard)
                        continue
                    response = await _handle_control_message(
                        backend,
                        message,
                        owned_sessions=owned_sessions,
                        runtimes=runtimes,
                    )
                except WebSocketDisconnect:
                    break
                except Exception as error:
                    response = _error_response(error, request_id=request_id)
                await _locked_send(websocket, send_lock, response)
        finally:
            for runtime in tuple(runtimes.values()):
                with contextlib.suppress(Exception):
                    await runtime.close(disconnect=True)
            for task in tuple(delivery_tasks):
                task.cancel()
            for task in tuple(delivery_tasks):
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return app


async def _handle_control_message(
    backend: _VLABackend,
    message: Mapping[str, Any],
    *,
    owned_sessions: set[str],
    runtimes: dict[str, _SessionRuntime],
) -> dict[str, Any]:
    operation = message.get("type")
    if not isinstance(operation, str):
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "message type must be a string")
    operation = operation.upper()
    request_id = _request_id(message)
    if operation == "OPEN":
        return await _open_session(
            backend,
            message,
            owned_sessions=owned_sessions,
            runtimes=runtimes,
            request_id=request_id,
        )

    session_id = _required_string(message, "session_id")
    if session_id not in owned_sessions:
        raise VLAProtocolError(VLAErrorCode.UNKNOWN_SESSION, f"session is not open on this connection: {session_id!r}")
    if operation == "PREDICT":
        raise RuntimeError("PREDICT must be scheduled by the connection receiver")
    if operation == "RESET":
        episode_id = _optional_string(message, "episode_id")
        await runtimes[session_id].reset(episode_id)
        return _success_response("RESET", session_id, request_id)
    if operation == "CLOSE":
        await runtimes[session_id].close()
        owned_sessions.remove(session_id)
        runtimes.pop(session_id).state.disconnect()
        return _success_response("CLOSE", session_id, request_id)
    raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, f"unsupported VLA operation: {operation!r}")


async def _open_session(
    backend: _VLABackend,
    message: Mapping[str, Any],
    *,
    owned_sessions: set[str],
    runtimes: dict[str, _SessionRuntime],
    request_id: str | int | None,
) -> dict[str, Any]:
    version = message.get("protocol_version")
    if version != VLA_SESSION_PROTOCOL_VERSION:
        raise VLAProtocolError(VLAErrorCode.UNSUPPORTED_VERSION, f"unsupported protocol_version: {version!r}")
    session_id = _required_string(message, "session_id")
    if session_id in owned_sessions:
        raise VLAProtocolError(VLAErrorCode.SESSION_EXISTS, f"VLA session is already open: {session_id!r}")
    opened = await backend.open(message)
    owned_sessions.add(session_id)
    max_horizon = opened.get("max_horizon")
    if not isinstance(max_horizon, int) or isinstance(max_horizon, bool) or max_horizon < 1:
        await backend.close(session_id)
        owned_sessions.remove(session_id)
        raise RuntimeError("VLA backend returned an invalid max_horizon")
    state = ActionChunkStateMachine(
        _required_string(message, "episode_id"),
        execute_horizon=_optional_positive_int(message, "execute_horizon") or max_horizon,
        max_observation_age_ns=_optional_positive_int(message, "max_observation_age_ns"),
    )
    runtimes[session_id] = _SessionRuntime(
        backend,
        session_id,
        state,
        stateful=bool(opened.get("stateful", False)),
    )
    return {
        "type": "OPENED",
        "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
        "request_id": request_id,
        **opened,
    }


def _validate_expected_action_space(payload: Mapping[str, Any], robot_action_space: ActionSpaceSpec) -> None:
    expected_payload = payload.get("expected_robot_action_space")
    if expected_payload is None:
        return
    if not isinstance(expected_payload, Mapping):
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "expected_robot_action_space must be an object")
    action_space_from_wire(expected_payload).require_compatible(
        robot_action_space,
        context="client and embodiment robot action space",
    )


def _validate_expected_observation_space(
    payload: Mapping[str, Any],
    robot_observation_space: ObservationSpaceSpec,
) -> None:
    expected_payload = payload.get("expected_robot_observation_space")
    if expected_payload is None:
        return
    if not isinstance(expected_payload, Mapping):
        raise VLAProtocolError(VLAErrorCode.INVALID_MESSAGE, "expected_robot_observation_space must be an object")
    observation_space_from_wire(expected_payload).require_compatible(
        robot_observation_space,
        context="client and embodiment robot observation space",
    )


async def _send(websocket: WebSocket, payload: Mapping[str, Any]) -> None:
    await websocket.send_text(dumps_wire_message(payload))


async def _locked_send(
    websocket: WebSocket,
    lock: asyncio.Lock,
    payload: Mapping[str, Any],
) -> None:
    async with lock:
        await _send(websocket, payload)


async def _deliver_prediction(
    websocket: WebSocket,
    lock: asyncio.Lock,
    response: asyncio.Future[dict[str, Any]],
    *,
    request_id: str | int | None,
) -> None:
    try:
        payload = await response
    except Exception as error:
        payload = _error_response(error, request_id=request_id)
    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
        await _locked_send(websocket, lock, payload)


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
    if isinstance(error, ReplicaVLAError):
        if error.error_type == "KeyError":
            message = str(error).lower()
            if "model_id" in message or "embodiment_id" in message:
                return VLAErrorCode.UNKNOWN_COMPONENT
            return VLAErrorCode.UNKNOWN_SESSION
        if error.error_type in {"ValueError", "TypeError"}:
            return _value_error_code(str(error))
        return VLAErrorCode.INTERNAL_ERROR
    if isinstance(error, KeyError):
        return VLAErrorCode.UNKNOWN_SESSION
    if isinstance(error, ValueError):
        return _value_error_code(str(error))
    if isinstance(error, RuntimeError) and "replica" in str(error).lower():
        return VLAErrorCode.REPLICA_UNAVAILABLE
    return VLAErrorCode.INTERNAL_ERROR


def _value_error_code(message: str) -> VLAErrorCode:
    normalized = message.lower()
    if "observation space" in normalized or "observation-space" in normalized:
        return VLAErrorCode.OBSERVATION_SPACE_MISMATCH
    if "action space" in normalized or "action-space" in normalized:
        return VLAErrorCode.ACTION_SPACE_MISMATCH
    if "sequence_id" in normalized or "must increase" in normalized:
        return VLAErrorCode.OUT_OF_ORDER
    if "stale" in normalized or "expired" in normalized:
        return VLAErrorCode.EXPIRED
    if "already open" in normalized:
        return VLAErrorCode.SESSION_EXISTS
    return VLAErrorCode.INVALID_MESSAGE


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
