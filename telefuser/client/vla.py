"""Async client for the generic TeleFuser VLA WebSocket protocol."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from typing import Any

from telefuser.vla.contracts import ActionSpaceSpec, ObservationSpaceSpec, RobotActionChunk, RobotObservation
from telefuser.vla.serialization import (
    VLA_SESSION_PROTOCOL_VERSION,
    VLA_WIRE_ENCODING,
    action_space_to_wire,
    dumps_wire_message,
    loads_wire_message,
    observation_space_to_wire,
    robot_action_chunk_from_wire,
    robot_observation_to_wire,
)

DEFAULT_MAX_VLA_MESSAGE_BYTES = 64 * 1024 * 1024


class VLAClientError(RuntimeError):
    """Error returned by the VLA session service."""

    def __init__(self, code: str, message: str, *, request_id: str | int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.request_id = request_id


class AsyncVLAClient:
    """Maintain one multiplexed connection to a VLA session service."""

    def __init__(
        self,
        url: str,
        *,
        max_message_bytes: int = DEFAULT_MAX_VLA_MESSAGE_BYTES,
        open_timeout_s: float = 10.0,
    ) -> None:
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if not isinstance(max_message_bytes, int) or isinstance(max_message_bytes, bool) or max_message_bytes < 1:
            raise ValueError("max_message_bytes must be a positive integer")
        if isinstance(open_timeout_s, bool) or not isinstance(open_timeout_s, (int, float)) or open_timeout_s <= 0:
            raise ValueError("open_timeout_s must be positive")
        self.url = url
        self.max_message_bytes = max_message_bytes
        self.open_timeout_s = float(open_timeout_s)
        self.hello: dict[str, Any] | None = None
        self._websocket: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._sessions: set[str] = set()
        self._send_lock = asyncio.Lock()
        self._request_sequence = 0

    async def connect(self) -> dict[str, Any]:
        """Connect and validate protocol capabilities."""
        if self._websocket is not None:
            raise RuntimeError("VLA client is already connected")
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            from websockets import connect

        websocket = await connect(
            self.url,
            max_size=self.max_message_bytes,
            open_timeout=self.open_timeout_s,
        )
        try:
            hello = loads_wire_message(await websocket.recv(), max_message_bytes=self.max_message_bytes)
            if hello.get("type") != "HELLO":
                raise VLAClientError("invalid_message", "VLA service did not send HELLO")
            if hello.get("protocol_version") != VLA_SESSION_PROTOCOL_VERSION:
                raise VLAClientError("unsupported_version", "VLA service protocol version is not supported")
            if hello.get("encoding") != VLA_WIRE_ENCODING:
                raise VLAClientError("unsupported_encoding", "VLA service wire encoding is not supported")
        except Exception:
            await websocket.close()
            raise
        self._websocket = websocket
        self.hello = hello
        self._reader = asyncio.create_task(self._read_responses())
        return dict(hello)

    async def open_session(
        self,
        session_id: str,
        *,
        model_id: str,
        embodiment_id: str,
        episode_id: str,
        execute_horizon: int | None = None,
        max_observation_age_ns: int | None = None,
        expected_robot_action_space: ActionSpaceSpec | None = None,
        expected_robot_observation_space: ObservationSpaceSpec | None = None,
    ) -> dict[str, Any]:
        """Open one session and optionally negotiate robot contracts."""
        payload: dict[str, Any] = {
            "type": "OPEN",
            "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
            "session_id": session_id,
            "model_id": model_id,
            "embodiment_id": embodiment_id,
            "episode_id": episode_id,
        }
        if execute_horizon is not None:
            payload["execute_horizon"] = execute_horizon
        if max_observation_age_ns is not None:
            payload["max_observation_age_ns"] = max_observation_age_ns
        if expected_robot_action_space is not None:
            payload["expected_robot_action_space"] = action_space_to_wire(expected_robot_action_space)
        if expected_robot_observation_space is not None:
            payload["expected_robot_observation_space"] = observation_space_to_wire(expected_robot_observation_space)
        response = await self._request(payload)
        self._sessions.add(session_id)
        return response

    async def predict(
        self,
        session_id: str,
        observation: RobotObservation,
        instruction: str,
        sequence_id: int,
        *,
        seed: int | None = None,
        request_ttl_ms: float | None = None,
        observation_clock_now_ns: int | None = None,
    ) -> RobotActionChunk:
        """Submit one observation and return its semantic robot action chunk."""
        payload: dict[str, Any] = {
            "type": "PREDICT",
            "session_id": session_id,
            "sequence_id": sequence_id,
            "observation_timestamp_ns": observation.state.timestamp_ns,
            "instruction": instruction,
            "observation": robot_observation_to_wire(observation),
        }
        if seed is not None:
            payload["seed"] = seed
        if request_ttl_ms is not None:
            payload["request_ttl_ms"] = request_ttl_ms
        if observation_clock_now_ns is not None:
            payload["observation_clock_now_ns"] = observation_clock_now_ns
        response = await self._request(payload)
        chunk = response.get("chunk")
        if not isinstance(chunk, Mapping):
            raise VLAClientError("invalid_message", "ACTION_CHUNK response is missing chunk")
        return robot_action_chunk_from_wire(chunk)

    async def reset_session(self, session_id: str, *, episode_id: str | None = None) -> None:
        """Reset model history and queued action chunks for one session."""
        payload: dict[str, Any] = {"type": "RESET", "session_id": session_id}
        if episode_id is not None:
            payload["episode_id"] = episode_id
        await self._request(payload)

    async def close_session(self, session_id: str) -> None:
        """Close one server-side session while retaining the connection."""
        await self._request({"type": "CLOSE", "session_id": session_id})
        self._sessions.discard(session_id)

    async def aclose(self) -> None:
        """Close owned sessions and the WebSocket connection."""
        if self._websocket is None:
            return
        for session_id in tuple(self._sessions):
            with contextlib.suppress(Exception):
                await self.close_session(session_id)
        websocket = self._websocket
        self._websocket = None
        await websocket.close()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        self._fail_pending(VLAClientError("connection_closed", "VLA client connection closed"))

    async def __aenter__(self) -> "AsyncVLAClient":
        await self.connect()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        websocket = self._websocket
        if websocket is None:
            raise RuntimeError("VLA client is not connected")
        self._request_sequence += 1
        request_id = f"vla-{self._request_sequence}"
        payload["request_id"] = request_id
        response = asyncio.get_running_loop().create_future()
        self._pending[request_id] = response
        try:
            async with self._send_lock:
                await websocket.send(dumps_wire_message(payload))
            return await response
        except BaseException:
            self._pending.pop(request_id, None)
            raise

    async def _read_responses(self) -> None:
        try:
            while True:
                payload = loads_wire_message(await self._websocket.recv(), max_message_bytes=self.max_message_bytes)
                request_id = payload.get("request_id")
                if not isinstance(request_id, str):
                    raise VLAClientError("invalid_message", "VLA response requires a string request_id")
                response = self._pending.pop(request_id, None)
                if response is None:
                    continue
                if payload.get("type") == "ERROR":
                    error = payload.get("error")
                    if not isinstance(error, Mapping):
                        response.set_exception(VLAClientError("invalid_message", "invalid VLA error response"))
                    else:
                        response.set_exception(
                            VLAClientError(
                                str(error.get("code", "internal_error")),
                                str(error.get("message", "VLA request failed")),
                                request_id=request_id,
                            )
                        )
                else:
                    response.set_result(payload)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail_pending(
                error if isinstance(error, VLAClientError) else VLAClientError("connection_closed", str(error))
            )

    def _fail_pending(self, error: Exception) -> None:
        for response in self._pending.values():
            if not response.done():
                response.set_exception(error)
        self._pending.clear()
