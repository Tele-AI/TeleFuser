"""Tests for the generic asynchronous VLA WebSocket client."""

from __future__ import annotations

import asyncio

import pytest
import torch

from telefuser.client.vla import AsyncVLAClient, VLAClientError
from telefuser.vla import ActionSpaceSpec, RobotActionChunk, RobotObservation, RobotState
from telefuser.vla.serialization import (
    VLA_SESSION_PROTOCOL_VERSION,
    VLA_WIRE_ENCODING,
    dumps_wire_message,
    loads_wire_message,
    robot_action_chunk_to_wire,
)

SPACE = ActionSpaceSpec("joint_position", ("joint",), ("radian",), None, 10.0, False)


class _FakeWebSocket:
    def __init__(self, *, fail_predict: bool = False) -> None:
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.incoming.put_nowait(
            dumps_wire_message(
                {
                    "type": "HELLO",
                    "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
                    "encoding": VLA_WIRE_ENCODING,
                }
            )
        )
        self.fail_predict = fail_predict
        self.delayed_prediction: str | None = None
        self.closed = False

    async def recv(self) -> str:
        return await self.incoming.get()

    async def send(self, encoded: str) -> None:
        request = loads_wire_message(encoded, max_message_bytes=1_000_000)
        operation = request["type"]
        response = {
            "type": f"{operation}_ACK",
            "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
            "request_id": request["request_id"],
            "session_id": request["session_id"],
        }
        if operation == "OPEN":
            response["type"] = "OPENED"
        elif operation == "PREDICT":
            if self.fail_predict:
                response = {
                    "type": "ERROR",
                    "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
                    "request_id": request["request_id"],
                    "error": {"code": "expired", "message": "observation expired"},
                }
            else:
                response = {
                    "type": "ACTION_CHUNK",
                    "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
                    "request_id": request["request_id"],
                    "session_id": request["session_id"],
                    "chunk": robot_action_chunk_to_wire(
                        RobotActionChunk(
                            torch.tensor([[float(request["sequence_id"])]]),
                            SPACE,
                            1,
                            request["observation_timestamp_ns"],
                            request["sequence_id"],
                            "episode",
                        )
                    ),
                }
        encoded_response = dumps_wire_message(response)
        if operation == "PREDICT" and request["sequence_id"] == 1 and not self.fail_predict:
            self.delayed_prediction = encoded_response
            return
        self.incoming.put_nowait(encoded_response)
        if operation == "PREDICT" and self.delayed_prediction is not None:
            self.incoming.put_nowait(self.delayed_prediction)
            self.delayed_prediction = None

    async def close(self) -> None:
        self.closed = True


def _observation(timestamp_ns: int) -> RobotObservation:
    return RobotObservation(RobotState(torch.tensor([0.0]), ("joint",), timestamp_ns), {})


@pytest.mark.asyncio
async def test_client_negotiates_lifecycle_and_correlates_out_of_order_responses(monkeypatch) -> None:
    websocket = _FakeWebSocket()

    async def connect(*_args, **_kwargs):
        return websocket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    client = AsyncVLAClient("ws://example.test/v1/vla/session")
    hello = await client.connect()
    assert hello["type"] == "HELLO"
    await client.open_session("session", model_id="fake", embodiment_id="robot", episode_id="episode")

    first, second = await asyncio.gather(
        client.predict("session", _observation(101), "move", 1),
        client.predict("session", _observation(102), "move", 2),
    )
    assert first.sequence_id == 1
    assert second.sequence_id == 2
    await client.reset_session("session")
    await client.close_session("session")
    await client.aclose()
    assert websocket.closed is True


@pytest.mark.asyncio
async def test_client_exposes_stable_server_error(monkeypatch) -> None:
    websocket = _FakeWebSocket(fail_predict=True)

    async def connect(*_args, **_kwargs):
        return websocket

    monkeypatch.setattr("websockets.asyncio.client.connect", connect)
    async with AsyncVLAClient("ws://example.test/v1/vla/session") as client:
        await client.open_session("session", model_id="fake", embodiment_id="robot", episode_id="episode")
        with pytest.raises(VLAClientError, match="observation expired") as error:
            await client.predict("session", _observation(101), "move", 1)
        assert error.value.code == "expired"
