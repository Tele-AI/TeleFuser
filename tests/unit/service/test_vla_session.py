"""Loopback tests for the generic versioned VLA WebSocket protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
import torch
from fastapi.testclient import TestClient

from telefuser.service.core.replica_worker import ReplicaDeadError
from telefuser.service.vla_session import VLA_SESSION_PROTOCOL_VERSION, create_vla_session_app
from telefuser.vla import (
    ActionSpaceSpec,
    ModelActionChunk,
    ModelObservation,
    RobotActionChunk,
    RobotObservation,
    RobotState,
    VLACapabilities,
    VLARegistry,
    VLASessionManager,
    action_space_to_wire,
    robot_action_chunk_from_wire,
    robot_observation_to_wire,
)

MODEL_SPACE = ActionSpaceSpec("joint_delta", ("joint",), ("radian",), None, 10.0, False)
ROBOT_SPACE = ActionSpaceSpec("joint_position", ("joint",), ("radian",), None, 10.0, False)


class _Policy:
    def __init__(self, *, delay_s: float = 0, failure: Exception | None = None) -> None:
        self.delay_s = delay_s
        self.failure = failure
        self.resets: list[str] = []

    def capabilities(self) -> VLACapabilities:
        return VLACapabilities("fake", MODEL_SPACE, max_horizon=3)

    def predict(self, request) -> ModelActionChunk:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.failure is not None:
            raise self.failure
        return ModelActionChunk(
            torch.tensor([[0.1], [0.2], [0.3]]),
            MODEL_SPACE,
            3,
            request.observation_timestamp_ns,
            request.sequence_id,
            request.episode_id,
        )

    def reset(self, episode_id: str) -> None:
        self.resets.append(episode_id)


@dataclass
class _Embodiment:
    embodiment_id: str = "fake-robot"
    model_action_space: ActionSpaceSpec = MODEL_SPACE
    robot_action_space: ActionSpaceSpec = ROBOT_SPACE

    def encode_observation(self, observation: RobotObservation) -> ModelObservation:
        return ModelObservation(observation.state.values, observation.images)

    def decode_actions(self, actions: ModelActionChunk, robot_state: RobotState) -> RobotActionChunk:
        return RobotActionChunk(
            actions.actions + robot_state.values,
            self.robot_action_space,
            actions.valid_length,
            actions.observation_timestamp_ns,
            actions.sequence_id,
            actions.episode_id,
        )


def _manager(policy: _Policy | None = None) -> tuple[VLASessionManager, _Policy]:
    policy = policy or _Policy()
    registry = VLARegistry()
    registry.register_policy("fake", policy)
    registry.register_embodiment("fake-robot", _Embodiment())
    return VLASessionManager(registry), policy


def _open(**overrides) -> dict:
    return {
        "type": "OPEN",
        "protocol_version": VLA_SESSION_PROTOCOL_VERSION,
        "request_id": "open",
        "session_id": "session",
        "episode_id": "episode",
        "model_id": "fake",
        "embodiment_id": "fake-robot",
        **overrides,
    }


def _predict(sequence_id: int, **overrides) -> dict:
    timestamp_ns = overrides.pop("observation_timestamp_ns", 100 + sequence_id)
    observation = RobotObservation(
        RobotState(torch.tensor([1.0]), ("joint",), timestamp_ns),
        {"front": torch.zeros((2, 2, 3), dtype=torch.uint8)},
    )
    return {
        "type": "PREDICT",
        "request_id": f"predict-{sequence_id}",
        "session_id": "session",
        "sequence_id": sequence_id,
        "observation_timestamp_ns": timestamp_ns,
        "instruction": "move",
        "seed": 7,
        "observation": robot_observation_to_wire(observation),
        **overrides,
    }


def test_protocol_negotiates_capabilities_and_runs_full_session_lifecycle() -> None:
    manager, policy = _manager()
    with TestClient(create_vla_session_app(manager)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        with client.websocket_connect("/v1/vla/session") as websocket:
            hello = websocket.receive_json()
            assert hello["type"] == "HELLO"
            assert hello["protocol_version"] == "1.0"
            assert hello["operations"] == ["OPEN", "PREDICT", "RESET", "CLOSE"]
            assert hello["model_ids"] == ["fake"]
            assert hello["embodiment_ids"] == ["fake-robot"]

            websocket.send_json(_open(expected_robot_action_space=action_space_to_wire(ROBOT_SPACE)))
            opened = websocket.receive_json()
            assert opened["type"] == "OPENED"
            assert opened["supports_seed"] is True
            assert opened["robot_action_space"] == action_space_to_wire(ROBOT_SPACE)

            websocket.send_json(_predict(1))
            response = websocket.receive_json()
            assert response["type"] == "ACTION_CHUNK"
            chunk = robot_action_chunk_from_wire(response["chunk"])
            assert chunk.actions.flatten().tolist() == pytest.approx([1.1, 1.2, 1.3])
            assert chunk.sequence_id == 1

            websocket.send_json({"type": "RESET", "session_id": "session", "episode_id": "episode-2"})
            assert websocket.receive_json()["type"] == "RESET_ACK"
            websocket.send_json(_predict(0, observation_timestamp_ns=200))
            assert websocket.receive_json()["type"] == "ACTION_CHUNK"

            websocket.send_json({"type": "CLOSE", "session_id": "session"})
            assert websocket.receive_json()["type"] == "CLOSE_ACK"

    assert manager.session_ids() == ()
    assert policy.resets == ["episode", "episode-2"]


def test_protocol_returns_stable_errors_for_version_components_contract_order_and_age() -> None:
    manager, _ = _manager()
    with TestClient(create_vla_session_app(manager)) as client:
        with client.websocket_connect("/v1/vla/session") as websocket:
            websocket.receive_json()
            websocket.send_json(_open(protocol_version="2.0"))
            assert websocket.receive_json()["error"]["code"] == "unsupported_version"

            websocket.send_json(_open(model_id="missing"))
            assert websocket.receive_json()["error"]["code"] == "unknown_component"

            mismatch = ActionSpaceSpec("velocity", ("joint",), ("radian_per_second",), None, 10.0, False)
            websocket.send_json(_open(expected_robot_action_space=action_space_to_wire(mismatch)))
            assert websocket.receive_json()["error"]["code"] == "action_space_mismatch"

            websocket.send_json(_open(max_observation_age_ns=10))
            assert websocket.receive_json()["type"] == "OPENED"
            websocket.send_json(_predict(2, observation_clock_now_ns=102))
            assert websocket.receive_json()["type"] == "ACTION_CHUNK"
            websocket.send_json(_predict(2))
            assert websocket.receive_json()["error"]["code"] == "out_of_order"
            websocket.send_json(_predict(3, observation_clock_now_ns=1_000))
            assert websocket.receive_json()["error"]["code"] == "expired"


def test_protocol_reports_timeout_recovers_session_and_maps_replica_failure() -> None:
    manager, _ = _manager(_Policy(delay_s=0.02))
    with TestClient(create_vla_session_app(manager)) as client:
        with client.websocket_connect("/v1/vla/session") as websocket:
            websocket.receive_json()
            websocket.send_json(_open())
            websocket.receive_json()
            websocket.send_json(_predict(1, request_ttl_ms=1))
            assert websocket.receive_json()["error"]["code"] == "timeout"
            websocket.send_json({"type": "RESET", "session_id": "session"})
            assert websocket.receive_json()["type"] == "RESET_ACK"

    failed_manager, _ = _manager(_Policy(failure=ReplicaDeadError("replica exited")))
    with TestClient(create_vla_session_app(failed_manager)) as client:
        with client.websocket_connect("/v1/vla/session") as websocket:
            websocket.receive_json()
            websocket.send_json(_open())
            websocket.receive_json()
            websocket.send_json(_predict(1))
            assert websocket.receive_json()["error"]["code"] == "replica_unavailable"


def test_disconnect_closes_connection_owned_sessions() -> None:
    manager, policy = _manager()
    with TestClient(create_vla_session_app(manager)) as client:
        with client.websocket_connect("/v1/vla/session") as websocket:
            websocket.receive_json()
            websocket.send_json(_open())
            assert websocket.receive_json()["type"] == "OPENED"
        assert manager.session_ids() == ()
    assert policy.resets == ["episode"]
