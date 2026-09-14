"""Worker-local adapter between replica RPC and semantic VLA sessions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from telefuser.vla import VLASessionManager
from telefuser.vla.serialization import (
    DEFAULT_MAX_TENSOR_BYTES,
    action_space_from_wire,
    action_space_to_wire,
    robot_action_chunk_to_wire,
    robot_observation_from_wire,
)


class VLAReplicaProvider:
    """Dispatch VLA lifecycle operations inside one pipeline replica."""

    def __init__(
        self,
        sessions: VLASessionManager,
        *,
        max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
    ) -> None:
        if not isinstance(max_tensor_bytes, int) or isinstance(max_tensor_bytes, bool) or max_tensor_bytes < 1:
            raise ValueError("max_tensor_bytes must be a positive integer")
        self.sessions = sessions
        self.max_tensor_bytes = max_tensor_bytes

    def metadata(self) -> dict[str, Any]:
        """Return stable component discovery metadata without opening a session."""
        return {
            "model_ids": list(self.sessions.registry.model_ids()),
            "embodiment_ids": list(self.sessions.registry.embodiment_ids()),
        }

    def dispatch(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one validated worker-local VLA operation."""
        if not isinstance(operation, str):
            raise ValueError("VLA replica operation must be a string")
        if not isinstance(payload, Mapping):
            raise ValueError("VLA replica payload must be an object")
        operation = operation.upper()
        if operation == "OPEN":
            return self._open(payload)
        session_id = _required_string(payload, "session_id")
        if operation == "PREDICT":
            return self._predict(session_id, payload)
        if operation == "RESET":
            self.sessions.reset(session_id, _optional_string(payload, "episode_id"))
            return {"session_id": session_id}
        if operation == "CLOSE":
            self.sessions.close(session_id)
            return {"session_id": session_id}
        raise ValueError(f"unsupported VLA replica operation: {operation!r}")

    def close(self) -> None:
        """Release every worker-local session during replica shutdown."""
        for session_id in self.sessions.session_ids():
            self.sessions.close(session_id)

    def _open(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        session_id = _required_string(payload, "session_id")
        model_id = _required_string(payload, "model_id")
        embodiment_id = _required_string(payload, "embodiment_id")
        policy = self.sessions.registry.get_policy(model_id)
        embodiment = self.sessions.registry.get_embodiment(embodiment_id)
        expected_payload = payload.get("expected_robot_action_space")
        if expected_payload is not None:
            if not isinstance(expected_payload, Mapping):
                raise ValueError("expected_robot_action_space must be an object")
            action_space_from_wire(expected_payload).require_compatible(
                embodiment.robot_action_space,
                context="client and embodiment robot action space",
            )
        self.sessions.open(
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
            "max_horizon": capabilities.max_horizon,
            "stateful": capabilities.stateful,
            "supports_seed": capabilities.supports_seed,
        }

    def _predict(self, session_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        observation_payload = payload.get("observation")
        if not isinstance(observation_payload, Mapping):
            raise ValueError("PREDICT requires an observation object")
        max_tensor_bytes = payload.get("_max_tensor_bytes", self.max_tensor_bytes)
        if not isinstance(max_tensor_bytes, int) or isinstance(max_tensor_bytes, bool) or max_tensor_bytes < 1:
            raise ValueError("_max_tensor_bytes must be a positive integer")
        observation = robot_observation_from_wire(observation_payload, max_tensor_bytes=max_tensor_bytes)
        timestamp_ns = _required_nonnegative_int(payload, "observation_timestamp_ns")
        if timestamp_ns != observation.state.timestamp_ns:
            raise ValueError("PREDICT observation_timestamp_ns must match observation.state.timestamp_ns")
        chunk = self.sessions.get(session_id).predict(
            observation,
            _required_string(payload, "instruction"),
            _required_nonnegative_int(payload, "sequence_id"),
            seed=_optional_integer(payload, "seed"),
            now_ns=_optional_nonnegative_int(payload, "observation_clock_now_ns"),
        )
        return {"session_id": session_id, "chunk": robot_action_chunk_to_wire(chunk)}


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_nonnegative_int(payload: Mapping[str, Any], field: str) -> int:
    value = _optional_nonnegative_int(payload, field)
    if value is None:
        raise ValueError(f"{field} is required")
    return value


def _optional_nonnegative_int(payload: Mapping[str, Any], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_positive_int(payload: Mapping[str, Any], field: str) -> int | None:
    value = _optional_nonnegative_int(payload, field)
    if value is not None and value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _optional_integer(payload: Mapping[str, Any], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value
