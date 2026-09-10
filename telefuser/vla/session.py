"""Session lifecycle for semantic VLA inference and action preparation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock

from .contracts import ModelActionChunk, RobotActionChunk, RobotObservation, VLARequest
from .embodiment import EmbodimentAdapter
from .policy import VLAPolicy
from .registry import VLARegistry
from .runtime.executor import ChunkExecutor
from .runtime.safety import ActionSafetyPolicy


@dataclass(frozen=True)
class VLASessionContract:
    """Immutable component and action-space selection made during OPEN."""

    session_id: str
    episode_id: str
    model_id: str
    embodiment_id: str


class VLASession:
    """Bind one loaded policy and embodiment for a continuous episode."""

    def __init__(
        self,
        contract: VLASessionContract,
        policy: VLAPolicy,
        embodiment: EmbodimentAdapter,
        executor: ChunkExecutor,
        policy_lock: RLock | None = None,
    ) -> None:
        self.contract = contract
        self.policy = policy
        self.embodiment = embodiment
        self.executor = executor
        self._policy_lock = policy_lock or RLock()
        self._last_sequence_id: int | None = None
        self._lock = RLock()

    def predict(
        self,
        observation: RobotObservation,
        instruction: str,
        sequence_id: int,
        *,
        seed: int | None = None,
        now_ns: int | None = None,
        timings: dict[str, float] | None = None,
    ) -> RobotActionChunk:
        """Run one ordered observation through policy, embodiment, and runtime."""
        with self._lock:
            if not isinstance(sequence_id, int) or isinstance(sequence_id, bool) or sequence_id < 0:
                raise ValueError("sequence_id must be a non-negative integer")
            if self._last_sequence_id is not None and sequence_id <= self._last_sequence_id:
                raise ValueError(
                    f"sequence_id must increase within a session: got {sequence_id}, "
                    f"last accepted {self._last_sequence_id}"
                )
            capabilities = self.policy.capabilities()
            if seed is not None and not capabilities.supports_seed:
                raise ValueError(f"VLA policy {capabilities.model_id!r} does not support seeded inference")
            stage_started_at = time.monotonic()
            model_observation = self.embodiment.encode_observation(observation)
            if timings is not None:
                timings["encode_observation_ms"] = (time.monotonic() - stage_started_at) * 1000.0
            request = VLARequest(
                observation=model_observation,
                instruction=instruction,
                episode_id=self.contract.episode_id,
                sequence_id=sequence_id,
                observation_timestamp_ns=observation.state.timestamp_ns,
                seed=seed,
            )
            stage_started_at = time.monotonic()
            with self._policy_lock:
                model_chunk = self.policy.predict(request)
            if timings is not None:
                timings["policy_ms"] = (time.monotonic() - stage_started_at) * 1000.0
            if not isinstance(model_chunk, ModelActionChunk):
                raise TypeError("VLA policy must return ModelActionChunk")
            capabilities.output_action_space.require_compatible(
                model_chunk.action_space,
                context="policy output action space",
            )
            if model_chunk.valid_length > capabilities.max_horizon:
                raise ValueError("policy output exceeds its declared maximum horizon")
            if model_chunk.episode_id != request.episode_id:
                raise ValueError("policy output episode_id does not match the request")
            if model_chunk.sequence_id != request.sequence_id:
                raise ValueError("policy output sequence_id does not match the request")
            if model_chunk.observation_timestamp_ns != request.observation_timestamp_ns:
                raise ValueError("policy output observation timestamp does not match the request")
            stage_started_at = time.monotonic()
            robot_chunk = self.embodiment.decode_actions(model_chunk, observation.state)
            if timings is not None:
                timings["decode_actions_ms"] = (time.monotonic() - stage_started_at) * 1000.0
            if not isinstance(robot_chunk, RobotActionChunk):
                raise TypeError("embodiment adapter must return RobotActionChunk")
            if robot_chunk.episode_id != model_chunk.episode_id:
                raise ValueError("embodiment output episode_id does not match the model chunk")
            if robot_chunk.sequence_id != model_chunk.sequence_id:
                raise ValueError("embodiment output sequence_id does not match the model chunk")
            if robot_chunk.observation_timestamp_ns != model_chunk.observation_timestamp_ns:
                raise ValueError("embodiment output observation timestamp does not match the model chunk")
            stage_started_at = time.monotonic()
            prepared = self.executor.prepare(robot_chunk, observation.state, now_ns=now_ns)
            if timings is not None:
                timings["prepare_actions_ms"] = (time.monotonic() - stage_started_at) * 1000.0
            self._last_sequence_id = sequence_id
            return prepared

    def reset(self, episode_id: str | None = None) -> None:
        """Clear temporal policy and ordering state for the bound episode."""
        with self._lock:
            old_episode_id = self.contract.episode_id
            with self._policy_lock:
                self.policy.reset(old_episode_id)
            if episode_id is not None:
                if not isinstance(episode_id, str) or not episode_id:
                    raise ValueError("episode_id must be a non-empty string")
                self.contract = VLASessionContract(
                    session_id=self.contract.session_id,
                    episode_id=episode_id,
                    model_id=self.contract.model_id,
                    embodiment_id=self.contract.embodiment_id,
                )
            self._last_sequence_id = None

    def close(self) -> None:
        """Release policy-owned state associated with this session."""
        with self._lock:
            with self._policy_lock:
                self.policy.reset(self.contract.episode_id)
            self._last_sequence_id = None


class VLASessionManager:
    """Implement OPEN, PREDICT, RESET, and CLOSE without transport concerns."""

    def __init__(self, registry: VLARegistry) -> None:
        self.registry = registry
        self._sessions: dict[str, VLASession] = {}
        self._policy_locks: dict[int, RLock] = {}
        self._lock = RLock()

    def open(
        self,
        session_id: str,
        *,
        model_id: str,
        embodiment_id: str,
        episode_id: str,
        execute_horizon: int | None = None,
        safety_policy: ActionSafetyPolicy | None = None,
        max_observation_age_ns: int | None = None,
    ) -> VLASession:
        """Bind one session to loaded component instances and validate semantics."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a non-empty string")
        with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"VLA session is already open: {session_id!r}")
            policy = self.registry.get_policy(model_id)
            embodiment = self.registry.get_embodiment(embodiment_id)
            embodiment.model_action_space.require_compatible(
                policy.capabilities().output_action_space,
                context="policy and embodiment action space",
            )
            contract = VLASessionContract(
                session_id=session_id,
                episode_id=episode_id,
                model_id=model_id,
                embodiment_id=embodiment_id,
            )
            session = VLASession(
                contract,
                policy,
                embodiment,
                ChunkExecutor(
                    embodiment.robot_action_space,
                    execute_horizon=execute_horizon,
                    safety_policy=safety_policy,
                    max_observation_age_ns=max_observation_age_ns,
                ),
                self._policy_locks.setdefault(id(policy), RLock()),
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> VLASession:
        """Return one active session."""
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise KeyError(f"unknown VLA session_id: {session_id!r}") from error

    def reset(self, session_id: str, episode_id: str | None = None) -> None:
        """Reset ordering and policy history for one active session."""
        self.get(session_id).reset(episode_id)

    def close(self, session_id: str) -> None:
        """Close one session and release its policy state."""
        with self._lock:
            try:
                session = self._sessions.pop(session_id)
            except KeyError as error:
                raise KeyError(f"unknown VLA session_id: {session_id!r}") from error
        session.close()

    def session_ids(self) -> tuple[str, ...]:
        """List active session IDs in deterministic order."""
        with self._lock:
            return tuple(sorted(self._sessions))
