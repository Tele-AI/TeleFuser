"""Client-side action lifecycle for simulator execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts import ActionSpaceSpec, RobotActionChunk
from .chunk_state import (
    ActionChunkStateMachine,
    ChunkStatus,
    DisconnectPolicy,
    RemainderPolicy,
    RuntimeState,
)
from .executor import ChunkExecutor

if TYPE_CHECKING:
    from telefuser.integrations.sim.base import SimulatorAdapter


class SimulatorChunkRuntime:
    """Own READY/EXECUTING/EXECUTED states beside a simulator connection."""

    def __init__(
        self,
        simulator: SimulatorAdapter,
        action_space: ActionSpaceSpec,
        episode_id: str,
        *,
        execute_horizon: int,
        remainder_policy: RemainderPolicy = RemainderPolicy.DISCARD,
        disconnect_policy: DisconnectPolicy = DisconnectPolicy.HOLD,
        max_observation_age_ns: int | None = None,
    ) -> None:
        self.simulator = simulator
        self.action_space = action_space
        self.state_machine = ActionChunkStateMachine(
            episode_id,
            execute_horizon=execute_horizon,
            remainder_policy=remainder_policy,
            disconnect_policy=disconnect_policy,
            max_observation_age_ns=max_observation_age_ns,
        )

    @property
    def state(self) -> RuntimeState:
        """Return the current simulator-side runtime state."""
        return self.state_machine.state

    def accept(
        self,
        chunk: RobotActionChunk,
        *,
        observation_clock_now_ns: int | None = None,
    ) -> ChunkStatus:
        """Accept a server-ready chunk without claiming that it was executed."""
        self.action_space.require_compatible(chunk.action_space, context="simulator action space")
        ticket = self.state_machine.submit(
            chunk.sequence_id,
            chunk.observation_timestamp_ns,
            observation_clock_now_ns=observation_clock_now_ns,
        )
        if self.state_machine.status(ticket) is not ChunkStatus.PENDING:
            return self.state_machine.status(ticket)
        self.state_machine.mark_inference_started(ticket)
        return self.state_machine.complete(
            ticket,
            chunk,
            observation_clock_now_ns=observation_clock_now_ns,
        )

    def execute_ready(self) -> int:
        """Execute one configured horizon and return the submitted step count."""
        chunk = self.state_machine.begin_execution()
        if chunk is None:
            return 0
        executed = 0
        try:
            for action in ChunkExecutor.iter_actions(chunk):
                self.simulator.execute(action)
                executed += 1
        except Exception:
            self.state_machine.finish_execution(success=False)
            self.state_machine.disconnect()
            raise
        self.state_machine.finish_execution(success=True)
        return executed

    def reset(self, episode_id: str | None = None) -> None:
        """Clear buffered action state after the remote session is reset."""
        self.state_machine.reset(episode_id)

    def disconnect(self) -> RuntimeState:
        """Discard buffered actions and enter the configured hold or stop state."""
        return self.state_machine.disconnect()
