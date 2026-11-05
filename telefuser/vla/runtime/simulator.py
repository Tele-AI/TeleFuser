"""Client-side action lifecycle for simulator execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ChunkExecutionReport:
    """Small, transport-neutral report emitted by simulator-side execution.

    The report deliberately does not become a WebSocket operation. A remote
    simulator can serialize it on its existing control channel while local
    callers can inspect it directly. ``sequence_id`` is ``None`` when no
    ready chunk was available.
    """

    status: str
    sequence_id: int | None
    episode_id: str
    executed_steps: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"executed", "failed", "no_action"}:
            raise ValueError("status must be executed, failed, or no_action")
        if self.sequence_id is not None and (
            not isinstance(self.sequence_id, int) or isinstance(self.sequence_id, bool) or self.sequence_id < 0
        ):
            raise ValueError("sequence_id must be a non-negative integer or None")
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be a non-empty string")
        if not isinstance(self.executed_steps, int) or isinstance(self.executed_steps, bool) or self.executed_steps < 0:
            raise ValueError("executed_steps must be a non-negative integer")
        if self.status == "no_action" and self.executed_steps != 0:
            raise ValueError("no_action reports must have zero executed_steps")
        if self.status == "failed" and not self.reason:
            raise ValueError("failed reports require a reason")


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
        report = self._execute_ready(raise_errors=True)
        return report.executed_steps

    def execute_ready_with_report(self) -> ChunkExecutionReport:
        """Execute one horizon and return a stable result for remote reporting.

        Unlike :meth:`execute_ready`, this method converts simulator failures
        into a ``failed`` report after entering the configured fail-closed
        state. The legacy method remains exception-based for compatibility.
        """
        return self._execute_ready(raise_errors=False)

    async def execute_ready_async(self) -> ChunkExecutionReport:
        """Run execution off the event loop for inference/execution overlap.

        Prediction scheduling remains the caller's responsibility. This
        additive helper only prevents a synchronous simulator adapter from
        blocking an async VLA client while a ready chunk is being applied.
        """
        return await asyncio.to_thread(self.execute_ready_with_report)

    def _execute_ready(self, *, raise_errors: bool) -> ChunkExecutionReport:
        chunk = self.state_machine.begin_execution()
        if chunk is None:
            return ChunkExecutionReport("no_action", None, self.state_machine.episode_id, 0)
        executed = 0
        try:
            for action in ChunkExecutor.iter_actions(chunk):
                self.simulator.execute(action)
                executed += 1
        except Exception as error:
            self.state_machine.finish_execution(success=False)
            self.state_machine.disconnect()
            report = ChunkExecutionReport(
                "failed",
                chunk.sequence_id,
                chunk.episode_id,
                executed,
                reason=str(error) or error.__class__.__name__,
            )
            if raise_errors:
                raise
            return report
        self.state_machine.finish_execution(success=True)
        return ChunkExecutionReport("executed", chunk.sequence_id, chunk.episode_id, executed)

    def reset(self, episode_id: str | None = None) -> None:
        """Clear buffered action state after the remote session is reset."""
        self.state_machine.reset(episode_id)

    def disconnect(self) -> RuntimeState:
        """Discard buffered actions and enter the configured hold or stop state."""
        return self.state_machine.disconnect()
