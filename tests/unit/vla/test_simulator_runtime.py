"""Tests for simulator-side action execution state."""

from __future__ import annotations

import pytest
import torch

from telefuser.vla import ActionSpaceSpec, RobotActionChunk
from telefuser.vla.runtime import ChunkStatus, RemainderPolicy, RuntimeState, SimulatorChunkRuntime

SPACE = ActionSpaceSpec("joint_position", ("joint",), ("radian",), None, 10.0, False)


class _Simulator:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.actions = []

    def execute(self, action) -> None:
        if self.fail_at == action.step_index:
            raise RuntimeError("simulator failed")
        self.actions.append(action)


def _chunk(sequence_id: int = 1) -> RobotActionChunk:
    return RobotActionChunk(
        torch.tensor([[1.0], [2.0], [3.0]]),
        SPACE,
        3,
        100,
        sequence_id,
        "episode",
    )


def test_client_runtime_owns_ready_execution_and_remainder_states() -> None:
    simulator = _Simulator()
    runtime = SimulatorChunkRuntime(
        simulator,
        SPACE,
        "episode",
        execute_horizon=2,
        remainder_policy=RemainderPolicy.RETAIN,
    )

    assert runtime.accept(_chunk()) is ChunkStatus.READY
    assert runtime.state is RuntimeState.READY
    assert runtime.execute_ready() == 2
    assert runtime.state is RuntimeState.READY
    assert runtime.execute_ready() == 1
    assert runtime.state is RuntimeState.EMPTY
    assert [action.values.item() for action in simulator.actions] == [1.0, 2.0, 3.0]


def test_client_runtime_rejects_mismatch_and_fails_closed_on_execution_error() -> None:
    simulator = _Simulator(fail_at=1)
    runtime = SimulatorChunkRuntime(simulator, SPACE, "episode", execute_horizon=3)
    mismatch = ActionSpaceSpec("velocity", ("joint",), ("radian_per_second",), None, 10.0, False)
    bad_chunk = RobotActionChunk(torch.tensor([[1.0]]), mismatch, 1, 100, 1, "episode")

    with pytest.raises(ValueError, match="action space"):
        runtime.accept(bad_chunk)

    assert runtime.accept(_chunk()) is ChunkStatus.READY
    with pytest.raises(RuntimeError, match="simulator failed"):
        runtime.execute_ready()
    assert runtime.state is RuntimeState.HOLDING


def test_client_runtime_exposes_transport_neutral_execution_reports() -> None:
    simulator = _Simulator()
    runtime = SimulatorChunkRuntime(simulator, SPACE, "episode", execute_horizon=2)

    no_action = runtime.execute_ready_with_report()
    assert no_action.status == "no_action"
    assert no_action.sequence_id is None
    assert no_action.executed_steps == 0

    assert runtime.accept(_chunk()) is ChunkStatus.READY
    report = runtime.execute_ready_with_report()
    assert report.status == "executed"
    assert report.sequence_id == 1
    assert report.executed_steps == 2


@pytest.mark.asyncio
async def test_client_runtime_can_execute_without_blocking_async_caller() -> None:
    simulator = _Simulator()
    runtime = SimulatorChunkRuntime(simulator, SPACE, "episode", execute_horizon=3)
    assert runtime.accept(_chunk()) is ChunkStatus.READY

    report = await runtime.execute_ready_async()

    assert report.status == "executed"
    assert report.executed_steps == 3
    assert runtime.state is RuntimeState.EMPTY


def test_client_runtime_reports_simulator_failure_without_raising() -> None:
    simulator = _Simulator(fail_at=1)
    runtime = SimulatorChunkRuntime(simulator, SPACE, "episode", execute_horizon=3)
    assert runtime.accept(_chunk()) is ChunkStatus.READY

    report = runtime.execute_ready_with_report()

    assert report.status == "failed"
    assert report.sequence_id == 1
    assert report.executed_steps == 1
    assert report.reason == "simulator failed"
    assert runtime.state is RuntimeState.HOLDING
