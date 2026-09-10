"""Deterministic tests for the generic action chunk state machine."""

from __future__ import annotations

import torch

from telefuser.vla import ActionSpaceSpec, RobotActionChunk
from telefuser.vla.runtime import (
    ActionChunkStateMachine,
    ChunkStatus,
    DisconnectPolicy,
    RemainderPolicy,
    RuntimeState,
)

SPACE = ActionSpaceSpec("joint_position", ("joint",), ("radian",), None, 10.0, False)


class _Clock:
    def __init__(self) -> None:
        self.now = 1.0

    def __call__(self) -> float:
        return self.now


def _chunk(sequence_id: int, timestamp_ns: int, values: tuple[float, ...] = (1, 2, 3, 4)) -> RobotActionChunk:
    return RobotActionChunk(
        torch.tensor(values).reshape(-1, 1),
        SPACE,
        len(values),
        timestamp_ns,
        sequence_id,
        "episode",
    )


def test_latest_sequence_supersedes_pending_and_ready_chunks() -> None:
    runtime = ActionChunkStateMachine("episode", execute_horizon=2)
    first = runtime.submit(1, 100)
    second = runtime.submit(2, 200)

    assert runtime.status(first) is ChunkStatus.SUPERSEDED
    assert runtime.complete(first, _chunk(1, 100)) is ChunkStatus.SUPERSEDED
    assert runtime.complete(second, _chunk(2, 200)) is ChunkStatus.READY

    third = runtime.submit(3, 300)
    assert runtime.status(second) is ChunkStatus.SUPERSEDED
    assert runtime.state is RuntimeState.PENDING
    assert runtime.status(third) is ChunkStatus.PENDING

    duplicate = runtime.submit(3, 300)
    assert runtime.status(duplicate) is ChunkStatus.REJECTED


def test_observation_age_and_network_ttl_use_independent_clocks() -> None:
    clock = _Clock()
    runtime = ActionChunkStateMachine(
        "episode",
        execute_horizon=2,
        max_observation_age_ns=20,
        clock=clock,
    )

    stale = runtime.submit(1, 100, request_ttl_ms=10_000, observation_clock_now_ns=121)
    assert runtime.status(stale) is ChunkStatus.EXPIRED
    assert runtime.reason(stale) == "observation timestamp is stale"

    network_expired = runtime.submit(2, 200, request_ttl_ms=10, observation_clock_now_ns=200)
    clock.now += 0.011
    assert (
        runtime.complete(
            network_expired,
            _chunk(2, 200),
            observation_clock_now_ns=200,
        )
        is ChunkStatus.EXPIRED
    )
    assert runtime.reason(network_expired) == "request_ttl_ms elapsed before completion"

    inference_stale = runtime.submit(3, 300, request_ttl_ms=10_000, observation_clock_now_ns=300)
    assert (
        runtime.complete(
            inference_stale,
            _chunk(3, 300),
            observation_clock_now_ns=321,
        )
        is ChunkStatus.EXPIRED
    )
    assert runtime.reason(inference_stale) == "observation became stale before completion"


def test_execute_horizon_can_retain_or_discard_remainder() -> None:
    retained = ActionChunkStateMachine(
        "episode",
        execute_horizon=2,
        remainder_policy=RemainderPolicy.RETAIN,
    )
    ticket = retained.submit(1, 100)
    assert retained.complete(ticket, _chunk(1, 100)) is ChunkStatus.READY

    first = retained.begin_execution()
    assert first is not None
    assert first.actions.flatten().tolist() == [1, 2]
    assert retained.finish_execution() is RuntimeState.READY
    second = retained.begin_execution()
    assert second is not None
    assert second.actions.flatten().tolist() == [3, 4]
    assert retained.finish_execution() is RuntimeState.EMPTY
    assert retained.status(ticket) is ChunkStatus.EXECUTED

    discarded = ActionChunkStateMachine("episode", execute_horizon=2)
    discarded_ticket = discarded.submit(1, 100)
    discarded.complete(discarded_ticket, _chunk(1, 100))
    assert discarded.begin_execution() is not None
    assert discarded.finish_execution() is RuntimeState.EMPTY
    assert discarded.status(discarded_ticket) is ChunkStatus.EXECUTED


def test_reset_disconnect_and_stateful_discard_recovery_are_explicit() -> None:
    recovered: list[str] = []
    runtime = ActionChunkStateMachine(
        "episode",
        execute_horizon=2,
        stateful_policy=True,
        recover_stateful_policy=recovered.append,
    )
    discarded = runtime.submit(1, 100)
    runtime.mark_inference_started(discarded)
    newest = runtime.submit(2, 200)
    assert runtime.complete(discarded, _chunk(1, 100)) is ChunkStatus.SUPERSEDED
    assert recovered == ["episode"]

    runtime.mark_inference_started(newest)
    runtime.reset("episode-2")
    assert recovered == ["episode", "episode"]
    assert runtime.state is RuntimeState.EMPTY
    assert runtime.latest_sequence_id is None
    assert runtime.submit(0, 0).sequence_id == 0

    assert runtime.disconnect() is RuntimeState.HOLDING
    rejected = runtime.submit(1, 1)
    assert runtime.status(rejected) is ChunkStatus.REJECTED

    stopped = ActionChunkStateMachine(
        "episode",
        execute_horizon=1,
        disconnect_policy=DisconnectPolicy.STOP,
    )
    assert stopped.disconnect() is RuntimeState.STOPPED


def test_terminal_history_preserves_inflight_ticket_until_discard_recovery() -> None:
    recovered: list[str] = []
    runtime = ActionChunkStateMachine(
        "episode",
        execute_horizon=1,
        stateful_policy=True,
        recover_stateful_policy=recovered.append,
        terminal_history=1,
    )
    inflight = runtime.submit(1, 100)
    runtime.mark_inference_started(inflight)
    runtime.submit(2, 200)
    for _ in range(3):
        rejected = runtime.submit(2, 200)
        assert runtime.status(rejected) is ChunkStatus.REJECTED

    assert runtime.status(inflight) is ChunkStatus.SUPERSEDED
    assert runtime.complete(inflight, _chunk(1, 100)) is ChunkStatus.SUPERSEDED
    assert recovered == ["episode"]
