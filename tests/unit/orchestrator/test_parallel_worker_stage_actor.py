from __future__ import annotations

from telefuser.orchestrator.parallel_worker_stage_actor import ParallelWorkerStageActor
from telefuser.orchestrator.streaming_pipeline_orchestrator import (
    StreamingActorState,
    StreamingSessionCloseReason,
    StreamingSessionContext,
    StreamingStageInvocation,
    StreamingTaskKey,
)


class _FakeParallelWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.closed = False
        self.failed = False
        self.failure_reason: str | None = None

    def process(self, *args: object, **kwargs: object):
        self.calls.append((args, kwargs))
        return lambda: {"value": args[0]}

    def close(self) -> None:
        self.closed = True


def test_parallel_worker_stage_actor_correlates_invocation_to_future() -> None:
    worker = _FakeParallelWorker()
    actor = ParallelWorkerStageActor(
        worker,
        "process",
        input_builder=lambda invocation: ((invocation.inputs["value"],), {"sequence": invocation.key.sequence_id}),
        output_builder=lambda result, _: {"output": result["value"]},
    )
    invocation = StreamingStageInvocation(
        key=StreamingTaskKey("session", 1, 3, "stage", "request"),
        inputs={"value": "payload"},
        is_first=False,
        is_last=False,
    )
    try:
        assert actor.submit(invocation).result(timeout=1) == {"output": "payload"}
        assert worker.calls == [(("payload",), {"sequence": 3})]
        actor.barrier(timeout=1)
        assert actor.health().state == StreamingActorState.RUNNING
        assert actor.health().pending_invocations == 0
    finally:
        actor.close()

    assert worker.closed is True
    assert actor.health().state == StreamingActorState.CLOSED


def test_parallel_worker_stage_actor_reports_worker_failure() -> None:
    worker = _FakeParallelWorker()
    actor = ParallelWorkerStageActor(
        worker,
        "process",
        input_builder=lambda _: ((), {}),
        output_builder=lambda result, _: {"output": result},
    )
    worker.failed = True
    worker.failure_reason = "worker exited"
    try:
        health = actor.health()
        assert health.state == StreamingActorState.FAILED
        assert health.failure_reason == "worker exited"
    finally:
        actor.close()


def test_parallel_worker_stage_actor_runs_session_cleanup_in_actor_order() -> None:
    worker = _FakeParallelWorker()
    cleanup_calls: list[tuple[str, StreamingSessionCloseReason]] = []
    actor = ParallelWorkerStageActor(
        worker,
        "process",
        input_builder=lambda _: ((), {}),
        output_builder=lambda result, _: {"output": result},
        session_closer=lambda context, reason: cleanup_calls.append((context.session_id, reason)),
    )
    try:
        actor.close_session(
            StreamingSessionContext("session", 1),
            StreamingSessionCloseReason.CANCELLED,
            timeout=1,
        )
        actor.barrier(timeout=1)
        assert cleanup_calls == [("session", StreamingSessionCloseReason.CANCELLED)]
    finally:
        actor.close()
