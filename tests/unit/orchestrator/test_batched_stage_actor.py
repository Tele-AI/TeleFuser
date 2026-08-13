from __future__ import annotations

import threading

from telefuser.orchestrator import (
    BatchedLocalStageActor,
    StreamingSessionCloseReason,
    StreamingSessionContext,
    StreamingStageInvocation,
    StreamingTaskKey,
)


def _invocation(session_id: str, sequence_id: int, bucket: str = "same") -> StreamingStageInvocation:
    return StreamingStageInvocation(
        key=StreamingTaskKey(session_id, 1, sequence_id, "stage", f"{session_id}-{sequence_id}"),
        inputs={"value": sequence_id, "bucket": bucket},
        is_first=sequence_id == 0,
        is_last=False,
    )


def test_actor_coalesces_compatible_cross_session_invocations() -> None:
    observed_batches: list[list[str]] = []
    release = threading.Event()

    def handler(invocations: list[StreamingStageInvocation]) -> list[dict[str, object]]:
        observed_batches.append([item.key.session_id for item in invocations])
        release.wait(timeout=1)
        return [{"output": item.inputs["value"]} for item in invocations]

    actor = BatchedLocalStageActor(
        handler,
        batch_key=lambda item: item.inputs["bucket"],
        max_batch_size=4,
        batching_window_seconds=0.05,
    )
    first = actor.submit(_invocation("a", 0))
    second = actor.submit(_invocation("b", 0))
    release.set()
    try:
        assert first.result(timeout=1) == {"output": 0}
        assert second.result(timeout=1) == {"output": 0}
        assert observed_batches == [["a", "b"]]
        assert actor.batch_metrics()["max_batch_size"] == 2
    finally:
        actor.close()


def test_actor_does_not_batch_two_strict_items_from_the_same_session() -> None:
    batch_sizes: list[int] = []

    def handler(invocations: list[StreamingStageInvocation]) -> list[dict[str, object]]:
        batch_sizes.append(len(invocations))
        return [{"output": item.key.sequence_id} for item in invocations]

    actor = BatchedLocalStageActor(handler, batching_window_seconds=0.01)
    first = actor.submit(_invocation("a", 0))
    second = actor.submit(_invocation("a", 1))
    try:
        assert first.result(timeout=1) == {"output": 0}
        assert second.result(timeout=1) == {"output": 1}
        assert batch_sizes == [1, 1]
    finally:
        actor.close()


def test_session_cleanup_runs_after_preceding_actor_work() -> None:
    events: list[str] = []

    def handler(invocations: list[StreamingStageInvocation]) -> list[dict[str, object]]:
        events.append("batch")
        return [{"output": 1} for _ in invocations]

    def close_session(context: StreamingSessionContext, reason: StreamingSessionCloseReason) -> None:
        events.append(f"close:{context.session_id}:{reason.value}")

    actor = BatchedLocalStageActor(handler, session_closer=close_session)
    future = actor.submit(_invocation("a", 0))
    try:
        assert future.result(timeout=1) == {"output": 1}
        actor.close_session(
            StreamingSessionContext("a", 1),
            StreamingSessionCloseReason.CLOSED,
        )
        assert events == ["batch", "close:a:closed"]
    finally:
        actor.close()
