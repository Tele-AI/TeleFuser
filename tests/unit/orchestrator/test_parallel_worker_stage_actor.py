from __future__ import annotations

from telefuser.orchestrator.parallel_worker_stage_actor import ParallelWorkerStageActor
from telefuser.orchestrator.streaming_pipeline_orchestrator import StreamingStageInvocation, StreamingTaskKey


class _FakeParallelWorker:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.closed = False

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
    finally:
        actor.close()

    assert worker.closed is True
