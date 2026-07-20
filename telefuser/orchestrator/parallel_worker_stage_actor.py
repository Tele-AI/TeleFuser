"""Actor adapter that safely drives one existing ParallelWorker."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from typing import Any

from telefuser.worker.parallel_worker import ParallelWorker

from .streaming_pipeline_orchestrator import LocalStageActor, StreamingStageInvocation


class ParallelWorkerStageActor:
    """Expose one exclusively owned ParallelWorker as a long-lived stage actor.

    A ParallelWorker instance must be wrapped by exactly one adapter for its
    lifetime. The adapter may serve many sessions, but serializes worker
    submission and result consumption in one actor loop, so the worker has at
    most one outstanding invocation. This preserves the ordered ParallelWorker
    output-queue contract while giving StreamingPipelineOrchestrator a Future per
    task key. Independent model stages must receive independent adapters and
    worker instances.
    """

    def __init__(
        self,
        worker: ParallelWorker,
        method_name: str,
        input_builder: Callable[[StreamingStageInvocation], tuple[tuple[Any, ...], dict[str, Any]]],
        output_builder: Callable[[Any, StreamingStageInvocation], Mapping[str, object]],
        mailbox_capacity: int = 1,
        close_worker: bool = True,
    ) -> None:
        self.worker = worker
        self.method_name = method_name
        self._input_builder = input_builder
        self._output_builder = output_builder
        self._close_worker = close_worker
        self._actor = LocalStageActor(
            self._invoke,
            mailbox_capacity=mailbox_capacity,
            name=f"parallel-worker-actor-{method_name}",
        )

    def submit(self, invocation: StreamingStageInvocation) -> Future[Mapping[str, object]]:
        """Submit one invocation and retain its scheduler-owned Future."""
        return self._actor.submit(invocation)

    def close(self) -> None:
        """Drain the adapter before deterministically closing its worker."""
        self._actor.close()
        if self._close_worker:
            self.worker.close()

    def _invoke(self, invocation: StreamingStageInvocation) -> Mapping[str, object]:
        args, kwargs = self._input_builder(invocation)
        result = getattr(self.worker, self.method_name)(*args, **kwargs)
        if callable(result):
            result = result()
        return self._output_builder(result, invocation)
