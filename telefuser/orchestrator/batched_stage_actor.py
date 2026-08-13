"""Continuous-batching stage actor for stateful streaming pipelines."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from .streaming_pipeline_orchestrator import (
    StreamingActorBusyError,
    StreamingActorHealth,
    StreamingActorState,
    StreamingSessionCloseReason,
    StreamingSessionContext,
    StreamingStageInvocation,
)


@dataclass
class _BatchInvocationMessage:
    invocation: StreamingStageInvocation
    future: Future[Mapping[str, object]]


@dataclass
class _BatchSessionCloseMessage:
    context: StreamingSessionContext
    reason: StreamingSessionCloseReason
    future: Future[None]


class BatchedLocalStageActor:
    """Own one stage and coalesce compatible cross-session invocations."""

    def __init__(
        self,
        batch_handler: Callable[
            [Sequence[StreamingStageInvocation]],
            Sequence[Mapping[str, object]],
        ],
        *,
        batch_key: Callable[[StreamingStageInvocation], object] | None = None,
        max_batch_size: int = 8,
        batching_window_seconds: float = 0.002,
        mailbox_capacity: int = 64,
        name: str = "batched-local-stage-actor",
        session_closer: Callable[[StreamingSessionContext, StreamingSessionCloseReason], None] | None = None,
    ) -> None:
        if max_batch_size < 1 or mailbox_capacity < 1 or batching_window_seconds < 0:
            raise ValueError("Invalid batched actor capacity or batching window")
        self._batch_handler = batch_handler
        self._batch_key = batch_key or (lambda _invocation: None)
        self._max_batch_size = int(max_batch_size)
        self._batching_window_seconds = float(batching_window_seconds)
        self._session_closer = session_closer
        self._mailbox: queue.Queue[_BatchInvocationMessage | _BatchSessionCloseMessage | None] = queue.Queue(
            maxsize=mailbox_capacity
        )
        self._backlog: deque[_BatchInvocationMessage | _BatchSessionCloseMessage | None] = deque()
        self._closed = False
        self._failure_reason: str | None = None
        self._pending_invocations = 0
        self._pending_session_closes = 0
        self._batch_count = 0
        self._batch_item_count = 0
        self._max_observed_batch_size = 0
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._run, daemon=True, name=name)
        self._thread.start()

    def submit(self, invocation: StreamingStageInvocation) -> Future[Mapping[str, object]]:
        with self._idle:
            if self._closed:
                raise RuntimeError("Stage actor is closed")
            if self._failure_reason is not None:
                raise RuntimeError(f"Stage actor has failed: {self._failure_reason}")
            future: Future[Mapping[str, object]] = Future()
            try:
                self._mailbox.put_nowait(_BatchInvocationMessage(invocation, future))
            except queue.Full as exc:
                raise StreamingActorBusyError("Stage actor mailbox is full") from exc
            self._pending_invocations += 1
            return future

    def health(self) -> StreamingActorHealth:
        with self._idle:
            if self._failure_reason is not None:
                state = StreamingActorState.FAILED
            elif self._closed:
                state = StreamingActorState.CLOSED
            else:
                state = StreamingActorState.RUNNING
            return StreamingActorHealth(state, self._pending_invocations, self._failure_reason)

    def batch_metrics(self) -> dict[str, int | float]:
        with self._idle:
            mean = self._batch_item_count / self._batch_count if self._batch_count else 0.0
            return {
                "batch_count": self._batch_count,
                "batch_items": self._batch_item_count,
                "max_batch_size": self._max_observed_batch_size,
                "mean_batch_size": mean,
            }

    def barrier(self, timeout: float = 5.0) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._idle:
            while self._pending_invocations or self._pending_session_closes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for batched stage actor quiescence")
                self._idle.wait(remaining)

    def close_session(
        self,
        context: StreamingSessionContext,
        reason: StreamingSessionCloseReason,
        timeout: float = 5.0,
    ) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        future: Future[None] = Future()
        with self._idle:
            if self._closed:
                raise RuntimeError("Stage actor is closed")
            if self._failure_reason is not None:
                raise RuntimeError(f"Stage actor has failed: {self._failure_reason}")
            self._pending_session_closes += 1
        try:
            self._mailbox.put(_BatchSessionCloseMessage(context, reason, future), timeout=timeout)
        except queue.Full as exc:
            with self._idle:
                self._pending_session_closes -= 1
                self._idle.notify_all()
            raise TimeoutError("Timed out submitting stage session cleanup") from exc
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        except FutureTimeoutError as exc:
            raise TimeoutError("Timed out waiting for stage session cleanup") from exc

    def close(self) -> None:
        with self._idle:
            if self._closed:
                return
            self._closed = True
        self._mailbox.put(None)
        self._thread.join()

    def _next_message(
        self,
        timeout: float | None = None,
    ) -> _BatchInvocationMessage | _BatchSessionCloseMessage | None:
        if self._backlog:
            return self._backlog.popleft()
        return self._mailbox.get(timeout=timeout)

    def _run(self) -> None:
        try:
            while True:
                message = self._next_message()
                if message is None:
                    return
                if isinstance(message, _BatchSessionCloseMessage):
                    self._run_close(message)
                    continue
                batch = self._collect_batch(message)
                self._run_batch(batch)
        except BaseException as exc:
            self._fail(exc)

    def _collect_batch(self, first: _BatchInvocationMessage) -> list[_BatchInvocationMessage]:
        batch = [first]
        key = self._batch_key(first.invocation)
        sessions = {first.invocation.key.session_id}
        deadline = time.monotonic() + self._batching_window_seconds
        while len(batch) < self._max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                candidate = self._next_message(timeout=remaining)
            except queue.Empty:
                break
            if candidate is None or isinstance(candidate, _BatchSessionCloseMessage):
                self._backlog.appendleft(candidate)
                break
            candidate_session = candidate.invocation.key.session_id
            if self._batch_key(candidate.invocation) != key or candidate_session in sessions:
                self._backlog.append(candidate)
                break
            batch.append(candidate)
            sessions.add(candidate_session)
        return batch

    def _run_batch(self, batch: Sequence[_BatchInvocationMessage]) -> None:
        runnable = [message for message in batch if message.future.set_running_or_notify_cancel()]
        if not runnable:
            self._finish_invocations(len(batch))
            return
        try:
            results = list(self._batch_handler([message.invocation for message in runnable]))
            if len(results) != len(runnable):
                raise RuntimeError("Batched stage handler returned the wrong number of results")
        except BaseException as exc:
            for message in runnable:
                message.future.set_exception(exc)
        else:
            for message, result in zip(runnable, results):
                message.future.set_result(result)
            with self._idle:
                self._batch_count += 1
                self._batch_item_count += len(runnable)
                self._max_observed_batch_size = max(self._max_observed_batch_size, len(runnable))
        finally:
            self._finish_invocations(len(batch))

    def _run_close(self, message: _BatchSessionCloseMessage) -> None:
        try:
            if self._session_closer is not None:
                self._session_closer(message.context, message.reason)
        except BaseException as exc:
            message.future.set_exception(exc)
        else:
            message.future.set_result(None)
        finally:
            with self._idle:
                self._pending_session_closes -= 1
                self._idle.notify_all()

    def _finish_invocations(self, count: int) -> None:
        with self._idle:
            self._pending_invocations -= count
            self._idle.notify_all()

    def _fail(self, exc: BaseException) -> None:
        failure_reason = f"{type(exc).__name__}: {exc}"
        with self._idle:
            self._failure_reason = failure_reason
            pending = list(self._backlog)
            self._backlog.clear()
            while True:
                try:
                    pending.append(self._mailbox.get_nowait())
                except queue.Empty:
                    break
            for message in pending:
                if message is None or message.future.done():
                    continue
                try:
                    message.future.set_exception(RuntimeError(f"Stage actor failed: {failure_reason}"))
                except InvalidStateError:
                    pass
                if isinstance(message, _BatchSessionCloseMessage):
                    self._pending_session_closes -= 1
                else:
                    self._pending_invocations -= 1
            self._idle.notify_all()
