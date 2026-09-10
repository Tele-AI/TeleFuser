"""Bounded latest-wins scheduling for action chunk inference."""

from __future__ import annotations

import asyncio
import math
import operator
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(slots=True)
class _ScheduledAction:
    request: Mapping[str, Any]
    session_key: str
    generation: int
    received_at: float
    deadline_at: float | None
    future: asyncio.Future[dict[str, Any]]


class ActionChunkScheduler:
    """Serialize inference while retaining only the newest pending chunk per session."""

    def __init__(
        self,
        infer: Callable[[Mapping[str, Any]], dict[str, Any]],
        *,
        max_pending_sessions: int = 32,
    ) -> None:
        if max_pending_sessions < 1:
            raise ValueError("max_pending_sessions must be positive")
        self._infer = infer
        self._max_pending_sessions = max_pending_sessions
        self._pending: OrderedDict[str, _ScheduledAction] = OrderedDict()
        self._latest_generation: dict[str, int] = {}
        self._latest_sequence: dict[str, int] = {}
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def metadata(self) -> dict[str, Any]:
        """Describe the additive scheduling controls accepted by the endpoint."""
        return {
            "scheduling": {
                "mode": "latest_wins",
                "max_pending_per_session": 1,
                "max_pending_sessions": self._max_pending_sessions,
                "sequence_field": "sequence_id",
                "ttl_field": "request_ttl_ms",
                "inflight_cancellation": False,
            }
        }

    async def start(self) -> None:
        """Start the single inference worker on the current event loop."""
        if self._worker is not None:
            return
        if self._closed:
            raise RuntimeError("action scheduler is closed")
        self._worker = asyncio.create_task(self._run(), name="vla-action-chunk-scheduler")

    def submit(self, request: Mapping[str, Any], *, session_key: str) -> asyncio.Future[dict[str, Any]]:
        """Admit one request and return a future without waiting for inference."""
        if self._worker is None or self._closed:
            raise RuntimeError("action scheduler is not running")
        if not session_key:
            raise ValueError("session_key must be non-empty")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        received_at = time.monotonic()
        ttl_ms = self._optional_ttl_ms(request)
        sequence_id = self._optional_sequence_id(request)
        previous = self._pending.get(session_key)
        if previous is None and len(self._pending) >= self._max_pending_sessions:
            future.set_result(
                self._discarded_response(
                    request,
                    status="overloaded",
                    message="the action scheduler has no free pending-session slot",
                )
            )
            return future
        if sequence_id is not None:
            latest_sequence = self._latest_sequence.get(session_key)
            if latest_sequence is not None and sequence_id <= latest_sequence:
                future.set_result(
                    self._discarded_response(
                        request,
                        status="stale_sequence",
                        message=f"sequence_id={sequence_id} is not newer than {latest_sequence}",
                    )
                )
                return future
            self._latest_sequence[session_key] = sequence_id

        generation = self._latest_generation.get(session_key, 0) + 1
        self._latest_generation[session_key] = generation
        deadline_at = None if ttl_ms is None else received_at + ttl_ms / 1000.0
        job = _ScheduledAction(
            request=dict(request),
            session_key=session_key,
            generation=generation,
            received_at=received_at,
            deadline_at=deadline_at,
            future=future,
        )
        if previous is not None:
            self._resolve(
                previous,
                self._discarded_response(
                    previous.request,
                    status="superseded",
                    message="a newer observation replaced this pending action request",
                ),
            )
        self._pending[session_key] = job
        self._wake.set()
        return future

    def release_session(self, session_key: str) -> None:
        """Discard pending work and sequence state for a disconnected session."""
        pending = self._pending.pop(session_key, None)
        if pending is not None:
            self._resolve(
                pending,
                self._discarded_response(
                    pending.request,
                    status="session_closed",
                    message="the client session closed before inference",
                ),
            )
        self._latest_generation.pop(session_key, None)
        self._latest_sequence.pop(session_key, None)

    async def close(self) -> None:
        """Drain an in-flight call and reject work that has not started."""
        if self._closed:
            return
        self._closed = True
        for job in self._pending.values():
            self._resolve(
                job,
                self._discarded_response(
                    job.request,
                    status="server_stopping",
                    message="the action scheduler is stopping",
                ),
            )
        self._pending.clear()
        self._wake.set()
        if self._worker is not None:
            await self._worker
            self._worker = None

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            if self._closed and not self._pending:
                return
            if not self._pending:
                self._wake.clear()
                continue

            _, job = self._pending.popitem(last=False)
            if not self._pending:
                self._wake.clear()
            if job.future.done():
                continue
            discard = self._discard_reason(job)
            if discard is not None:
                self._resolve(job, discard)
                continue

            inference_started_at = time.monotonic()
            try:
                response = await asyncio.to_thread(self._infer, job.request)
            except Exception as error:
                if not job.future.done():
                    job.future.set_exception(error)
                continue
            inference_ms = (time.monotonic() - inference_started_at) * 1000.0

            discard = self._discard_reason(job)
            if discard is not None:
                self._resolve(job, discard)
                continue
            completed_at = time.monotonic()
            response = dict(response)
            server_timing = dict(response.get("server_timing", {}))
            server_timing.update(
                queue_wait_ms=(inference_started_at - job.received_at) * 1000.0,
                infer_ms=inference_ms,
                scheduler_total_ms=(completed_at - job.received_at) * 1000.0,
            )
            response.update(scheduler_status="completed", server_timing=server_timing)
            for field in ("request_id", "episode_id", "sequence_id"):
                if field in job.request:
                    response.setdefault(field, job.request[field])
            if job.deadline_at is not None:
                response["request_ttl_ms"] = (job.deadline_at - job.received_at) * 1000.0
            self._resolve(job, response)

    def _discard_reason(self, job: _ScheduledAction) -> dict[str, Any] | None:
        if job.deadline_at is not None and time.monotonic() >= job.deadline_at:
            return self._discarded_response(
                job.request,
                status="expired",
                message="the action request exceeded request_ttl_ms",
            )
        if self._latest_generation.get(job.session_key) != job.generation:
            return self._discarded_response(
                job.request,
                status="superseded",
                message="a newer observation superseded this action result",
            )
        return None

    @staticmethod
    def _resolve(job: _ScheduledAction, response: dict[str, Any]) -> None:
        if not job.future.done():
            job.future.set_result(response)

    @staticmethod
    def _optional_sequence_id(request: Mapping[str, Any]) -> int | None:
        value = request.get("sequence_id")
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("sequence_id must be a non-negative integer")
        try:
            sequence_id = operator.index(value)
        except TypeError as error:
            raise ValueError("sequence_id must be a non-negative integer") from error
        if sequence_id < 0:
            raise ValueError("sequence_id must be a non-negative integer")
        return sequence_id

    @staticmethod
    def _optional_ttl_ms(request: Mapping[str, Any]) -> float | None:
        value = request.get("request_ttl_ms")
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("request_ttl_ms must be a positive finite number")
        ttl_ms = float(value)
        if not math.isfinite(ttl_ms) or ttl_ms <= 0:
            raise ValueError("request_ttl_ms must be a positive finite number")
        return ttl_ms

    @staticmethod
    def _discarded_response(
        request: Mapping[str, Any],
        *,
        status: str,
        message: str,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "action": None,
            "scheduler_status": status,
            "error": {"code": status, "message": message},
        }
        for field in ("request_id", "episode_id", "sequence_id"):
            if field in request:
                response[field] = request[field]
        return response
