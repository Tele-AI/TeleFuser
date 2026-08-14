"""Process-isolated LiveKit model workers."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import uuid
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.process import BaseProcess
from typing import Any

from telefuser.service.security.security_validator import SecurityLevel
from telefuser.utils.logging import logger

from .config import LiveKitServeConfig
from .session_registry import SessionRecord
from .worker import WorkerEventSink

_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 60.0
_WORKER_START_TIMEOUT_SECONDS = 600.0
_COMMAND_TIMEOUT_SECONDS = 15.0
_PROCESS_JOIN_TIMEOUT_SECONDS = 10.0
_PROCESS_MONITOR_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True)
class ProcessWorkerSpec:
    """Serializable configuration for one isolated model worker."""

    worker_id: str
    gpu_ids: list[str]


@dataclass
class _ProcessHandle:
    spec: ProcessWorkerSpec
    commands: Any
    process: BaseProcess


class ProcessLiveKitWorkerPool:
    """Run one model replica per spawned process and keep the API process model-free."""

    def __init__(
        self,
        specs: list[ProcessWorkerSpec],
        *,
        config: LiveKitServeConfig,
        pipeline_file: str,
        event_sink: WorkerEventSink,
        security_level: SecurityLevel | None = None,
        initial_workers: int | None = None,
        context: BaseContext | None = None,
        worker_target: Any = None,
    ) -> None:
        if not specs:
            raise ValueError("Process worker pool requires at least one worker")
        if initial_workers is not None and not 1 <= initial_workers <= len(specs):
            raise ValueError("initial_workers must be within the configured worker pool")
        self._specs = {spec.worker_id: spec for spec in specs}
        self._config = config
        self._pipeline_file = pipeline_file
        self._event_sink = event_sink
        self._security_level = security_level
        self._initial_workers = initial_workers
        self._context = context or multiprocessing.get_context("spawn")
        self._worker_target = worker_target or _process_worker_main
        self._events = self._context.Queue()
        self._handles: dict[str, _ProcessHandle] = {}
        self._active_workers: set[str] = set()
        self._stopping_workers: set[str] = set()
        self._session_workers: dict[str, str] = {}
        self._pipeline_routes: dict[str, str] = {}
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._pending_workers: dict[str, str] = {}
        self._startup: dict[str, asyncio.Future[None]] = {}
        self._event_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._scale_lock = asyncio.Lock()
        self._started = False
        self._closing = False
        self._skip_validation = False

    async def start(self, *, skip_validation: bool = False) -> None:
        """Spawn and wait for the configured initial replica set."""
        if self._started:
            return
        self._started = True
        self._skip_validation = skip_validation
        self._event_task = asyncio.create_task(self._event_loop(), name="livekit-process-events")
        self._monitor_task = asyncio.create_task(self._monitor_loop(), name="livekit-process-monitor")
        target = self._initial_workers or len(self._specs)
        try:
            await self.scale_to(target)
        except Exception:
            await self.aclose()
            raise
        for worker_id in self._specs:
            if worker_id not in self._active_workers:
                self._event_sink.on_worker_status(worker_id, "stopped")

    def start_session(self, record: SessionRecord) -> None:
        """Submit one retained LiveKit session to its owning process."""
        if not self._started or self._closing:
            raise RuntimeError("LiveKit process worker pool is not accepting sessions")
        if record.worker_id is None:
            raise RuntimeError(f"Session {record.session_id} has no assigned worker")
        if record.worker_id not in self._active_workers:
            raise RuntimeError(f"Worker {record.worker_id} is not active")
        if record.session_id in self._session_workers:
            raise RuntimeError(f"Session {record.session_id} is already running")
        self._session_workers[record.session_id] = record.worker_id
        try:
            self._send(
                record.worker_id,
                {
                    "type": "start_session",
                    "record": record.model_dump(mode="python"),
                },
            )
        except Exception:
            self._session_workers.pop(record.session_id, None)
            raise

    async def stop_session(self, session_id: str) -> None:
        """Stop a child-owned room and wait for model-state cleanup."""
        worker_id = self._session_workers.get(session_id)
        if worker_id is None or worker_id not in self._active_workers:
            return
        await self._request(worker_id, "stop_session", session_id=session_id)

    async def scale_to(self, target_workers: int) -> int:
        """Spawn replicas or retire idle replicas until the target is reached."""
        if not self._started:
            raise RuntimeError("LiveKit process worker pool is not started")
        if not 1 <= target_workers <= len(self._specs):
            raise ValueError("target_workers must be within the configured worker pool")
        async with self._scale_lock:
            while len(self._active_workers) < target_workers:
                worker_id = next(worker_id for worker_id in self._specs if worker_id not in self._active_workers)
                await self._start_worker(worker_id)
            while len(self._active_workers) > target_workers:
                candidate = self._scale_in_candidate()
                if candidate is None:
                    break
                await self._stop_worker(candidate)
            return len(self._active_workers)

    def active_worker_count(self) -> int:
        return len(self._active_workers)

    def turboserve_snapshot(self) -> dict[str, object]:
        retained = {worker_id: 0 for worker_id in self._specs}
        for worker_id in self._session_workers.values():
            retained[worker_id] += 1
        return {
            "routes": dict(self._pipeline_routes),
            "retained_sessions_by_worker": retained,
            "active_workers": sorted(self._active_workers),
            "configured_workers": len(self._specs),
            "migration_supported": False,
        }

    async def aclose(self) -> None:
        """Stop children, terminate unresponsive processes, and close IPC resources."""
        if self._closing:
            return
        self._closing = True
        for worker_id in tuple(self._active_workers):
            with contextlib.suppress(Exception):
                await self._stop_worker(worker_id)
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task
        if self._event_task is not None:
            self._events.put({"type": "pool_stop"})
            with contextlib.suppress(asyncio.CancelledError):
                await self._event_task
        self._monitor_task = None
        self._event_task = None
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("LiveKit process worker pool closed"))
        self._pending.clear()
        self._pending_workers.clear()
        self._startup.clear()
        _close_queue(self._events)
        self._started = False
        self._closing = False

    async def _start_worker(self, worker_id: str) -> None:
        spec = self._specs[worker_id]
        commands = self._context.Queue()
        security_name = self._security_level.name if self._security_level is not None else None
        process = self._context.Process(
            target=self._worker_target,
            name=f"telefuser-{worker_id}",
            args=(
                spec,
                self._config.model_dump(mode="python"),
                self._pipeline_file,
                self._skip_validation,
                security_name,
                commands,
                self._events,
            ),
        )
        future = asyncio.get_running_loop().create_future()
        self._startup[worker_id] = future
        self._handles[worker_id] = _ProcessHandle(spec, commands, process)
        try:
            process.start()
        except Exception:
            self._startup.pop(worker_id, None)
            self._terminate_handle(worker_id)
            raise
        try:
            await asyncio.wait_for(future, timeout=_WORKER_START_TIMEOUT_SECONDS)
        except Exception:
            self._terminate_handle(worker_id)
            raise
        finally:
            self._startup.pop(worker_id, None)
        self._active_workers.add(worker_id)

    async def _stop_worker(self, worker_id: str) -> None:
        if worker_id not in self._handles:
            return
        self._stopping_workers.add(worker_id)
        try:
            try:
                await self._request(worker_id, "shutdown", timeout=_WORKER_SHUTDOWN_TIMEOUT_SECONDS)
            except Exception as exc:
                logger.warning(f"Process worker did not shut down cleanly: worker={worker_id} error={exc}")
            handle = self._handles.get(worker_id)
            if handle is not None:
                await asyncio.to_thread(handle.process.join, _PROCESS_JOIN_TIMEOUT_SECONDS)
                if handle.process.is_alive():
                    self._terminate_handle(worker_id)
                else:
                    self._discard_handle(worker_id)
        finally:
            self._active_workers.discard(worker_id)
            self._stopping_workers.discard(worker_id)

    async def _request(
        self,
        worker_id: str,
        command_type: str,
        *,
        timeout: float = _COMMAND_TIMEOUT_SECONDS,
        **payload: Any,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._pending_workers[request_id] = worker_id
        try:
            self._send(worker_id, {"type": command_type, "request_id": request_id, **payload})
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending.pop(request_id, None)
            self._pending_workers.pop(request_id, None)

    def _send(self, worker_id: str, command: dict[str, Any]) -> None:
        handle = self._handles.get(worker_id)
        if handle is None or not handle.process.is_alive():
            raise RuntimeError(f"Worker process {worker_id} is not alive")
        handle.commands.put(command)

    async def _event_loop(self) -> None:
        while True:
            event = await asyncio.to_thread(self._events.get)
            if event.get("type") == "pool_stop":
                return
            self._dispatch_event(event)

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        worker_id = event.get("worker_id")
        if event_type == "worker_ready":
            future = self._startup.get(worker_id)
            if future is not None and not future.done():
                future.set_result(None)
            return
        if event_type == "worker_start_failed":
            error = RuntimeError(str(event.get("error", "worker startup failed")))
            future = self._startup.get(worker_id)
            if future is not None and not future.done():
                future.set_exception(error)
            self._event_sink.on_worker_status(worker_id, "failed")
            return
        if event_type == "command_result":
            future = self._pending.get(event.get("request_id"))
            if future is not None and not future.done():
                if event.get("error") is not None:
                    future.set_exception(RuntimeError(str(event["error"])))
                else:
                    future.set_result(event)
            return
        if event_type == "worker_status":
            self._event_sink.on_worker_status(worker_id, event["status"])
        elif event_type == "worker_capacity":
            self._event_sink.on_worker_capacity(worker_id, int(event["capacity"]), event.get("profile"))
        elif event_type == "session_status":
            self._event_sink.on_session_status(event["session_id"], event["status"], event.get("error"))
        elif event_type == "pipeline_session":
            session_id = event["session_id"]
            pipeline_session_id = event["pipeline_session_id"]
            self._pipeline_routes[pipeline_session_id] = worker_id
            self._event_sink.on_pipeline_session(session_id, pipeline_session_id)
        elif event_type == "session_finished":
            session_id = event["session_id"]
            self._session_workers.pop(session_id, None)
            for pipeline_session_id, owner in tuple(self._pipeline_routes.items()):
                if owner == worker_id and pipeline_session_id == event.get("pipeline_session_id"):
                    self._pipeline_routes.pop(pipeline_session_id, None)
            self._event_sink.on_session_finished(worker_id, session_id, event.get("error"))
        elif event_type == "control_received":
            callback = getattr(self._event_sink, "on_control_received", None)
            if callable(callback):
                callback(worker_id, event["session_id"])
        elif event_type == "chunk_published":
            callback = getattr(self._event_sink, "on_chunk_published", None)
            if callable(callback):
                callback(
                    worker_id,
                    event["session_id"],
                    int(event.get("frames", 0)),
                    event.get("first_frame_at"),
                )
        elif event_type == "model_output":
            callback = getattr(self._event_sink, "on_model_output", None)
            if callable(callback):
                callback(
                    worker_id,
                    event["session_id"],
                    event["payload"],
                    runtime_metrics=event.get("runtime_metrics"),
                    session_runtime_metrics=event.get("session_runtime_metrics"),
                )

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(_PROCESS_MONITOR_INTERVAL_SECONDS)
            for worker_id, handle in tuple(self._handles.items()):
                if handle.process.is_alive():
                    continue
                startup = self._startup.get(worker_id)
                if startup is not None:
                    if not startup.done():
                        startup.set_exception(
                            RuntimeError(f"Worker process exited during startup with code {handle.process.exitcode}")
                        )
                    self._discard_handle(worker_id)
                    continue
                if worker_id in self._stopping_workers:
                    error = None
                    if handle.process.exitcode != 0:
                        error = f"Worker process exited during shutdown with code {handle.process.exitcode}"
                    self._resolve_pending_requests(worker_id, error)
                    self._discard_handle(worker_id)
                    continue
                if worker_id not in self._active_workers:
                    continue
                self._handle_unexpected_exit(worker_id, handle.process.exitcode)

    def _handle_unexpected_exit(self, worker_id: str, exitcode: int | None) -> None:
        self._active_workers.discard(worker_id)
        self._event_sink.on_worker_status(worker_id, "failed")
        error = f"Worker process exited unexpectedly with code {exitcode}"
        self._resolve_pending_requests(worker_id, error)
        for session_id, owner in tuple(self._session_workers.items()):
            if owner == worker_id:
                self._session_workers.pop(session_id, None)
                self._event_sink.on_session_finished(worker_id, session_id, error)
        for pipeline_session_id, owner in tuple(self._pipeline_routes.items()):
            if owner == worker_id:
                self._pipeline_routes.pop(pipeline_session_id, None)
        self._discard_handle(worker_id)

    def _scale_in_candidate(self) -> str | None:
        busy = set(self._session_workers.values())
        return next(
            (
                worker_id
                for worker_id in reversed(tuple(self._specs))
                if worker_id in self._active_workers and worker_id not in busy
            ),
            None,
        )

    def _resolve_pending_requests(self, worker_id: str, error: str | None) -> None:
        for request_id, owner in tuple(self._pending_workers.items()):
            if owner != worker_id:
                continue
            future = self._pending.get(request_id)
            if future is None or future.done():
                continue
            if error is None:
                future.set_result({"type": "command_result", "worker_id": worker_id})
            else:
                future.set_exception(RuntimeError(error))

    def _terminate_handle(self, worker_id: str) -> None:
        handle = self._handles.get(worker_id)
        if handle is None:
            return
        if handle.process.is_alive():
            handle.process.terminate()
            handle.process.join(_PROCESS_JOIN_TIMEOUT_SECONDS)
            if handle.process.is_alive() and hasattr(handle.process, "kill"):
                handle.process.kill()
                handle.process.join(_PROCESS_JOIN_TIMEOUT_SECONDS)
        self._discard_handle(worker_id)

    def _discard_handle(self, worker_id: str) -> None:
        handle = self._handles.pop(worker_id, None)
        if handle is not None:
            _close_queue(handle.commands, join=False)


def _process_worker_main(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    """Child entrypoint; imports model-facing modules only after process spawn."""
    try:
        asyncio.run(
            _run_process_worker(
                spec,
                config_values,
                pipeline_file,
                skip_validation,
                security_name,
                commands,
                events,
            )
        )
    except BaseException as exc:
        events.put({"type": "worker_start_failed", "worker_id": spec.worker_id, "error": repr(exc)})
        raise
    finally:
        _close_queue(commands, join=False)
        _close_queue(events)


def _close_queue(ipc_queue: Any, *, join: bool = True) -> None:
    """Close one multiprocessing queue and wait for its local feeder thread."""
    if not join:
        with contextlib.suppress(Exception):
            ipc_queue.cancel_join_thread()
    with contextlib.suppress(Exception):
        ipc_queue.close()
    if join:
        with contextlib.suppress(Exception):
            ipc_queue.join_thread()


async def _run_process_worker(
    spec: ProcessWorkerSpec,
    config_values: dict[str, Any],
    pipeline_file: str,
    skip_validation: bool,
    security_name: str | None,
    commands: Any,
    events: Any,
) -> None:
    from .multi_session_worker import MultiSessionLiveKitWorker
    from .pipeline_adapter import LiveKitPipelineAdapter
    from .token_service import LiveKitTokenService

    config = LiveKitServeConfig(**config_values)
    security_level = SecurityLevel[security_name] if security_name is not None else None
    sink = _ProcessEventSink(spec.worker_id, events)
    token_service = LiveKitTokenService(
        api_key=config.livekit_api_key,
        api_secret=config.livekit_api_secret,
        token_ttl=config.token_ttl,
    )
    worker = MultiSessionLiveKitWorker(
        worker_id=spec.worker_id,
        config=config,
        pipeline_file=pipeline_file,
        token_service=token_service,
        event_sink=sink,
        pipeline_adapter=LiveKitPipelineAdapter(security_level=security_level),
        gpu_num=max(1, len(spec.gpu_ids)),
        gpu_ids=spec.gpu_ids or None,
    )
    tasks: dict[str, asyncio.Task[None]] = {}
    await worker.start(skip_validation=skip_validation)
    events.put({"type": "worker_ready", "worker_id": spec.worker_id})
    try:
        while True:
            command = await asyncio.to_thread(commands.get)
            command_type = command["type"]
            request_id = command.get("request_id")
            try:
                if command_type == "start_session":
                    record = SessionRecord.model_validate(command["record"])
                    task = asyncio.create_task(worker.run_session(record), name=f"room-{record.session_id}")
                    tasks[record.session_id] = task
                    task.add_done_callback(
                        lambda done, sid=record.session_id: _consume_session_task(
                            tasks,
                            sid,
                            done,
                            worker_id=spec.worker_id,
                            events=events,
                        )
                    )
                elif command_type == "stop_session":
                    session_id = command["session_id"]
                    await worker.stop_session(session_id)
                    task = tasks.get(session_id)
                    if task is not None:
                        try:
                            await asyncio.wait_for(asyncio.shield(task), timeout=_COMMAND_TIMEOUT_SECONDS)
                        except asyncio.TimeoutError:
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task
                elif command_type == "shutdown":
                    for session_id in tuple(tasks):
                        await worker.stop_session(session_id)
                    pending = set()
                    if tasks:
                        _, pending = await asyncio.wait(tasks.values(), timeout=_COMMAND_TIMEOUT_SECONDS)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    await worker.stop()
                else:
                    raise ValueError(f"Unknown process-worker command {command_type!r}")
            except Exception as exc:
                if request_id is not None:
                    events.put(
                        {
                            "type": "command_result",
                            "worker_id": spec.worker_id,
                            "request_id": request_id,
                            "error": repr(exc),
                        }
                    )
                if command_type == "shutdown":
                    raise
            else:
                if request_id is not None:
                    events.put(
                        {
                            "type": "command_result",
                            "worker_id": spec.worker_id,
                            "request_id": request_id,
                        }
                    )
                if command_type == "shutdown":
                    return
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        with contextlib.suppress(Exception):
            await worker.stop()


def _consume_session_task(
    tasks: dict[str, asyncio.Task[None]],
    session_id: str,
    task: asyncio.Task[None],
    *,
    worker_id: str,
    events: Any,
) -> None:
    tasks.pop(session_id, None)
    if task.cancelled():
        return
    error = task.exception()
    if error is None:
        return
    events.put(
        {
            "type": "session_status",
            "worker_id": worker_id,
            "session_id": session_id,
            "status": "failed",
            "error": repr(error),
        }
    )
    events.put(
        {
            "type": "session_finished",
            "worker_id": worker_id,
            "session_id": session_id,
            "pipeline_session_id": None,
            "error": repr(error),
        }
    )


class _ProcessEventSink:
    """Serialize child lifecycle callbacks onto the parent event queue."""

    def __init__(self, worker_id: str, events: Any) -> None:
        self.worker_id = worker_id
        self.events = events
        self._pipeline_sessions: dict[str, str] = {}

    def on_worker_status(self, worker_id: str, status: str) -> None:
        self.events.put({"type": "worker_status", "worker_id": worker_id, "status": status})

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        self.events.put(
            {
                "type": "worker_capacity",
                "worker_id": worker_id,
                "capacity": capacity,
                "profile": profile,
            }
        )

    def on_session_status(self, session_id: str, status: str, error: str | None = None) -> None:
        self.events.put(
            {
                "type": "session_status",
                "worker_id": self.worker_id,
                "session_id": session_id,
                "status": status,
                "error": error,
            }
        )

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        self._pipeline_sessions[session_id] = pipeline_session_id
        self.events.put(
            {
                "type": "pipeline_session",
                "worker_id": self.worker_id,
                "session_id": session_id,
                "pipeline_session_id": pipeline_session_id,
            }
        )

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        self.events.put(
            {
                "type": "session_finished",
                "worker_id": worker_id,
                "session_id": session_id,
                "pipeline_session_id": self._pipeline_sessions.pop(session_id, None),
                "error": error,
            }
        )

    def on_control_received(self, worker_id: str, session_id: str) -> None:
        self.events.put(
            {"type": "control_received", "worker_id": worker_id, "session_id": session_id}
        )

    def on_chunk_published(
        self, worker_id: str, session_id: str, frames: int, first_frame_at: float | None = None
    ) -> None:
        self.events.put(
            {
                "type": "chunk_published",
                "worker_id": worker_id,
                "session_id": session_id,
                "frames": frames,
                "first_frame_at": first_frame_at,
            }
        )

    def on_model_output(
        self,
        worker_id: str,
        session_id: str,
        payload: dict,
        runtime_metrics: dict | None = None,
        session_runtime_metrics: dict | None = None,
    ) -> None:
        self.events.put(
            {
                "type": "model_output",
                "worker_id": worker_id,
                "session_id": session_id,
                "payload": payload,
                "runtime_metrics": runtime_metrics,
                "session_runtime_metrics": session_runtime_metrics,
            }
        )
