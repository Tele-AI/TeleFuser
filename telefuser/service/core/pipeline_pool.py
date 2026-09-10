"""Pool of pipeline replicas in separate subprocesses.

Each replica has exclusive GPU access via CUDA_VISIBLE_DEVICES. The pool
is duck-type compatible with PipelineService for use in MediaGenerationService.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp_stdlib
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger

from .replica_worker import (
    _STARTUP_TIMEOUT_S,
    ReplicaDeadError,
    ReplicaHandle,
    _replica_main,
)

_spawn_ctx = mp_stdlib.get_context("spawn")

if TYPE_CHECKING:
    from .config import ServerConfig
    from .task_manager import TaskManager


@dataclass
class _ReplicaSessionLease:
    replica_id: int
    lock: asyncio.Lock
    closing: bool = False


class PipelinePool:
    """Pool of pipeline replicas in separate subprocesses.

    Holds a reference to TaskManager to dynamically adjust claim capacity
    when replicas die.
    """

    def __init__(
        self,
        num_replicas: int,
        replica_device_ids: list[list[str]],
        security_level_name: str,
        config: ServerConfig | None = None,
        task_manager: TaskManager | None = None,
    ) -> None:
        self._num_replicas = num_replicas
        self._replica_device_ids = replica_device_ids
        self._security_level_name = security_level_name
        self._server_config_data = config.model_dump(mode="json") if config is not None else None
        self._task_manager = task_manager
        self._handles: list[ReplicaHandle] = []
        self._available: asyncio.Queue[int] = asyncio.Queue()
        self._instance_status: list[str] = []
        self._live_count = num_replicas
        self._status_lock = threading.Lock()
        self._session_leases: dict[str, _ReplicaSessionLease] = {}
        self._opening_sessions: set[str] = set()
        self._session_leases_lock = asyncio.Lock()
        self._session_shutdown = False
        self._session_shutdown_event = asyncio.Event()
        self.is_running = False

        self._cached_server_metadata: dict[str, Any] = {}
        self._cached_supported_tasks: tuple[str, ...] = ()
        self._cached_task_contracts: dict[str, Any] = {}

    def start_all(
        self,
        ppl_file: str,
        parallelism_per_replica: int,
        task: str,
        skip_validation: bool,
    ) -> bool:
        """Start all replica subprocesses. Returns True on success."""
        device_env_var = current_platform.device_control_env_var
        original_cvd = os.environ.get(device_env_var)

        try:
            return self._start_all_replicas(ppl_file, parallelism_per_replica, task, skip_validation, device_env_var)
        finally:
            self._restore_cvd(device_env_var, original_cvd)

    def _start_all_replicas(
        self,
        ppl_file: str,
        parallelism_per_replica: int,
        task: str,
        skip_validation: bool,
        device_env_var: str,
    ) -> bool:
        for i, device_ids in enumerate(self._replica_device_ids):
            visible_devices = ",".join(device_ids)
            logger.info(f"Starting replica {i}/{self._num_replicas}: CVD={visible_devices}")

            os.environ[device_env_var] = visible_devices

            parent_conn, child_conn = _spawn_ctx.Pipe()
            cancel_event = _spawn_ctx.Event()

            p = _spawn_ctx.Process(
                target=_replica_main,
                args=(
                    i,
                    ppl_file,
                    parallelism_per_replica,
                    task,
                    visible_devices,
                    device_env_var,
                    child_conn,
                    cancel_event,
                    self._security_level_name,
                    skip_validation,
                    self._server_config_data,
                ),
                daemon=False,
            )
            p.start()
            child_conn.close()

            if not parent_conn.poll(_STARTUP_TIMEOUT_S):
                if not p.is_alive():
                    logger.error(f"Replica {i} process died during startup (exit code {p.exitcode})")
                else:
                    logger.error(f"Replica {i} startup timed out after {_STARTUP_TIMEOUT_S}s")
                    p.terminate()
                self._cleanup_started()
                return False

            try:
                tag, payload = parent_conn.recv()
            except Exception as e:
                logger.error(f"Replica {i} startup communication failed: {e}")
                p.terminate()
                self._cleanup_started()
                return False

            if tag != "ready":
                logger.error(f"Replica {i} startup failed: {payload}")
                p.terminate()
                self._cleanup_started()
                return False

            if i == 0:
                self._cached_server_metadata = payload.get("server_metadata", {})
                self._cached_supported_tasks = tuple(payload.get("supported_tasks", []))
                self._cached_task_contracts = payload.get("task_contracts", {})

            handle = ReplicaHandle(
                replica_id=i,
                process=p,
                conn=parent_conn,
                cancel_event=cancel_event,
                metadata=payload,
            )
            self._handles.append(handle)
            self._instance_status.append("idle")
            self._available.put_nowait(i)

        self.is_running = True
        self._session_shutdown = False
        self._session_shutdown_event.clear()
        logger.info(f"Pipeline pool started: num_replicas={self._num_replicas}")
        return True

    def _cleanup_started(self) -> None:
        """Cleanup already-started replicas on failure."""
        for h in self._handles:
            h.shutdown()
        self._handles.clear()
        self._instance_status.clear()

    @staticmethod
    def _restore_cvd(env_var: str, original: str | None) -> None:
        if original is not None:
            os.environ[env_var] = original
        else:
            os.environ.pop(env_var, None)

    def _evict_replica(self, idx: int, reason: str) -> None:
        """Evict a dead replica: mark status, shutdown, shrink capacity."""
        with self._status_lock:
            if self._instance_status[idx] == "dead":
                return
            self._instance_status[idx] = "dead"
            self._live_count -= 1
            live = self._live_count
        self._handles[idx].shutdown()
        logger.error(f"Replica {idx} evicted ({reason}). Live replicas: {live}/{self._num_replicas}")

        if self._task_manager is not None:
            self._task_manager.set_max_concurrent_processing(live)

        if live == 0:
            logger.critical("All replicas dead. No inference capacity remaining.")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[ReplicaHandle]:
        """Acquire an idle replica (blocks until available).

        Dead replicas are evicted and the next available one is tried.
        Raises RuntimeError if all replicas are dead.
        """
        while True:
            with self._status_lock:
                if self._live_count == 0:
                    raise RuntimeError("All pipeline replicas are dead; no inference capacity")

            try:
                idx = await asyncio.wait_for(self._available.get(), timeout=5.0)
            except asyncio.TimeoutError:
                with self._status_lock:
                    if self._live_count == 0:
                        raise RuntimeError("All pipeline replicas are dead; no inference capacity")
                continue

            handle = self._handles[idx]
            process = getattr(handle, "process", None)
            process_dead = process is not None and not process.is_alive()
            if handle._dead or process_dead:
                reason = "process exited" if process_dead else "pre-existing dead state"
                self._evict_replica(idx, reason)
                continue
            break

        with self._status_lock:
            self._instance_status[idx] = "busy"
        try:
            yield handle
        except ReplicaDeadError:
            self._evict_replica(idx, "died during task execution")
            raise
        else:
            with self._status_lock:
                self._instance_status[idx] = "idle"
            self._available.put_nowait(idx)

    async def open_session(self, session_id: str, *, timeout_s: float | None = None) -> int:
        """Reserve one healthy replica for a long-lived session."""
        self._validate_session_id(session_id)
        if timeout_s is not None and (
            isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be None or a positive number")
        async with self._session_leases_lock:
            if self._session_shutdown:
                raise RuntimeError("pipeline pool is shutting down")
            if session_id in self._session_leases or session_id in self._opening_sessions:
                raise ValueError(f"pipeline session is already open: {session_id!r}")
            self._opening_sessions.add(session_id)

        idx: int | None = None
        registered = False
        try:
            idx = await self._take_available_replica(timeout_s=None if timeout_s is None else float(timeout_s))
            async with self._session_leases_lock:
                if self._session_shutdown:
                    raise RuntimeError("pipeline pool is shutting down")
                self._session_leases[session_id] = _ReplicaSessionLease(idx, asyncio.Lock())
                self._opening_sessions.discard(session_id)
                registered = True
            with self._status_lock:
                self._instance_status[idx] = "session_reserved"
            return idx
        finally:
            if not registered:
                async with self._session_leases_lock:
                    self._opening_sessions.discard(session_id)
                if idx is not None:
                    self._return_available_replica(idx)

    @asynccontextmanager
    async def acquire_session(self, session_id: str) -> AsyncIterator[ReplicaHandle]:
        """Acquire the replica pinned to a session, serializing its requests."""
        lease = await self._get_session_lease(session_id)
        async with lease.lock:
            async with self._session_leases_lock:
                if self._session_leases.get(session_id) is not lease or lease.closing:
                    raise KeyError(f"unknown or closing pipeline session: {session_id!r}")

            idx = lease.replica_id
            handle = self._handles[idx]
            if self._handle_is_dead(handle):
                async with self._session_leases_lock:
                    self._session_leases.pop(session_id, None)
                self._evict_replica(idx, "session replica process exited")
                raise ReplicaDeadError(f"Replica {idx} assigned to session {session_id!r} is not alive")

            with self._status_lock:
                self._instance_status[idx] = "session_busy"
            try:
                yield handle
            except ReplicaDeadError:
                async with self._session_leases_lock:
                    self._session_leases.pop(session_id, None)
                self._evict_replica(idx, "died during session execution")
                raise
            else:
                with self._status_lock:
                    if self._instance_status[idx] != "dead":
                        self._instance_status[idx] = "session_reserved"

    async def close_session(self, session_id: str) -> int:
        """Release a session lease and return its live replica to the idle pool."""
        lease = await self._get_session_lease(session_id)
        async with self._session_leases_lock:
            if self._session_leases.get(session_id) is not lease or lease.closing:
                raise KeyError(f"unknown or closing pipeline session: {session_id!r}")
            lease.closing = True

        async with lease.lock:
            async with self._session_leases_lock:
                if self._session_leases.get(session_id) is not lease:
                    raise KeyError(f"unknown pipeline session: {session_id!r}")
                self._session_leases.pop(session_id)
            self._return_available_replica(lease.replica_id)
            return lease.replica_id

    async def session_replica_id(self, session_id: str) -> int:
        """Return the stable replica assignment for an open session."""
        return (await self._get_session_lease(session_id)).replica_id

    async def session_bindings(self) -> dict[str, int]:
        """Return a monitoring snapshot of active session-to-replica bindings."""
        async with self._session_leases_lock:
            return {session_id: lease.replica_id for session_id, lease in sorted(self._session_leases.items())}

    async def run_task_with_stop_event(
        self,
        task_data: dict,
        stop_event: threading.Event,
        timeout_s: float | None = None,
        output_root: str | None = None,
    ) -> dict:
        """Duck-type compatible with PipelineService.run_task_with_stop_event."""
        async with self.acquire() as handle:
            return await handle.run_task(task_data, stop_event, timeout_s, output_root)

    async def aclose(self) -> None:
        """Shutdown all replicas."""
        async with self._session_leases_lock:
            self._session_shutdown = True
            self._session_shutdown_event.set()
            leases = tuple(self._session_leases.values())
            for lease in leases:
                lease.closing = True
        for lease in leases:
            async with lease.lock:
                pass
        async with self._session_leases_lock:
            self._session_leases.clear()
            self._opening_sessions.clear()
        for h in self._handles:
            h.shutdown()
        self._handles.clear()
        self._instance_status.clear()
        self.is_running = False

    async def _get_session_lease(self, session_id: str) -> _ReplicaSessionLease:
        self._validate_session_id(session_id)
        async with self._session_leases_lock:
            try:
                lease = self._session_leases[session_id]
            except KeyError as error:
                raise KeyError(f"unknown pipeline session: {session_id!r}") from error
            if lease.closing:
                raise KeyError(f"unknown or closing pipeline session: {session_id!r}")
            return lease

    async def _take_available_replica(self, timeout_s: float | None) -> int:
        loop = asyncio.get_running_loop()
        deadline = None if timeout_s is None else loop.time() + timeout_s
        while True:
            if self._session_shutdown:
                raise RuntimeError("pipeline pool is shutting down")
            with self._status_lock:
                if self._live_count == 0:
                    raise RuntimeError("All pipeline replicas are dead; no inference capacity")
            wait_s = 5.0 if deadline is None else min(5.0, max(0.0, deadline - loop.time()))
            if wait_s == 0:
                raise TimeoutError("timed out waiting for an available pipeline replica")
            available = asyncio.create_task(self._available.get())
            shutdown = asyncio.create_task(self._session_shutdown_event.wait())
            try:
                done, _ = await asyncio.wait(
                    (available, shutdown),
                    timeout=wait_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if shutdown in done:
                    raise RuntimeError("pipeline pool is shutting down")
                if available not in done:
                    if deadline is not None and loop.time() >= deadline:
                        raise TimeoutError("timed out waiting for an available pipeline replica")
                    continue
                idx = available.result()
            finally:
                for task in (available, shutdown):
                    if not task.done():
                        task.cancel()
                for task in (available, shutdown):
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            handle = self._handles[idx]
            if self._handle_is_dead(handle):
                self._evict_replica(idx, "unavailable while opening session")
                continue
            return idx

    def _return_available_replica(self, idx: int) -> None:
        if idx >= len(self._handles):
            return
        handle = self._handles[idx]
        if self._handle_is_dead(handle):
            if idx < len(self._instance_status) and self._instance_status[idx] != "dead":
                self._evict_replica(idx, "session closed after replica exit")
            return
        with self._status_lock:
            self._instance_status[idx] = "idle"
        self._available.put_nowait(idx)

    @staticmethod
    def _handle_is_dead(handle: ReplicaHandle) -> bool:
        process = getattr(handle, "process", None)
        return handle._dead or process is not None and not process.is_alive()

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")

    # --- Metadata proxy (pool overlay) ---

    def server_metadata(self) -> dict:
        """Return server metadata with pool overlay."""
        if not self._cached_server_metadata:
            return {}
        base = dict(self._cached_server_metadata)
        per_instance_mode = base.get("execution_mode", "serial_single_pipeline")
        with self._status_lock:
            live = self._live_count
        base["effective_max_concurrent_tasks"] = live
        base["execution_mode"] = "concurrent_pipeline_pool"
        base["pool"] = {
            "num_replicas": self._num_replicas,
            "live_replicas": live,
            "replica_device_ids": self._replica_device_ids,
            "per_instance_execution_mode": per_instance_mode,
        }
        return base

    def supported_tasks(self) -> tuple[str, ...]:
        """Return tasks declared by the loaded pipeline contract."""
        return self._cached_supported_tasks

    def get_task_contract(self, task: str) -> Any:
        """Return the task-level contract for a declared task, if available."""
        return self._cached_task_contracts.get(task)

    def pool_status(self) -> list[dict]:
        """Return per-replica status for monitoring."""
        with self._status_lock:
            return [
                {
                    "id": i,
                    "device_ids": self._replica_device_ids[i],
                    "status": self._instance_status[i],
                }
                for i in range(self._num_replicas)
            ]
