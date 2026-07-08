"""Pool of pipeline replicas in separate subprocesses.

Each replica has exclusive GPU access via CUDA_VISIBLE_DEVICES. The pool
is duck-type compatible with PipelineService for use in MediaGenerationService.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp_stdlib
import os
import threading
from contextlib import asynccontextmanager
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
            if handle._dead:
                self._evict_replica(idx, "pre-existing dead state")
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
        for h in self._handles:
            h.shutdown()
        self._handles.clear()
        self._instance_status.clear()
        self.is_running = False

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
