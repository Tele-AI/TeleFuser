"""Replica subprocess worker and main-process handle for PipelinePool.

Each replica runs in its own subprocess with exclusive GPU access via
CUDA_VISIBLE_DEVICES. The main process communicates with replicas through
mp.Pipe connections using a simple message protocol:
    startup: replica -> main: ("ready", metadata)
    task:    main -> replica: ("task", data, timeout, root)
             replica -> main: ("ok", result) | ("error", msg)
    cancel:  main sets mp.Event, replica checks it
    shutdown: main -> replica: ("shutdown",)
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp_stdlib
import os
import threading
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any

_STARTUP_TIMEOUT_S = 300
_TASK_IPC_MARGIN_S = 30


class ReplicaDeadError(RuntimeError):
    """Raised when a replica is unresponsive or has crashed."""

    pass


class ReplicaVLAError(RuntimeError):
    """Structured application error returned by a live VLA replica."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------


def _replica_main(
    replica_id: int,
    ppl_file: str,
    parallelism: int,
    task: str,
    visible_devices: str,
    device_env_var: str,
    conn: Connection,
    cancel_event: mp_stdlib.Event,
    security_level_name: str,
    skip_validation: bool,
    server_config_data: dict[str, Any] | None = None,
    vla_provider_factory_name: str | None = None,
) -> None:
    """Entry point for a replica subprocess.

    Sets device visibility env var BEFORE any torch import, creates the
    persistent event loop BEFORE start_pipeline() so PipelineRunner's
    asyncio.Lock binds to the correct loop from construction.
    """
    os.environ[device_env_var] = visible_devices

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    from telefuser.service.core.config import ServerConfig
    from telefuser.service.core.pipeline_service import PipelineService
    from telefuser.service.security.security_validator import SecurityLevel
    from telefuser.service_types import TaskType
    from telefuser.utils.logging import logger

    logger.info(f"Replica {replica_id} started: {device_env_var}={visible_devices}, parallelism={parallelism}")

    security_level = SecurityLevel[security_level_name]
    config = ServerConfig.model_validate(server_config_data) if server_config_data is not None else None
    svc = PipelineService(security_level=security_level, config=config)

    try:
        success = svc.start_pipeline(
            ppl_file=ppl_file,
            parallelism=parallelism,
            task=TaskType(task),
            skip_validation=skip_validation,
        )
    except Exception as e:
        conn.send(("error", f"Replica {replica_id} init failed: {e}\n{traceback.format_exc()}"))
        conn.close()
        loop.close()
        return

    if not success:
        conn.send(("error", f"Replica {replica_id} start_pipeline returned False"))
        conn.close()
        loop.close()
        return

    vla_provider = None
    if vla_provider_factory_name is not None:
        try:
            module = svc._module
            if module is None or not hasattr(module, vla_provider_factory_name):
                raise RuntimeError(
                    f"Pipeline file must define optional VLA provider factory {vla_provider_factory_name}(pipeline)"
                )
            factory = getattr(module, vla_provider_factory_name)
            vla_provider = factory(svc.pipeline)
            if not callable(getattr(vla_provider, "dispatch", None)):
                raise TypeError("VLA provider must define dispatch(operation, payload)")
            if not callable(getattr(vla_provider, "metadata", None)):
                raise TypeError("VLA provider must define metadata()")
        except Exception as e:
            conn.send(("error", f"Replica {replica_id} VLA provider init failed: {e}\n{traceback.format_exc()}"))
            loop.run_until_complete(svc.aclose())
            conn.close()
            loop.close()
            return

    try:
        metadata = _collect_metadata(svc, vla_provider=vla_provider)
    except Exception as e:
        conn.send(("error", f"Replica {replica_id} metadata collection failed: {e}\n{traceback.format_exc()}"))
        if vla_provider is not None and callable(getattr(vla_provider, "close", None)):
            vla_provider.close()
        loop.run_until_complete(svc.aclose())
        conn.close()
        loop.close()
        return
    conn.send(("ready", metadata))
    logger.info(f"Replica {replica_id} ready, entering task loop")

    try:
        loop.run_until_complete(_task_loop(replica_id, svc, conn, cancel_event, logger, vla_provider=vla_provider))
    except Exception as e:
        logger.error(f"Replica {replica_id} loop crashed: {e}")
    finally:
        if vla_provider is not None and callable(getattr(vla_provider, "close", None)):
            try:
                vla_provider.close()
            except Exception as e:
                logger.warning(f"Replica {replica_id} VLA provider cleanup failed: {e}")
        loop.run_until_complete(svc.aclose())
        loop.close()
        conn.close()
        logger.info(f"Replica {replica_id} exited")


def _collect_metadata(svc: Any, *, vla_provider: Any | None = None) -> dict:
    """Collect metadata from initialized PipelineService for pool caching."""
    task_contracts: dict[str, Any] = {}
    for t in svc.supported_tasks():
        tc = svc.get_task_contract(t)
        if tc is not None and hasattr(tc, "to_metadata"):
            task_contracts[t] = tc.to_metadata()
        elif isinstance(tc, dict):
            task_contracts[t] = tc
    metadata = {
        "server_metadata": svc.server_metadata(),
        "supported_tasks": list(svc.supported_tasks()),
        "task_contracts": task_contracts,
    }
    if vla_provider is not None:
        vla_metadata = vla_provider.metadata()
        if not isinstance(vla_metadata, dict):
            raise TypeError("VLA provider metadata must be a dictionary")
        metadata["vla"] = vla_metadata
    return metadata


def _cancel_watcher_fn(
    cancel_event: mp_stdlib.Event,
    stop_event: threading.Event,
    forwarder_done: threading.Event,
) -> None:
    """Bridge mp.Event -> threading.Event for cancellation in subprocess."""
    cancel_event.wait()
    stop_event.set()
    forwarder_done.set()


async def _task_loop(
    replica_id: int,
    svc: Any,
    conn: Connection,
    cancel_event: mp_stdlib.Event,
    logger: Any,
    *,
    vla_provider: Any | None = None,
) -> None:
    """Persistent async task loop inside the replica subprocess."""
    loop = asyncio.get_running_loop()

    while True:
        msg = await loop.run_in_executor(None, _recv_with_poll, conn, 5.0)
        if msg is None:
            continue

        if msg[0] == "shutdown":
            logger.info(f"Replica {replica_id}: shutdown requested")
            break

        if msg[0] == "task":
            _, task_data, timeout_s, output_root = msg
            cancel_event.clear()

            stop_event = threading.Event()
            forwarder_done = threading.Event()

            watcher = threading.Thread(
                target=_cancel_watcher_fn,
                args=(cancel_event, stop_event, forwarder_done),
                daemon=True,
            )
            watcher.start()

            try:
                result = await svc.run_task_with_stop_event(
                    task_data,
                    stop_event,
                    timeout_s=timeout_s,
                    output_root=output_root,
                )
                conn.send(("ok", result))
            except Exception as e:
                conn.send(("error", str(e)))
            finally:
                cancel_event.set()
                forwarder_done.wait(1.0)
            continue

        if msg[0] == "vla":
            _, operation, payload = msg
            if vla_provider is None:
                conn.send(("vla_error", {"type": "RuntimeError", "message": "replica has no VLA provider"}))
                continue
            try:
                result = await asyncio.to_thread(vla_provider.dispatch, operation, payload)
                conn.send(("ok", result))
            except Exception as e:
                conn.send(("vla_error", {"type": type(e).__name__, "message": str(e)}))


def _recv_with_poll(conn: Connection, timeout: float) -> Any:
    """Receive from connection with bounded poll. Returns None on timeout."""
    if conn.poll(timeout):
        return conn.recv()
    return None


# ---------------------------------------------------------------------------
# Main-process handle
# ---------------------------------------------------------------------------


def _forward_cancel_fn(
    cancel_event: mp_stdlib.Event,
    stop_event: threading.Event,
    forwarder_exit: threading.Event,
    forwarder_wake: threading.Event,
    forwarder_done: threading.Event,
) -> None:
    """Forward main-process stop_event to subprocess cancel_event (polled)."""
    while not forwarder_exit.is_set():
        if stop_event.is_set():
            cancel_event.set()
            break
        forwarder_wake.wait(timeout=0.5)
        forwarder_wake.clear()
    forwarder_done.set()


class ReplicaHandle:
    """Main-process handle for communicating with a replica subprocess."""

    def __init__(
        self,
        replica_id: int,
        process: mp_stdlib.Process,
        conn: Connection,
        cancel_event: mp_stdlib.Event,
        metadata: dict,
    ) -> None:
        self.replica_id = replica_id
        self.process = process
        self.conn = conn
        self.cancel_event = cancel_event
        self.metadata = metadata
        self._dead = False

    async def run_task(
        self,
        task_data: dict,
        stop_event: threading.Event,
        timeout_s: float | None,
        output_root: str | None,
    ) -> dict:
        """Send task to replica and await result with bounded wait + health check."""
        self.cancel_event.clear()

        forwarder_exit = threading.Event()
        forwarder_wake = threading.Event()
        forwarder_done = threading.Event()

        forwarder = threading.Thread(
            target=_forward_cancel_fn,
            args=(self.cancel_event, stop_event, forwarder_exit, forwarder_wake, forwarder_done),
            daemon=True,
        )
        forwarder.start()

        loop = asyncio.get_running_loop()
        ipc_timeout = (timeout_s or 600) + _TASK_IPC_MARGIN_S

        try:
            if not self.process.is_alive():
                self._dead = True
                raise ReplicaDeadError(f"Replica {self.replica_id} process is not alive")
            try:
                self.conn.send(("task", task_data, timeout_s, output_root))
                result = await loop.run_in_executor(None, self._recv_with_health_check, ipc_timeout)
            except (EOFError, OSError) as error:
                self._dead = True
                raise ReplicaDeadError(f"Replica {self.replica_id} IPC failed: {error}") from error
        finally:
            forwarder_exit.set()
            forwarder_wake.set()
            forwarder_done.wait(2.0)

        if result is None:
            self._dead = True
            raise ReplicaDeadError(
                f"Replica {self.replica_id} did not respond within {ipc_timeout}s "
                f"(process alive: {self.process.is_alive()})"
            )

        tag, payload = result
        if tag == "error":
            raise RuntimeError(f"Replica {self.replica_id}: {payload}")
        return payload

    async def run_vla_operation(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Send one session-affine VLA operation to this replica."""
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be a non-empty string")
        if timeout_s is not None and (
            isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be None or a positive number")
        ipc_timeout = (float(timeout_s) if timeout_s is not None else 600.0) + _TASK_IPC_MARGIN_S
        loop = asyncio.get_running_loop()
        if not self.process.is_alive():
            self._dead = True
            raise ReplicaDeadError(f"Replica {self.replica_id} process is not alive")
        try:
            self.conn.send(("vla", operation, payload))
            result = await loop.run_in_executor(None, self._recv_with_health_check, ipc_timeout)
        except (EOFError, OSError) as error:
            self._dead = True
            raise ReplicaDeadError(f"Replica {self.replica_id} IPC failed: {error}") from error
        if result is None:
            self._dead = True
            raise ReplicaDeadError(
                f"Replica {self.replica_id} did not respond within {ipc_timeout}s "
                f"(process alive: {self.process.is_alive()})"
            )
        tag, response = result
        if tag == "vla_error":
            if not isinstance(response, dict):
                raise RuntimeError(f"Replica {self.replica_id} returned an invalid VLA error")
            raise ReplicaVLAError(str(response.get("type", "RuntimeError")), str(response.get("message", "")))
        if tag == "error":
            raise RuntimeError(f"Replica {self.replica_id}: {response}")
        if tag != "ok" or not isinstance(response, dict):
            raise RuntimeError(f"Replica {self.replica_id} returned an invalid VLA response")
        return response

    def _recv_with_health_check(self, total_timeout: float) -> tuple[str, Any] | None:
        """Receive with periodic health checks. Returns (tag, payload) or None."""
        from telefuser.utils.logging import logger

        deadline = time.monotonic() + total_timeout
        poll_interval = 5.0

        while time.monotonic() < deadline:
            remaining = min(poll_interval, deadline - time.monotonic())
            if remaining <= 0:
                break
            if self.conn.poll(remaining):
                return self.conn.recv()
            if not self.process.is_alive():
                logger.error(f"Replica {self.replica_id} process died (exit code {self.process.exitcode})")
                return None

        return None

    def shutdown(self) -> None:
        """Gracefully shutdown the replica subprocess."""
        try:
            self.conn.send(("shutdown",))
        except (OSError, BrokenPipeError):
            pass
        self.process.join(timeout=30)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
