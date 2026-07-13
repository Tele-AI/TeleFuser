"""Stream pipeline service for real-time generation.

Replaces PipelineService for stream-mode serving. Instead of
get_pipeline() + run_with_file() (request-response), stream pipelines
expose get_service() which returns an object with start/stop/serve methods.

Two interaction modes are supported:

* SERVER_PUSH   – single request in, continuous chunks out  (WebRTC media tracks)
* BIDIRECTIONAL – continuous input & output                 (WebRTC DataChannel + media tracks)

Pipeline ``serve()`` methods may contain blocking calls (GPU inference,
``time.sleep``, etc.). ``stream_task()`` runs them on a dedicated thread
with its own event loop so the server's main loop stays responsive.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from telefuser.metrics.runtime import (
    RuntimeMeasurement,
    collect_runtime_environment,
    finish_runtime_measurement,
    start_runtime_measurement,
    visible_cuda_devices,
)
from telefuser.utils.logging import logger

from ..security.security_validator import (
    PipelineSecurityValidator,
    SecurityError,
    SecurityLevel,
)
from .config import server_config
from .pipeline_loader import load_pipeline_module, unload_pipeline_module, validate_pipeline_file

# ---------------------------------------------------------------------------
# Stream mode constants
# ---------------------------------------------------------------------------

STREAM_MODE_SERVER_PUSH = "server_push"
STREAM_MODE_BIDIRECTIONAL = "bidirectional"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Service protocols – pipeline authors implement one of these
# ---------------------------------------------------------------------------


@runtime_checkable
class ServerPushService(Protocol):
    """Protocol for single-input / continuous-output pipelines."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    async def serve(self, request: dict) -> AsyncGenerator[dict, None]: ...


@runtime_checkable
class BidirectionalService(Protocol):
    """Protocol for continuous-input / continuous-output pipelines."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def create_session(self, config: dict) -> str: ...
    def push_chunk(self, session_id: str, chunk: dict) -> None: ...
    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]: ...
    def close_session(self, session_id: str) -> None: ...


# ---------------------------------------------------------------------------
# StreamPipelineService
# ---------------------------------------------------------------------------


class StreamPipelineService:
    """Loads a stream pipeline module and drives streaming execution.

    Pipeline file convention:
        def get_service() -> ServerPushService | BidirectionalService
    """

    def __init__(self, security_level: SecurityLevel | None = None) -> None:
        self.is_running = False
        self.service: ServerPushService | BidirectionalService | None = None
        self.stream_mode: str | None = None
        self.ppl_file: str | None = None
        self._module: ModuleType | None = None
        self._module_name: str | None = None
        self._startup_measurement: dict[str, Any] | None = None
        self._runtime_environment: dict[str, Any] = {}

        self.security_level = security_level or getattr(server_config, "security_level", SecurityLevel.STRICT)
        self.security_validator = PipelineSecurityValidator(
            security_level=self.security_level,
            max_file_size=getattr(server_config, "max_ppl_file_size", 1024 * 1024),
        )
        logger.info(f"StreamPipelineService initialized with security_level={self.security_level.name}")

    # -- lifecycle -----------------------------------------------------------

    def start_service(self, ppl_file: str, skip_validation: bool = False) -> bool:
        """Load module, call get_service(), detect mode, and start."""
        if self.is_running:
            logger.warning("Stream service is already running")
            return True

        devices = visible_cuda_devices()
        started_at = time.perf_counter()
        measurement: RuntimeMeasurement | None = None
        try:
            measurement = start_runtime_measurement(
                devices,
                capture_peak_memory=True,
            )
        except Exception as exc:
            logger.warning(f"Could not initialize stream startup metrics: {exc}")

        try:
            if not skip_validation:
                validate_pipeline_file(ppl_file, self.security_level, self.security_validator)
            else:
                logger.warning("Skipping security validation for pipeline file")

            self.ppl_file = ppl_file
            self._module, self._module_name = load_pipeline_module(ppl_file, prefix="telefuser_stream_ppl")

            if not hasattr(self._module, "get_service"):
                raise RuntimeError(
                    "Stream pipeline file must define get_service() "
                    "returning a ServerPushService or BidirectionalService"
                )

            self.service = self._module.get_service()
            self.stream_mode = self._detect_mode(self.service)
            self.service.start()
            self.is_running = True
            try:
                facts = (
                    finish_runtime_measurement(measurement)
                    if measurement is not None
                    else {
                        "seconds": max(time.perf_counter() - started_at, 0.0),
                        "memory": [],
                    }
                )
            except Exception as exc:
                logger.warning(f"Could not finish stream startup metrics: {exc}")
                facts = {
                    "seconds": max(time.perf_counter() - started_at, 0.0),
                    "memory": [],
                }
            self._startup_measurement = {"name": "pipeline_init", **facts}
            try:
                self._runtime_environment = collect_runtime_environment(
                    devices,
                    repo_root=_REPO_ROOT,
                )
            except Exception as exc:
                logger.warning(f"Could not collect stream runtime environment: {exc}")
            logger.info(f"Stream service started: mode={self.stream_mode}, file={ppl_file}")
            return True

        except SecurityError as e:
            logger.error(f"Security validation failed: {e}")
            return False
        except Exception as e:
            logger.exception(f"Failed to start stream service: {e}")
            return False

    async def aclose(self) -> None:
        if not self.is_running:
            return
        try:
            if self.service is not None:
                self.service.stop()
        except Exception as e:
            logger.warning(f"Error during stream service shutdown: {e}")
        finally:
            self.service = None
            unload_pipeline_module(getattr(self, "_module_name", None))
            self._module = None
            self._module_name = None
            self.stream_mode = None
            self.is_running = False
            logger.info("Stream service stopped")

    # -- server-push --------------------------------------------------------

    async def stream_task(self, task_data: dict) -> AsyncGenerator[dict, None]:
        """Yield chunks for a single-request / continuous-output task.

        The pipeline's ``serve()`` generator runs on a dedicated thread with
        its own event loop so that blocking calls (GPU inference, time.sleep,
        etc.) do not stall the server's main loop.

        A ``threading.Event`` stop flag is set when the consumer stops
        iterating (e.g. WebRTC disconnect), so the producer thread can
        break out of the pipeline's ``serve()`` loop promptly instead of
        running to completion.
        """
        self._ensure_running()
        if self.stream_mode != STREAM_MODE_SERVER_PUSH:
            raise RuntimeError(f"stream_task requires ServerPushService, got {type(self.service).__name__}")

        main_loop = asyncio.get_running_loop()
        out_queue: asyncio.Queue[dict | BaseException | None] = asyncio.Queue()
        stop_event = threading.Event()

        def _run() -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_drain())
            finally:
                loop.close()

        async def _drain() -> None:
            try:
                async for chunk in self.service.serve(task_data):
                    if stop_event.is_set():
                        break
                    main_loop.call_soon_threadsafe(out_queue.put_nowait, chunk)
            except Exception as exc:
                if not stop_event.is_set():
                    main_loop.call_soon_threadsafe(out_queue.put_nowait, exc)
            finally:
                main_loop.call_soon_threadsafe(out_queue.put_nowait, None)

        thread = threading.Thread(target=_run, daemon=True, name="stream-task")
        thread.start()

        try:
            while True:
                item = await out_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop_event.set()

    # -- bidirectional ------------------------------------------------------

    def _ensure_bidirectional(self) -> BidirectionalService:
        self._ensure_running()
        if self.stream_mode != STREAM_MODE_BIDIRECTIONAL:
            raise RuntimeError(f"Operation requires BidirectionalService, got {type(self.service).__name__}")
        return self.service

    def create_session(self, config: dict) -> str:
        return self._ensure_bidirectional().create_session(config)

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self._ensure_bidirectional().push_chunk(session_id, chunk)

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        svc = self._ensure_bidirectional()
        async for chunk in svc.pull_chunks(session_id):
            yield chunk

    def close_session(self, session_id: str) -> None:
        self._ensure_bidirectional().close_session(session_id)

    def has_session(self, session_id: str) -> bool:
        """Check if a bidirectional session exists (duck-typed)."""
        if self.stream_mode != STREAM_MODE_BIDIRECTIONAL or self.service is None:
            return False
        checker = getattr(self.service, "has_session", None)
        if callable(checker):
            return checker(session_id)
        sessions = getattr(self.service, "_sessions", None)
        if isinstance(sessions, dict):
            return session_id in sessions
        return False

    # -- metadata ------------------------------------------------------------

    def server_metadata(self) -> dict[str, Any]:
        metadata = {
            "service_type": "stream",
            "stream_mode": self.stream_mode,
            "pipeline_file": self.ppl_file,
            "security_level": self.security_level.name if self.security_level else "NONE",
            "runner": "StreamPipelineService",
        }
        startup_measurement = getattr(self, "_startup_measurement", None)
        if startup_measurement is not None:
            metadata["performance"] = {
                "phases": [startup_measurement],
            }
        runtime_environment = getattr(self, "_runtime_environment", {})
        if runtime_environment:
            metadata["environment"] = runtime_environment
        return metadata

    # -- internals -----------------------------------------------------------

    def _ensure_running(self) -> None:
        if not self.is_running or self.service is None:
            raise RuntimeError("Stream service is not started")

    @staticmethod
    def _detect_mode(service: Any) -> str:
        if isinstance(service, BidirectionalService):
            return STREAM_MODE_BIDIRECTIONAL
        if isinstance(service, ServerPushService):
            return STREAM_MODE_SERVER_PUSH
        raise RuntimeError("Service must implement ServerPushService or BidirectionalService protocol")
