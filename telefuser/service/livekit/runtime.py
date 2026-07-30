"""Runtime coordinator for LiveKit-backed ``telefuser stream-serve``."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from telefuser.service.security.security_validator import SecurityLevel

from .config import LiveKitServeConfig
from .multi_session_worker import MultiSessionLiveKitWorker as LiveKitWorker
from .pipeline_adapter import LiveKitPipelineAdapter
from .scheduler import LiveKitScheduler, SchedulerAdmission
from .schemas import (
    LiveKitHealthResponse,
    SessionCreateRequest,
    SessionStatus,
    SessionStatusResponse,
    SessionTokenRequest,
    new_session_id,
)
from .session_registry import TERMINAL_SESSION_STATUSES, SessionRecord, SessionRegistry
from .token_service import LiveKitTokenService
from .worker_pool import InProcessLiveKitWorkerPool, WorkerPool


@dataclass(frozen=True)
class CreateSessionResult:
    """Internal result for a session creation request."""

    record: SessionRecord
    token: str
    admission: SchedulerAdmission


class LiveKitServeRuntime:
    """Coordinates LiveKit token/session APIs and TeleFuser worker capacity."""

    def __init__(
        self,
        *,
        config: LiveKitServeConfig,
        pipeline_file: str,
        token_service: LiveKitTokenService | None = None,
        registry: SessionRegistry | None = None,
        scheduler: LiveKitScheduler | None = None,
        worker_pool: WorkerPool | None = None,
        skip_validation: bool = False,
        security_level: SecurityLevel | str | None = None,
    ) -> None:
        self.config = config
        self.pipeline_file = pipeline_file
        self.registry = registry or SessionRegistry()
        self.scheduler = scheduler or LiveKitScheduler(
            num_workers=config.num_workers,
            gpu_groups=config.worker_gpu_groups(),
            queue_size=config.queue_size,
            max_sessions_per_worker=config.max_sessions_per_worker,
        )
        self.token_service = token_service or LiveKitTokenService(
            api_key=config.livekit_api_key,
            api_secret=config.livekit_api_secret,
            token_ttl=config.token_ttl,
        )
        self.skip_validation = skip_validation
        self.security_level = security_level
        self.worker_pool = worker_pool or self._create_worker_pool()
        self._started = False
        self._closing = False
        self._closed = False
        self._finished_sessions: set[str] = set()
        self._lock = threading.RLock()

    @property
    def is_ready(self) -> bool:
        """Return whether workers are started and able to accept sessions."""
        return self._started and not self._closing and self.health().status != "unhealthy"

    async def start(self) -> None:
        """Start runtime-owned workers and load their pipelines."""
        with self._lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("LiveKit runtime is already closed")
            if self.config.worker_mode != "in-process":
                raise NotImplementedError("stream-serve currently supports only worker_mode='in-process'")
            if self.config.num_workers != 1:
                raise NotImplementedError("stream-serve currently supports exactly one in-process worker")
        await self.worker_pool.start(skip_validation=self.skip_validation)
        with self._lock:
            self._started = True

    def create_session(self, request: SessionCreateRequest) -> CreateSessionResult:
        """Create a session record, mint a controller token, and reserve capacity."""

        session_id = new_session_id()
        room_name = f"tf-world-{session_id}"
        session_config = dict(request.config)
        session_config["session_id"] = session_id
        session_config["control_idle_timeout"] = self.config.control_idle_timeout
        if request.prompt is not None:
            session_config["prompt"] = request.prompt
        if request.image_path is not None:
            session_config["image_path"] = request.image_path

        record = self.registry.create(
            session_id=session_id,
            room_name=room_name,
            controller_identity=request.identity,
            config=session_config,
            timeout_s=self.config.session_timeout,
        )
        admission = self.scheduler.assign(session_id=session_id, room_name=room_name)
        if admission.status == "rejected":
            self.registry.delete(session_id)
            return CreateSessionResult(record=record, token="", admission=admission)
        if admission.status == "assigned" and admission.worker_id is not None:
            record = self.registry.assign_worker(session_id, admission.worker_id)
        else:
            record = self.registry.update_status(session_id, "queued")

        try:
            token = self.token_service.create_token(
                identity=request.identity,
                room_name=room_name,
                role="controller",
            )
        except Exception:
            self.scheduler.release_session(session_id)
            self.registry.delete(session_id)
            raise

        if admission.status == "assigned":
            try:
                self.worker_pool.start_session(record)
            except Exception as exc:
                self.scheduler.release_session(session_id)
                self.registry.fail(session_id, str(exc))
                raise
        return CreateSessionResult(record=record, token=token, admission=admission)

    def create_viewer_token(self, session_id: str, request: SessionTokenRequest) -> tuple[SessionRecord, str]:
        """Mint a subscribe-only viewer token for an existing session."""
        record = self.registry.require(session_id)
        if record.status in TERMINAL_SESSION_STATUSES:
            raise ValueError(f"Session {session_id} is not active")
        token = self.token_service.create_token(
            identity=request.identity,
            room_name=record.room_name,
            role="viewer",
        )
        return record, token

    def get_session_response(self, session_id: str) -> SessionStatusResponse:
        """Return public status for one session."""
        return session_record_to_response(self.registry.require(session_id))

    async def delete_session(self, session_id: str) -> SessionRecord:
        """Stop a session, close its room worker, and release capacity."""
        record = self.registry.require(session_id)
        if record.status in TERMINAL_SESSION_STATUSES:
            return record
        self.registry.update_status(session_id, "draining")
        await self.worker_pool.stop_session(session_id)
        return self._finish_session(session_id)

    def on_worker_status(self, worker_id: str, status: str) -> None:
        """Apply a worker lifecycle callback to scheduler state."""
        self.scheduler.update_worker_status(worker_id, status)

    def on_session_status(self, session_id: str, status: SessionStatus, error: str | None = None) -> None:
        """Apply a worker-reported public session state."""
        if self.registry.require(session_id).status not in TERMINAL_SESSION_STATUSES:
            self.registry.update_status(session_id, status, error=error)

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        """Record the pipeline session created by a worker."""
        self.registry.set_pipeline_session(session_id, pipeline_session_id)

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        """Release capacity after a worker session exits."""
        del worker_id
        self._finish_session(session_id, error=error)

    def health(self) -> LiveKitHealthResponse:
        """Return service health based on current scheduler state."""
        snapshot = self.scheduler.health_snapshot()
        workers_total = snapshot["workers_total"]
        workers_failed = snapshot["workers_failed"]
        status = "healthy"
        if workers_total and workers_failed == workers_total:
            status = "unhealthy"
        elif workers_failed:
            status = "degraded"
        connected_statuses = {"starting_pipeline", "running", "draining"}
        return LiveKitHealthResponse(
            status=status,
            livekit_connected=any(worker.status in connected_statuses for worker in self.scheduler.workers()),
            **snapshot,
        )

    def metadata(self) -> dict:
        """Return runtime metadata for `/v1/service/metadata`."""
        health = self.health()
        return {
            "service_type": "stream",
            "transport": "livekit",
            "pipeline_file": self.pipeline_file,
            "livekit_url": self.config.livekit_url,
            "num_workers": self.config.num_workers,
            "max_sessions_per_worker": self.config.max_sessions_per_worker,
            "control_idle_timeout": self.config.control_idle_timeout,
            "worker_mode": self.config.worker_mode,
            "queue_size": self.config.queue_size,
            **health.model_dump(),
        }

    async def aclose(self) -> None:
        """Stop runtime-owned background resources."""
        with self._lock:
            if self._closed or self._closing:
                return
            self._closing = True
        try:
            await self.worker_pool.aclose()
            for record in self.registry.list_records():
                if record.status not in TERMINAL_SESSION_STATUSES:
                    self._finish_session(record.session_id, error="runtime closed")
        finally:
            with self._lock:
                self._started = False
                self._closing = False
                self._closed = True

    def _create_worker_pool(self) -> WorkerPool:
        security_level = self.security_level
        if isinstance(security_level, str):
            security_level = SecurityLevel[security_level.upper()]
        workers: dict[str, LiveKitWorker] = {}
        for worker_state in self.scheduler.workers():
            workers[worker_state.worker_id] = LiveKitWorker(
                worker_id=worker_state.worker_id,
                config=self.config,
                pipeline_file=self.pipeline_file,
                token_service=self.token_service,
                event_sink=self,
                pipeline_adapter=LiveKitPipelineAdapter(security_level=security_level),
                gpu_num=max(1, len(worker_state.gpu_ids)),
            )
        return InProcessLiveKitWorkerPool(workers)

    def _finish_session(self, session_id: str, *, error: str | None = None) -> SessionRecord:
        with self._lock:
            current = self.registry.require(session_id)
            if session_id in self._finished_sessions:
                return current
            self._finished_sessions.add(session_id)

            if current.status in TERMINAL_SESSION_STATUSES:
                record = current
            elif error is not None and error != "cancelled":
                record = self.registry.fail(session_id, error)
            else:
                record = self.registry.close(session_id)
            admission = self.scheduler.release_session(session_id)
            if admission is not None and not self._closing:
                self._start_queued_session(admission)
            return record

    def _start_queued_session(self, admission: SchedulerAdmission) -> None:
        if admission.worker_id is None or admission.session_id is None:
            return
        session_id = admission.session_id
        try:
            record = self.registry.assign_worker(session_id, admission.worker_id)
            self.worker_pool.start_session(record)
        except Exception as exc:
            self.registry.fail(session_id, str(exc))
            self._finish_session(session_id, error=str(exc))


def session_record_to_response(record: SessionRecord) -> SessionStatusResponse:
    """Convert an internal session record to public response schema."""
    return SessionStatusResponse(
        session_id=record.session_id,
        room=record.room_name,
        status=record.status,
        worker_id=record.worker_id,
        pipeline_session_id=record.pipeline_session_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=record.expires_at,
        participant_count=record.participant_count,
        error=record.error,
    )
