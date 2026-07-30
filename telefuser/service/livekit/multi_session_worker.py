"""Model worker that multiplexes retained LiveKit sessions over one pipeline."""

from __future__ import annotations

from collections.abc import Callable

from telefuser.service.core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL

from .config import LiveKitServeConfig
from .pipeline_adapter import LiveKitPipelineAdapter
from .room_client import LiveKitRoomClient, RoomClient
from .schemas import SessionStatus
from .session_registry import SessionRecord
from .token_service import LiveKitTokenService
from .worker import LiveKitWorker as LiveKitSessionRunner
from .worker import NullWorkerEventSink, WorkerEventSink


class _SessionWorkerEventSink:
    """Bind worker-level callbacks from one runner to its retained session."""

    def __init__(self, owner: MultiSessionLiveKitWorker, session_id: str) -> None:
        self._owner = owner
        self._session_id = session_id

    def on_worker_status(self, worker_id: str, status: str) -> None:
        del worker_id
        self._owner._on_session_worker_status(self._session_id, status)

    def on_session_status(self, session_id: str, status: SessionStatus, error: str | None = None) -> None:
        self._owner.event_sink.on_session_status(session_id, status, error)

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        self._owner.event_sink.on_pipeline_session(session_id, pipeline_session_id)

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        self._owner.event_sink.on_session_finished(worker_id, session_id, error)


class MultiSessionLiveKitWorker:
    """Load one model pipeline and retain multiple independent room sessions."""

    def __init__(
        self,
        *,
        worker_id: str,
        config: LiveKitServeConfig,
        pipeline_file: str,
        token_service: LiveKitTokenService,
        event_sink: WorkerEventSink | None = None,
        pipeline_adapter: LiveKitPipelineAdapter | None = None,
        room_client_factory: Callable[[], RoomClient] | None = None,
        gpu_num: int = 1,
    ) -> None:
        self.worker_id = worker_id
        self.config = config
        self.pipeline_file = pipeline_file
        self.token_service = token_service
        self.event_sink = event_sink or NullWorkerEventSink()
        self.pipeline_adapter = pipeline_adapter or LiveKitPipelineAdapter()
        self.room_client_factory = room_client_factory or LiveKitRoomClient
        self.gpu_num = gpu_num
        self._sessions: dict[str, LiveKitSessionRunner] = {}
        self._session_worker_statuses: dict[str, str] = {}
        self._started = False

    async def start(self, *, skip_validation: bool = False) -> None:
        """Load the shared pipeline exactly once."""
        if self._started:
            return
        self.pipeline_adapter.start(
            self.pipeline_file,
            skip_validation=skip_validation,
            gpu_num=self.gpu_num,
        )
        if self.config.max_sessions_per_worker > 1 and self.pipeline_adapter.stream_mode != STREAM_MODE_BIDIRECTIONAL:
            await self.pipeline_adapter.aclose()
            raise RuntimeError("Multiple retained sessions require a BidirectionalService pipeline")
        self._started = True
        self.event_sink.on_worker_status(self.worker_id, "idle")

    async def run_session(self, record: SessionRecord) -> None:
        """Run one room session against the shared pipeline adapter."""
        if not self._started:
            raise RuntimeError(f"Worker {self.worker_id} is not started")
        if record.session_id in self._sessions:
            raise RuntimeError(f"Session {record.session_id} is already running")
        if len(self._sessions) >= self.config.max_sessions_per_worker:
            raise RuntimeError(f"Worker {self.worker_id} retained-session capacity is full")

        self._session_worker_statuses[record.session_id] = "assigned"
        runner = LiveKitSessionRunner(
            worker_id=self.worker_id,
            config=self.config,
            pipeline_file=self.pipeline_file,
            token_service=self.token_service,
            event_sink=_SessionWorkerEventSink(self, record.session_id),
            pipeline_adapter=self.pipeline_adapter,
            room_client=self.room_client_factory(),
            gpu_num=self.gpu_num,
        )
        self._sessions[record.session_id] = runner
        try:
            await runner.run_session(record)
        finally:
            self._sessions.pop(record.session_id, None)
            self._session_worker_statuses.pop(record.session_id, None)
            self._publish_aggregate_worker_status()

    async def stop_session(self, session_id: str) -> None:
        """Request one retained session to stop without affecting its peers."""
        runner = self._sessions.get(session_id)
        if runner is not None:
            await runner.stop_session(session_id)

    async def stop(self) -> None:
        """Stop admission and close the shared pipeline after sessions drain."""
        for session_id in tuple(self._sessions):
            await self.stop_session(session_id)
        self._started = False
        await self.pipeline_adapter.aclose()
        self.event_sink.on_worker_status(self.worker_id, "stopped")

    def _on_session_worker_status(self, session_id: str, status: str) -> None:
        if session_id not in self._session_worker_statuses:
            return
        self._session_worker_statuses[session_id] = status
        self._publish_aggregate_worker_status()

    def _publish_aggregate_worker_status(self) -> None:
        statuses = set(self._session_worker_statuses.values())
        aggregate = next(
            (
                status
                for status in ("running", "draining", "starting_pipeline", "joining_room", "assigned", "starting")
                if status in statuses
            ),
            "idle",
        )
        self.event_sink.on_worker_status(self.worker_id, aggregate)
