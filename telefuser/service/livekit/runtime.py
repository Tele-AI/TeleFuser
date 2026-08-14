"""Runtime coordinator for LiveKit-backed ``telefuser stream-serve``."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass

from telefuser.service.security.security_validator import SecurityLevel
from telefuser.utils.logging import logger

from .config import LiveKitServeConfig
from .metrics import LiveKitServingMetrics
from .multi_session_worker import MultiSessionLiveKitWorker as LiveKitWorker
from .nccl_process_worker_pool import NCCLProcessLiveKitWorkerPool
from .pipeline_adapter import LiveKitPipelineAdapter
from .pipeline_router import TurboServePipelineRouter
from .process_worker_pool import ProcessLiveKitWorkerPool, ProcessWorkerSpec
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
from .turboserve import (
    TurboServeAutoscalingController,
    TurboServeClusterScheduler,
    TurboServeMigrationPlan,
    TurboServeOwnership,
    TurboServePlacementController,
    TurboServeRuntimeCalibration,
    TurboServeScaleDecision,
    TurboServeSchedulerConfig,
    TurboServeSchedulingSnapshot,
    TurboServeSessionDemand,
    TurboServeSessionView,
    TurboServeWorkerLoad,
)
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
            max_sessions_per_worker=config.session_capacity_limit() or 1,
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
        self._reported_worker_capacities: dict[str, int] = {}
        self._worker_capacity_profiles: dict[str, dict[str, object]] = {}
        self._autoscaling_controller = TurboServeAutoscalingController(
            sessions_per_worker=config.session_capacity_limit() or 1,
            target_utilization=config.autoscaling_target_utilization,
            hysteresis=config.autoscaling_hysteresis,
            cooldown_seconds=config.autoscaling_cooldown_seconds,
            min_workers=config.autoscaling_min_workers,
            max_workers=config.num_workers,
        )
        self._placement_controller = TurboServePlacementController(
            migration_bandwidth_bytes_per_second=config.turboserve_migration_bandwidth_gbps * 1_000_000_000,
            migration_penalty=config.turboserve_migration_penalty,
        )
        self._autoscale_task: asyncio.Task | None = None
        self._cluster_scheduler = TurboServeClusterScheduler(
            TurboServeSchedulerConfig(
                enable_autoscaling=config.autoscaling_enabled,
                enable_migration=config.turboserve_rebalance_enabled,
                min_workers=config.autoscaling_min_workers,
                max_workers=config.num_workers,
                capacity_per_worker=config.session_capacity_limit() or 1,
                target_utilization=config.autoscaling_target_utilization,
                scale_in_hold_seconds=config.turboserve_scale_in_hold_seconds,
                migration_eta=config.turboserve_migration_eta,
                min_gain_ms=config.turboserve_min_migration_gain_ms,
                rebalance_iteration_limit=config.turboserve_rebalance_iteration_limit,
            )
        )
        self._last_scale_decision: TurboServeScaleDecision | None = None
        self._last_migration_plan: TurboServeMigrationPlan | None = None
        self._last_migration_error: str | None = None
        self._serving_metrics = LiveKitServingMetrics()
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
            worker_groups = self.config.worker_gpu_groups()
            if self.config.worker_mode == "process" and self.config.num_workers > 1 and any(
                not group for group in worker_groups
            ):
                raise ValueError("worker_gpu_map is required for multiple process workers")
            groups = [gpu_id for group in worker_groups for gpu_id in group]
            if len(groups) != len(set(groups)):
                raise ValueError("worker_gpu_map assigns a GPU to more than one worker")
        await self.worker_pool.start(skip_validation=self.skip_validation)
        with self._lock:
            self._started = True
        if self.config.autoscaling_enabled or self.config.turboserve_rebalance_enabled:
            self._autoscale_task = asyncio.create_task(self._autoscale_loop(), name="livekit-turboserve-control")

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
        self._serving_metrics.record_admission(admission.status)
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

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        """Apply worker-local hardware capacity before the runtime becomes ready."""
        self.scheduler.update_worker_capacity(worker_id, capacity)
        with self._lock:
            self._reported_worker_capacities[worker_id] = capacity
            if profile is not None:
                self._worker_capacity_profiles[worker_id] = dict(profile)
            effective_capacity = min(self._reported_worker_capacities.values())
            self._autoscaling_controller.sessions_per_worker = effective_capacity
            # Keep the source-aligned controller on the hardware-measured
            # capacity too. Otherwise an auto-capacity deployment silently
            # schedules every worker as if its capacity were one.
            self._cluster_scheduler.config = TurboServeSchedulerConfig(
                **{
                    **self._cluster_scheduler.config.__dict__,
                    "capacity_per_worker": effective_capacity,
                }
            )

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
        record = self._finish_session(session_id, error=error)
        self._serving_metrics.record_session_finished(record.status, error=error)

    def on_control_received(self, worker_id: str, session_id: str) -> None:
        """Record a validated controller action entering the serving pipeline."""
        self._serving_metrics.on_control_received(worker_id, session_id)

    def on_chunk_published(
        self,
        worker_id: str,
        session_id: str,
        frames: int,
        first_frame_at: float | None = None,
    ) -> None:
        """Record frames accepted by the LiveKit video publisher."""
        self._serving_metrics.on_chunk_published(
            worker_id=worker_id,
            session_id=session_id,
            frames=frames,
            first_frame_at=first_frame_at,
        )

    def on_model_output(
        self,
        worker_id: str,
        session_id: str,
        payload: dict,
        runtime_metrics: dict | None = None,
        session_runtime_metrics: dict | None = None,
    ) -> None:
        """Ingest a child-model output forwarded by the process-NCCL pool."""
        self._serving_metrics.on_model_output(
            self,
            worker_id=worker_id,
            pipeline_session_id=session_id,
            payload=payload,
            runtime_metrics=runtime_metrics,
            session_runtime_metrics=session_runtime_metrics,
        )

    def prometheus_metrics(self) -> str:
        """Render runtime, scheduler, session, and pipeline serving metrics."""
        return self._serving_metrics.render_prometheus(self)

    def serving_metrics_snapshot(self) -> dict:
        """Return aggregate serving metrics for the JSON endpoint and experiments."""
        return self._serving_metrics.json_snapshot(self)

    def health(self) -> LiveKitHealthResponse:
        """Return service health based on current scheduler state."""
        snapshot = self.scheduler.health_snapshot()
        workers_failed = snapshot["workers_failed"]
        workers = self.scheduler.workers()
        serving_workers = [worker for worker in workers if worker.status not in {"failed", "stopped"}]
        status = "healthy"
        if not serving_workers:
            status = "unhealthy"
        elif workers_failed:
            status = "degraded"
        connected_statuses = {"starting_pipeline", "running", "draining"}
        return LiveKitHealthResponse(
            status=status,
            livekit_connected=any(worker.status in connected_statuses for worker in workers),
            **snapshot,
        )

    def metadata(self) -> dict:
        """Return runtime metadata for `/v1/service/metadata`."""
        health = self.health()
        with self._lock:
            reported_capacities = tuple(self._reported_worker_capacities.values())
            capacity_profiles = dict(self._worker_capacity_profiles)
        max_sessions_per_worker = (
            min(reported_capacities)
            if reported_capacities
            else min(worker.session_capacity for worker in self.scheduler.workers())
        )
        metadata = {
            "service_type": "stream",
            "transport": "livekit",
            "pipeline_file": self.pipeline_file,
            "livekit_url": self.config.livekit_url,
            "num_workers": self.config.num_workers,
            "max_sessions_per_worker": max_sessions_per_worker,
            "configured_max_sessions_per_worker": self.config.max_sessions_per_worker,
            "control_idle_timeout": self.config.control_idle_timeout,
            "worker_mode": self.config.worker_mode,
            "queue_size": self.config.queue_size,
            "autoscaling_enabled": self.config.autoscaling_enabled,
            **health.model_dump(),
        }
        if capacity_profiles:
            metadata["session_capacity"] = capacity_profiles
        snapshot = getattr(self.worker_pool, "turboserve_snapshot", None)
        if callable(snapshot) and (routing := snapshot()) is not None:
            metadata["turboserve_routing"] = routing
        if self._last_scale_decision is not None:
            metadata["autoscaling"] = {
                "current_workers": self._last_scale_decision.current_workers,
                "target_workers": self._last_scale_decision.target_workers,
                "action": self._last_scale_decision.action,
                "target_utilization": self._last_scale_decision.target_utilization,
                "reason": self._last_scale_decision.reason,
            }
        if self._last_migration_plan is not None or self._last_migration_error is not None:
            metadata["turboserve_rebalance"] = {
                "enabled": self.config.turboserve_rebalance_enabled,
                "last_plan": (
                    {
                        "session_id": self._last_migration_plan.session_id,
                        "source_worker_id": self._last_migration_plan.source_worker_id,
                        "target_worker_id": self._last_migration_plan.target_worker_id,
                        "gain_seconds": self._last_migration_plan.gain_seconds,
                        "migration_cost_seconds": self._last_migration_plan.migration_cost_seconds,
                    }
                    if self._last_migration_plan is not None
                    else None
                ),
                "last_error": self._last_migration_error,
            }
        return metadata

    async def migrate_session(self, session_id: str, target_worker_id: str) -> TurboServeOwnership:
        """Migrate ABot model state without reconnecting the LiveKit room."""
        record = self.registry.require(session_id)
        if record.pipeline_session_id is None:
            raise RuntimeError(f"Session {session_id} has not created its pipeline state yet")
        target = next((worker for worker in self.scheduler.workers() if worker.worker_id == target_worker_id), None)
        if target is None:
            raise KeyError(f"Unknown migration target {target_worker_id}")
        if target_worker_id != record.worker_id and len(target.session_ids) >= target.session_capacity:
            raise RuntimeError(f"Migration target {target_worker_id} has no retained-session capacity")
        migrate = getattr(self.worker_pool, "migrate_session", None)
        if not callable(migrate):
            raise RuntimeError("Configured worker pool does not support TurboServe migration")
        migration_started_at = asyncio.get_running_loop().time()
        try:
            ownership = await migrate(record.pipeline_session_id, target_worker_id)
        except Exception as exc:
            self._serving_metrics.record_migration(success=False, error=str(exc))
            raise
        self._serving_metrics.record_migration(
            success=True,
            duration_seconds=asyncio.get_running_loop().time() - migration_started_at,
        )
        self.scheduler.reassign_session(session_id, target_worker_id)
        self.registry.assign_worker(session_id, target_worker_id)
        return ownership

    async def aclose(self) -> None:
        """Stop runtime-owned background resources."""
        with self._lock:
            if self._closed or self._closing:
                return
            self._closing = True
        try:
            if self._autoscale_task is not None:
                self._autoscale_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._autoscale_task
                self._autoscale_task = None
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
        initial_workers = self.config.autoscaling_min_workers if self.config.autoscaling_enabled else None
        if self.config.worker_mode in {"process", "process-nccl"}:
            specs = [
                ProcessWorkerSpec(worker_id=state.worker_id, gpu_ids=list(state.gpu_ids))
                for state in self.scheduler.workers()
            ]
            pool_type = (
                NCCLProcessLiveKitWorkerPool
                if self.config.worker_mode == "process-nccl"
                else ProcessLiveKitWorkerPool
            )
            return pool_type(
                specs,
                config=self.config,
                pipeline_file=self.pipeline_file,
                event_sink=self,
                security_level=security_level,
                initial_workers=initial_workers,
            )
        worker_states = self.scheduler.workers()
        backends = {
            state.worker_id: LiveKitPipelineAdapter(security_level=security_level) for state in worker_states
        }
        router = TurboServePipelineRouter(backends)
        workers: dict[str, LiveKitWorker] = {}
        for worker_state in worker_states:
            workers[worker_state.worker_id] = LiveKitWorker(
                worker_id=worker_state.worker_id,
                config=self.config,
                pipeline_file=self.pipeline_file,
                token_service=self.token_service,
                event_sink=self,
                pipeline_adapter=router.worker_view(
                    worker_state.worker_id, gpu_ids=worker_state.gpu_ids or None
                ),
                gpu_num=max(1, len(worker_state.gpu_ids)),
                gpu_ids=worker_state.gpu_ids or None,
            )
        return InProcessLiveKitWorkerPool(workers, router=router, initial_workers=initial_workers)

    async def _autoscale_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.autoscaling_interval_seconds)
            try:
                await self._turboserve_control_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception(f"LiveKit autoscaling iteration failed: {exc}")

    async def _autoscale_once(self) -> TurboServeScaleDecision:
        active_count = getattr(self.worker_pool, "active_worker_count", None)
        scale_to = getattr(self.worker_pool, "scale_to", None)
        if not callable(active_count) or not callable(scale_to):
            raise RuntimeError("Configured worker pool does not support autoscaling")
        snapshot = self.scheduler.health_snapshot()
        workload = self._workload_snapshot()
        retained = sum(len(worker.session_ids) for worker in self.scheduler.workers())
        demand = max(retained + snapshot["queued_sessions"], workload["active_sessions"] + snapshot["queued_sessions"])
        decision = self._autoscaling_controller.decide(
            demand,
            active_count(),
            activation_volatility=workload["activation_volatility"],
        )
        actual = await scale_to(decision.target_workers)
        if actual > decision.current_workers:
            for admission in self.scheduler.drain_queue():
                self._start_queued_session(admission)
        await self._rebalance_once()
        self._last_scale_decision = TurboServeScaleDecision(
            current_workers=decision.current_workers,
            target_workers=actual,
            action=decision.action if actual != decision.current_workers else "hold",
            target_utilization=decision.target_utilization,
            reason=decision.reason,
        )
        return self._last_scale_decision

    async def _turboserve_control_once(self) -> None:
        """Run one source-compatible closed-loop scheduling decision."""

        snapshot_fn = getattr(self.worker_pool, "turboserve_snapshot", None)
        active_count = getattr(self.worker_pool, "active_worker_count", None)
        scale_to = getattr(self.worker_pool, "scale_to", None)
        if not callable(snapshot_fn) or not callable(active_count) or not callable(scale_to):
            if self.config.autoscaling_enabled:
                await self._autoscale_once()
            else:
                await self._rebalance_once()
            return
        routing = snapshot_fn()
        if not isinstance(routing, dict):
            return
        worker_order = tuple(worker.worker_id for worker in self.scheduler.workers())
        if not worker_order:
            return
        capacity = min(
            self._reported_worker_capacities.values(),
            default=min(worker.session_capacity for worker in self.scheduler.workers()),
        )
        session_metrics = routing.get("session_runtime_metrics", {})
        if not isinstance(session_metrics, dict):
            session_metrics = {}
        sessions: dict[str, TurboServeSessionView] = {}
        placement: dict[str, str | None] = {}
        for record in self.registry.list_records():
            if record.status in TERMINAL_SESSION_STATUSES or record.pipeline_session_id is None:
                continue
            metrics = session_metrics.get(record.pipeline_session_id, {})
            if not isinstance(metrics, dict):
                metrics = {}
            profile = self._worker_capacity_profiles.get(record.worker_id or "", {})
            bytes_ = int(profile.get("estimated_session_bytes", 1)) if isinstance(profile, dict) else 1
            sessions[record.session_id] = TurboServeSessionView(
                session_id=record.session_id,
                active=bool(metrics.get("active", record.status == "running")),
                state_size_mb=max(1.0, bytes_ / (1024 * 1024)),
                frame_count=int(metrics.get("emitted_frames", 9)),
            )
            placement[record.session_id] = record.worker_id
        calibration = routing.get("migration_calibration", {})
        if not isinstance(calibration, dict):
            calibration = {}
        worker_metrics = routing.get("worker_runtime_metrics", {})
        if not isinstance(worker_metrics, dict):
            worker_metrics = {}
        base_latency_ms = max(
            (
                float(values.get("p95_chunk_seconds", 0.0)) * 1000
                for values in worker_metrics.values()
                if isinstance(values, dict)
            ),
            default=0.0,
        )
        decision = self._cluster_scheduler.decide(
            TurboServeSchedulingSnapshot(
                time_seconds=asyncio.get_running_loop().time(),
                sessions=sessions,
                placement=placement,
                current_workers=active_count(),
                worker_order=worker_order,
                capacity_per_worker=max(1, capacity),
                runtime_calibration=TurboServeRuntimeCalibration(
                    average_migration_total_ms=float(calibration.get("average_total_ms", 0.0)),
                    base_chunk_latency_ms=base_latency_ms,
                ),
            )
        )
        current_workers = active_count()
        if decision.worker_budget > current_workers:
            actual = await scale_to(decision.worker_budget)
            for admission in self.scheduler.drain_queue():
                self._start_queued_session(admission)
        else:
            actual = current_workers
        migrations = 0
        for session_id, target_worker_id in decision.placement.items():
            record = self.registry.require(session_id)
            if (
                target_worker_id is None
                or target_worker_id == record.worker_id
                or record.pipeline_session_id is None
            ):
                continue
            try:
                await self.migrate_session(session_id, target_worker_id)
                migrations += 1
            except Exception as exc:
                self._last_migration_error = str(exc)
                logger.warning(
                    "TurboServe placement move failed: session=%s target=%s error=%s",
                    session_id,
                    target_worker_id,
                    exc,
                )
        if decision.worker_budget < actual:
            actual = await scale_to(decision.worker_budget)
        action = str(decision.metadata["autoscale_action"])
        self._last_scale_decision = TurboServeScaleDecision(
            current_workers=current_workers,
            target_workers=actual,
            action=action if action in {"scale_out", "scale_in"} else "hold",
            target_utilization=self.config.autoscaling_target_utilization,
            reason=action,
        )
        if migrations:
            self._last_migration_error = None

        """Aggregate live ABot control activity when the pool can expose it."""
    def _workload_snapshot(self) -> dict[str, float | int]:
        snapshot = getattr(self.worker_pool, "turboserve_snapshot", None)
        routing = snapshot() if callable(snapshot) else None
        per_worker = routing.get("worker_runtime_metrics", {}) if isinstance(routing, dict) else {}
        if not isinstance(per_worker, dict) or not per_worker:
            return {"active_sessions": 0, "activation_volatility": 0.0}
        active = 0
        volatility = 0.0
        for values in per_worker.values():
            if not isinstance(values, dict):
                continue
            active += int(values.get("active_sessions", 0))
            volatility = max(volatility, float(values.get("activation_volatility", 0.0)))
        return {"active_sessions": active, "activation_volatility": volatility}

    async def _rebalance_once(self) -> None:
        """Commit at most one profitable chunk-boundary migration per control tick."""
        if not self.config.turboserve_rebalance_enabled:
            return
        migrate = getattr(self.worker_pool, "migrate_session", None)
        snapshot_fn = getattr(self.worker_pool, "turboserve_snapshot", None)
        if not callable(migrate) or not callable(snapshot_fn):
            return
        routing = snapshot_fn()
        if not isinstance(routing, dict) or routing.get("migration_supported") is False:
            return
        metrics = routing.get("worker_runtime_metrics", {})
        if not isinstance(metrics, dict):
            return
        workers: list[TurboServeWorkerLoad] = []
        for state in self.scheduler.workers():
            values = metrics.get(state.worker_id, {})
            if not isinstance(values, dict):
                values = {}
            p95 = float(values.get("p95_chunk_seconds", 0.0))
            mean = float(values.get("mean_chunk_seconds", 0.0))
            workers.append(
                TurboServeWorkerLoad(
                    worker_id=state.worker_id,
                    capacity=state.session_capacity,
                    active_sessions=int(values.get("active_sessions", 0)),
                    retained_sessions=len(state.session_ids),
                    predicted_chunk_latency_seconds=max(p95, mean, 1e-6),
                    ready=state.status not in {"failed", "stopped"},
                    draining=state.status == "draining",
                )
            )
        if len(workers) < 2:
            return
        profiles = self._worker_capacity_profiles
        sessions: list[TurboServeSessionDemand] = []
        for record in self.registry.list_records():
            if (
                record.status in TERMINAL_SESSION_STATUSES
                or record.pipeline_session_id is None
                or record.worker_id is None
            ):
                continue
            profile = profiles.get(record.worker_id, {})
            state_bytes = int(profile.get("estimated_session_bytes", 1)) if isinstance(profile, dict) else 1
            sessions.append(
                TurboServeSessionDemand(
                    session_id=record.session_id,
                    active=True,
                    state_bytes=max(1, state_bytes),
                    owner_worker_id=record.worker_id,
                )
            )
        plans = self._placement_controller.plan_rebalance(sessions, workers)
        if not plans:
            return
        plan = plans[0]
        try:
            await self.migrate_session(plan.session_id, plan.target_worker_id)
        except Exception as exc:
            self._last_migration_error = str(exc)
            logger.warning(
                f"TurboServe rebalance migration failed: session={plan.session_id} "
                f"source={plan.source_worker_id} target={plan.target_worker_id} error={exc}"
            )
            return
        self._last_migration_plan = plan
        self._last_migration_error = None

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
