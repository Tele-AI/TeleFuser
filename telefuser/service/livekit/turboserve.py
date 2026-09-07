"""TurboServe workload, placement, migration, and autoscaling controllers."""

from __future__ import annotations

import math
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TurboServeSessionDemand:
    """Scheduler-visible demand and residency facts for one streaming session."""

    session_id: str
    active: bool
    state_bytes: int
    owner_worker_id: str | None = None
    migration_bytes: int | None = None


@dataclass(frozen=True)
class TurboServeWorkerLoad:
    """One worker's measured and predicted load."""

    worker_id: str
    capacity: int
    active_sessions: int
    retained_sessions: int
    predicted_chunk_latency_seconds: float
    ready: bool = True
    draining: bool = False


@dataclass(frozen=True)
class TurboServePlacementDecision:
    """Placement result with its predicted global bottleneck latency."""

    worker_id: str
    predicted_bottleneck_seconds: float


@dataclass(frozen=True)
class TurboServeMigrationPlan:
    """A beneficial chunk-boundary state movement."""

    session_id: str
    source_worker_id: str
    target_worker_id: str
    gain_seconds: float
    migration_cost_seconds: float


@dataclass(frozen=True)
class TurboServeWorkloadSnapshot:
    """Sliding-window demand statistics used by placement and autoscaling."""

    active_sessions: int
    arrivals_per_second: float
    activation_volatility: float
    mean_chunk_seconds: float
    p95_chunk_seconds: float
    observed_at: float


@dataclass(frozen=True)
class TurboServeScaleDecision:
    """Desired worker count and the reason for changing or retaining it."""

    current_workers: int
    target_workers: int
    action: Literal["scale_out", "scale_in", "hold"]
    target_utilization: float
    reason: str


class TurboServeWorkloadDetector:
    """Track long-lived arrivals, active/idle transitions, and chunk service time."""

    def __init__(self, window_seconds: float = 60.0, volatility_bins: int = 10) -> None:
        if window_seconds <= 0 or volatility_bins < 2:
            raise ValueError("window_seconds must be positive and volatility_bins at least two")
        self.window_seconds = float(window_seconds)
        self.volatility_bins = int(volatility_bins)
        self._events: deque[tuple[float, str, str]] = deque()
        self._chunks: deque[tuple[float, float]] = deque()
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def record_arrival(self, session_id: str, now: float | None = None) -> None:
        self._record(session_id, "arrival", now)

    def record_active(self, session_id: str, now: float | None = None) -> None:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            self._active.add(session_id)
            self._events.append((observed_at, "active", session_id))
            self._trim(observed_at)

    def record_idle(self, session_id: str, now: float | None = None) -> None:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            self._active.discard(session_id)
            self._events.append((observed_at, "idle", session_id))
            self._trim(observed_at)

    def record_departure(self, session_id: str, now: float | None = None) -> None:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            self._active.discard(session_id)
            self._events.append((observed_at, "departure", session_id))
            self._trim(observed_at)

    def record_chunk(self, duration_seconds: float, now: float | None = None) -> None:
        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            self._chunks.append((observed_at, float(duration_seconds)))
            self._trim(observed_at)

    def snapshot(self, now: float | None = None) -> TurboServeWorkloadSnapshot:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            self._trim(observed_at)
            arrivals = sum(1 for _, event, _ in self._events if event == "arrival")
            durations = sorted(duration for _, duration in self._chunks)
            mean_chunk = statistics.fmean(durations) if durations else 0.0
            p95_index = max(0, math.ceil(len(durations) * 0.95) - 1)
            p95_chunk = durations[p95_index] if durations else 0.0
            volatility = self._volatility(observed_at)
            return TurboServeWorkloadSnapshot(
                active_sessions=len(self._active),
                arrivals_per_second=arrivals / self.window_seconds,
                activation_volatility=volatility,
                mean_chunk_seconds=mean_chunk,
                p95_chunk_seconds=p95_chunk,
                observed_at=observed_at,
            )

    def _record(self, session_id: str, event: str, now: float | None) -> None:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            self._events.append((observed_at, event, session_id))
            self._trim(observed_at)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        while self._chunks and self._chunks[0][0] < cutoff:
            self._chunks.popleft()

    def _volatility(self, now: float) -> float:
        bin_width = self.window_seconds / self.volatility_bins
        counts = [0] * self.volatility_bins
        cutoff = now - self.window_seconds
        for observed_at, event, _ in self._events:
            if event not in {"active", "idle"}:
                continue
            index = min(self.volatility_bins - 1, max(0, int((observed_at - cutoff) / bin_width)))
            counts[index] += 1
        mean = statistics.fmean(counts)
        return statistics.pstdev(counts) / mean if mean > 0 else 0.0


class TurboServePlacementController:
    """Minimize predicted bottleneck latency and migration-aware rebalance cost."""

    def __init__(self, migration_bandwidth_bytes_per_second: float, migration_penalty: float = 1.0) -> None:
        if migration_bandwidth_bytes_per_second <= 0 or migration_penalty < 0:
            raise ValueError("migration bandwidth must be positive and migration_penalty non-negative")
        self.migration_bandwidth_bytes_per_second = float(migration_bandwidth_bytes_per_second)
        self.migration_penalty = float(migration_penalty)

    def place(
        self,
        demand: TurboServeSessionDemand,
        workers: list[TurboServeWorkerLoad],
    ) -> TurboServePlacementDecision:
        candidates = [
            worker
            for worker in workers
            if worker.ready and not worker.draining and worker.retained_sessions < worker.capacity
        ]
        if not candidates:
            raise RuntimeError("No TurboServe worker has retained-session capacity")
        if demand.owner_worker_id is not None:
            retained = next((item for item in candidates if item.worker_id == demand.owner_worker_id), None)
            if retained is not None:
                return TurboServePlacementDecision(
                    retained.worker_id,
                    max(item.predicted_chunk_latency_seconds for item in workers),
                )
        best = min(
            candidates,
            key=lambda candidate: (
                self._predicted_bottleneck_after_add(candidate.worker_id, workers),
                candidate.active_sessions,
                candidate.retained_sessions,
                candidate.worker_id,
            ),
        )
        return TurboServePlacementDecision(
            best.worker_id,
            self._predicted_bottleneck_after_add(best.worker_id, workers),
        )

    def plan_rebalance(
        self,
        sessions: list[TurboServeSessionDemand],
        workers: list[TurboServeWorkerLoad],
    ) -> list[TurboServeMigrationPlan]:
        if len(workers) < 2:
            return []
        bottleneck = max(workers, key=lambda worker: worker.predicted_chunk_latency_seconds)
        old_max = bottleneck.predicted_chunk_latency_seconds
        best: TurboServeMigrationPlan | None = None
        for session in sessions:
            if session.owner_worker_id != bottleneck.worker_id:
                continue
            for target in workers:
                if (
                    target.worker_id == bottleneck.worker_id
                    or not target.ready
                    or target.draining
                    or target.retained_sessions >= target.capacity
                ):
                    continue
                new_max = self._predicted_bottleneck_after_move(bottleneck, target, workers)
                migration_bytes = session.migration_bytes or session.state_bytes
                migration_cost = migration_bytes / self.migration_bandwidth_bytes_per_second
                gain = old_max - new_max - self.migration_penalty * migration_cost
                candidate = TurboServeMigrationPlan(
                    session.session_id,
                    bottleneck.worker_id,
                    target.worker_id,
                    gain,
                    migration_cost,
                )
                if gain > 0 and (best is None or candidate.gain_seconds > best.gain_seconds):
                    best = candidate
        return [best] if best is not None else []

    @staticmethod
    def _predicted_bottleneck_after_add(worker_id: str, workers: list[TurboServeWorkerLoad]) -> float:
        predictions = []
        for worker in workers:
            latency = worker.predicted_chunk_latency_seconds
            if worker.worker_id == worker_id:
                latency *= (worker.active_sessions + 1) / max(1, worker.active_sessions)
            predictions.append(latency)
        return max(predictions)

    @staticmethod
    def _predicted_bottleneck_after_move(
        source: TurboServeWorkerLoad,
        target: TurboServeWorkerLoad,
        workers: list[TurboServeWorkerLoad],
    ) -> float:
        predictions = []
        for worker in workers:
            latency = worker.predicted_chunk_latency_seconds
            if worker.worker_id == source.worker_id:
                latency *= max(0, worker.active_sessions - 1) / max(1, worker.active_sessions)
            elif worker.worker_id == target.worker_id:
                latency *= (worker.active_sessions + 1) / max(1, worker.active_sessions)
            predictions.append(latency)
        return max(predictions)


class TurboServeAutoscalingController:
    """Compute a hysteretic worker target from active demand and profiled capacity."""

    def __init__(
        self,
        sessions_per_worker: int,
        target_utilization: float = 0.75,
        hysteresis: float = 0.10,
        cooldown_seconds: float = 30.0,
        min_workers: int = 1,
        max_workers: int = 64,
    ) -> None:
        if sessions_per_worker < 1 or not 0 < target_utilization <= 1 or not 0 <= hysteresis < 1:
            raise ValueError("Invalid TurboServe autoscaling capacity or utilization")
        if cooldown_seconds < 0 or not 1 <= min_workers <= max_workers:
            raise ValueError("Invalid TurboServe autoscaling cooldown or worker bounds")
        self.sessions_per_worker = int(sessions_per_worker)
        self.target_utilization = float(target_utilization)
        self.hysteresis = float(hysteresis)
        self.cooldown_seconds = float(cooldown_seconds)
        self.min_workers = int(min_workers)
        self.max_workers = int(max_workers)
        self._last_scale_at: float | None = None

    def decide(
        self,
        active_sessions: int,
        current_workers: int,
        *,
        activation_volatility: float = 0.0,
        now: float | None = None,
    ) -> TurboServeScaleDecision:
        if active_sessions < 0 or current_workers < 1:
            raise ValueError("active_sessions must be non-negative and current_workers positive")
        observed_at = time.monotonic() if now is None else now
        volatility_margin = min(0.25, max(0.0, activation_volatility) * 0.05)
        utilization = max(0.25, self.target_utilization - volatility_margin)
        raw_target = math.ceil(active_sessions / (self.sessions_per_worker * utilization)) if active_sessions else 1
        raw_target = min(self.max_workers, max(self.min_workers, raw_target))
        current_capacity = current_workers * self.sessions_per_worker
        current_utilization = active_sessions / current_capacity
        upper = min(1.0, utilization + self.hysteresis)
        lower = max(0.0, utilization - self.hysteresis)
        if self._last_scale_at is not None and observed_at - self._last_scale_at < self.cooldown_seconds:
            return TurboServeScaleDecision(current_workers, current_workers, "hold", utilization, "cooldown")
        if raw_target > current_workers and current_utilization > upper:
            self._last_scale_at = observed_at
            return TurboServeScaleDecision(current_workers, raw_target, "scale_out", utilization, "above_upper_band")
        if raw_target < current_workers and current_utilization < lower:
            self._last_scale_at = observed_at
            return TurboServeScaleDecision(current_workers, raw_target, "scale_in", utilization, "below_lower_band")
        return TurboServeScaleDecision(current_workers, current_workers, "hold", utilization, "within_hysteresis")


@dataclass(frozen=True)
class TurboServeOwnership:
    """Committed owner and monotonically increasing epoch for one session."""

    session_id: str
    worker_id: str
    epoch: int


@dataclass(frozen=True)
class TurboServeMigrationToken:
    """Opaque prepare record used to commit or abort a migration transaction."""

    token_id: str
    session_id: str
    source_worker_id: str
    target_worker_id: str
    source_epoch: int


class TurboServeOwnershipTable:
    """Serialize prepare/commit migration ownership at chunk boundaries."""

    def __init__(self) -> None:
        self._owners: dict[str, TurboServeOwnership] = {}
        self._pending: dict[str, TurboServeMigrationToken] = {}
        self._lock = threading.RLock()

    def register(self, session_id: str, worker_id: str) -> TurboServeOwnership:
        with self._lock:
            if session_id in self._owners:
                raise ValueError(f"TurboServe session {session_id!r} already has an owner")
            ownership = TurboServeOwnership(session_id, worker_id, 1)
            self._owners[session_id] = ownership
            return ownership

    def owner(self, session_id: str) -> TurboServeOwnership:
        with self._lock:
            return self._owners[session_id]

    def prepare_migration(
        self,
        session_id: str,
        source_worker_id: str,
        target_worker_id: str,
    ) -> TurboServeMigrationToken:
        with self._lock:
            owner = self._owners[session_id]
            if owner.worker_id != source_worker_id:
                raise RuntimeError("Migration source does not own the session")
            if session_id in self._pending:
                raise RuntimeError("Session already has a pending migration")
            token = TurboServeMigrationToken(
                str(uuid.uuid4()),
                session_id,
                source_worker_id,
                target_worker_id,
                owner.epoch,
            )
            self._pending[session_id] = token
            return token

    def commit_migration(self, token: TurboServeMigrationToken) -> TurboServeOwnership:
        with self._lock:
            if self._pending.get(token.session_id) != token:
                raise RuntimeError("Migration token is stale or already completed")
            owner = self._owners[token.session_id]
            if owner.worker_id != token.source_worker_id or owner.epoch != token.source_epoch:
                raise RuntimeError("Session ownership changed while migration was prepared")
            committed = TurboServeOwnership(token.session_id, token.target_worker_id, owner.epoch + 1)
            self._owners[token.session_id] = committed
            del self._pending[token.session_id]
            return committed

    def abort_migration(self, token: TurboServeMigrationToken) -> None:
        with self._lock:
            if self._pending.get(token.session_id) == token:
                del self._pending[token.session_id]

    def release(self, session_id: str) -> None:
        with self._lock:
            self._pending.pop(session_id, None)
            self._owners.pop(session_id, None)


# The classes below intentionally mirror the closed-loop scheduler in the
# TurboServe reference implementation. The older controllers above are kept
# for API compatibility with the first TeleFuser prototype.


@dataclass(frozen=True)
class TurboServeSchedulerConfig:
    """Knobs of TurboServe's budget sizing and migration-aware placement."""

    enable_autoscaling: bool = True
    enable_migration: bool = True
    min_workers: int = 1
    max_workers: int = 64
    capacity_per_worker: int = 1
    target_utilization: float = 0.9
    scale_in_hold_seconds: float = 5.0
    migration_eta: float = 0.35
    min_gain_ms: float = 40.0
    rebalance_iteration_limit: int = 3


@dataclass(frozen=True)
class TurboServeSessionView:
    """One session as seen by the cluster controller at a control boundary."""

    session_id: str
    active: bool
    state_size_mb: float
    chunk_compute_units: float = 1.0
    prompt_tokens: int = 256
    resolution: str = "480p"
    frame_count: int = 9


@dataclass(frozen=True)
class TurboServeRuntimeCalibration:
    """Measured values that replace the migration-model cold estimate."""

    average_migration_total_ms: float = 0.0
    base_chunk_latency_ms: float = 0.0


@dataclass(frozen=True)
class TurboServeSchedulingSnapshot:
    """Complete input to one source-compatible scheduling decision."""

    time_seconds: float
    sessions: dict[str, TurboServeSessionView]
    placement: dict[str, str | None]
    current_workers: int
    worker_order: tuple[str, ...]
    capacity_per_worker: int
    runtime_calibration: TurboServeRuntimeCalibration = field(default_factory=TurboServeRuntimeCalibration)


@dataclass(frozen=True)
class TurboServeSchedulingDecision:
    """Worker budget and requested active-session placement for one tick."""

    worker_budget: int
    placement: dict[str, str | None]
    metadata: dict[str, object]


@dataclass
class TurboServeLatencyModel:
    """The reference analytic chunk and migration latency model, in ms."""

    migration_alpha_ms: float = 8.0
    migration_bandwidth_mb_per_ms: float = 32.0
    base_chunk_latency_ms: float = 180.0
    load_penalty_ms: float = 55.0
    quadratic_load_penalty_ms: float = 18.0
    prompt_token_penalty_ms: float = 0.015
    frame_reference_count: float = 64.0
    frame_exponent: float = 0.5
    min_frame_factor: float = 0.5
    resolution_factors: dict[str, float] = field(
        default_factory=lambda: {
            "360p": 0.45,
            "480p": 0.65,
            "720p": 1.0,
            "1080p": 1.7,
            "4k": 3.2,
        }
    )

    def migration_cost_ms(self, session: TurboServeSessionView, calibration: TurboServeRuntimeCalibration) -> float:
        if calibration.average_migration_total_ms > 0:
            return calibration.average_migration_total_ms
        return self.migration_alpha_ms + session.state_size_mb / max(1e-9, self.migration_bandwidth_mb_per_ms)

    def session_latency_ms(
        self,
        session: TurboServeSessionView,
        colocated_sessions: int,
        capacity_per_worker: int,
        calibration: TurboServeRuntimeCalibration,
    ) -> float:
        base = calibration.base_chunk_latency_ms or self.base_chunk_latency_ms
        load = max(1, colocated_sessions)
        normalized = load / max(1, capacity_per_worker)
        compute = base * session.chunk_compute_units
        compute *= self.resolution_factors.get(session.resolution, 1.0)
        compute *= max(
            self.min_frame_factor,
            (max(1, session.frame_count) / self.frame_reference_count) ** self.frame_exponent,
        )
        return (
            compute
            + self.prompt_token_penalty_ms * session.prompt_tokens
            + self.load_penalty_ms * (load - 1)
            + self.quadratic_load_penalty_ms * normalized * normalized
        )


class TurboServeClusterScheduler:
    """Source-aligned closed-loop budget and migration-aware placement."""

    def __init__(
        self, config: TurboServeSchedulerConfig | None = None, latency_model: TurboServeLatencyModel | None = None
    ) -> None:
        self.config = config or TurboServeSchedulerConfig()
        self.latency_model = latency_model or TurboServeLatencyModel()
        self._scale_in_target: int | None = None
        self._scale_in_deadline_seconds: float | None = None

    def decide(self, snapshot: TurboServeSchedulingSnapshot) -> TurboServeSchedulingDecision:
        budget, action = self._autoscale_budget(snapshot)
        placement, metadata = self._place_at_budget(snapshot, budget)
        metadata.update(
            {
                "scheduler": "turboserve",
                "autoscale_action": action,
                "worker_budget": budget,
                "active_sessions": sum(session.active for session in snapshot.sessions.values()),
                "target_utilization": self.config.target_utilization,
            }
        )
        return TurboServeSchedulingDecision(budget, placement, metadata)

    def _autoscale_budget(self, snapshot: TurboServeSchedulingSnapshot) -> tuple[int, str]:
        current = self._clamp(snapshot.current_workers, len(snapshot.worker_order))
        if not self.config.enable_autoscaling:
            self._scale_in_target = self._scale_in_deadline_seconds = None
            return current, "disabled"
        active = sum(session.active for session in snapshot.sessions.values())
        capacity = max(1, min(snapshot.capacity_per_worker, self.config.capacity_per_worker))
        target = self._target_budget(active, capacity, len(snapshot.worker_order))
        if not self.config.enable_migration:
            for session_id, session in snapshot.sessions.items():
                owner = snapshot.placement.get(session_id)
                if session.active and owner in snapshot.worker_order:
                    target = max(target, snapshot.worker_order.index(owner) + 1)
        if target > current:
            self._scale_in_target = self._scale_in_deadline_seconds = None
            return target, "scale_out"
        if target < current:
            if self._scale_in_target != target or self._scale_in_deadline_seconds is None:
                self._scale_in_target = target
                self._scale_in_deadline_seconds = snapshot.time_seconds + max(0.0, self.config.scale_in_hold_seconds)
            if snapshot.time_seconds >= self._scale_in_deadline_seconds:
                self._scale_in_target = self._scale_in_deadline_seconds = None
                return target, "scale_in"
            return current, "hold_scale_in"
        self._scale_in_target = self._scale_in_deadline_seconds = None
        return current, "hold"

    def _target_budget(self, active: int, capacity: int, maximum: int) -> int:
        if active <= 0:
            return self._clamp(self.config.min_workers, maximum)
        utilization = min(1.0, max(0.01, self.config.target_utilization))
        hard = math.ceil(active / capacity)
        target = math.ceil(active / (capacity * utilization))
        return self._clamp(max(hard, target), maximum)

    def _clamp(self, value: int, maximum: int) -> int:
        return min(maximum, self.config.max_workers, max(self.config.min_workers, int(value)))

    def _place_at_budget(
        self, snapshot: TurboServeSchedulingSnapshot, budget: int
    ) -> tuple[dict[str, str | None], dict[str, object]]:
        workers = tuple(snapshot.worker_order[:budget])
        capacity = max(1, min(snapshot.capacity_per_worker, self.config.capacity_per_worker))
        loads: dict[str, list[str]] = {worker: [] for worker in workers}
        placement: dict[str, str | None] = {}
        pending: list[str] = []
        for session_id in sorted(snapshot.sessions):
            session = snapshot.sessions[session_id]
            owner = snapshot.placement.get(session_id)
            if not session.active:
                placement[session_id] = None
            elif owner in loads and (not self.config.enable_migration or len(loads[owner]) < capacity):
                placement[session_id] = owner
                loads[owner].append(session_id)
            else:
                placement[session_id] = None
                pending.append(session_id)
        for session_id in pending:
            feasible = [worker for worker, sessions in loads.items() if len(sessions) < capacity]
            if feasible:
                target = min(feasible, key=lambda worker: (len(loads[worker]), worker))
                placement[session_id] = target
                loads[target].append(session_id)
        before = self._bottleneck(snapshot, loads, capacity)
        moves = evaluations = 0
        if self.config.enable_migration:
            moves, evaluations = self._rebalance(snapshot, loads, placement, capacity)
        after = self._bottleneck(snapshot, loads, capacity)
        unplaced = sum(
            session.active and placement.get(session_id) is None for session_id, session in snapshot.sessions.items()
        )
        rho_max = max((len(items) / capacity for items in loads.values()), default=0.0)
        return placement, {
            "algorithm": "least_load_with_optional_rebalance",
            "capacity_per_worker": capacity,
            "rebalance_moves": moves,
            "candidate_evaluations": evaluations,
            "unplaced_active": unplaced,
            "bottleneck_before_ms": round(before, 3),
            "bottleneck_after_ms": round(after, 3),
            "rho_max": round(rho_max, 4),
        }

    def _rebalance(
        self,
        snapshot: TurboServeSchedulingSnapshot,
        loads: dict[str, list[str]],
        placement: dict[str, str | None],
        capacity: int,
    ) -> tuple[int, int]:
        moves = evaluations = 0
        for _ in range(self.config.rebalance_iteration_limit):
            if not loads:
                break
            source = max(
                loads,
                key=lambda worker: (self._worker_worst(snapshot, loads[worker], capacity), len(loads[worker]), worker),
            )
            if not loads[source]:
                break
            current = self._bottleneck(snapshot, loads, capacity)
            best: tuple[tuple[float, float, str, str], str, str] | None = None
            for session_id in tuple(loads[source]):
                session = snapshot.sessions[session_id]
                migration = self.latency_model.migration_cost_ms(session, snapshot.runtime_calibration)
                for target, target_load in loads.items():
                    if target == source or len(target_load) >= capacity:
                        continue
                    evaluations += 1
                    candidate = {key: list(value) for key, value in loads.items()}
                    candidate[source].remove(session_id)
                    candidate[target].append(session_id)
                    gain = current - self._bottleneck(snapshot, candidate, capacity)
                    gain -= self.config.migration_eta * migration
                    score = (gain, -session.state_size_mb, target, session_id)
                    if best is None or score > best[0]:
                        best = (score, session_id, target)
            if best is None or best[0][0] <= self.config.min_gain_ms:
                break
            _, session_id, target = best
            loads[source].remove(session_id)
            loads[target].append(session_id)
            placement[session_id] = target
            moves += 1
        return moves, evaluations

    def _bottleneck(self, snapshot: TurboServeSchedulingSnapshot, loads: dict[str, list[str]], capacity: int) -> float:
        return max((self._worker_worst(snapshot, sessions, capacity) for sessions in loads.values()), default=0.0)

    def _worker_worst(self, snapshot: TurboServeSchedulingSnapshot, sessions: list[str], capacity: int) -> float:
        if not sessions:
            return 0.0
        return max(
            self.latency_model.session_latency_ms(
                snapshot.sessions[session_id], len(sessions), capacity, snapshot.runtime_calibration
            )
            for session_id in sessions
        )
