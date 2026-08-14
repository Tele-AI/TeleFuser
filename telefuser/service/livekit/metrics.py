"""Low-cardinality Prometheus metrics for LiveKit stream serving.

The API process is the Prometheus scrape target in a process-NCCL deployment.
This collector consumes scheduler state and model-output events forwarded to that
process. It deliberately never exposes session IDs as labels: per-session
performance is exported as a distribution instead.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .runtime import LiveKitServeRuntime


_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0, 10.0)
_BATCH_BUCKETS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, 32.0)
_FPS_WINDOW_SECONDS = 30.0
_TERMINAL_STATUSES = frozenset({"closed", "failed", "expired"})


def _labels(labels: dict[str, object] | None = None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items()))


def _label_text(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""

    def escape(value: str) -> str:
        return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    return "{" + ",".join(f'{key}="{escape(value)}"' for key, value in labels) + "}"


def _number(value: float | int) -> str:
    number = float(value)
    return f"{number:.12g}" if math.isfinite(number) else "0"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * q) - 1))]


@dataclass
class _Histogram:
    buckets: tuple[float, ...]
    counts: list[int] = field(init=False)
    count: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        self.counts = [0 for _ in (*self.buckets, float("inf"))]

    def observe(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        self.count += 1
        self.total += value
        for index, upper_bound in enumerate((*self.buckets, float("inf"))):
            if value <= upper_bound:
                self.counts[index] += 1


class LiveKitServingMetrics:
    """Cumulative serving measurements plus scrape-time scheduler gauges."""

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._lock = threading.RLock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}
        self._pending_action_at: dict[str, float] = {}
        self._frame_events: deque[tuple[float, str, int]] = deque()

    def record_admission(self, result: str) -> None:
        self._inc("telefuser_serving_session_admissions_total", {"result": result})

    def record_session_finished(self, status: str, error: str | None = None) -> None:
        result = "failed" if status == "failed" or error else "closed"
        self._inc("telefuser_serving_sessions_finished_total", {"result": result})
        if error:
            self._record_error(error)

    def record_migration(
        self,
        *,
        success: bool,
        duration_seconds: float | None = None,
        error: str | None = None,
    ) -> None:
        self._inc("telefuser_serving_migrations_total", {"result": "success" if success else "error"})
        if duration_seconds is not None:
            self._observe("telefuser_serving_migration_duration_seconds", {}, duration_seconds)
        if error:
            self._record_error(error)

    def on_control_received(self, worker_id: str, session_id: str, received_at: float | None = None) -> None:
        """Anchor A2F at validated controller-action ingress."""
        del worker_id
        with self._lock:
            self._pending_action_at[session_id] = time.monotonic() if received_at is None else float(received_at)
            self._inc_locked("telefuser_serving_actions_total", {})

    def on_model_output(
        self,
        runtime: LiveKitServeRuntime,
        *,
        worker_id: str,
        pipeline_session_id: str,
        payload: dict[str, Any],
        runtime_metrics: dict[str, Any] | None = None,
        session_runtime_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Ingest a model output before network transport and frame pacing."""
        del worker_id, runtime_metrics, session_runtime_metrics
        if not isinstance(payload, dict):
            return
        if payload.get("type") == "error":
            self._inc("telefuser_serving_model_outputs_total", {"result": "error"})
            self._record_error(str(payload.get("error", "model output error")))
            return
        if payload.get("type") != "chunk":
            return

        scheduler = payload.get("scheduler")
        if not isinstance(scheduler, dict):
            nested = payload.get("data")
            scheduler = (
                nested.get("scheduler")
                if isinstance(nested, dict) and isinstance(nested.get("scheduler"), dict)
                else {}
            )
        frames = payload.get("frames")
        if not isinstance(frames, (list, tuple)):
            nested = payload.get("data")
            frames = nested.get("frames") if isinstance(nested, dict) else None
        if isinstance(frames, (list, tuple)):
            frame_count = len(frames)
        else:
            frame_count = int(self._nonnegative(payload.get("frame_count")) or 0)
        batch_size = self._positive(scheduler.get("batch_size"), default=1.0)
        compute_seconds = self._nonnegative(scheduler.get("compute_seconds"))
        queue_wait_seconds = self._nonnegative(scheduler.get("queue_wait_seconds"))
        taew_decode = self._taew_decode_measurement(scheduler)

        with self._lock:
            self._inc_locked("telefuser_serving_model_outputs_total", {"result": "chunk"})
            self._inc_locked("telefuser_serving_chunks_total", {"result": "processed"})
            # Every member observes the same batch. Fractional counting makes
            # the total exactly one per coalesced execution after all members.
            self._inc_locked("telefuser_serving_batches_total", {}, amount=1.0 / batch_size)
            self._inc_locked("telefuser_serving_batch_items_total", {})
            self._observe_locked("telefuser_serving_batch_size", {}, batch_size, buckets=_BATCH_BUCKETS)
            if taew_decode is not None:
                mode, items, invocations = taew_decode
                # Each session output carries the same batch-level facts.  Count
                # one logical item per output and split native decoder calls
                # across those items, so a DiT B=2 serial fallback contributes
                # two TAeW invocations while a synchronized decode contributes one.
                self._inc_locked(f"telefuser_serving_taew_decode_{mode}_items_total", {})
                self._inc_locked(
                    f"telefuser_serving_taew_decode_{mode}_executions_total",
                    {},
                    amount=invocations / items,
                )
            if compute_seconds is not None:
                self._observe_locked("telefuser_serving_chunk_latency_seconds", {}, compute_seconds)
            if queue_wait_seconds is not None:
                self._observe_locked("telefuser_serving_queue_wait_seconds", {}, queue_wait_seconds)
            for key, stage in (
                ("input_prepare_seconds", "input_prepare"),
                ("cache_collate_seconds", "cache_collate"),
                ("denoise_seconds", "dit"),
                ("cache_scatter_seconds", "cache_scatter"),
                ("vae_encode_seconds", "vae_encode"),
                ("vae_decode_seconds", "vae_decode"),
                ("postprocess_seconds", "postprocess"),
            ):
                value = self._nonnegative(scheduler.get(key))
                if value is not None:
                    self._observe_locked("telefuser_serving_pipeline_stage_latency_seconds", {"stage": stage}, value)

            fps = self._session_fps(runtime, pipeline_session_id, payload)
            elapsed = (compute_seconds or 0.0) + (queue_wait_seconds or 0.0)
            if frame_count and fps is not None and elapsed > 0:
                budget = frame_count / fps
                self._observe_locked("telefuser_serving_slo_budget_seconds", {}, budget)
                self._inc_locked(
                    "telefuser_serving_slo_chunks_total",
                    {"result": "met" if elapsed <= budget else "missed"},
                )

    def on_chunk_published(
        self,
        *,
        worker_id: str,
        session_id: str,
        frames: int,
        first_frame_at: float | None = None,
    ) -> None:
        """Record frames handed to LiveKit's video publisher.

        This intentionally happens after model completion and video pacing. It
        is the server-side measure closest to client-visible end-to-end FPS;
        receiver decode and network jitter remain client-side observability.
        """
        del worker_id
        if frames <= 0:
            return
        published_at = time.monotonic()
        first_at = published_at if first_frame_at is None else float(first_frame_at)
        with self._lock:
            self._inc_locked("telefuser_serving_chunks_total", {"result": "published"})
            self._inc_locked("telefuser_serving_frames_total", {"result": "published"}, amount=float(frames))
            action_at = self._pending_action_at.pop(session_id, None)
            if action_at is not None and first_at >= action_at:
                self._observe_locked("telefuser_serving_action_to_first_frame_seconds", {}, first_at - action_at)
            self._frame_events.append((published_at, session_id, int(frames)))
            self._trim_frame_events_locked(published_at)

    def render_prometheus(self, runtime: LiveKitServeRuntime) -> str:
        """Render a low-cardinality Prometheus exposition fragment."""
        state = self._state(runtime)
        lines: list[str] = []
        self._render_gauges(lines, state["gauges"])
        self._render_counters(lines, state["counters"])
        self._render_histograms(lines, state["histograms"])
        return "\n".join(lines)

    def json_snapshot(self, runtime: LiveKitServeRuntime) -> dict[str, Any]:
        """Return a compact aggregate-only JSON view for experiment tooling."""
        state = self._state(runtime)
        return {
            "summary": state["summary"],
            "counters": {name + _label_text(labels): value for (name, labels), value in state["counters"].items()},
        }

    def _state(self, runtime: LiveKitServeRuntime) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            self._trim_frame_events_locked(now)
            counters = dict(self._counters)
            histograms = dict(self._histograms)
            frame_events = tuple(self._frame_events)

        health = runtime.health().model_dump()
        workers = runtime.scheduler.workers()
        records = runtime.registry.list_records()
        snapshot_fn = getattr(runtime.worker_pool, "turboserve_snapshot", None)
        routing = snapshot_fn() if callable(snapshot_fn) else {}
        routing = routing if isinstance(routing, dict) else {}
        worker_metrics = routing.get("worker_runtime_metrics", {})
        worker_metrics = worker_metrics if isinstance(worker_metrics, dict) else {}
        session_metrics = routing.get("session_runtime_metrics", {})
        session_metrics = session_metrics if isinstance(session_metrics, dict) else {}

        active = 0
        retained = 0
        status_counts: dict[str, int] = defaultdict(int)
        for record in records:
            status_counts[str(record.status)] += 1
            if record.status in _TERMINAL_STATUSES:
                continue
            if record.worker_id is not None:
                retained += 1
            metrics = session_metrics.get(record.pipeline_session_id, {}) if record.pipeline_session_id else {}
            if isinstance(metrics, dict) and bool(metrics.get("active", 0)):
                active += 1
            elif record.status == "running":
                active += 1
        idle = max(0, retained - active)

        gauges: list[tuple[str, str, tuple[tuple[str, str], ...], float]] = []

        def gauge(name: str, description: str, value: float | int, labels: dict[str, object] | None = None) -> None:
            gauges.append((name, description, _labels(labels), float(value)))

        gauge(
            "telefuser_serving_uptime_seconds",
            "Seconds since the LiveKit serving metrics collector started",
            now - self._started_at,
        )
        for state, value in (
            ("retained", retained),
            ("active", active),
            ("idle", idle),
            ("waiting", health["queued_sessions"]),
        ):
            gauge(
                "telefuser_serving_sessions",
                "Current sessions grouped by scheduler state",
                int(value),
                {"state": state},
            )
        for status, count in sorted(status_counts.items()):
            gauge(
                "telefuser_serving_session_status",
                "Current sessions grouped by public lifecycle status",
                count,
                {"status": status},
            )
        gauge(
            "telefuser_serving_queue_depth",
            "Current scheduler queue depth",
            health["queued_sessions"],
            {"queue": "admission"},
        )
        for state, value in (
            ("configured", health["workers_total"]),
            ("busy", health["workers_busy"]),
            ("idle", health["workers_idle"]),
            ("failed", health["workers_failed"]),
        ):
            gauge("telefuser_serving_workers", "Workers grouped by current state", value, {"state": state})

        scheduler_mode = "unknown"
        for worker in workers:
            values = worker_metrics.get(worker.worker_id, {})
            values = values if isinstance(values, dict) else {}
            scheduler_mode = str(values.get("scheduler_mode", scheduler_mode))
            gpu_ids = worker.gpu_ids or ["unassigned"]
            for gpu_id in gpu_ids:
                labels = {"worker_id": worker.worker_id, "gpu": gpu_id}
                gauge(
                    "telefuser_serving_worker_sessions",
                    "Retained sessions assigned to a worker/GPU group",
                    len(worker.session_ids),
                    labels,
                )
                gauge(
                    "telefuser_serving_worker_capacity",
                    "Retained-session capacity for a worker/GPU group",
                    worker.session_capacity,
                    labels,
                )
                gauge(
                    "telefuser_serving_worker_up",
                    "Whether a configured worker is available",
                    int(worker.status not in {"failed", "stopped"}),
                    labels,
                )
                if worker.session_capacity:
                    gauge(
                        "telefuser_serving_worker_busy_ratio",
                        "Scheduler retained-session occupancy for a worker/GPU group",
                        len(worker.session_ids) / worker.session_capacity,
                        labels,
                    )
                for metric_key, metric_name, description, extra_labels in (
                    (
                        "active_sessions",
                        "telefuser_serving_worker_active_sessions",
                        "Active model sessions reported by a worker",
                        {},
                    ),
                    (
                        "maximum_batch_size",
                        "telefuser_serving_worker_maximum_batch_size",
                        "Largest batch observed by a worker",
                        {},
                    ),
                    (
                        "mean_chunk_seconds",
                        "telefuser_serving_worker_chunk_latency_seconds",
                        "Worker-reported chunk latency",
                        {"stat": "mean"},
                    ),
                    (
                        "p95_chunk_seconds",
                        "telefuser_serving_worker_chunk_latency_seconds",
                        "Worker-reported chunk latency",
                        {"stat": "p95"},
                    ),
                ):
                    value = self._nonnegative(values.get(metric_key))
                    if value is not None:
                        gauge(metric_name, description, value, {**labels, **extra_labels})
                for metric_key, metric_name, description in (
                    (
                        "taew_decode_items",
                        "telefuser_serving_worker_taew_decode_items",
                        "Logical chunks in the latest TAeW LightVAE decode",
                    ),
                    (
                        "taew_decode_batch_size",
                        "telefuser_serving_worker_taew_decode_batch_size",
                        "Effective native batch size in the latest TAeW LightVAE decode",
                    ),
                    (
                        "taew_decode_invocations",
                        "telefuser_serving_worker_taew_decode_invocations",
                        "Native TAeW LightVAE decode calls in the latest model batch",
                    ),
                    (
                        "taew_decode_mode",
                        "telefuser_serving_worker_taew_decode_mode",
                        "TAeW decode mode: 0 singleton, 1 synchronized batch, 2 safe serial fallback",
                    ),
                ):
                    value = self._nonnegative(values.get(metric_key))
                    if value is not None:
                        gauge(metric_name, description, value, labels)
                for stage_key, stage in (
                    ("input_prepare_seconds", "input_prepare"),
                    ("cache_collate_seconds", "cache_collate"),
                    ("denoise_seconds", "dit"),
                    ("cache_scatter_seconds", "cache_scatter"),
                    ("vae_encode_seconds", "vae_encode"),
                    ("vae_decode_seconds", "vae_decode"),
                    ("postprocess_seconds", "postprocess"),
                ):
                    value = self._nonnegative(values.get(stage_key))
                    if value is not None:
                        gauge(
                            "telefuser_serving_worker_pipeline_stage_last_latency_seconds",
                            "Latest worker-reported pipeline stage latency",
                            value,
                            {**labels, "stage": stage},
                        )
        gauge(
            "telefuser_serving_scheduler_mode_info",
            "One for the ABot scheduler mode selected by each worker",
            1,
            {"mode": scheduler_mode},
        )

        batches = counters.get(("telefuser_serving_batches_total", ()), 0.0)
        batch_items = counters.get(("telefuser_serving_batch_items_total", ()), 0.0)
        if batches:
            gauge("telefuser_serving_mean_batch_size", "Mean observed coalesced batch size", batch_items / batches)
        taew_items = sum(
            counters.get((f"telefuser_serving_taew_decode_{mode}_items_total", ()), 0.0)
            for mode in ("singleton", "synchronized", "serial_fallback")
        )
        taew_executions = sum(
            counters.get((f"telefuser_serving_taew_decode_{mode}_executions_total", ()), 0.0)
            for mode in ("singleton", "synchronized", "serial_fallback")
        )
        if taew_executions:
            gauge(
                "telefuser_serving_taew_decode_mean_native_batch_size",
                "Mean effective native TAeW LightVAE batch size across observed decoder calls",
                taew_items / taew_executions,
            )

        met = counters.get(("telefuser_serving_slo_chunks_total", _labels({"result": "met"})), 0.0)
        missed = counters.get(("telefuser_serving_slo_chunks_total", _labels({"result": "missed"})), 0.0)
        if met + missed:
            gauge(
                "telefuser_serving_slo_attainment_ratio",
                "Fraction of chunks meeting their FPS-derived queue-plus-compute budget",
                met / (met + missed),
            )

        fps = self._fps_summary(frame_events, now)
        for scope, value in fps.items():
            gauge(
                "telefuser_serving_published_fps",
                "Published video frame rate over the trailing 30 seconds",
                value,
                {"scope": scope},
            )

        return {
            "gauges": gauges,
            "counters": counters,
            "histograms": histograms,
            "summary": {
                "sessions": {
                    "retained": retained,
                    "active": active,
                    "idle": idle,
                    "waiting": int(health["queued_sessions"]),
                },
                "published_fps": fps,
                "scheduler_mode": scheduler_mode,
                "worker_runtime_metrics": {
                    str(worker_id): dict(values)
                    for worker_id, values in worker_metrics.items()
                    if isinstance(values, dict)
                },
            },
        }

    def _session_fps(
        self, runtime: LiveKitServeRuntime, pipeline_session_id: str, payload: dict[str, Any]
    ) -> float | None:
        for record in runtime.registry.list_records():
            if record.pipeline_session_id == pipeline_session_id:
                value = record.config.get("fps", payload.get("fps", runtime.config.default_fps))
                break
        else:
            value = payload.get("fps", runtime.config.default_fps)
        try:
            fps = float(value)
        except (TypeError, ValueError):
            return None
        return fps if fps > 0 and math.isfinite(fps) else None

    def _fps_summary(self, events: tuple[tuple[float, str, int], ...], now: float) -> dict[str, float]:
        window_start = max(self._started_at, now - _FPS_WINDOW_SECONDS)
        duration = max(1e-6, now - window_start)
        total = 0
        per_session: dict[str, int] = defaultdict(int)
        for timestamp, session_id, frames in events:
            if timestamp >= window_start:
                total += frames
                per_session[session_id] += frames
        rates = [frames / duration for frames in per_session.values()]
        return {
            "aggregate": total / duration,
            "per_active_session_mean": sum(rates) / len(rates) if rates else 0.0,
            "per_active_session_p50": _quantile(rates, 0.50),
            "per_active_session_p95": _quantile(rates, 0.95),
            "per_active_session_min": min(rates) if rates else 0.0,
        }

    def _record_error(self, error: str) -> None:
        normalized = error.lower()
        kind = "oom" if "out of memory" in normalized or "cuda oom" in normalized else "model"
        self._inc("telefuser_serving_errors_total", {"kind": kind})

    def _inc(self, name: str, labels: dict[str, object], amount: float = 1.0) -> None:
        with self._lock:
            self._inc_locked(name, labels, amount)

    def _inc_locked(self, name: str, labels: dict[str, object], amount: float = 1.0) -> None:
        if math.isfinite(amount) and amount >= 0:
            self._counters[(name, _labels(labels))] += amount

    def _observe(
        self,
        name: str,
        labels: dict[str, object],
        value: float,
        *,
        buckets: tuple[float, ...] = _LATENCY_BUCKETS,
    ) -> None:
        with self._lock:
            self._observe_locked(name, labels, value, buckets=buckets)

    def _observe_locked(
        self,
        name: str,
        labels: dict[str, object],
        value: float,
        *,
        buckets: tuple[float, ...] = _LATENCY_BUCKETS,
    ) -> None:
        key = (name, _labels(labels))
        series = self._histograms.get(key)
        if series is None:
            series = self._histograms[key] = _Histogram(buckets=buckets)
        series.observe(float(value))

    def _trim_frame_events_locked(self, now: float) -> None:
        cutoff = now - _FPS_WINDOW_SECONDS
        while self._frame_events and self._frame_events[0][0] < cutoff:
            self._frame_events.popleft()

    @staticmethod
    def _nonnegative(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number >= 0 else None

    @classmethod
    def _positive(cls, value: object, *, default: float) -> float:
        number = cls._nonnegative(value)
        return number if number is not None and number > 0 else default

    @classmethod
    def _taew_decode_measurement(cls, scheduler: dict[str, Any]) -> tuple[str, int, int] | None:
        """Validate batch-local TAeW facts before exporting cumulative counters."""
        items = cls._positive_integer(scheduler.get("taew_decode_items"))
        native_batch_size = cls._positive_integer(scheduler.get("taew_decode_batch_size"))
        invocations = cls._positive_integer(scheduler.get("taew_decode_invocations"))
        mode = cls._nonnegative(scheduler.get("taew_decode_mode"))
        if items is None or native_batch_size is None or invocations is None or mode is None:
            return None
        if not mode.is_integer():
            return None
        if int(mode) == 0 and (items, native_batch_size, invocations) == (1, 1, 1):
            return ("singleton", items, invocations)
        if int(mode) == 1 and items > 1 and (native_batch_size, invocations) == (items, 1):
            return ("synchronized", items, invocations)
        if int(mode) == 2 and items > 1 and (native_batch_size, invocations) == (1, items):
            return ("serial_fallback", items, invocations)
        return None

    @classmethod
    def _positive_integer(cls, value: object) -> int | None:
        number = cls._nonnegative(value)
        if number is None or number <= 0 or not number.is_integer():
            return None
        return int(number)

    @staticmethod
    def _render_gauges(
        lines: list[str],
        gauges: list[tuple[str, str, tuple[tuple[str, str], ...], float]],
    ) -> None:
        grouped: dict[str, tuple[str, list[tuple[tuple[tuple[str, str], ...], float]]]] = {}
        for name, description, labels, value in gauges:
            if name not in grouped:
                grouped[name] = (description, [])
            grouped[name][1].append((labels, value))
        for name, (description, samples) in sorted(grouped.items()):
            lines.extend((f"# HELP {name} {description}", f"# TYPE {name} gauge"))
            lines.extend(f"{name}{_label_text(labels)} {_number(value)}" for labels, value in sorted(samples))

    @staticmethod
    def _render_counters(
        lines: list[str],
        counters: dict[tuple[str, tuple[tuple[str, str], ...]], float],
    ) -> None:
        descriptions = {
            "telefuser_serving_actions_total": "Validated control actions accepted by the serving transport",
            "telefuser_serving_batch_items_total": "Session chunks included in coalesced batches",
            "telefuser_serving_batches_total": "Coalesced model batch executions",
            "telefuser_serving_chunks_total": "Processed or published chunks",
            "telefuser_serving_errors_total": "Serving errors grouped by bounded error class",
            "telefuser_serving_frames_total": "Frames handed to the LiveKit publisher",
            "telefuser_serving_migrations_total": "Session migration attempts grouped by outcome",
            "telefuser_serving_model_outputs_total": "Model output messages grouped by result",
            "telefuser_serving_session_admissions_total": "Session admissions grouped by scheduler result",
            "telefuser_serving_sessions_finished_total": "Terminal sessions grouped by outcome",
            "telefuser_serving_slo_chunks_total": "Chunks grouped by whether their FPS-derived budget was met",
            "telefuser_serving_taew_decode_serial_fallback_executions_total": (
                "Native TAeW LightVAE decode calls made after safe serial fallback"
            ),
            "telefuser_serving_taew_decode_serial_fallback_items_total": (
                "Logical session chunks decoded through safe TAeW serial fallback"
            ),
            "telefuser_serving_taew_decode_singleton_executions_total": (
                "Native TAeW LightVAE decode calls for singleton chunks"
            ),
            "telefuser_serving_taew_decode_singleton_items_total": (
                "Logical singleton session chunks decoded by TAeW LightVAE"
            ),
            "telefuser_serving_taew_decode_synchronized_executions_total": (
                "Native synchronized TAeW LightVAE decode calls"
            ),
            "telefuser_serving_taew_decode_synchronized_items_total": (
                "Logical session chunks decoded in synchronized native TAeW batches"
            ),
        }
        grouped: dict[str, list[tuple[tuple[tuple[str, str], ...], float]]] = defaultdict(list)
        for (name, labels), value in counters.items():
            grouped[name].append((labels, value))
        for name, samples in sorted(grouped.items()):
            lines.extend((f"# HELP {name} {descriptions.get(name, name)}", f"# TYPE {name} counter"))
            lines.extend(f"{name}{_label_text(labels)} {_number(value)}" for labels, value in sorted(samples))

    @staticmethod
    def _render_histograms(
        lines: list[str],
        histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram],
    ) -> None:
        descriptions = {
            "telefuser_serving_action_to_first_frame_seconds": (
                "Validated action ingress to first frame handed to LiveKit"
            ),
            "telefuser_serving_batch_size": "Observed coalesced model batch size",
            "telefuser_serving_chunk_latency_seconds": "Scheduler compute time for one session chunk",
            "telefuser_serving_migration_duration_seconds": "End-to-end migration time",
            "telefuser_serving_pipeline_stage_latency_seconds": "Model pipeline stage latency",
            "telefuser_serving_queue_wait_seconds": "Time a ready session waits before a chunk starts",
            "telefuser_serving_slo_budget_seconds": "FPS-derived queue-plus-compute budget",
        }
        grouped: dict[str, list[tuple[tuple[tuple[str, str], ...], _Histogram]]] = defaultdict(list)
        for (name, labels), series in histograms.items():
            grouped[name].append((labels, series))
        for name, series_list in sorted(grouped.items()):
            lines.extend((f"# HELP {name} {descriptions.get(name, name)}", f"# TYPE {name} histogram"))
            for labels, series in sorted(series_list):
                for bound, count in zip((*series.buckets, float("inf")), series.counts):
                    bucket_labels = _labels({**dict(labels), "le": "+Inf" if math.isinf(bound) else str(bound)})
                    lines.append(f"{name}_bucket{_label_text(bucket_labels)} {count}")
                lines.append(f"{name}_sum{_label_text(labels)} {_number(series.total)}")
                lines.append(f"{name}_count{_label_text(labels)} {series.count}")
