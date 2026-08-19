#!/usr/bin/env python3
"""Run a phase-based, black-box ABot-World LiveKit workload.

The driver intentionally talks only to the public serving interfaces:

* ``POST /v1/stream/sessions`` for admission;
* a real LiveKit/WebRTC room for keyboard controls and video consumption; and
* ``DELETE /v1/stream/sessions/{session_id}`` for departure.

It never selects a GPU, reaches into a worker process, or calls an ABot
pipeline object.  It is therefore suitable for measuring the four-GPU serving
system as a black box.  The LiveKit client flow is the same one used by the
TeleFuser AIPerf adapter, but this tool adds phase-based user arrivals and
departures for long-lived world-model sessions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import json
import os
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

_CONTROL_TOPIC = "tf.control"
_METRICS_TOPIC = "tf.metrics"
_STATUS_TOPIC = "tf.status"
_PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ScenarioError(ValueError):
    """Raised when a user-wave scenario is malformed."""


@dataclass(frozen=True)
class Phase:
    """One target-concurrency interval in a user-wave experiment."""

    name: str
    duration_seconds: float
    target_users: int
    arrival_window_seconds: float = 0.0
    departure_window_seconds: float = 0.0
    active_input_fraction: float = 1.0
    input_transition_window_seconds: float = 0.0


@dataclass(frozen=True)
class ControlConfig:
    """Independent keyboard activity generated for each connected client."""

    interval_seconds: float
    jitter_seconds: float
    idle_probability: float
    idle_min_seconds: float
    idle_max_seconds: float
    action_states: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SessionConfig:
    """ABot request and user-visible playback settings."""

    prompt: str
    image_path: str
    fps: float
    control_latent_frames: int
    delivery_mode: str
    expected_preview_frames: int
    control: ControlConfig


@dataclass(frozen=True)
class AdmissionExpectation:
    """Public-admission contract required for a workload to be valid."""

    require_immediate_assignment: bool = False
    expected_max_sessions_per_worker: int | None = None
    expected_queue_size: int | None = None


@dataclass(frozen=True)
class DiagnosticInitialControlBarrier:
    """Synthetic synchronized-first-control configuration for a diagnostic trace."""

    phase_name: str
    expected_connected_sessions: int
    timeout_seconds: float


@dataclass(frozen=True)
class LifecycleTraceEvent:
    """One explicit user lifecycle transition in a black-box replay."""

    offset_seconds: float
    sequence: int
    event: str
    trace_session_id: str
    source_session_id: int | None
    source_user_id: int | None
    input_enabled: bool | None


@dataclass(frozen=True)
class LifecycleTrace:
    """An exact per-session lifecycle trace, unlike aggregate phase fractions."""

    kind: str
    duration_seconds: float
    events: tuple[LifecycleTraceEvent, ...]


@dataclass(frozen=True)
class Scenario:
    """Validated configuration of one complete black-box experiment."""

    name: str
    server_url: str
    session: SessionConfig
    phases: tuple[Phase, ...]
    sample_interval_seconds: float
    connect_timeout_seconds: float
    http_timeout_seconds: float
    shutdown_timeout_seconds: float
    first_generation_grace_seconds: float
    slo_fps_tolerance: float
    seed: int
    expected_worker_mode: str | None
    admission: AdmissionExpectation
    expected_num_workers: int | None
    diagnostic_initial_control_barrier: DiagnosticInitialControlBarrier | None
    lifecycle_trace: LifecycleTrace | None
    raw: dict[str, Any]


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * quantile + 0.999999) - 1))
    return float(ordered[index])


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 6) if values else 0.0,
        "p50": round(_percentile(values, 0.50), 6),
        "p95": round(_percentile(values, 0.95), 6),
        "p99": round(_percentile(values, 0.99), 6),
        "maximum": round(max(values), 6) if values else 0.0,
    }


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ScenarioError(f"{label} must be an object")
    return dict(value)


def _require_positive_float(value: object, label: str, *, allow_zero: bool = False) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ScenarioError(f"{label} must be a number")
    parsed = float(value)
    if parsed < 0 or (not allow_zero and parsed == 0):
        comparison = "non-negative" if allow_zero else "positive"
        raise ScenarioError(f"{label} must be {comparison}")
    return parsed


def _require_non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScenarioError(f"{label} must be a non-negative integer")
    return int(value)


def _resolve_image_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioError("session.image_path must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    if not path.is_file():
        raise ScenarioError(f"session.image_path does not exist: {path}")
    return str(path)


def _parse_control(value: object) -> ControlConfig:
    raw = _require_mapping(value, "session.control")
    interval = _require_positive_float(raw.get("interval_seconds", 0.5), "session.control.interval_seconds")
    jitter = _require_positive_float(raw.get("jitter_seconds", 0.0), "session.control.jitter_seconds", allow_zero=True)
    idle_probability = _require_positive_float(
        raw.get("idle_probability", 0.0), "session.control.idle_probability", allow_zero=True
    )
    if idle_probability > 1:
        raise ScenarioError("session.control.idle_probability must be at most one")
    idle_min = _require_positive_float(
        raw.get("idle_min_seconds", 0.2), "session.control.idle_min_seconds", allow_zero=True
    )
    idle_max = _require_positive_float(
        raw.get("idle_max_seconds", 1.0), "session.control.idle_max_seconds", allow_zero=True
    )
    if idle_max < idle_min:
        raise ScenarioError("session.control.idle_max_seconds must be at least idle_min_seconds")
    raw_states = raw.get("action_states", [["KeyW"], ["KeyW", "KeyA"], ["KeyW", "KeyD"], ["KeyI"]])
    if not isinstance(raw_states, list) or not raw_states:
        raise ScenarioError("session.control.action_states must be a non-empty list of key lists")
    action_states: list[tuple[str, ...]] = []
    for index, state in enumerate(raw_states):
        if not isinstance(state, list) or not state or not all(isinstance(key, str) and key for key in state):
            raise ScenarioError(f"session.control.action_states[{index}] must be a non-empty list of keys")
        action_states.append(tuple(state))
    return ControlConfig(
        interval_seconds=interval,
        jitter_seconds=jitter,
        idle_probability=idle_probability,
        idle_min_seconds=idle_min,
        idle_max_seconds=idle_max,
        action_states=tuple(action_states),
    )


def _parse_admission(value: object) -> AdmissionExpectation:
    """Parse optional public-admission expectations for a user wave."""
    raw = _require_mapping(value, "admission")
    require_immediate_assignment = raw.get("require_immediate_assignment", False)
    if not isinstance(require_immediate_assignment, bool):
        raise ScenarioError("admission.require_immediate_assignment must be a boolean")

    def optional_positive_int(key: str) -> int | None:
        field = raw.get(key)
        if field is None:
            return None
        parsed = _require_non_negative_int(field, f"admission.{key}")
        if parsed < 1:
            raise ScenarioError(f"admission.{key} must be positive when supplied")
        return parsed

    expected_queue_size = raw.get("expected_queue_size")
    if expected_queue_size is not None:
        expected_queue_size = _require_non_negative_int(expected_queue_size, "admission.expected_queue_size")

    return AdmissionExpectation(
        require_immediate_assignment=require_immediate_assignment,
        expected_max_sessions_per_worker=optional_positive_int("expected_max_sessions_per_worker"),
        expected_queue_size=expected_queue_size,
    )


def _parse_diagnostic_initial_control_barrier(
    value: object,
    phases: Sequence[Phase],
) -> DiagnosticInitialControlBarrier | None:
    """Parse an explicitly synthetic synchronized-first-control diagnostic.

    This is deliberately restricted to a fresh first phase.  A later phase
    would contain users that have already sent controls, so it could not be
    accurately described as an initial-control alignment.
    """
    raw = _require_mapping(value, "diagnostic")
    barrier_value = raw.get("initial_control_barrier")
    if barrier_value is None:
        return None
    barrier = _require_mapping(barrier_value, "diagnostic.initial_control_barrier")
    enabled = barrier.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ScenarioError("diagnostic.initial_control_barrier.enabled must be a boolean")
    if not enabled:
        return None
    if barrier.get("kind") != "phase_aligned_initial_control":
        raise ScenarioError("diagnostic.initial_control_barrier.kind must be phase_aligned_initial_control")
    if barrier.get("not_a_real_user_trace") is not True:
        raise ScenarioError("diagnostic.initial_control_barrier.not_a_real_user_trace must be true")
    phase_name = barrier.get("phase")
    if not isinstance(phase_name, str) or not phase_name:
        raise ScenarioError("diagnostic.initial_control_barrier.phase must be a non-empty string")
    phase_index = next((index for index, phase in enumerate(phases) if phase.name == phase_name), None)
    if phase_index is None:
        raise ScenarioError("diagnostic.initial_control_barrier.phase must name a scenario phase")
    if phase_index != 0:
        raise ScenarioError(
            "diagnostic.initial_control_barrier only supports the first phase; "
            "later phases are not initial-control traces"
        )
    phase = phases[phase_index]
    expected = _require_non_negative_int(
        barrier.get("expected_connected_sessions"),
        "diagnostic.initial_control_barrier.expected_connected_sessions",
    )
    if expected < 1:
        raise ScenarioError("diagnostic.initial_control_barrier.expected_connected_sessions must be positive")
    if expected != phase.target_users:
        raise ScenarioError(
            "diagnostic.initial_control_barrier.expected_connected_sessions must equal the fresh phase target_users"
        )
    if phase.active_input_fraction != 1.0:
        raise ScenarioError(
            "diagnostic.initial_control_barrier requires active_input_fraction=1.0 "
            "so every released first control is active"
        )
    timeout = _require_positive_float(
        barrier.get("timeout_seconds"),
        "diagnostic.initial_control_barrier.timeout_seconds",
    )
    if timeout >= phase.duration_seconds:
        raise ScenarioError("diagnostic.initial_control_barrier.timeout_seconds must be shorter than its phase")
    return DiagnosticInitialControlBarrier(
        phase_name=phase.name,
        expected_connected_sessions=expected,
        timeout_seconds=timeout,
    )


def load_scenario(path: Path, *, server_url_override: str | None = None) -> Scenario:
    """Load and validate a JSON user-wave scenario."""
    try:
        raw_value = json.loads(path.read_text())
    except OSError as exc:
        raise ScenarioError(f"Could not read scenario {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"Scenario is not valid JSON: {exc}") from exc
    raw = _require_mapping(raw_value, "scenario")
    name = raw.get("name", path.stem)
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError("scenario.name must be a non-empty string")
    server_url = server_url_override or raw.get("server_url", "http://127.0.0.1:8088")
    if not isinstance(server_url, str) or not server_url.startswith(("http://", "https://")):
        raise ScenarioError("server_url must be an http(s) URL")
    server_url = server_url.rstrip("/")

    session_raw = _require_mapping(raw.get("session"), "session")
    prompt = session_raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ScenarioError("session.prompt must be a non-empty string")
    fps = _require_positive_float(session_raw.get("fps", 12), "session.fps")
    control_latent_frames = _require_non_negative_int(
        session_raw.get("control_latent_frames", 3), "session.control_latent_frames"
    )
    if control_latent_frames not in {1, 2, 3}:
        raise ScenarioError("session.control_latent_frames must be 1, 2, or 3")
    delivery_mode = session_raw.get("delivery_mode", "latest")
    if delivery_mode not in {"latest", "lossless"}:
        raise ScenarioError("session.delivery_mode must be 'latest' or 'lossless'")
    expected_preview_frames = _require_non_negative_int(
        session_raw.get("expected_preview_frames", 1), "session.expected_preview_frames"
    )
    admission = _parse_admission(raw.get("admission", {}))
    session = SessionConfig(
        prompt=prompt,
        image_path=_resolve_image_path(session_raw.get("image_path")),
        fps=fps,
        control_latent_frames=control_latent_frames,
        delivery_mode=delivery_mode,
        expected_preview_frames=expected_preview_frames,
        control=_parse_control(session_raw.get("control", {})),
    )

    phases_raw = raw.get("phases")
    if not isinstance(phases_raw, list) or not phases_raw:
        raise ScenarioError("phases must be a non-empty list")
    phases: list[Phase] = []
    phase_names: set[str] = set()
    for index, phase_value in enumerate(phases_raw):
        phase_raw = _require_mapping(phase_value, f"phases[{index}]")
        phase_name = phase_raw.get("name", f"phase-{index}")
        if not isinstance(phase_name, str) or not phase_name.strip():
            raise ScenarioError(f"phases[{index}].name must be a non-empty string")
        if phase_name in phase_names:
            raise ScenarioError(f"Duplicate phase name: {phase_name}")
        phase_names.add(phase_name)
        duration = _require_positive_float(phase_raw.get("duration_seconds"), f"phases[{index}].duration_seconds")
        arrival_window = _require_positive_float(
            phase_raw.get("arrival_window_seconds", 0.0),
            f"phases[{index}].arrival_window_seconds",
            allow_zero=True,
        )
        departure_window = _require_positive_float(
            phase_raw.get("departure_window_seconds", 0.0),
            f"phases[{index}].departure_window_seconds",
            allow_zero=True,
        )
        active_input_fraction = _require_positive_float(
            phase_raw.get("active_input_fraction", 1.0),
            f"phases[{index}].active_input_fraction",
            allow_zero=True,
        )
        if active_input_fraction > 1:
            raise ScenarioError(f"phases[{index}].active_input_fraction must be at most one")
        input_transition_window = _require_positive_float(
            phase_raw.get("input_transition_window_seconds", 0.0),
            f"phases[{index}].input_transition_window_seconds",
            allow_zero=True,
        )
        if arrival_window > duration or departure_window > duration or input_transition_window > duration:
            raise ScenarioError(f"phases[{index}] transition windows cannot exceed duration")
        phases.append(
            Phase(
                name=phase_name,
                duration_seconds=duration,
                target_users=_require_non_negative_int(phase_raw.get("target_users"), f"phases[{index}].target_users"),
                arrival_window_seconds=arrival_window,
                departure_window_seconds=departure_window,
                active_input_fraction=active_input_fraction,
                input_transition_window_seconds=input_transition_window,
            )
        )

    diagnostic_initial_control_barrier = _parse_diagnostic_initial_control_barrier(
        raw.get("diagnostic", {}),
        phases,
    )

    measurement = _require_mapping(raw.get("measurement", {}), "measurement")
    expected_worker_mode = raw.get("expected_worker_mode")
    if expected_worker_mode is not None and not isinstance(expected_worker_mode, str):
        raise ScenarioError("expected_worker_mode must be a string when supplied")
    expected_num_workers = raw.get("expected_num_workers")
    if expected_num_workers is not None:
        expected_num_workers = _require_non_negative_int(expected_num_workers, "expected_num_workers")
        if expected_num_workers < 1:
            raise ScenarioError("expected_num_workers must be positive")
    seed = raw.get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ScenarioError("seed must be an integer")
    slo_fps_tolerance = _require_positive_float(
        measurement.get("slo_fps_tolerance", 0.25),
        "measurement.slo_fps_tolerance",
        allow_zero=True,
    )
    if slo_fps_tolerance >= session.fps:
        raise ScenarioError("measurement.slo_fps_tolerance must be smaller than session.fps")
    return Scenario(
        name=name,
        server_url=server_url,
        session=session,
        phases=tuple(phases),
        sample_interval_seconds=_require_positive_float(
            measurement.get("sample_interval_seconds", 1.0), "measurement.sample_interval_seconds"
        ),
        connect_timeout_seconds=_require_positive_float(
            measurement.get("connect_timeout_seconds", 60.0), "measurement.connect_timeout_seconds"
        ),
        http_timeout_seconds=_require_positive_float(
            measurement.get("http_timeout_seconds", 30.0), "measurement.http_timeout_seconds"
        ),
        shutdown_timeout_seconds=_require_positive_float(
            measurement.get("shutdown_timeout_seconds", 20.0), "measurement.shutdown_timeout_seconds"
        ),
        admission=admission,
        first_generation_grace_seconds=_require_positive_float(
            measurement.get("first_generation_grace_seconds", 15.0),
            "measurement.first_generation_grace_seconds",
            allow_zero=True,
        ),
        slo_fps_tolerance=slo_fps_tolerance,
        seed=seed,
        expected_worker_mode=expected_worker_mode,
        expected_num_workers=expected_num_workers,
        diagnostic_initial_control_barrier=diagnostic_initial_control_barrier,
        lifecycle_trace=None,
        raw=raw,
    )


def _disable_proxy_for_loopback(url: str) -> None:
    """Avoid accidentally sending local LiveKit traffic through a proxy."""
    host = urlsplit(url).hostname
    if host is None:
        return
    try:
        loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if loopback:
        for name in _PROXY_ENV_NAMES:
            os.environ.pop(name, None)


def _load_livekit_rtc() -> Any:
    try:
        from livekit import rtc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The black-box LiveKit workload requires the TeleFuser 'livekit' runtime dependency. "
            "Use the Telefuser serving environment."
        ) from exc
    return rtc


def _safe_json_message(payload: bytes | str) -> dict[str, Any] | None:
    try:
        decoded = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


@dataclass
class LiveKitWaveSession:
    """One real browser-equivalent ABot client in a user-wave experiment."""

    index: int
    scenario: Scenario
    http: httpx.AsyncClient
    rtc: Any
    record_event: Any
    started_at: float
    trace_session_id: str | None = None
    source_trace_session_id: int | None = None
    source_trace_user_id: int | None = None
    diagnostic_initial_control_barrier_phase: str | None = None
    initial_control_gate: asyncio.Event | None = field(default=None, repr=False)
    _room: Any | None = field(default=None, init=False, repr=False)
    _video_streams: list[Any] = field(default_factory=list, init=False, repr=False)
    _video_tasks: list[asyncio.Task[None]] = field(default_factory=list, init=False, repr=False)
    _control_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _rng: random.Random = field(init=False, repr=False)
    scheduled_at: float | None = field(default=None, init=False)
    create_started_at: float | None = field(default=None, init=False)
    created_at: float | None = field(default=None, init=False)
    connected_at: float | None = field(default=None, init=False)
    initial_control_barrier_arrived_at: float | None = field(default=None, init=False)
    initial_control_barrier_released_at: float | None = field(default=None, init=False)
    first_media_frame_at: float | None = field(default=None, init=False)
    first_generated_frame_at: float | None = field(default=None, init=False)
    last_generated_frame_at: float | None = field(default=None, init=False)
    first_active_control_at: float | None = field(default=None, init=False)
    stopped_at: float | None = field(default=None, init=False)
    server_session_id: str | None = field(default=None, init=False)
    worker_id: str | None = field(default=None, init=False)
    admission_status: str | None = field(default=None, init=False)
    queue_position: int | None = field(default=None, init=False)
    admission_violation: str | None = field(default=None, init=False)
    frames_received: int = field(default=0, init=False)
    generated_frames_received: int = field(default=0, init=False)
    control_messages_sent: int = field(default=0, init=False)
    status_messages_received: int = field(default=0, init=False)
    error: str | None = field(default=None, init=False)
    stop_requested: bool = field(default=False, init=False)
    departure_scheduled: bool = field(default=False, init=False)
    remote_session_deleted: bool = field(default=False, init=False)
    connected: bool = field(default=False, init=False)
    current_controls: tuple[str, ...] = field(default_factory=tuple, init=False)
    input_enabled: bool = field(default=True, init=False)
    input_pauses: int = field(default=0, init=False)
    input_resumes: int = field(default=0, init=False)
    input_pause_started_at: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.scenario.seed + self.index)

    @property
    def logical_id(self) -> str:
        return self.trace_session_id or f"wave-{self.index:03d}"

    @property
    def active_controls(self) -> bool:
        return bool(self.current_controls) and self.input_enabled and not self.stop_requested

    async def set_input_enabled(self, enabled: bool, *, reason: str) -> None:
        """Pause or resume controller input without dropping the LiveKit session.

        This models a user temporarily releasing all keys or switching away from
        the browser. It preserves the public serving session, KV/decoder state,
        and WebRTC subscription; actual session departure still uses ``stop``.
        """
        if self.stop_requested or self.input_enabled == enabled:
            return
        now = time.perf_counter()
        self.input_enabled = enabled
        if enabled:
            self.input_resumes += 1
            paused_for = (
                max(0.0, now - self.input_pause_started_at) if self.input_pause_started_at is not None else None
            )
            self.input_pause_started_at = None
            self.record_event(
                "input_resumed",
                session=self.logical_id,
                reason=reason,
                paused_seconds=round(paused_for, 6) if paused_for is not None else None,
            )
            controls = self._rng.choice(self.scenario.session.control.action_states)
        else:
            self.input_pauses += 1
            self.input_pause_started_at = now
            self.record_event("input_paused", session=self.logical_id, reason=reason)
            controls = ()

        # Do not wait for the next heartbeat to clear a stale key state.
        if (
            self.connected
            and self._room is not None
            and (self.initial_control_gate is None or self.initial_control_gate.is_set())
        ):
            try:
                await self._publish_control_state(controls)
            except Exception as exc:  # noqa: BLE001 - a transition failure is a workload fact
                detail = f"{type(exc).__name__}: {exc}"
                if self.error is None:
                    self.error = f"InputTransitionPublishError: {detail}"
                self.record_event("input_transition_publish_error", session=self.logical_id, error=detail)

    async def start(self) -> None:
        """Create a public session, join its room, and start keyboard heartbeats."""
        if self.stop_requested:
            return
        self.create_started_at = time.perf_counter()
        identity = f"abot-wave-{self.index}-{self.scenario.seed}"
        request = {
            "identity": identity,
            "role": "controller",
            "prompt": self.scenario.session.prompt,
            "image_path": self.scenario.session.image_path,
            "config": {
                "fps": self.scenario.session.fps,
                "control_latent_frames": self.scenario.session.control_latent_frames,
                "delivery_mode": self.scenario.session.delivery_mode,
                "seed": self.scenario.seed + self.index,
            },
        }
        self.record_event("session_create_started", session=self.logical_id)
        try:
            response = await self.http.post(
                f"{self.scenario.server_url}/v1/stream/sessions",
                json=request,
            )
            response.raise_for_status()
            created = response.json()
            if not isinstance(created, Mapping):
                raise RuntimeError("session-create response is not an object")
            required = ("session_id", "livekit_url", "token")
            missing = [key for key in required if not isinstance(created.get(key), str)]
            if missing:
                raise RuntimeError("session-create response is missing " + ", ".join(missing))
            self.server_session_id = str(created["session_id"])
            worker_id = created.get("worker_id")
            self.worker_id = worker_id if isinstance(worker_id, str) else None
            status = created.get("status")
            self.admission_status = status if isinstance(status, str) else None
            queue_position = created.get("queue_position")
            self.queue_position = queue_position if isinstance(queue_position, int) else None
            if self.scenario.admission.require_immediate_assignment and self.admission_status != "assigned":
                self.admission_violation = f"Expected immediate assignment, got {self.admission_status!r}"
                self.record_event(
                    "admission_contract_violation",
                    session=self.logical_id,
                    violation=self.admission_violation,
                )
            self.created_at = time.perf_counter()
            self.record_event(
                "session_created",
                session=self.logical_id,
                server_session_id=self.server_session_id,
                worker_id=self.worker_id,
                status=self.admission_status,
                queue_position=self.queue_position,
                offer_rtt_seconds=round(self.created_at - self.create_started_at, 6),
            )
            if self.stop_requested:
                await self._delete_remote_session()
                return
            await self._connect_room(str(created["livekit_url"]), str(created["token"]))
        except Exception as exc:  # noqa: BLE001 - externally visible workload outcome
            self.error = f"{type(exc).__name__}: {exc}"
            self.record_event("session_start_failed", session=self.logical_id, error=self.error)
            await self._delete_remote_session()

    async def _connect_room(self, livekit_url: str, token: str) -> None:
        _disable_proxy_for_loopback(livekit_url)
        room = self.rtc.Room()
        self._room = room

        @room.on("data_received")
        def on_data_received(packet: Any) -> None:
            topic = getattr(packet, "topic", "") or ""
            if topic not in {_STATUS_TOPIC, _METRICS_TOPIC}:
                return
            self.status_messages_received += 1
            payload = _safe_json_message(getattr(packet, "data", b""))
            if payload is None:
                return
            data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
            stage = data.get("stage") if isinstance(data, Mapping) else None
            if stage in {"worker_running", "runtime_ready"}:
                self.record_event("worker_ready", session=self.logical_id, stage=stage)

        @room.on("track_subscribed")
        def on_track_subscribed(track: Any, _publication: Any, _participant: Any) -> None:
            if getattr(track, "kind", None) != self.rtc.TrackKind.KIND_VIDEO:
                return
            stream = self.rtc.VideoStream(track)
            self._video_streams.append(stream)
            self._video_tasks.append(asyncio.create_task(self._consume_video(stream)))
            self.record_event("video_track_subscribed", session=self.logical_id)

        @room.on("disconnected")
        def on_disconnected(reason: Any) -> None:
            if not self.stop_requested and self.error is None:
                self.error = f"LiveKit disconnected: {reason}"
                self.record_event("room_disconnected", session=self.logical_id, reason=str(reason))

        options = self.rtc.RoomOptions(auto_subscribe=True, connect_timeout=self.scenario.connect_timeout_seconds)
        await room.connect(livekit_url, token, options)
        self.connected = True
        self.connected_at = time.perf_counter()
        self.record_event(
            "room_connected",
            session=self.logical_id,
            connected_seconds=round(self.connected_at - (self.create_started_at or self.connected_at), 6),
        )
        if not self.stop_requested:
            if self.initial_control_gate is not None:
                self.initial_control_barrier_arrived_at = self.connected_at
                self.record_event(
                    "diagnostic_initial_control_barrier_arrived",
                    session=self.logical_id,
                    phase=self.diagnostic_initial_control_barrier_phase,
                )
            self._start_control_task()

    def _start_control_task(self) -> None:
        """Start the heartbeat once; a diagnostic gate may hold its first send."""
        if self.stop_requested or self._control_task is not None:
            return
        self._control_task = asyncio.create_task(self._send_controls(), name=f"abot-controls-{self.logical_id}")

    async def _consume_video(self, stream: Any) -> None:
        try:
            async for _ in stream:
                now = time.perf_counter()
                self.frames_received += 1
                if self.first_media_frame_at is None:
                    self.first_media_frame_at = now
                    self.record_event("first_media_frame", session=self.logical_id)
                if self.frames_received > self.scenario.session.expected_preview_frames:
                    self.generated_frames_received += 1
                    if self.first_generated_frame_at is None:
                        self.first_generated_frame_at = now
                        self.record_event(
                            "first_generated_frame",
                            session=self.logical_id,
                            action_to_first_generated_seconds=(
                                round(now - self.first_active_control_at, 6)
                                if self.first_active_control_at is not None
                                else None
                            ),
                        )
                    self.last_generated_frame_at = now
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - WebRTC termination is a workload fact
            if not self.stop_requested and self.error is None:
                self.error = f"VideoStreamError: {type(exc).__name__}: {exc}"
                self.record_event("video_stream_error", session=self.logical_id, error=self.error)

    async def _send_controls(self) -> None:
        gate = self.initial_control_gate
        if gate is not None:
            await gate.wait()
        if self.stop_requested:
            return
        assert self._room is not None
        control = self.scenario.session.control
        idle_until = 0.0
        sent_non_idle_control = False
        try:
            while not self.stop_requested:
                now = time.perf_counter()
                if not self.input_enabled or now < idle_until:
                    controls: tuple[str, ...] = ()
                else:
                    if sent_non_idle_control and self._rng.random() < control.idle_probability:
                        idle_until = now + self._rng.uniform(control.idle_min_seconds, control.idle_max_seconds)
                        controls = ()
                    else:
                        controls = self._rng.choice(control.action_states)
                try:
                    await self._publish_control_state(controls)
                    sent_non_idle_control = sent_non_idle_control or bool(controls)
                except Exception as exc:  # noqa: BLE001 - network write error is data
                    if not self.stop_requested and self.error is None:
                        self.error = f"ControlPublishError: {type(exc).__name__}: {exc}"
                        self.record_event("control_publish_error", session=self.logical_id, error=self.error)
                interval = control.interval_seconds + self._rng.uniform(-control.jitter_seconds, control.jitter_seconds)
                await asyncio.sleep(max(interval, 0.01))
        except asyncio.CancelledError:
            raise

    async def _publish_control_state(self, controls: tuple[str, ...]) -> None:
        """Publish one reliable control heartbeat and retain its public state."""
        if self._room is None:
            return
        gate = self.initial_control_gate
        if gate is not None and not gate.is_set():
            return
        payload = {"type": "control_state", "controls": list(controls)}
        await self._room.local_participant.publish_data(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            topic=_CONTROL_TOPIC,
            reliable=True,
        )
        self.control_messages_sent += 1
        self.current_controls = controls
        if controls and self.first_active_control_at is None:
            self.first_active_control_at = time.perf_counter()
            self.record_event("first_active_control", session=self.logical_id, controls=list(controls))

    async def stop(self) -> None:
        """Stop only this client/session; sibling sessions keep running."""
        if self.stop_requested and self.stopped_at is not None:
            return
        self.stop_requested = True
        self.departure_scheduled = False
        self.current_controls = ()
        control_task = self._control_task
        if control_task is not None and not control_task.done():
            control_task.cancel()
            await asyncio.gather(control_task, return_exceptions=True)
        room = self._room
        if room is not None and self.connected:
            with contextlib.suppress(Exception):
                await room.local_participant.publish_data(b'{"type":"stop"}', topic=_CONTROL_TOPIC, reliable=True)
        await self._delete_remote_session()
        for stream in self._video_streams:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stream.aclose(), timeout=self.scenario.shutdown_timeout_seconds)
        for task in self._video_tasks:
            if not task.done():
                task.cancel()
        if self._video_tasks:
            await asyncio.gather(*self._video_tasks, return_exceptions=True)
        if room is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(room.disconnect(), timeout=self.scenario.shutdown_timeout_seconds)
        self.connected = False
        self.stopped_at = time.perf_counter()
        self.record_event("session_stopped", session=self.logical_id)

    async def _delete_remote_session(self) -> None:
        if self.server_session_id is None or self.remote_session_deleted:
            return
        session_id = self.server_session_id
        self.remote_session_deleted = True
        try:
            await self.http.delete(
                f"{self.scenario.server_url}/v1/stream/sessions/{session_id}",
            )
            self.record_event("session_deleted", session=self.logical_id, server_session_id=session_id)
        except Exception as exc:  # noqa: BLE001 - cleanup is best effort
            self.record_event(
                "session_delete_error",
                session=self.logical_id,
                server_session_id=session_id,
                error=f"{type(exc).__name__}: {exc}",
            )

    def snapshot(self, now: float) -> dict[str, Any]:
        """Return bounded client-side facts; no model-internal state is read."""
        action_to_first = (
            self.first_generated_frame_at - self.first_active_control_at
            if self.first_generated_frame_at is not None and self.first_active_control_at is not None
            else None
        )
        creation_to_first = (
            self.first_generated_frame_at - self.create_started_at
            if self.first_generated_frame_at is not None and self.create_started_at is not None
            else None
        )
        return {
            "logical_session_id": self.logical_id,
            "source_trace_session_id": self.source_trace_session_id,
            "source_trace_user_id": self.source_trace_user_id,
            "server_session_id": self.server_session_id,
            "worker_id_at_admission": self.worker_id,
            "admission_status": self.admission_status,
            "queue_position": self.queue_position,
            "scheduled": self.scheduled_at is not None,
            "admission_contract_violation": self.admission_violation,
            "connected": self.connected,
            "stop_requested": self.stop_requested,
            "departure_scheduled": self.departure_scheduled,
            "remote_session_deleted": self.remote_session_deleted,
            "diagnostic_initial_control_barrier_phase": self.diagnostic_initial_control_barrier_phase,
            "diagnostic_initial_control_barrier_waiting": (
                self.initial_control_gate is not None and not self.initial_control_gate.is_set()
            ),
            "diagnostic_initial_control_barrier_arrived_offset_seconds": (
                round(self.initial_control_barrier_arrived_at - self.started_at, 6)
                if self.initial_control_barrier_arrived_at is not None
                else None
            ),
            "diagnostic_initial_control_barrier_released_offset_seconds": (
                round(self.initial_control_barrier_released_at - self.started_at, 6)
                if self.initial_control_barrier_released_at is not None
                else None
            ),
            "input_enabled": self.input_enabled,
            "active_controls": self.active_controls,
            "input_pauses": self.input_pauses,
            "input_resumes": self.input_resumes,
            "frames_received": self.frames_received,
            "generated_frames_received": self.generated_frames_received,
            "control_messages_sent": self.control_messages_sent,
            "status_messages_received": self.status_messages_received,
            "offer_rtt_seconds": (
                round(self.created_at - self.create_started_at, 6)
                if self.created_at is not None and self.create_started_at is not None
                else None
            ),
            "connected_seconds": (
                round(self.connected_at - self.create_started_at, 6)
                if self.connected_at is not None and self.create_started_at is not None
                else None
            ),
            "action_to_first_generated_seconds": round(action_to_first, 6) if action_to_first is not None else None,
            "creation_to_first_generated_seconds": (
                round(creation_to_first, 6) if creation_to_first is not None else None
            ),
            "last_generated_frame_age_seconds": (
                round(now - self.last_generated_frame_at, 6) if self.last_generated_frame_at is not None else None
            ),
            "error": self.error,
        }


class LiveKitWaveRunner:
    """Coordinate a black-box user wave and persist client-delivery facts."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.rtc = _load_livekit_rtc()
        self.started_at = 0.0
        # ``offset_seconds`` is measured with perf_counter(). Keep a sampled
        # wall-clock/monotonic triplet from the same workload origin so an
        # external per-dispatch trace can align client events without guessing
        # from process start time.
        self._trace_started_monotonic_seconds = 0.0
        self._trace_started_unix_seconds = 0.0
        self._sessions: list[LiveKitWaveSession] = []
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._monitor_task: asyncio.Task[None] | None = None
        self._monitoring = False
        self._phase_name = "startup"
        self._phase_target_users = 0
        self._phase_active_input_fraction = 1.0
        self._samples: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._phase_results: list[dict[str, Any]] = []
        self._previous_generated_frames: dict[str, int] = {}
        self._previous_sample_at: float | None = None
        self._server_metadata: list[dict[str, Any]] = []
        self._warnings: list[str] = []
        # These records are kept separate from user-wave results because a
        # synchronized first action is a harness diagnostic, not user behavior.
        self._diagnostic_initial_control_barrier_results: list[dict[str, Any]] = []

    def record_event(self, event: str, **values: Any) -> None:
        now = time.perf_counter()
        self._events.append(
            {
                "offset_seconds": round(max(0.0, now - self.started_at), 6),
                "event": event,
                **values,
            }
        )

    async def run(self) -> dict[str, Any]:
        """Run all phases and return a complete JSON-serializable artifact."""
        _disable_proxy_for_loopback(self.scenario.server_url)
        timeout = httpx.Timeout(self.scenario.http_timeout_seconds)
        # User arrivals in the supplied wave are intentionally several seconds
        # apart. Uvicorn can close an idle HTTP/1.1 keep-alive socket at the
        # same boundary, which otherwise turns an unrelated client-side stale
        # connection into a false "session_start_failed" observation. Session
        # admission is not idempotent, so retrying a POST after a read failure
        # could create a duplicate world state. Use a fresh loopback HTTP
        # connection for each public API request instead.
        limits = httpx.Limits(max_keepalive_connections=0)
        async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as http:
            self._http = http
            self.started_at = time.perf_counter()
            self._trace_started_monotonic_seconds = time.monotonic()
            self._trace_started_unix_seconds = time.time()
            await self._capture_server_metadata("before_workload")
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor(), name="abot-livekit-wave-monitor")
            try:
                for phase in self.scenario.phases:
                    await self._run_phase(phase)
            finally:
                self._monitoring = False
                if self._monitor_task is not None:
                    self._monitor_task.cancel()
                    await asyncio.gather(self._monitor_task, return_exceptions=True)
                await self._stop_all_sessions()
                await self._capture_server_metadata("after_workload")
        completed_at = time.perf_counter()
        return {
            "schema_version": "abot_livekit_user_wave_v1",
            "scenario": self.scenario.raw,
            "effective_scenario": {
                "name": self.scenario.name,
                "server_url": self.scenario.server_url,
                "target_fps_per_active_session": self.scenario.session.fps,
                "control_latent_frames": self.scenario.session.control_latent_frames,
                "delivery_mode": self.scenario.session.delivery_mode,
                "slo_fps_tolerance": self.scenario.slo_fps_tolerance,
                "admission": {
                    "require_immediate_assignment": self.scenario.admission.require_immediate_assignment,
                    "expected_max_sessions_per_worker": self.scenario.admission.expected_max_sessions_per_worker,
                    "expected_queue_size": self.scenario.admission.expected_queue_size,
                },
                "diagnostic_initial_control_barrier": (
                    {
                        "enabled": True,
                        "kind": "phase_aligned_initial_control",
                        "not_a_real_user_trace": True,
                        "phase": self.scenario.diagnostic_initial_control_barrier.phase_name,
                        "expected_connected_sessions": (
                            self.scenario.diagnostic_initial_control_barrier.expected_connected_sessions
                        ),
                        "timeout_seconds": self.scenario.diagnostic_initial_control_barrier.timeout_seconds,
                    }
                    if self.scenario.diagnostic_initial_control_barrier is not None
                    else {"enabled": False}
                ),
                "phases": [
                    {
                        "name": phase.name,
                        "duration_seconds": phase.duration_seconds,
                        "target_users": phase.target_users,
                        "arrival_window_seconds": phase.arrival_window_seconds,
                        "departure_window_seconds": phase.departure_window_seconds,
                        "active_input_fraction": phase.active_input_fraction,
                        "input_transition_window_seconds": phase.input_transition_window_seconds,
                    }
                    for phase in self.scenario.phases
                ],
            },
            "trace_clock": {
                "offset_clock": "time.perf_counter",
                "origin_performance_counter_seconds": round(self.started_at, 9),
                "origin_monotonic_seconds": round(self._trace_started_monotonic_seconds, 9),
                "origin_unix_seconds": round(self._trace_started_unix_seconds, 9),
                "offset_to_unix_seconds": "origin_unix_seconds + offset_seconds",
                "offset_to_monotonic_seconds": "origin_monotonic_seconds + offset_seconds",
            },
            "started_at_unix_seconds": round(self._trace_started_unix_seconds, 9),
            "elapsed_seconds": round(completed_at - self.started_at, 6),
            "server_metadata": self._server_metadata,
            "warnings": self._warnings,
            "diagnostic_initial_control_barrier_results": self._diagnostic_initial_control_barrier_results,
            "phase_results": self._phase_results,
            "sessions": [session.snapshot(completed_at) for session in self._sessions],
            "samples": self._samples,
            "events": self._events,
        }

    async def _run_phase(self, phase: Phase) -> None:
        phase_started = time.perf_counter()
        sample_start = len(self._samples)
        self._phase_name = phase.name
        self._phase_target_users = phase.target_users
        self._phase_active_input_fraction = phase.active_input_fraction
        self.record_event(
            "phase_started",
            phase=phase.name,
            target_users=phase.target_users,
            active_input_fraction=phase.active_input_fraction,
        )
        await self._capture_server_metadata(f"phase_start:{phase.name}")
        self._schedule_transition(phase)
        self._schedule_input_activity(phase)
        await asyncio.sleep(phase.duration_seconds)
        phase_completed = time.perf_counter()
        await self._capture_server_metadata(f"phase_end:{phase.name}")
        result = self._summarize_phase(
            phase,
            phase_started=phase_started,
            phase_completed=phase_completed,
            samples=self._samples[sample_start:],
        )
        self._phase_results.append(result)
        self.record_event("phase_completed", phase=phase.name, summary=result["summary"])

    def _schedule_transition(self, phase: Phase) -> None:
        present = [
            session for session in self._sessions if not session.stop_requested and not session.departure_scheduled
        ]
        difference = phase.target_users - len(present)
        barrier = self.scenario.diagnostic_initial_control_barrier
        gate: asyncio.Event | None = None
        if barrier is not None and barrier.phase_name == phase.name:
            if present or difference != barrier.expected_connected_sessions:
                raise RuntimeError("Diagnostic initial-control barrier no longer describes a fresh target population")
            gate = asyncio.Event()
        if difference > 0:
            barrier_sessions: list[LiveKitWaveSession] = []
            for ordinal in range(difference):
                session = LiveKitWaveSession(
                    index=len(self._sessions),
                    scenario=self.scenario,
                    http=self._http,
                    rtc=self.rtc,
                    record_event=self.record_event,
                    started_at=self.started_at,
                    diagnostic_initial_control_barrier_phase=phase.name if gate is not None else None,
                    initial_control_gate=gate,
                )
                session.scheduled_at = time.perf_counter()
                self._sessions.append(session)
                barrier_sessions.append(session)
                offset = self._spread_offset(ordinal + 1, difference, phase.arrival_window_seconds)
                self._spawn_background(self._delayed_start(session, offset))
            if gate is not None:
                assert barrier is not None
                self.record_event(
                    "diagnostic_initial_control_barrier_opened",
                    phase=phase.name,
                    expected_connected_sessions=barrier.expected_connected_sessions,
                    not_a_real_user_trace=True,
                )
                self._spawn_background(
                    self._run_diagnostic_initial_control_barrier(phase, barrier, barrier_sessions, gate)
                )
            return
        if difference < 0:
            # Newest users leave first. This also removes still-queued arrivals before
            # disrupting sessions that have already accumulated a world state.
            departing = present[-(-difference):]
            for ordinal, session in enumerate(reversed(departing)):
                session.departure_scheduled = True
                offset = self._spread_offset(ordinal + 1, -difference, phase.departure_window_seconds)
                self._spawn_background(self._delayed_stop(session, offset))

    async def _run_diagnostic_initial_control_barrier(
        self,
        phase: Phase,
        barrier: DiagnosticInitialControlBarrier,
        sessions: Sequence[LiveKitWaveSession],
        gate: asyncio.Event,
    ) -> None:
        """Release held control tasks only after the synthetic cohort connects.

        A timeout or failed admission deliberately opens the gate for already
        connected clients, so the harness never leaves serving sessions held
        forever.  The artifact records that result as unaligned and invalid for
        a phase-alignment comparison.
        """
        opened_at = time.perf_counter()
        deadline = opened_at + barrier.timeout_seconds
        status = "released_aligned"
        warning: str | None = None
        connected: list[LiveKitWaveSession] = []
        try:
            while True:
                now = time.perf_counter()
                connected = [session for session in sessions if session.connected and not session.stop_requested]
                failed = [session for session in sessions if session.stop_requested or session.error is not None]
                if failed:
                    status = "released_unaligned_failure"
                    warning = (
                        "Diagnostic initial-control barrier saw failed or stopped sessions; "
                        "released connected sessions without phase alignment."
                    )
                    break
                if len(connected) == barrier.expected_connected_sessions:
                    break
                if now >= deadline:
                    status = "released_unaligned_timeout"
                    warning = (
                        "Diagnostic initial-control barrier timed out before every session connected; "
                        "released connected sessions without phase alignment."
                    )
                    break
                await asyncio.sleep(min(0.02, max(0.001, deadline - now)))
        except asyncio.CancelledError:
            cancelled_at = time.perf_counter()
            result = {
                "kind": "phase_aligned_initial_control",
                "not_a_real_user_trace": True,
                "phase": phase.name,
                "expected_connected_sessions": barrier.expected_connected_sessions,
                "connected_sessions_at_release": len(connected),
                "status": "cancelled",
                "opened_offset_seconds": round(opened_at - self.started_at, 6),
                "released_offset_seconds": round(cancelled_at - self.started_at, 6),
                "wait_seconds": round(cancelled_at - opened_at, 6),
            }
            self._diagnostic_initial_control_barrier_results.append(result)
            self.record_event("diagnostic_initial_control_barrier_cancelled", phase=phase.name)
            raise

        released_at = time.perf_counter()
        for session in connected:
            session.initial_control_barrier_released_at = released_at
        gate.set()
        result = {
            "kind": "phase_aligned_initial_control",
            "not_a_real_user_trace": True,
            "phase": phase.name,
            "expected_connected_sessions": barrier.expected_connected_sessions,
            "connected_sessions_at_release": len(connected),
            "status": status,
            "opened_offset_seconds": round(opened_at - self.started_at, 6),
            "released_offset_seconds": round(released_at - self.started_at, 6),
            "wait_seconds": round(released_at - opened_at, 6),
        }
        self._diagnostic_initial_control_barrier_results.append(result)
        self.record_event(
            "diagnostic_initial_control_barrier_released",
            phase=phase.name,
            status=status,
            connected_sessions=len(connected),
            expected_connected_sessions=barrier.expected_connected_sessions,
            not_a_real_user_trace=True,
        )
        if warning is not None:
            self._warnings.append(warning)

    def _schedule_input_activity(self, phase: Phase) -> None:
        """Schedule long input pauses/resumes for sessions present in this phase.

        Arrivals default to active input. The supplied real-user trace uses input
        fractions only after its target population has already arrived; this keeps
        an arrival's first action deterministic and observable.
        """
        present = [
            session for session in self._sessions if not session.stop_requested and not session.departure_scheduled
        ]
        if not present:
            return
        active_count = round(len(present) * phase.active_input_fraction)
        phase_offset = sum(ord(character) for character in phase.name)
        ordered = sorted(
            present,
            key=lambda session: (((session.index + 1) * 17 + phase_offset) % len(present), session.index),
        )
        active_ids = {session.logical_id for session in ordered[:active_count]}
        changes = [session for session in present if session.input_enabled != (session.logical_id in active_ids)]
        for ordinal, session in enumerate(changes):
            offset = self._spread_offset(ordinal + 1, len(changes), phase.input_transition_window_seconds)
            self._spawn_background(
                self._delayed_set_input_enabled(
                    session,
                    session.logical_id in active_ids,
                    offset,
                    reason=f"phase:{phase.name}",
                )
            )

    @staticmethod
    def _spread_offset(position: int, count: int, window_seconds: float) -> float:
        if count <= 1 or window_seconds <= 0:
            return 0.0
        return window_seconds * (position - 1) / (count - 1)

    def _spawn_background(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _delayed_start(self, session: LiveKitWaveSession, offset_seconds: float) -> None:
        await asyncio.sleep(offset_seconds)
        if session.stop_requested or session.departure_scheduled:
            return
        try:
            await asyncio.wait_for(session.start(), timeout=self.scenario.connect_timeout_seconds)
        except asyncio.TimeoutError:
            session.error = f"SessionStartTimeout after {self.scenario.connect_timeout_seconds:g}s"
            self.record_event("session_start_timeout", session=session.logical_id, error=session.error)

    async def _delayed_stop(self, session: LiveKitWaveSession, offset_seconds: float) -> None:
        await asyncio.sleep(offset_seconds)
        await session.stop()

    async def _delayed_set_input_enabled(
        self, session: LiveKitWaveSession, enabled: bool, offset_seconds: float, *, reason: str
    ) -> None:
        await asyncio.sleep(offset_seconds)
        await session.set_input_enabled(enabled, reason=reason)

    def _session_delivery_fps(
        self, session: LiveKitWaveSession, *, now: float, interval: float, delta: int
    ) -> tuple[float | None, float | None]:
        """Return visible delivery FPS and the SLO observation, if either is eligible.

        The SLO path deliberately emits a zero for a connected, continuously
        controlled user that is still waiting beyond the first-generation grace
        period.  The first return is restricted to active controls because it is
        the user-facing per-session FPS shown in phase summaries.
        """
        if interval <= 0 or session.stop_requested:
            return None, None
        controlled_after_grace = (
            session.connected
            and session.active_controls
            and session.first_active_control_at is not None
            and now - session.first_active_control_at >= self.scenario.first_generation_grace_seconds
        )
        if session.first_generated_frame_at is not None and session.active_controls:
            fps = delta / interval
            return fps, fps if controlled_after_grace else None
        if controlled_after_grace:
            return 0.0, 0.0
        return None, None

    def _session_requested_delivery_fps(
        self, session: LiveKitWaveSession, *, now: float, interval: float, delta: int
    ) -> float | None:
        """Return FPS for every requested user once its grace interval expires."""
        if interval <= 0 or session.stop_requested or session.create_started_at is None:
            return None
        service_started_at = session.first_active_control_at or session.create_started_at
        if now - service_started_at < self.scenario.first_generation_grace_seconds:
            return None
        if session.first_generated_frame_at is None:
            return 0.0
        return delta / interval

    async def _monitor(self) -> None:
        while self._monitoring:
            now = time.perf_counter()
            elapsed = max(0.0, now - self.started_at)
            previous_at = self._previous_sample_at
            interval = now - previous_at if previous_at is not None else 0.0
            session_fps: dict[str, float] = {}
            slo_session_fps: dict[str, float] = {}
            demand_slo_session_fps: dict[str, float] = {}
            active_session_fps: list[float] = []
            requested_session_fps: dict[str, float] = {}
            requested_session_values: list[float] = []
            connected_sessions = 0
            active_control_sessions = 0
            input_enabled_sessions = 0
            requested_sessions = 0
            generated_frames_delta = 0
            for session in self._sessions:
                snapshot = session.snapshot(now)
                logical_id = session.logical_id
                previous_frames = self._previous_generated_frames.get(logical_id, session.generated_frames_received)
                delta = max(0, session.generated_frames_received - previous_frames)
                self._previous_generated_frames[logical_id] = session.generated_frames_received
                if session.connected:
                    connected_sessions += 1
                if session.active_controls:
                    active_control_sessions += 1
                if session.input_enabled and not session.stop_requested:
                    input_enabled_sessions += 1
                if session.create_started_at is not None and not session.stop_requested:
                    requested_sessions += 1
                if interval > 0 and not session.stop_requested:
                    # Aggregate delivery counts every generated video frame, including
                    # frames from users that are temporarily idle between controls.
                    if session.first_generated_frame_at is not None:
                        generated_frames_delta += delta
                    delivery_fps, demand_slo_fps = self._session_delivery_fps(
                        session, now=now, interval=interval, delta=delta
                    )
                    if delivery_fps is not None:
                        session_fps[logical_id] = round(delivery_fps, 6)
                        active_session_fps.append(delivery_fps)
                    if demand_slo_fps is not None:
                        demand_slo_session_fps[logical_id] = round(demand_slo_fps, 6)
                requested_fps = self._session_requested_delivery_fps(session, now=now, interval=interval, delta=delta)
                if requested_fps is not None:
                    requested_session_fps[logical_id] = round(requested_fps, 6)
                    requested_session_values.append(requested_fps)
                    slo_session_fps[logical_id] = round(requested_fps, 6)
                snapshot["generated_frames_delta"] = delta
                snapshot["delivery_fps"] = session_fps.get(logical_id)
                snapshot["requested_delivery_fps"] = requested_session_fps.get(logical_id)
            aggregate_fps = generated_frames_delta / interval if interval > 0 else 0.0
            self._samples.append(
                {
                    "offset_seconds": round(elapsed, 6),
                    "phase": self._phase_name,
                    "target_users": self._phase_target_users,
                    "target_active_input_fraction": self._phase_active_input_fraction,
                    "requested_users": requested_sessions,
                    "connected_users": connected_sessions,
                    "input_enabled_users": input_enabled_sessions,
                    "active_control_users": active_control_sessions,
                    "aggregate_delivery_fps": round(aggregate_fps, 6),
                    "per_requested_session_delivery_fps": round(statistics.fmean(requested_session_values), 6)
                    if requested_session_values
                    else None,
                    "per_active_session_delivery_fps": round(statistics.fmean(active_session_fps), 6)
                    if active_session_fps
                    else None,
                    "requested_session_delivery_fps": requested_session_fps,
                    "session_delivery_fps": session_fps,
                    "slo_session_delivery_fps": slo_session_fps,
                    "slo_observation_sessions": len(slo_session_fps),
                    "demand_slo_session_delivery_fps": demand_slo_session_fps,
                    "demand_slo_observation_sessions": len(demand_slo_session_fps),
                    "sessions": [session.snapshot(now) for session in self._sessions],
                }
            )
            self._previous_sample_at = now
            await asyncio.sleep(self.scenario.sample_interval_seconds)

    def _summarize_phase(
        self,
        phase: Phase,
        *,
        phase_started: float,
        phase_completed: float,
        samples: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        aggregate_fps = [float(sample["aggregate_delivery_fps"]) for sample in samples if sample["offset_seconds"] > 0]
        per_session_fps = [float(fps) for sample in samples for fps in sample["session_delivery_fps"].values()]
        active_session_fps = [
            float(sample["per_active_session_delivery_fps"])
            for sample in samples
            if sample["per_active_session_delivery_fps"] is not None
        ]
        requested_session_fps = [
            float(fps) for sample in samples for fps in sample["requested_session_delivery_fps"].values()
        ]
        requested_active_fps = [
            float(sample["per_requested_session_delivery_fps"])
            for sample in samples
            if sample["per_requested_session_delivery_fps"] is not None
        ]
        phase_sessions = [
            session
            for session in self._sessions
            if session.create_started_at is not None and phase_started <= session.create_started_at <= phase_completed
        ]
        phase_population = [
            session
            for session in self._sessions
            if session.create_started_at is not None
            and session.create_started_at <= phase_completed
            and (session.stopped_at is None or session.stopped_at >= phase_started)
        ]
        action_to_first = [
            session.first_generated_frame_at - session.first_active_control_at
            for session in phase_sessions
            if session.first_generated_frame_at is not None and session.first_active_control_at is not None
        ]
        # The all-user SLO denominator includes every requested session after
        # its grace interval, including rejected, disconnected, or stalled users
        # as zero-FPS observations.
        slo_observations = [float(fps) for sample in samples for fps in sample["slo_session_delivery_fps"].values()]
        demand_slo_observations = [
            float(fps) for sample in samples for fps in sample["demand_slo_session_delivery_fps"].values()
        ]
        slo_threshold = self.scenario.session.fps - self.scenario.slo_fps_tolerance
        slo_hits = sum(value >= slo_threshold for value in slo_observations)
        demand_slo_hits = sum(value >= slo_threshold for value in demand_slo_observations)
        assigned_sessions = sum(session.admission_status == "assigned" for session in phase_population)
        queued_sessions = sum(session.admission_status == "queued" for session in phase_population)
        unassigned_sessions = len(phase_population) - assigned_sessions - queued_sessions
        admission_contract_satisfied = not self.scenario.admission.require_immediate_assignment or all(
            session.admission_status == "assigned" for session in phase_population
        )
        summary = {
            "duration_seconds": round(phase_completed - phase_started, 6),
            "target_users": phase.target_users,
            "max_requested_users": max((int(sample["requested_users"]) for sample in samples), default=0),
            "max_connected_users": max((int(sample["connected_users"]) for sample in samples), default=0),
            "max_input_enabled_users": max((int(sample["input_enabled_users"]) for sample in samples), default=0),
            "max_active_control_users": max((int(sample["active_control_users"]) for sample in samples), default=0),
            "admission": {
                "require_immediate_assignment": self.scenario.admission.require_immediate_assignment,
                "phase_population_sessions": len(phase_population),
                "assigned_sessions": assigned_sessions,
                "queued_sessions": queued_sessions,
                "unassigned_or_failed_sessions": unassigned_sessions,
                "immediate_assignment_satisfied": admission_contract_satisfied,
            },
            "aggregate_delivery_fps": _summary(aggregate_fps),
            "per_requested_session_delivery_fps": _summary(requested_active_fps),
            "per_requested_user_delivery_fps": _summary(requested_session_fps),
            "per_active_session_delivery_fps": _summary(active_session_fps),
            "per_session_delivery_fps": _summary(per_session_fps),
            "action_to_first_generated_seconds": _summary(action_to_first),
            "slo_target_fps": self.scenario.session.fps,
            "slo_tolerance_fps": self.scenario.slo_fps_tolerance,
            "slo_threshold_fps": round(slo_threshold, 6),
            "slo_first_generation_grace_seconds": self.scenario.first_generation_grace_seconds,
            "slo_observation_samples": len(slo_observations),
            "slo_satisfied_samples": slo_hits,
            "slo_sample_attainment": (round(slo_hits / len(slo_observations), 6) if slo_observations else 0.0),
            "demand_slo_observation_samples": len(demand_slo_observations),
            "demand_slo_satisfied_samples": demand_slo_hits,
            "demand_slo_sample_attainment": (
                round(demand_slo_hits / len(demand_slo_observations), 6) if demand_slo_observations else 0.0
            ),
            "started_sessions": len(phase_sessions),
            "failed_sessions": sum(1 for session in phase_sessions if session.error is not None),
        }
        return {
            "phase": phase.name,
            "started_offset_seconds": round(phase_started - self.started_at, 6),
            "completed_offset_seconds": round(phase_completed - self.started_at, 6),
            "summary": summary,
        }

    async def _capture_server_metadata(self, label: str) -> None:
        try:
            response = await self._http.get(f"{self.scenario.server_url}/v1/service/metadata")
            response.raise_for_status()
            metadata = response.json()
            if not isinstance(metadata, Mapping):
                raise RuntimeError("metadata response is not an object")
            metadata_dict = dict(metadata)
            self._server_metadata.append(
                {
                    "label": label,
                    "offset_seconds": round(max(0.0, time.perf_counter() - self.started_at), 6),
                    "metadata": metadata_dict,
                }
            )
            self._validate_server_metadata(metadata_dict, label)
        except Exception as exc:  # noqa: BLE001 - metadata availability is an experiment fact
            warning = f"Could not collect server metadata at {label}: {type(exc).__name__}: {exc}"
            self._warnings.append(warning)
            self.record_event("metadata_error", label=label, warning=warning)

    def _validate_server_metadata(self, metadata: Mapping[str, Any], label: str) -> None:
        expected_mode = self.scenario.expected_worker_mode
        if expected_mode is not None and metadata.get("worker_mode") != expected_mode:
            warning = (
                f"Expected worker_mode={expected_mode!r}, got {metadata.get('worker_mode')!r} at {label}. "
                "This is not the intended four-GPU process-NCCL baseline."
            )
            self._warnings.append(warning)
        expected_workers = self.scenario.expected_num_workers
        if expected_workers is not None and metadata.get("num_workers") != expected_workers:
            warning = f"Expected num_workers={expected_workers}, got {metadata.get('num_workers')!r} at {label}."
            self._warnings.append(warning)
        admission = self.scenario.admission
        if (
            admission.expected_max_sessions_per_worker is not None
            and metadata.get("configured_max_sessions_per_worker") != admission.expected_max_sessions_per_worker
        ):
            warning = (
                "Expected configured_max_sessions_per_worker="
                f"{admission.expected_max_sessions_per_worker}, got "
                f"{metadata.get('configured_max_sessions_per_worker')!r} at {label}."
            )
            self._warnings.append(warning)
        if admission.expected_queue_size is not None and metadata.get("queue_size") != admission.expected_queue_size:
            warning = (
                f"Expected queue_size={admission.expected_queue_size}, got {metadata.get('queue_size')!r} at {label}."
            )
            self._warnings.append(warning)

    async def _stop_all_sessions(self) -> None:
        for session in self._sessions:
            session.stop_requested = True
        for task in tuple(self._background_tasks):
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*tuple(self._background_tasks), return_exceptions=True)
        await asyncio.gather(*(session.stop() for session in self._sessions), return_exceptions=True)


def _print_summary(result: Mapping[str, Any]) -> None:
    """Print a compact table suitable for an experiment log."""
    print("\\nABot LiveKit black-box user-wave results")
    print(
        "phase                         target  max-req  max-input  agg-FPS  FPS/demand  FPS/all-user  "
        "admitted  demand-SLO  all-user-SLO  A2F-p95(s)"
    )
    for phase in result["phase_results"]:
        summary = phase["summary"]
        aggregate = summary["aggregate_delivery_fps"]
        per_user = summary["per_requested_session_delivery_fps"]
        admission = summary["admission"]
        a2f = summary["action_to_first_generated_seconds"]
        print(
            f"{phase['phase'][:28]:28} "
            f"{summary['target_users']:>6} "
            f"{summary['max_requested_users']:>8} "
            f"{summary['max_input_enabled_users']:>10} "
            f"{aggregate['mean']:>8.3f} "
            f"{summary['per_active_session_delivery_fps']['mean']:>10.3f} "
            f"{per_user['mean']:>13.3f} "
            f"{admission['assigned_sessions']:>3}/{admission['phase_population_sessions']:<3} "
            f"{summary['demand_slo_sample_attainment'] * 100:>10.1f}% "
            f"{summary['slo_sample_attainment'] * 100:>12.1f}% "
            f"{a2f['p95']:>11.3f} "
        )
    warnings = result.get("warnings", [])
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_wave.json"),
        help="JSON phase/user-wave scenario, relative to the repository root by default.",
    )
    parser.add_argument("--server-url", help="Override scenario.server_url, for example http://127.0.0.1:8088")
    parser.add_argument("--output", type=Path, help="Write the complete JSON artifact to this path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the scenario without contacting the service.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_path = args.scenario.expanduser()
    if not scenario_path.is_absolute():
        scenario_path = (_REPO_ROOT / scenario_path).resolve()
    scenario = load_scenario(scenario_path, server_url_override=args.server_url)
    if args.dry_run:
        print(json.dumps(scenario.raw, indent=2, sort_keys=True))
        barrier = scenario.diagnostic_initial_control_barrier
        if barrier is not None:
            print(
                "\nDIAGNOSTIC ONLY: this scenario synchronizes the first active control after "
                f"{barrier.expected_connected_sessions} sessions connect in phase {barrier.phase_name!r}. "
                "It is not a real-user arrival trace."
            )
        print(f"\nValidated scenario: {scenario.name}")
        return
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (_REPO_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(LiveKitWaveRunner(scenario).run())
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _print_summary(result)
    print(f"\nWrote complete artifact: {output}")
    if scenario.admission.require_immediate_assignment:
        invalid_phases = [
            str(phase["phase"])
            for phase in result["phase_results"]
            if not phase["summary"]["admission"]["immediate_assignment_satisfied"]
        ]
        if invalid_phases:
            raise SystemExit(
                "Immediate-assignment contract failed in "
                + ", ".join(invalid_phases)
                + f"; artifact was preserved at {output}"
            )


if __name__ == "__main__":
    main()
