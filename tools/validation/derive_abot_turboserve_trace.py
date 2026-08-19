#!/usr/bin/env python3
"""Derive runnable ABot LiveKit traces from TurboServe's public demo trace.

TurboServe's ``traces/example_8gpu.json`` is a simulator lifecycle trace, not
an ABot action stream.  This tool deliberately preserves its wall-clock
arrival / active / idle / departure sequence while normalizing *retained
session concurrency* to an ABot serving capacity.  The generated scenario is
replayed by :mod:`benchmark_abot_livekit_burst` exclusively through the public
HTTP and LiveKit interfaces.

The transformation is intentionally deterministic and recorded verbatim in
each output's ``trace_contract``.  It must be described as a
``TurboServe-public-demo-trace-derived`` workload, never as TurboServe's
production trace or a reproduction of the paper's private T1--T6 traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _REPO_ROOT.parent
_DEFAULT_SOURCE = _WORKSPACE_ROOT / "TurboServe" / "traces" / "example_8gpu.json"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "tools" / "validation" / "workloads"
_DERIVATION_VERSION = "turboserve-public-demo-capacity-normalized-v1"
_SELECTION_SEED = 20260815


class TraceAdapterError(ValueError):
    """Raised when a source TurboServe lifecycle trace is malformed."""


@dataclass(frozen=True)
class SourceEvent:
    """The small source-trace subset required for lifecycle replay."""

    time_seconds: float
    sequence: int
    event_type: str
    session_id: int
    user_id: int | None
    active_on_arrival: bool | None


@dataclass
class SourceSession:
    """Mutable source session state during deterministic capacity normalization."""

    session_id: int
    user_id: int | None
    input_enabled: bool


@dataclass(frozen=True)
class DerivedTrace:
    """A complete explicit lifecycle replay plus reproducibility metadata."""

    events: tuple[dict[str, Any], ...]
    source_duration_seconds: float
    derived_duration_seconds: float
    source_peak_retained_sessions: int
    source_peak_active_sessions: int
    derived_peak_retained_sessions: int
    derived_peak_active_sessions: int
    source_event_counts: dict[str, int]
    derived_event_counts: dict[str, int]
    source_sha256: str
    selected_source_session_count: int
    derived_connection_count: int


def _as_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceAdapterError(f"{label} must be an object")
    return value


def _as_non_negative_float(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TraceAdapterError(f"{label} must be a number")
    result = float(value)
    if result < 0:
        raise TraceAdapterError(f"{label} must be non-negative")
    return result


def _as_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TraceAdapterError(f"{label} must be an integer")
    return int(value)


def _load_source_trace(path: Path) -> tuple[list[SourceEvent], dict[str, Any], str]:
    """Load and minimally validate the public TurboServe JSON trace."""
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise TraceAdapterError(f"Could not read source trace {path}: {exc}") from exc
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise TraceAdapterError(f"Source trace is not valid JSON: {exc}") from exc
    root = _as_mapping(document, "source trace")
    raw_events = root.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise TraceAdapterError("source trace.events must be a non-empty list")

    events: list[SourceEvent] = []
    for index, raw_event in enumerate(raw_events):
        event = _as_mapping(raw_event, f"source trace.events[{index}]")
        event_type = event.get("event_type")
        if event_type not in {"session_arrival", "user_active", "user_idle", "session_departure"}:
            raise TraceAdapterError(f"Unsupported source event type at index {index}: {event_type!r}")
        payload = _as_mapping(event.get("payload", {}), f"source trace.events[{index}].payload")
        active_on_arrival: bool | None = None
        if event_type == "session_arrival":
            candidate = payload.get("active", True)
            if not isinstance(candidate, bool):
                raise TraceAdapterError(f"source trace.events[{index}].payload.active must be boolean")
            active_on_arrival = candidate
        user_id = event.get("user_id")
        if user_id is not None:
            user_id = _as_int(user_id, f"source trace.events[{index}].user_id")
        events.append(
            SourceEvent(
                time_seconds=_as_non_negative_float(event.get("time_s"), f"source trace.events[{index}].time_s"),
                sequence=_as_int(event.get("sequence"), f"source trace.events[{index}].sequence"),
                event_type=str(event_type),
                session_id=_as_int(event.get("session_id"), f"source trace.events[{index}].session_id"),
                user_id=user_id,
                active_on_arrival=active_on_arrival,
            )
        )
    events.sort(key=lambda event: (event.time_seconds, event.sequence))
    config = dict(_as_mapping(root.get("config", {}), "source trace.config"))
    return events, config, hashlib.sha256(raw_bytes).hexdigest()


def _stable_rank(source_session_id: int, *, seed: int) -> int:
    """Return a stable pseudo-random rank without depending on Python hash salt."""
    digest = hashlib.sha256(f"{seed}:{source_session_id}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _scaled_target(source_retained: int, *, source_peak: int, target_peak: int) -> int:
    """Round source concurrency proportionally, guaranteeing the requested peak."""
    if source_retained <= 0:
        return 0
    # Source peak is observed from this exact file.  Half-up rounding makes the
    # rescaling independent of Python's banker-rounding implementation.
    target = (source_retained * target_peak * 2 + source_peak) // (source_peak * 2)
    return max(0, min(target_peak, int(target)))


def _trace_event(
    *,
    offset_seconds: float,
    event: str,
    trace_session_id: str,
    source_session: SourceSession,
    source_event: SourceEvent,
    derived_sequence: int,
    **extra: Any,
) -> dict[str, Any]:
    """Build a self-contained replay event with direct provenance."""
    return {
        "offset_seconds": round(offset_seconds, 6),
        "sequence": derived_sequence,
        "event": event,
        "trace_session_id": trace_session_id,
        "source_session_id": source_session.session_id,
        "source_user_id": source_session.user_id,
        "source_time_seconds": round(source_event.time_seconds, 6),
        "source_event_sequence": source_event.sequence,
        **extra,
    }


def _observed_source_peak(source_events: Iterable[SourceEvent]) -> int:
    """Return retained-session peak after replaying source lifecycle events."""
    present: set[int] = set()
    peak = 0
    for event in sorted(source_events, key=lambda item: (item.time_seconds, item.sequence)):
        if event.event_type == "session_arrival":
            present.add(event.session_id)
        elif event.event_type == "session_departure":
            present.discard(event.session_id)
        peak = max(peak, len(present))
    return peak


def derive_trace(
    source_events: Iterable[SourceEvent],
    *,
    target_peak: int,
    source_sha256: str,
    selection_seed: int = _SELECTION_SEED,
) -> DerivedTrace:
    """Capacity-normalize source lifecycles while retaining selected identities.

    The selected population is sticky: it is only evicted on a source
    departure or when the scaled target decreases.  When a scaled target grows
    we fill the free slots using a stable hash order among currently present
    source sessions.  That retains each selected source session's repeated
    ``user_idle``/``user_active`` sequence whenever capacity permits, rather
    than resampling a different cohort on every source event.
    """
    if target_peak < 1:
        raise TraceAdapterError("target_peak must be positive")
    ordered = sorted(source_events, key=lambda event: (event.time_seconds, event.sequence))
    if not ordered:
        raise TraceAdapterError("source_events must not be empty")

    source_peak_for_scaling = _observed_source_peak(ordered)
    if source_peak_for_scaling < 1:
        raise TraceAdapterError("source trace never retains a session")
    source_sessions: dict[int, SourceSession] = {}
    selected: dict[int, str] = {}
    generations: Counter[int] = Counter()
    derived_events: list[dict[str, Any]] = []
    source_event_counts: Counter[str] = Counter()
    derived_event_counts: Counter[str] = Counter()
    selected_source_ids: set[int] = set()
    source_peak_retained = 0
    source_peak_active = 0
    derived_peak_retained = 0
    derived_peak_active = 0
    derived_sequence = 0

    def emit(
        event: str,
        trace_session_id: str,
        source_session: SourceSession,
        source_event: SourceEvent,
        **extra: Any,
    ) -> None:
        nonlocal derived_sequence
        derived_events.append(
            _trace_event(
                offset_seconds=source_event.time_seconds,
                event=event,
                trace_session_id=trace_session_id,
                source_session=source_session,
                source_event=source_event,
                derived_sequence=derived_sequence,
                **extra,
            )
        )
        derived_sequence += 1
        derived_event_counts[event] += 1

    def remove_selection(
        source_session_id: int,
        source_event: SourceEvent,
        *,
        reason: str,
        source_session: SourceSession | None = None,
    ) -> None:
        trace_session_id = selected.pop(source_session_id, None)
        if trace_session_id is None:
            return
        source_session = source_session or source_sessions.get(source_session_id)
        if source_session is None:
            # A departing session is removed from ``source_sessions`` only
            # after its selection's corresponding departure is emitted.
            raise TraceAdapterError(f"Selected source session {source_session_id} disappeared before departure")
        emit(
            "session_departure",
            trace_session_id,
            source_session,
            source_event,
            departure_reason=reason,
        )

    def fill_selection(desired_count: int, source_event: SourceEvent) -> set[int]:
        """Fill scaled capacity with stable-ranked live source sessions."""
        newly_selected: set[int] = set()
        candidates = sorted(
            (session for session_id, session in source_sessions.items() if session_id not in selected),
            key=lambda session: (_stable_rank(session.session_id, seed=selection_seed), session.session_id),
        )
        for source_session in candidates[: max(0, desired_count - len(selected))]:
            generations[source_session.session_id] += 1
            trace_session_id = f"ts-{source_session.session_id:05d}-g{generations[source_session.session_id]:02d}"
            selected[source_session.session_id] = trace_session_id
            selected_source_ids.add(source_session.session_id)
            newly_selected.add(source_session.session_id)
            emit(
                "session_arrival",
                trace_session_id,
                source_session,
                source_event,
                input_enabled=source_session.input_enabled,
                arrival_reason=(
                    "source_session_arrival"
                    if (
                        source_event.event_type == "session_arrival"
                        and source_event.session_id == source_session.session_id
                    )
                    else "capacity_normalization_scale_up"
                ),
            )
        return newly_selected

    for source_event in ordered:
        source_event_counts[source_event.event_type] += 1
        source_session_id = source_event.session_id
        previous_input_enabled: bool | None = None
        if source_event.event_type == "session_arrival":
            if source_session_id in source_sessions:
                raise TraceAdapterError(f"Source session {source_session_id} arrived while already present")
            assert source_event.active_on_arrival is not None
            source_sessions[source_session_id] = SourceSession(
                session_id=source_session_id,
                user_id=source_event.user_id,
                input_enabled=source_event.active_on_arrival,
            )
        else:
            source_session = source_sessions.get(source_session_id)
            if source_session is None:
                raise TraceAdapterError(
                    f"Source event {source_event.event_type} references non-present session {source_session_id}"
                )
            previous_input_enabled = source_session.input_enabled
            if source_event.event_type == "user_active":
                source_session.input_enabled = True
            elif source_event.event_type == "user_idle":
                source_session.input_enabled = False
            elif source_event.event_type == "session_departure":
                remove_selection(source_session_id, source_event, reason="source_session_departure")
                del source_sessions[source_session_id]

        source_peak_retained = max(source_peak_retained, len(source_sessions))
        source_peak_active = max(
            source_peak_active,
            sum(session.input_enabled for session in source_sessions.values()),
        )
        desired_count = _scaled_target(
            len(source_sessions), source_peak=source_peak_for_scaling, target_peak=target_peak
        )
        # Remove least-preferred selected sessions only when scaled capacity
        # genuinely shrinks.  This avoids churn on ordinary source arrivals.
        excess = len(selected) - desired_count
        if excess > 0:
            evicted_ids = sorted(
                selected,
                key=lambda session_id: (_stable_rank(session_id, seed=selection_seed), session_id),
                reverse=True,
            )[:excess]
            for session_id in evicted_ids:
                remove_selection(session_id, source_event, reason="capacity_normalization_scale_down")

        newly_selected = fill_selection(desired_count, source_event)
        selected_trace_session_id = selected.get(source_session_id)
        if (
            source_event.event_type in {"user_active", "user_idle"}
            and selected_trace_session_id is not None
            and source_session_id not in newly_selected
            and previous_input_enabled != source_sessions[source_session_id].input_enabled
        ):
            emit(
                source_event.event_type,
                selected_trace_session_id,
                source_sessions[source_session_id],
                source_event,
                input_enabled=source_sessions[source_session_id].input_enabled,
            )

        derived_peak_retained = max(derived_peak_retained, len(selected))
        derived_peak_active = max(
            derived_peak_active,
            sum(source_sessions[session_id].input_enabled for session_id in selected),
        )

    if selected:
        raise TraceAdapterError(
            "Source trace ended with retained sessions; expected departure events before duration end"
        )
    if source_peak_retained != 186:
        raise TraceAdapterError(
            "This adapter pins the published example trace's observed retained-session peak at 186; "
            f"got {source_peak_retained}. Regenerate policy only after reviewing the source trace."
        )
    if derived_peak_retained != target_peak:
        raise TraceAdapterError(
            f"Capacity normalization did not reach requested peak {target_peak}; got {derived_peak_retained}"
        )
    source_duration = max(event.time_seconds for event in ordered)
    return DerivedTrace(
        events=tuple(derived_events),
        source_duration_seconds=source_duration,
        derived_duration_seconds=source_duration,
        source_peak_retained_sessions=source_peak_retained,
        source_peak_active_sessions=source_peak_active,
        derived_peak_retained_sessions=derived_peak_retained,
        derived_peak_active_sessions=derived_peak_active,
        source_event_counts=dict(sorted(source_event_counts.items())),
        derived_event_counts=dict(sorted(derived_event_counts.items())),
        source_sha256=source_sha256,
        selected_source_session_count=len(selected_source_ids),
        derived_connection_count=derived_event_counts["session_arrival"],
    )


def _scenario_payload(*, name: str, workers: int, target_peak: int, trace: DerivedTrace) -> dict[str, Any]:
    """Build the existing ABot workload format plus explicit lifecycle events."""
    expected_worker_mode = "process-nccl" if workers > 1 else "process"
    source_relative_path = "../../../TurboServe/traces/example_8gpu.json"
    return {
        "name": name,
        "trace_contract": {
            "kind": "turboserve_public_demo_trace_derived_abot_lifecycle",
            "derivation_version": _DERIVATION_VERSION,
            "not_a_turboserve_production_trace": True,
            "not_a_reproduction_of_private_paper_t1_to_t6_traces": True,
            "source": {
                "public_demo_repository_relative_path": source_relative_path,
                "public_demo_trace_filename": "example_8gpu.json",
                "sha256": trace.source_sha256,
                "source_event_counts": trace.source_event_counts,
                "source_duration_seconds": trace.source_duration_seconds,
                "source_peak_retained_sessions": trace.source_peak_retained_sessions,
                "source_peak_active_sessions": trace.source_peak_active_sessions,
            },
            "time_transform": {
                "kind": "identity_wall_clock",
                "source_to_derived_scale": 1.0,
                "derived_duration_seconds": trace.derived_duration_seconds,
                "description": (
                    "No time compression: arrival, active, idle, and departure offsets retain "
                    "the source 30-minute wall-clock scale."
                ),
            },
            "capacity_transform": {
                "kind": "sticky_capacity_normalized_session_sampling",
                "selection_seed": _SELECTION_SEED,
                "source_observed_peak_retained_sessions": trace.source_peak_retained_sessions,
                "target_peak_retained_sessions": target_peak,
                "target_workers": workers,
                "target_sessions_per_worker": 4,
                "scaling_rule": (
                    "round_half_up(source_retained_sessions * target_peak / source_peak); sticky selected "
                    "sessions are retained until source departure or a scaled capacity decrease; scale-up "
                    "uses stable SHA-256(seed:source_session_id) rank among currently present sessions."
                ),
                "derived_peak_retained_sessions": trace.derived_peak_retained_sessions,
                "derived_peak_active_sessions": trace.derived_peak_active_sessions,
                "selected_source_session_count": trace.selected_source_session_count,
                "derived_connection_count": trace.derived_connection_count,
            },
            "event_mapping": {
                "session_arrival": (
                    "create one ABot LiveKit session; arrival input_enabled follows source payload.active/current state"
                ),
                "user_active": ("resume that selected ABot client's action heartbeat without dropping its session"),
                "user_idle": (
                    "pause that selected ABot client's action heartbeat without dropping its session or retained state"
                ),
                "session_departure": "stop and delete that selected ABot LiveKit session",
            },
            "execution_contract": (
                "No diagnostic barrier. The black-box runner schedules each lifecycle event at its "
                "explicit source-derived offset and never assigns a GPU from the client side."
            ),
        },
        "server_url": "http://127.0.0.1:8088",
        "expected_worker_mode": expected_worker_mode,
        "expected_num_workers": workers,
        "seed": _SELECTION_SEED,
        "admission": {
            "require_immediate_assignment": True,
            "expected_max_sessions_per_worker": 4,
            "expected_queue_size": 0,
        },
        "session": {
            "prompt": "A smooth first-person exploration through a vivid natural landscape.",
            "image_path": "../ABot-World/web_client/datasets/images/84b90ad568b693d2.png",
            "fps": 12,
            "control_latent_frames": 3,
            "delivery_mode": "latest",
            "expected_preview_frames": 1,
            "control": {
                "interval_seconds": 0.5,
                "jitter_seconds": 0.15,
                "idle_probability": 0.0,
                "idle_min_seconds": 0.0,
                "idle_max_seconds": 0.0,
                "action_states": [["KeyW"], ["KeyW", "KeyA"], ["KeyW", "KeyD"], ["KeyI"]],
            },
        },
        "measurement": {
            "sample_interval_seconds": 1.0,
            "connect_timeout_seconds": 90.0,
            "http_timeout_seconds": 30.0,
            "shutdown_timeout_seconds": 20.0,
            "first_generation_grace_seconds": 15.0,
            "slo_fps_tolerance": 0.25,
        },
        "phases": [
            {
                "name": "turboserve_public_demo_lifecycle_replay",
                "duration_seconds": trace.derived_duration_seconds,
                "target_users": target_peak,
                "active_input_fraction": 1.0,
            }
        ],
        "lifecycle_trace": {
            "kind": "explicit_session_lifecycle_v1",
            "duration_seconds": trace.derived_duration_seconds,
            "events": list(trace.events),
        },
    }


def build_scenarios(source_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the canonical single-GPU and four-GPU scenario documents."""
    source_events, _source_config, source_sha256 = _load_source_trace(source_path)
    trace_1gpu = derive_trace(source_events, target_peak=4, source_sha256=source_sha256)
    trace_4gpu = derive_trace(source_events, target_peak=16, source_sha256=source_sha256)
    return (
        _scenario_payload(
            name="abot_livekit_1gpu_lf3_12fps_turboserve_public_demo_trace_peak4",
            workers=1,
            target_peak=4,
            trace=trace_1gpu,
        ),
        _scenario_payload(
            name="abot_livekit_4gpu_lf3_12fps_turboserve_public_demo_trace_peak16",
            workers=4,
            target_peak=16,
            trace=trace_4gpu,
        ),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE, help="TurboServe public demo JSON trace")
    parser.add_argument(
        "--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR, help="Directory for generated scenarios"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if canonical checked-in scenario files differ from deterministic regenerated content.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    scenario_1gpu, scenario_4gpu = build_scenarios(source)
    outputs = {
        output_dir / "abot_livekit_1gpu_lf3_12fps_turboserve_public_demo_trace_peak4.json": scenario_1gpu,
        output_dir / "abot_livekit_4gpu_lf3_12fps_turboserve_public_demo_trace_peak16.json": scenario_4gpu,
    }
    mismatches: list[Path] = []
    for path, payload in outputs.items():
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.check:
            try:
                actual = path.read_text(encoding="utf-8")
            except OSError:
                mismatches.append(path)
                continue
            if actual != rendered:
                mismatches.append(path)
            continue
        _write_json(path, payload)
        print(f"Wrote {path}")
    if args.check and mismatches:
        raise SystemExit("Derived scenario files are stale or missing: " + ", ".join(str(path) for path in mismatches))
    if args.check:
        print("Canonical TurboServe-public-demo-derived scenarios are current.")


if __name__ == "__main__":
    main()
