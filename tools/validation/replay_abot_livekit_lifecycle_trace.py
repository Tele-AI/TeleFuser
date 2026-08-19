#!/usr/bin/env python3
"""Replay an explicit per-session ABot LiveKit lifecycle trace.

Unlike ``benchmark_abot_livekit_burst.py``'s aggregate phase fractions, this
runner schedules the ``lifecycle_trace.events`` embedded in a scenario at their
declared offsets.  It still uses the same public HTTP and LiveKit client path;
the workload never selects a GPU or accesses model internals.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.validation import benchmark_abot_livekit_burst as wave


@dataclass(frozen=True)
class ExplicitLifecycleEvent:
    """One validated arrival, pause, resume, or departure."""

    offset_seconds: float
    sequence: int
    event: str
    trace_session_id: str
    source_session_id: int | None
    source_user_id: int | None
    input_enabled: bool | None


@dataclass(frozen=True)
class ExplicitLifecycleTrace:
    """Exact lifecycle schedule for one reporting phase."""

    duration_seconds: float
    events: tuple[ExplicitLifecycleEvent, ...]


class LifecycleTraceError(ValueError):
    """Raised before the service is contacted for an invalid replay trace."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LifecycleTraceError(f"{label} must be an object")
    return value


def _number(value: object, label: str, *, allow_zero: bool) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise LifecycleTraceError(f"{label} must be a number")
    parsed = float(value)
    if parsed < 0 or (not allow_zero and parsed == 0):
        raise LifecycleTraceError(f"{label} must be {'non-negative' if allow_zero else 'positive'}")
    return parsed


def _non_negative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LifecycleTraceError(f"{label} must be a non-negative integer")
    return int(value)


def load_explicit_lifecycle_trace(scenario: wave.Scenario) -> ExplicitLifecycleTrace:
    """Parse the generated trace before opening any HTTP or LiveKit connection."""
    raw = _mapping(scenario.raw.get("lifecycle_trace"), "lifecycle_trace")
    if raw.get("kind") != "explicit_session_lifecycle_v1":
        raise LifecycleTraceError("lifecycle_trace.kind must be explicit_session_lifecycle_v1")
    if scenario.diagnostic_initial_control_barrier is not None:
        raise LifecycleTraceError("explicit lifecycle replay cannot use diagnostic.initial_control_barrier")
    if len(scenario.phases) != 1:
        raise LifecycleTraceError("explicit lifecycle replay requires exactly one reporting phase")
    duration = _number(raw.get("duration_seconds"), "lifecycle_trace.duration_seconds", allow_zero=False)
    if abs(duration - scenario.phases[0].duration_seconds) > 1e-6:
        raise LifecycleTraceError("lifecycle_trace.duration_seconds must equal its reporting phase duration")
    raw_events = raw.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise LifecycleTraceError("lifecycle_trace.events must be a non-empty list")

    retained: dict[str, bool] = {}
    previous_key = (-1.0, -1)
    parsed_events: list[ExplicitLifecycleEvent] = []
    valid_events = {"session_arrival", "user_active", "user_idle", "session_departure"}
    for index, event_value in enumerate(raw_events):
        event = _mapping(event_value, f"lifecycle_trace.events[{index}]")
        offset = _number(
            event.get("offset_seconds"),
            f"lifecycle_trace.events[{index}].offset_seconds",
            allow_zero=True,
        )
        sequence = _non_negative_int(event.get("sequence"), f"lifecycle_trace.events[{index}].sequence")
        if offset > duration:
            raise LifecycleTraceError(f"lifecycle_trace.events[{index}] occurs after duration")
        if (offset, sequence) <= previous_key:
            raise LifecycleTraceError("lifecycle_trace.events must be strictly ordered by (offset_seconds, sequence)")
        previous_key = (offset, sequence)
        event_name = event.get("event")
        if event_name not in valid_events:
            raise LifecycleTraceError(f"Unsupported lifecycle event {event_name!r} at index {index}")
        trace_session_id = event.get("trace_session_id")
        if not isinstance(trace_session_id, str) or not trace_session_id:
            raise LifecycleTraceError(f"lifecycle_trace.events[{index}].trace_session_id must be a non-empty string")
        input_enabled = event.get("input_enabled")
        if event_name != "session_departure" and not isinstance(input_enabled, bool):
            raise LifecycleTraceError(f"lifecycle_trace.events[{index}].input_enabled must be boolean")
        if event_name == "session_arrival":
            if trace_session_id in retained:
                raise LifecycleTraceError(f"Trace session {trace_session_id!r} arrived while retained")
            retained[trace_session_id] = bool(input_enabled)
        elif event_name == "session_departure":
            if trace_session_id not in retained:
                raise LifecycleTraceError(f"Trace session {trace_session_id!r} departed before arrival")
            del retained[trace_session_id]
        else:
            current = retained.get(trace_session_id)
            if current is None:
                raise LifecycleTraceError(f"Trace session {trace_session_id!r} changed input before arrival")
            desired = bool(input_enabled)
            if current == desired:
                raise LifecycleTraceError(f"Trace session {trace_session_id!r} has a redundant {event_name}")
            retained[trace_session_id] = desired
        source_session_id = event.get("source_session_id")
        if source_session_id is not None:
            source_session_id = _non_negative_int(
                source_session_id, f"lifecycle_trace.events[{index}].source_session_id"
            )
        source_user_id = event.get("source_user_id")
        if source_user_id is not None:
            source_user_id = _non_negative_int(source_user_id, f"lifecycle_trace.events[{index}].source_user_id")
        parsed_events.append(
            ExplicitLifecycleEvent(
                offset_seconds=offset,
                sequence=sequence,
                event=str(event_name),
                trace_session_id=trace_session_id,
                source_session_id=source_session_id,
                source_user_id=source_user_id,
                input_enabled=input_enabled if isinstance(input_enabled, bool) else None,
            )
        )
    if retained:
        raise LifecycleTraceError("lifecycle_trace must depart every session before its duration ends")
    return ExplicitLifecycleTrace(duration_seconds=duration, events=tuple(parsed_events))


class ExplicitLifecycleRunner(wave.LiveKitWaveRunner):
    """Reuse the normal black-box runner while replacing phase scheduling."""

    def __init__(self, scenario: wave.Scenario, trace: ExplicitLifecycleTrace) -> None:
        super().__init__(scenario)
        self._explicit_trace = trace
        self._trace_sessions: dict[str, wave.LiveKitWaveSession] = {}
        # A trace departure means that the logical session is gone at its source
        # timestamp, but the public DELETE/LiveKit teardown is asynchronous. Keep
        # that physical teardown in the admission accounting until ``stop()`` has
        # completed, otherwise an arrival at the same (or a nearby) timestamp can
        # race the server's fixed session capacity and receive a false 429.
        self._lifecycle_start_tasks: dict[str, asyncio.Task[None]] = {}
        self._lifecycle_departure_tasks: dict[str, asyncio.Task[None]] = {}
        self._lifecycle_admission_capacity = self._derive_admission_capacity(trace)

    def _derive_admission_capacity(self, trace: ExplicitLifecycleTrace) -> int:
        """Return the public capacity used to guard lifecycle arrivals.

        A scenario with an explicit admission contract should use that contract.
        Older/free-form traces can still be replayed safely by treating their
        observed retained-session peak as the capacity limit.
        """
        per_worker = self.scenario.admission.expected_max_sessions_per_worker
        workers = self.scenario.expected_num_workers
        if per_worker is not None and workers is not None:
            return per_worker * workers

        retained = 0
        peak = 0
        for event in trace.events:
            if event.event == "session_arrival":
                retained += 1
                peak = max(peak, retained)
            elif event.event == "session_departure":
                retained -= 1
        if peak < 1:
            raise RuntimeError("Explicit lifecycle trace has no retained-session capacity")
        return peak

    async def _run_phase(self, phase: wave.Phase) -> None:
        """Run the single reporting phase from its exact event schedule."""
        trace = self._explicit_trace
        if abs(trace.duration_seconds - phase.duration_seconds) > 1e-6:
            raise RuntimeError("Explicit lifecycle trace and reporting phase duration diverged")
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
            lifecycle_trace_kind="explicit_session_lifecycle_v1",
            lifecycle_trace_event_count=len(trace.events),
        )
        await self._capture_server_metadata(f"phase_start:{phase.name}")
        self.record_event("lifecycle_trace_started", duration_seconds=trace.duration_seconds)
        for event in trace.events:
            remaining = event.offset_seconds - (time.perf_counter() - phase_started)
            if remaining > 0:
                await asyncio.sleep(remaining)
            await self._schedule_lifecycle_event(event)
        remaining = trace.duration_seconds - (time.perf_counter() - phase_started)
        if remaining > 0:
            await asyncio.sleep(remaining)
        await self._wait_for_all_lifecycle_departures()
        phase_completed = time.perf_counter()
        await self._capture_server_metadata(f"phase_end:{phase.name}")
        result = self._summarize_phase(
            phase,
            phase_started=phase_started,
            phase_completed=phase_completed,
            samples=self._samples[sample_start:],
        )
        self._phase_results.append(result)
        self.record_event("lifecycle_trace_completed", scheduled_event_count=len(trace.events))
        self.record_event("phase_completed", phase=phase.name, summary=result["summary"])

    def _spawn_lifecycle_task(
        self,
        tasks: dict[str, asyncio.Task[None]],
        trace_session_id: str,
        coroutine: Any,
    ) -> asyncio.Task[None]:
        """Track a lifecycle task both for cleanup and admission ordering."""
        task = asyncio.create_task(coroutine)
        tasks[trace_session_id] = task
        self._background_tasks.add(task)

        def _complete(completed: asyncio.Task[None]) -> None:
            self._background_tasks.discard(completed)
            if tasks.get(trace_session_id) is completed:
                del tasks[trace_session_id]

        task.add_done_callback(_complete)
        return task

    async def _wait_for_departures_before_arrival(self) -> None:
        """Wait only when unfinished teardowns still consume all capacity.

        The source trace's retained-session count is capacity-normalized, but
        public HTTP deletion/LiveKit disconnect are asynchronous. At a source
        timestamp that replaces a departing session, issuing the new POST while
        the old DELETE is still in flight produces a harness-only 429. Keep
        the source schedule concurrent unless that narrow physical-capacity
        collision exists.
        """
        closing = tuple(task for task in self._lifecycle_departure_tasks.values() if not task.done())
        physical_retained = len(self._trace_sessions) + len(closing)
        if physical_retained < self._lifecycle_admission_capacity:
            return
        self.record_event(
            "lifecycle_arrival_waiting_for_departure",
            physical_retained=physical_retained,
            admission_capacity=self._lifecycle_admission_capacity,
            closing_sessions=len(closing),
        )
        await asyncio.gather(*closing, return_exceptions=True)
        self.record_event(
            "lifecycle_arrival_departure_wait_completed",
            admission_capacity=self._lifecycle_admission_capacity,
            closing_sessions=len(closing),
        )

    async def _wait_for_all_lifecycle_departures(self) -> None:
        """Finish trace departures before recording the phase-end metadata."""
        closing = tuple(task for task in self._lifecycle_departure_tasks.values() if not task.done())
        if closing:
            await asyncio.gather(*closing, return_exceptions=True)

    async def _stop_after_start(
        self,
        session: wave.LiveKitWaveSession,
        start_task: asyncio.Task[None] | None,
    ) -> None:
        """Ensure an arrival already in progress is deleted before replacement."""
        if start_task is not None:
            await asyncio.gather(start_task, return_exceptions=True)
        await session.stop()

    async def _schedule_lifecycle_event(self, event: ExplicitLifecycleEvent) -> None:
        """Schedule one trace event, preserving public-session capacity."""
        shared = {
            "trace_session_id": event.trace_session_id,
            "source_trace_session_id": event.source_session_id,
            "source_trace_user_id": event.source_user_id,
            "trace_event_sequence": event.sequence,
            "trace_scheduled_offset_seconds": event.offset_seconds,
        }
        if event.event == "session_arrival":
            if event.trace_session_id in self._trace_sessions:
                raise RuntimeError(f"Trace session {event.trace_session_id!r} arrived twice")
            await self._wait_for_departures_before_arrival()
            session = wave.LiveKitWaveSession(
                index=len(self._sessions),
                scenario=self.scenario,
                http=self._http,
                rtc=self.rtc,
                record_event=self.record_event,
                started_at=self.started_at,
                trace_session_id=event.trace_session_id,
                source_trace_session_id=event.source_session_id,
                source_trace_user_id=event.source_user_id,
            )
            session.input_enabled = bool(event.input_enabled)
            session.scheduled_at = time.perf_counter()
            self._sessions.append(session)
            self._trace_sessions[event.trace_session_id] = session
            self.record_event("lifecycle_session_arrival_scheduled", input_enabled=session.input_enabled, **shared)
            self._spawn_lifecycle_task(
                self._lifecycle_start_tasks,
                event.trace_session_id,
                self._delayed_start(session, 0.0),
            )
            return

        session = self._trace_sessions.get(event.trace_session_id)
        if session is None:
            raise RuntimeError(f"Trace event references unknown session {event.trace_session_id!r}")
        if event.event == "session_departure":
            del self._trace_sessions[event.trace_session_id]
            session.departure_scheduled = True
            self.record_event("lifecycle_session_departure_scheduled", **shared)
            self._spawn_lifecycle_task(
                self._lifecycle_departure_tasks,
                event.trace_session_id,
                self._stop_after_start(
                    session,
                    self._lifecycle_start_tasks.get(event.trace_session_id),
                ),
            )
            return

        enabled = bool(event.input_enabled)
        self.record_event(
            "lifecycle_input_transition_scheduled",
            input_enabled=enabled,
            source_event=event.event,
            **shared,
        )
        self._spawn_background(
            self._delayed_set_input_enabled(
                session,
                enabled,
                0.0,
                reason=f"lifecycle_trace:{event.event}",
            )
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True, help="Explicit lifecycle-trace scenario JSON")
    parser.add_argument("--server-url", help="Override scenario.server_url")
    parser.add_argument("--output", type=Path, help="Write complete workload artifact")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without contacting the service")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenario_path = args.scenario.expanduser()
    if not scenario_path.is_absolute():
        scenario_path = (wave._REPO_ROOT / scenario_path).resolve()
    scenario = wave.load_scenario(scenario_path, server_url_override=args.server_url)
    trace = load_explicit_lifecycle_trace(scenario)
    if args.dry_run:
        counts: dict[str, int] = {}
        for event in trace.events:
            counts[event.event] = counts.get(event.event, 0) + 1
        print(
            json.dumps(
                {
                    "scenario": scenario.name,
                    "duration_seconds": trace.duration_seconds,
                    "event_count": len(trace.events),
                    "event_counts": counts,
                    "reporting_phase": scenario.phases[0].name,
                    "diagnostic_initial_control_barrier": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.output is None:
        raise SystemExit("--output is required unless --dry-run is used")
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (wave._REPO_ROOT / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(ExplicitLifecycleRunner(scenario, trace).run())
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wave._print_summary(result)
    print(f"Wrote complete artifact: {output}")


if __name__ == "__main__":
    main()
