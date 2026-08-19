from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path

from tools.validation import benchmark_abot_livekit_burst as wave
from tools.validation import derive_abot_turboserve_trace as adapter
from tools.validation import replay_abot_livekit_lifecycle_trace as replay

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = _REPO_ROOT.parent / "TurboServe" / "traces" / "example_8gpu.json"
_WORKLOADS = _REPO_ROOT / "tools" / "validation" / "workloads"


def _peak_retained(events: list[Mapping[str, object]]) -> int:
    retained: set[str] = set()
    peak = 0
    for event in events:
        trace_session_id = event["trace_session_id"]
        assert isinstance(trace_session_id, str)
        if event["event"] == "session_arrival":
            retained.add(trace_session_id)
        elif event["event"] == "session_departure":
            retained.remove(trace_session_id)
        peak = max(peak, len(retained))
    assert not retained
    return peak


def test_checked_in_turboserve_public_demo_scenarios_are_deterministic_and_runnable() -> None:
    expected_1gpu, expected_4gpu = adapter.build_scenarios(_SOURCE)
    for expected in (expected_1gpu, expected_4gpu):
        filename = f"{expected['name']}.json"
        actual = json.loads((_WORKLOADS / filename).read_text(encoding="utf-8"))
        assert actual == expected

        contract = actual["trace_contract"]
        assert contract["not_a_turboserve_production_trace"] is True
        assert contract["not_a_reproduction_of_private_paper_t1_to_t6_traces"] is True
        assert contract["time_transform"]["source_to_derived_scale"] == 1.0
        assert contract["execution_contract"].startswith("No diagnostic barrier")

        trace = actual["lifecycle_trace"]
        assert trace["kind"] == "explicit_session_lifecycle_v1"
        assert trace["duration_seconds"] == 1800.0
        events = trace["events"]
        assert isinstance(events, list)
        assert _peak_retained(events) == contract["capacity_transform"]["target_peak_retained_sessions"]

        scenario = wave.load_scenario(_WORKLOADS / filename)
        parsed = replay.load_explicit_lifecycle_trace(scenario)
        assert parsed.duration_seconds == 1800.0
        assert parsed.events
        assert scenario.diagnostic_initial_control_barrier is None


def test_public_demo_capacity_normalization_retains_real_pause_resume_events() -> None:
    one_gpu, four_gpu = adapter.build_scenarios(_SOURCE)
    for scenario, peak, expected_counts in (
        (one_gpu, 4, {"session_arrival": 61, "session_departure": 61, "user_active": 66, "user_idle": 69}),
        (four_gpu, 16, {"session_arrival": 300, "session_departure": 300, "user_active": 282, "user_idle": 323}),
    ):
        trace = scenario["lifecycle_trace"]
        assert isinstance(trace, Mapping)
        events = trace["events"]
        assert isinstance(events, list)
        actual_counts: dict[str, int] = {}
        for event in events:
            assert isinstance(event, Mapping)
            name = event["event"]
            assert isinstance(name, str)
            actual_counts[name] = actual_counts.get(name, 0) + 1
        assert actual_counts == expected_counts
        assert _peak_retained(events) == peak
        assert actual_counts["user_active"] > 0
        assert actual_counts["user_idle"] > 0


def test_lifecycle_replay_waits_for_inflight_departure_at_capacity() -> None:
    """A same-timestamp replacement cannot POST before DELETE completes."""

    async def check() -> None:
        runner = object.__new__(replay.ExplicitLifecycleRunner)
        runner.started_at = time.perf_counter()
        runner._events = []
        runner._trace_sessions = {"retained-0": object(), "retained-1": object(), "retained-2": object()}
        runner._lifecycle_admission_capacity = 4
        departure_completed = asyncio.Event()

        async def finish_departure() -> None:
            await asyncio.sleep(0)
            departure_completed.set()

        departure = asyncio.create_task(finish_departure())
        runner._lifecycle_departure_tasks = {"departing": departure}

        await runner._wait_for_departures_before_arrival()

        assert departure_completed.is_set()
        assert [event["event"] for event in runner._events] == [
            "lifecycle_arrival_waiting_for_departure",
            "lifecycle_arrival_departure_wait_completed",
        ]

    asyncio.run(check())
