from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tools.validation import benchmark_abot_livekit_burst as wave


def _scenario_payload(image_path: Path) -> dict[str, Any]:
    return {
        "name": "unit-wave",
        "server_url": "http://127.0.0.1:8088",
        "expected_worker_mode": "process-nccl",
        "expected_num_workers": 4,
        "seed": 7,
        "session": {
            "prompt": "unit-test prompt",
            "image_path": str(image_path),
            "fps": 12,
            "control_latent_frames": 3,
            "delivery_mode": "latest",
            "expected_preview_frames": 1,
            "control": {"action_states": [["KeyW"]]},
        },
        "measurement": {
            "sample_interval_seconds": 1,
            "connect_timeout_seconds": 90,
            "http_timeout_seconds": 30,
            "shutdown_timeout_seconds": 20,
            "first_generation_grace_seconds": 15,
        },
        "phases": [
            {"name": "warmup", "duration_seconds": 2, "target_users": 4, "arrival_window_seconds": 1},
            {"name": "recovery", "duration_seconds": 2, "target_users": 2, "departure_window_seconds": 1},
        ],
    }


def _load_scenario(tmp_path: Path) -> wave.Scenario:
    image = tmp_path / "initial.png"
    image.write_bytes(b"test image placeholder")
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(_scenario_payload(image)), encoding="utf-8")
    return wave.load_scenario(scenario_path)


def _runner_for_scheduling(scenario: wave.Scenario) -> tuple[wave.LiveKitWaveRunner, list[Any]]:
    runner = object.__new__(wave.LiveKitWaveRunner)
    runner.scenario = scenario
    runner._sessions = []
    runner._background_tasks = set()
    runner._http = object()
    runner.rtc = object()
    runner.started_at = 0.0
    runner.record_event = lambda *args, **kwargs: None
    scheduled: list[Any] = []
    runner._spawn_background = scheduled.append
    return runner, scheduled


def _session(index: int, scenario: wave.Scenario) -> wave.LiveKitWaveSession:
    return wave.LiveKitWaveSession(
        index=index,
        scenario=scenario,
        http=object(),
        rtc=object(),
        record_event=lambda *args, **kwargs: None,
        started_at=0.0,
    )


def test_load_scenario_validates_lf3_process_nccl_wave(tmp_path: Path) -> None:
    scenario = _load_scenario(tmp_path)

    assert scenario.expected_worker_mode == "process-nccl"
    assert scenario.expected_num_workers == 4
    assert scenario.session.fps == 12
    assert scenario.session.control_latent_frames == 3
    assert scenario.first_generation_grace_seconds == 15
    assert scenario.slo_fps_tolerance == 0.25
    assert [phase.target_users for phase in scenario.phases] == [4, 2]


def test_load_scenario_rejects_arrival_window_longer_than_phase(tmp_path: Path) -> None:
    image = tmp_path / "initial.png"
    image.write_bytes(b"test image placeholder")
    payload = _scenario_payload(image)
    payload["phases"][0]["arrival_window_seconds"] = 3
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(wave.ScenarioError, match="cannot exceed duration"):
        wave.load_scenario(scenario_path)


def test_scale_down_marks_newest_sessions_and_keeps_them_counted_until_stop(tmp_path: Path) -> None:
    scenario = _load_scenario(tmp_path)
    runner, scheduled = _runner_for_scheduling(scenario)
    runner._sessions = [_session(index, scenario) for index in range(4)]

    runner._schedule_transition(wave.Phase("down", 2, target_users=2, departure_window_seconds=1))

    assert [session.index for session in runner._sessions if session.departure_scheduled] == [2, 3]
    assert all(not session.stop_requested for session in runner._sessions)
    assert len(scheduled) == 2
    for coroutine in scheduled:
        coroutine.close()


def test_scale_up_spreads_only_new_arrivals_across_its_window(tmp_path: Path) -> None:
    scenario = _load_scenario(tmp_path)
    runner, scheduled = _runner_for_scheduling(scenario)
    runner._sessions = [_session(index, scenario) for index in range(4)]

    runner._schedule_transition(wave.Phase("up", 2, target_users=8, arrival_window_seconds=3))

    assert [session.index for session in runner._sessions] == list(range(8))
    assert len(scheduled) == 4
    for coroutine in scheduled:
        coroutine.close()


def test_slo_includes_zero_fps_after_first_generation_grace(tmp_path: Path) -> None:
    scenario = _load_scenario(tmp_path)
    runner, _ = _runner_for_scheduling(scenario)
    session = _session(0, scenario)
    session.connected = True
    session.current_controls = ("KeyW",)
    session.first_active_control_at = 0.0

    assert runner._session_delivery_fps(session, now=14.9, interval=1.0, delta=0) == (None, None)
    assert runner._session_delivery_fps(session, now=15.0, interval=1.0, delta=0) == (0.0, 0.0)

    session.first_generated_frame_at = 1.0
    assert runner._session_delivery_fps(session, now=15.0, interval=2.0, delta=24) == (12.0, 12.0)


def test_load_scenario_parses_all_active_admission_contract(tmp_path: Path) -> None:
    image = tmp_path / "initial.png"
    image.write_bytes(b"test image placeholder")
    payload = _scenario_payload(image)
    payload["admission"] = {
        "require_immediate_assignment": True,
        "expected_max_sessions_per_worker": 4,
        "expected_queue_size": 0,
    }
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(payload), encoding="utf-8")

    scenario = wave.load_scenario(scenario_path)

    assert scenario.admission.require_immediate_assignment is True
    assert scenario.admission.expected_max_sessions_per_worker == 4
    assert scenario.admission.expected_queue_size == 0


def test_requested_user_fps_counts_unserved_request_as_zero_after_grace(tmp_path: Path) -> None:
    scenario = _load_scenario(tmp_path)
    runner, _ = _runner_for_scheduling(scenario)
    session = _session(0, scenario)
    session.create_started_at = 0.0

    assert runner._session_requested_delivery_fps(session, now=14.9, interval=1.0, delta=0) is None
    assert runner._session_requested_delivery_fps(session, now=15.0, interval=1.0, delta=0) == 0.0

    session.first_generated_frame_at = 1.0
    assert runner._session_requested_delivery_fps(session, now=15.0, interval=2.0, delta=24) == 12.0


def test_peak16_trace_requires_all_active_users() -> None:
    scenario_path = (
        wave._REPO_ROOT / "tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_all_active_peak16_wave.json"
    )
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))

    assert payload["admission"] == {
        "require_immediate_assignment": True,
        "expected_max_sessions_per_worker": 4,
        "expected_queue_size": 0,
    }
    assert [phase["target_users"] for phase in payload["phases"]] == [4, 8, 12, 16, 8, 4]
    assert sum(phase["duration_seconds"] for phase in payload["phases"]) == 330.0


def test_phase_input_activity_pauses_and_resumes_without_departure(tmp_path: Path) -> None:
    scenario = _load_scenario(tmp_path)
    runner, scheduled = _runner_for_scheduling(scenario)
    runner._sessions = [_session(index, scenario) for index in range(8)]

    runner._schedule_input_activity(wave.Phase("pause", 2, target_users=8, active_input_fraction=0.5))

    assert len(scheduled) == 4
    for coroutine in scheduled:
        asyncio.run(coroutine)
    assert sum(session.input_enabled for session in runner._sessions) == 4
    assert all(not session.stop_requested for session in runner._sessions)
    assert all(not session.departure_scheduled for session in runner._sessions)

    scheduled.clear()
    runner._schedule_input_activity(wave.Phase("resume", 2, target_users=8, active_input_fraction=1.0))
    assert len(scheduled) == 4
    for coroutine in scheduled:
        asyncio.run(coroutine)
    assert all(session.input_enabled for session in runner._sessions)


def test_intermittent_peak16_trace_models_pauses_and_reengagement() -> None:
    scenario_path = (
        wave._REPO_ROOT / "tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_intermittent_input_peak16.json"
    )
    payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    phases = payload["phases"]

    assert payload["admission"] == {
        "require_immediate_assignment": True,
        "expected_max_sessions_per_worker": 4,
        "expected_queue_size": 0,
    }
    assert [phase["target_users"] for phase in phases] == [4, 8, 8, 16, 16, 16, 16, 8, 4]
    assert max(phase["target_users"] for phase in phases) == 16
    assert sum(phase["duration_seconds"] for phase in phases) == 385.0
    assert phases[2]["active_input_fraction"] == 0.5
    assert phases[5]["active_input_fraction"] == 0.5
    assert phases[6]["active_input_fraction"] == 1.0
