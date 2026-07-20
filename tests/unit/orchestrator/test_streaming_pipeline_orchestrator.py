from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest

from telefuser.orchestrator.streaming_pipeline_orchestrator import (
    LocalStageActor,
    StageOrdering,
    StreamingEdgeSpec,
    StreamingPipelineOrchestrator,
    StreamingPipelineSpec,
    StreamingSessionStatus,
    StreamingStageSpec,
)


def _wait_for_outputs(
    orchestrator: StreamingPipelineOrchestrator,
    session_id: str,
    expected_count: int,
) -> list[tuple[int, str, object]]:
    deadline = time.monotonic() + 2.0
    outputs: list[tuple[int, str, object]] = []
    while time.monotonic() < deadline:
        outputs.extend(orchestrator.poll_outputs(session_id))
        if len(outputs) >= expected_count:
            return outputs
        time.sleep(0.01)
    return outputs


def _orchestrator(
    encode_calls: list[tuple[int, bool, bool]],
    denoise_calls: list[tuple[int, bool, bool]],
) -> StreamingPipelineOrchestrator:
    def encode(invocation):
        encode_calls.append((invocation.key.sequence_id, invocation.is_first, invocation.is_last))
        return {"condition": f"condition-{invocation.inputs['image']}"}

    def denoise(invocation):
        denoise_calls.append((invocation.key.sequence_id, invocation.is_first, invocation.is_last))
        return {"frames": f"{invocation.inputs['condition']}:{invocation.inputs['control']}"}

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("encode", frozenset({"image"}), frozenset({"condition"})),
            StreamingStageSpec("denoise", frozenset({"condition", "control"}), frozenset({"frames"})),
        ),
        edges=(
            StreamingEdgeSpec("image", "encode"),
            StreamingEdgeSpec("condition", "denoise", source_stage="encode", capacity_per_session=1),
            StreamingEdgeSpec("control", "denoise"),
        ),
        output_artifacts=frozenset({"frames"}),
    )
    return StreamingPipelineOrchestrator(
        spec,
        {
            "encode": LocalStageActor(encode, name="test-encode"),
            "denoise": LocalStageActor(denoise, name="test-denoise"),
        },
    )


def test_streaming_orchestrator_joins_inputs_preserves_order_and_emits_outputs() -> None:
    encode_calls: list[tuple[int, bool, bool]] = []
    denoise_calls: list[tuple[int, bool, bool]] = []
    orchestrator = _orchestrator(encode_calls, denoise_calls)
    try:
        assert orchestrator.create_session("session", final_sequence_id=1) == 1

        orchestrator.push_input("session", 0, "image", "a")
        assert orchestrator.wait_until_idle("session")
        assert denoise_calls == []

        orchestrator.push_input("session", 0, "control", "left")
        assert orchestrator.wait_until_idle("session")
        assert _wait_for_outputs(orchestrator, "session", 1) == [(0, "frames", "condition-a:left")]

        orchestrator.push_input("session", 1, "image", "b")
        orchestrator.push_input("session", 1, "control", "right")
        assert orchestrator.wait_until_idle("session")
        assert _wait_for_outputs(orchestrator, "session", 1) == [(1, "frames", "condition-b:right")]

        assert encode_calls == [(0, True, False), (1, False, True)]
        assert denoise_calls == [(0, True, False), (1, False, True)]
    finally:
        orchestrator.close()


def test_bounded_condition_edge_applies_backpressure_until_downstream_consumes() -> None:
    encode_calls: list[tuple[int, bool, bool]] = []
    denoise_calls: list[tuple[int, bool, bool]] = []
    orchestrator = _orchestrator(encode_calls, denoise_calls)
    try:
        orchestrator.create_session("session")
        orchestrator.push_input("session", 0, "image", "a")
        assert orchestrator.wait_until_idle("session")
        assert encode_calls == [(0, True, False)]

        orchestrator.push_input("session", 1, "image", "b")
        assert orchestrator.wait_until_idle("session")
        assert encode_calls == [(0, True, False)]

        orchestrator.push_input("session", 0, "control", "left")
        assert orchestrator.wait_until_idle("session")
        assert _wait_for_outputs(orchestrator, "session", 1)
        assert orchestrator.wait_until_idle("session")
        assert encode_calls == [(0, True, False), (1, False, False)]
    finally:
        orchestrator.close()


def test_stage_failure_poison_session_and_prevents_future_input() -> None:
    spec = StreamingPipelineSpec(
        stages=(StreamingStageSpec("broken", frozenset({"value"}), frozenset({"output"})),),
        edges=(StreamingEdgeSpec("value", "broken"),),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {"broken": LocalStageActor(lambda _: {"wrong": "value"}, name="test-broken")},
    )
    try:
        orchestrator.create_session("session")
        orchestrator.push_input("session", 0, "value", 1)
        assert orchestrator.wait_until_idle("session")
        assert orchestrator.status("session") == StreamingSessionStatus.FAILED
        assert isinstance(orchestrator.error("session"), ValueError)
        with pytest.raises(RuntimeError, match="not accepting input"):
            orchestrator.push_input("session", 1, "value", 2)
    finally:
        orchestrator.close()


def test_validation_rejects_multiple_artifact_producers() -> None:
    first = StreamingStageSpec("first", frozenset({"input"}), frozenset({"shared"}))
    second = StreamingStageSpec("second", frozenset({"other"}), frozenset({"shared"}))
    target = StreamingStageSpec("target", frozenset({"shared"}), frozenset({"result"}))
    spec = StreamingPipelineSpec(
        stages=(first, second, target),
        edges=(
            StreamingEdgeSpec("input", "first"),
            StreamingEdgeSpec("other", "second"),
            StreamingEdgeSpec("shared", "target", source_stage="first"),
            StreamingEdgeSpec("shared", "target", source_stage="second"),
        ),
        output_artifacts=frozenset({"result"}),
    )
    actors = {
        "first": LocalStageActor(lambda _: {"shared": 1}, name="test-first"),
        "second": LocalStageActor(lambda _: {"shared": 2}, name="test-second"),
        "target": LocalStageActor(lambda _: {"result": 3}, name="test-target"),
    }
    try:
        with pytest.raises(ValueError, match="multiple producers"):
            StreamingPipelineOrchestrator(spec, actors)
    finally:
        for actor in actors.values():
            actor.close()


def test_stage_idle_metrics_attribute_a_dit_bubble_to_missing_condition() -> None:
    def encode(invocation):
        if invocation.key.sequence_id == 1:
            time.sleep(0.05)
        return {"condition": invocation.key.sequence_id}

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("encode", frozenset({"image"}), frozenset({"condition"})),
            StreamingStageSpec("denoise", frozenset({"condition", "control", "noise"}), frozenset({"frames"})),
        ),
        edges=(
            StreamingEdgeSpec("image", "encode", capacity_per_session=2),
            StreamingEdgeSpec("condition", "denoise", source_stage="encode", capacity_per_session=2),
            StreamingEdgeSpec("control", "denoise", capacity_per_session=2),
            StreamingEdgeSpec("noise", "denoise", capacity_per_session=2),
        ),
        output_artifacts=frozenset({"frames"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "encode": LocalStageActor(encode, name="timing-encode"),
            "denoise": LocalStageActor(
                lambda invocation: {"frames": invocation.key.sequence_id}, name="timing-denoise"
            ),
        },
    )
    try:
        orchestrator.create_session("session", final_sequence_id=1)
        for sequence_id in range(2):
            orchestrator.push_input("session", sequence_id, "image", sequence_id)
            orchestrator.push_input("session", sequence_id, "control", sequence_id)
            orchestrator.push_input("session", sequence_id, "noise", sequence_id)

        assert _wait_for_outputs(orchestrator, "session", 2) == [(0, "frames", 0), (1, "frames", 1)]
        timings = orchestrator.stage_timings("session", "denoise")
        intervals = orchestrator.stage_idle_intervals("session", "denoise")
    finally:
        orchestrator.close()

    assert len(timings) == 2
    assert all(timing.inputs_ready_at is not None for timing in timings)
    assert all(timing.admitted_at is not None and timing.completed_at is not None for timing in timings)
    assert len(intervals) == 1
    assert intervals[0].idle_seconds >= 0.03
    assert intervals[0].reason == "inputs_not_ready"
    assert intervals[0].missing_inputs == ("condition",)


def test_try_push_inputs_is_atomic_when_one_ingress_is_backpressured() -> None:
    spec = StreamingPipelineSpec(
        stages=(StreamingStageSpec("join", frozenset({"left", "right"}), frozenset({"result"})),),
        edges=(
            StreamingEdgeSpec("left", "join", capacity_per_session=1),
            StreamingEdgeSpec("right", "join", capacity_per_session=2),
        ),
        output_artifacts=frozenset({"result"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {"join": LocalStageActor(lambda invocation: {"result": invocation.inputs["left"]}, name="atomic-join")},
    )
    try:
        orchestrator.create_session("session", final_sequence_id=1)
        orchestrator.push_input("session", 0, "left", "first")

        assert not orchestrator.try_push_inputs("session", 1, {"left": "second", "right": "second"})
        assert orchestrator.can_push_input("session", "right")
        orchestrator.push_input("session", 1, "right", "second")

        orchestrator.push_input("session", 0, "right", "first")
        assert _wait_for_outputs(orchestrator, "session", 1) == [(0, "result", "first")]
        assert orchestrator.try_push_inputs("session", 1, {"left": "second"})
        assert _wait_for_outputs(orchestrator, "session", 1) == [(1, "result", "second")]
    finally:
        orchestrator.close()


def test_busy_actor_retries_ready_work_from_another_session() -> None:
    started = threading.Event()
    release = threading.Event()

    def process(invocation):
        started.set()
        release.wait(timeout=1)
        return {"output": invocation.inputs["value"]}

    spec = StreamingPipelineSpec(
        stages=(StreamingStageSpec("stage", frozenset({"value"}), frozenset({"output"})),),
        edges=(StreamingEdgeSpec("value", "stage"),),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {"stage": LocalStageActor(process, mailbox_capacity=1, name="multi-session")},
    )
    try:
        for session_id in ("first", "second", "third"):
            orchestrator.create_session(session_id, final_sequence_id=0)
        orchestrator.push_input("first", 0, "value", "first")
        assert started.wait(timeout=1)
        orchestrator.push_input("second", 0, "value", "second")
        orchestrator.push_input("third", 0, "value", "third")
        release.set()

        for session_id in ("first", "second", "third"):
            assert orchestrator.wait_until_idle(session_id)
            assert _wait_for_outputs(orchestrator, session_id, 1) == [(0, "output", session_id)]
    finally:
        release.set()
        orchestrator.close()


class _ManualActor:
    def __init__(self) -> None:
        self.invocations = []
        self.futures: list[Future] = []

    def submit(self, invocation):
        future = Future()
        self.invocations.append(invocation)
        self.futures.append(future)
        return future

    def close(self) -> None:
        return None


def test_in_flight_output_reserves_bounded_output_capacity() -> None:
    actor = _ManualActor()
    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec(
                "stage",
                frozenset({"value"}),
                frozenset({"output"}),
                ordering=StageOrdering.NONE,
                max_in_flight_per_session=2,
            ),
        ),
        edges=(StreamingEdgeSpec("value", "stage", capacity_per_session=2),),
        output_artifacts=frozenset({"output"}),
        output_capacity_per_session=1,
    )
    orchestrator = StreamingPipelineOrchestrator(spec, {"stage": actor})
    try:
        orchestrator.create_session("session", final_sequence_id=1)
        orchestrator.push_input("session", 0, "value", 0)
        orchestrator.push_input("session", 1, "value", 1)
        assert len(actor.invocations) == 1

        actor.futures[0].set_result({"output": 0})
        assert len(actor.invocations) == 1
        assert orchestrator.poll_outputs("session") == [(0, "output", 0)]
        assert len(actor.invocations) == 2

        actor.futures[1].set_result({"output": 1})
        assert orchestrator.poll_outputs("session") == [(1, "output", 1)]
    finally:
        orchestrator.close()


def test_validation_rejects_cycles() -> None:
    first_actor = _ManualActor()
    second_actor = _ManualActor()
    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("first", frozenset({"second_output"}), frozenset({"first_output"})),
            StreamingStageSpec("second", frozenset({"first_output"}), frozenset({"second_output"})),
        ),
        edges=(
            StreamingEdgeSpec("first_output", "second", source_stage="first"),
            StreamingEdgeSpec("second_output", "first", source_stage="second"),
        ),
        output_artifacts=frozenset({"first_output"}),
    )
    with pytest.raises(ValueError, match="contains a cycle"):
        StreamingPipelineOrchestrator(spec, {"first": first_actor, "second": second_actor})


def test_finish_inputs_rejects_a_final_marker_after_admission() -> None:
    actor = LocalStageActor(lambda _: {"output": 1}, name="late-final")
    spec = StreamingPipelineSpec(
        stages=(StreamingStageSpec("stage", frozenset({"value"}), frozenset({"output"})),),
        edges=(StreamingEdgeSpec("value", "stage"),),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(spec, {"stage": actor})
    try:
        orchestrator.create_session("session")
        orchestrator.push_input("session", 0, "value", 1)
        assert orchestrator.wait_until_idle("session")
        with pytest.raises(RuntimeError, match="before its first stage is admitted"):
            orchestrator.finish_inputs("session", 0)
    finally:
        orchestrator.close()
