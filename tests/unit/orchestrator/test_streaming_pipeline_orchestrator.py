from __future__ import annotations

import gc
import threading
import time
import weakref
from concurrent.futures import Future

import pytest

from telefuser.orchestrator.streaming_pipeline_orchestrator import (
    LocalStageActor,
    StageOrdering,
    StreamingActorFailedError,
    StreamingActorHealth,
    StreamingActorState,
    StreamingEdgeSpec,
    StreamingPipelineOrchestrator,
    StreamingPipelineSpec,
    StreamingResourceGroupSpec,
    StreamingSessionCloseReason,
    StreamingSessionContext,
    StreamingSessionStatus,
    StreamingStageInvocation,
    StreamingStageSpec,
    StreamingTaskKey,
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


def test_session_metrics_report_end_to_end_percentiles_after_output_polling() -> None:
    orchestrator = _orchestrator([], [])
    try:
        orchestrator.create_session("session", final_sequence_id=1)
        for sequence_id in range(2):
            assert orchestrator.try_push_inputs(
                "session",
                sequence_id,
                {"image": sequence_id, "control": sequence_id},
            )
            assert orchestrator.wait_until_idle("session")
            assert len(orchestrator.poll_outputs("session")) == 1

        metrics = orchestrator.session_metrics("session")
        assert [sequence_id for sequence_id, _ in metrics.ingress_accepted_at] == [0, 1]
        assert [sequence_id for sequence_id, _ in metrics.output_emitted_at] == [0, 1]
        assert metrics.first_output_latency_seconds is not None
        assert metrics.first_output_latency_seconds >= 0
        assert metrics.control_to_output_latency.count == 2
        assert metrics.control_to_output_latency.p50_seconds is not None
        assert metrics.control_to_output_latency.p95_seconds is not None
        assert metrics.chunk_period.count == 1
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
        assert orchestrator.actor_health("broken").state == StreamingActorState.RUNNING
        assert orchestrator.actor_health("broken").pending_invocations == 0
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
    assert intervals[0].reason == "condition"
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
                max_in_flight_global=2,
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


def test_actor_health_and_graph_barrier_track_pending_work() -> None:
    started = threading.Event()
    release = threading.Event()

    def process(invocation: StreamingStageInvocation) -> dict[str, object]:
        started.set()
        release.wait(timeout=1)
        return {"output": invocation.inputs["value"]}

    actor = LocalStageActor(process, name="health-barrier")
    spec = StreamingPipelineSpec(
        stages=(StreamingStageSpec("stage", frozenset({"value"}), frozenset({"output"})),),
        edges=(StreamingEdgeSpec("value", "stage"),),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(spec, {"stage": actor})
    try:
        orchestrator.create_session("session", final_sequence_id=0)
        orchestrator.push_input("session", 0, "value", "payload")
        assert started.wait(timeout=1)
        health = orchestrator.actor_health("stage")
        assert health.state == StreamingActorState.RUNNING
        assert health.pending_invocations == 1
        with pytest.raises(TimeoutError, match="graph quiescence"):
            orchestrator.barrier(timeout=0.01)

        release.set()
        orchestrator.barrier(timeout=1)
        assert orchestrator.actor_health("stage").pending_invocations == 0
        assert orchestrator.poll_outputs("session") == [(0, "output", "payload")]
    finally:
        release.set()
        orchestrator.close()

    assert actor.health().state == StreamingActorState.CLOSED


def test_close_attempts_every_actor_before_reporting_failure() -> None:
    closed: list[str] = []

    class CloseTrackingActor:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def submit(self, invocation: StreamingStageInvocation) -> Future:
            raise AssertionError(f"unexpected submission: {invocation}")

        def close(self) -> None:
            closed.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("first", frozenset({"first_input"}), frozenset({"first_output"})),
            StreamingStageSpec("second", frozenset({"second_input"}), frozenset({"second_output"})),
        ),
        edges=(
            StreamingEdgeSpec("first_input", "first"),
            StreamingEdgeSpec("second_input", "second"),
        ),
        output_artifacts=frozenset({"first_output", "second_output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "first": CloseTrackingActor("first", fail=True),
            "second": CloseTrackingActor("second"),
        },
    )

    with pytest.raises(RuntimeError, match="one or more streaming stage actors") as exc_info:
        orchestrator.close()

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert closed == ["second", "first"]


def test_local_actor_failure_resolves_every_queued_future_once() -> None:
    started = threading.Event()
    release = threading.Event()

    def process(invocation: StreamingStageInvocation) -> dict[str, object]:
        started.set()
        release.wait(timeout=1)
        return {"output": invocation.key.sequence_id}

    actor = LocalStageActor(process, mailbox_capacity=2, name="failure-resolution")
    first = StreamingStageInvocation(
        StreamingTaskKey("session", 1, 0, "stage", "first"),
        {},
        True,
        False,
    )
    second = StreamingStageInvocation(
        StreamingTaskKey("session", 1, 1, "stage", "second"),
        {},
        False,
        True,
    )
    callbacks = {"first": 0, "second": 0}
    first_future = actor.submit(first)
    assert started.wait(timeout=1)
    second_future = actor.submit(second)
    first_future.add_done_callback(lambda _: callbacks.__setitem__("first", callbacks["first"] + 1))
    second_future.add_done_callback(lambda _: callbacks.__setitem__("second", callbacks["second"] + 1))

    first_future.set_result({"output": "injected"})
    release.set()
    actor.barrier(timeout=1)
    try:
        assert first_future.result() == {"output": "injected"}
        with pytest.raises(RuntimeError, match="Stage actor failed"):
            second_future.result()
        health = actor.health()
        assert health.state == StreamingActorState.FAILED
        assert health.pending_invocations == 0
        assert callbacks == {"first": 1, "second": 1}
    finally:
        release.set()
        actor.close()


def test_resource_group_serializes_independent_stage_actors() -> None:
    lock = threading.Lock()
    active = 0
    max_active = 0
    first_started = threading.Event()
    second_started = threading.Event()
    first_release = threading.Event()
    second_release = threading.Event()

    def handler(
        name: str,
        started: threading.Event,
        release: threading.Event,
        output: str,
    ):
        def process(_: StreamingStageInvocation) -> dict[str, object]:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            started.set()
            release.wait(timeout=1)
            with lock:
                active -= 1
            return {output: name}

        return process

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec(
                "first",
                frozenset({"first_input"}),
                frozenset({"first_output"}),
                resource_group="shared",
            ),
            StreamingStageSpec(
                "second",
                frozenset({"second_input"}),
                frozenset({"second_output"}),
                resource_group="shared",
            ),
        ),
        edges=(
            StreamingEdgeSpec("first_input", "first"),
            StreamingEdgeSpec("second_input", "second"),
        ),
        output_artifacts=frozenset({"first_output", "second_output"}),
        resource_groups=(StreamingResourceGroupSpec("shared", max_concurrency=1),),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "first": LocalStageActor(
                handler("first", first_started, first_release, "first_output"),
                name="resource-first",
            ),
            "second": LocalStageActor(
                handler("second", second_started, second_release, "second_output"),
                name="resource-second",
            ),
        },
    )
    try:
        orchestrator.create_session("session", final_sequence_id=0)
        orchestrator.push_input("session", 0, "first_input", 1)
        assert first_started.wait(timeout=1)
        orchestrator.push_input("session", 0, "second_input", 2)
        assert not second_started.wait(timeout=0.05)
        assert orchestrator.resource_group_usage("shared") == 1

        first_release.set()
        assert second_started.wait(timeout=1)
        assert orchestrator.resource_group_usage("shared") == 1
        second_release.set()
        orchestrator.barrier(timeout=1)

        assert max_active == 1
        assert orchestrator.resource_group_usage("shared") == 0
        assert orchestrator.poll_outputs("session") == [
            (0, "first_output", "first"),
            (0, "second_output", "second"),
        ]
    finally:
        first_release.set()
        second_release.set()
        orchestrator.close()


def test_artifact_reference_count_and_prompt_release() -> None:
    class Prompt:
        pass

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec(
                "first",
                frozenset({"prompt", "first_control"}),
                frozenset({"first_output"}),
            ),
            StreamingStageSpec(
                "second",
                frozenset({"prompt", "second_control"}),
                frozenset({"second_output"}),
            ),
        ),
        edges=(
            StreamingEdgeSpec("prompt", "first"),
            StreamingEdgeSpec("prompt", "second"),
            StreamingEdgeSpec("first_control", "first"),
            StreamingEdgeSpec("second_control", "second"),
        ),
        output_artifacts=frozenset({"first_output", "second_output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "first": LocalStageActor(lambda _: {"first_output": "first"}, name="artifact-first"),
            "second": LocalStageActor(lambda _: {"second_output": "second"}, name="artifact-second"),
        },
    )
    try:
        orchestrator.create_session("session", final_sequence_id=0)
        prompt = Prompt()
        prompt_ref = weakref.ref(prompt)
        orchestrator.push_input("session", 0, "prompt", prompt)
        assert orchestrator.artifact_stats("session").slot_count == 1
        assert orchestrator.artifact_stats("session").reference_count == 2

        del prompt
        gc.collect()
        assert prompt_ref() is not None

        orchestrator.push_input("session", 0, "first_control", 1)
        orchestrator.barrier(timeout=1)
        assert orchestrator.artifact_stats("session").slot_count == 1
        assert orchestrator.artifact_stats("session").reference_count == 1
        assert prompt_ref() is not None

        orchestrator.push_input("session", 0, "second_control", 2)
        orchestrator.barrier(timeout=1)
        assert orchestrator.poll_outputs("session") == [
            (0, "first_output", "first"),
            (0, "second_output", "second"),
        ]
        assert orchestrator.artifact_stats("session").slot_count == 0
        assert orchestrator.artifact_stats("session").reference_count == 0
        gc.collect()
        assert prompt_ref() is None
    finally:
        orchestrator.close()


def test_resource_group_and_global_limit_validation() -> None:
    with pytest.raises(ValueError, match="non-empty name"):
        StreamingResourceGroupSpec("")
    with pytest.raises(ValueError, match="max_concurrency"):
        StreamingResourceGroupSpec("shared", max_concurrency=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        StreamingStageSpec(
            "stage",
            frozenset({"input"}),
            frozenset({"output"}),
            ordering=StageOrdering.NONE,
            max_in_flight_per_session=2,
            max_in_flight_global=1,
        )

    actor = LocalStageActor(lambda _: {"output": 1}, name="unknown-resource")
    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec(
                "stage",
                frozenset({"input"}),
                frozenset({"output"}),
                resource_group="missing",
            ),
        ),
        edges=(StreamingEdgeSpec("input", "stage"),),
        output_artifacts=frozenset({"output"}),
    )
    try:
        with pytest.raises(ValueError, match="Unknown streaming resource groups"):
            StreamingPipelineOrchestrator(spec, {"stage": actor})
    finally:
        actor.close()


def test_resource_group_permit_is_released_after_stage_failure() -> None:
    failing_started = threading.Event()
    failing_release = threading.Event()
    healthy_started = threading.Event()

    def fail(_: StreamingStageInvocation) -> dict[str, object]:
        failing_started.set()
        failing_release.wait(timeout=1)
        raise RuntimeError("injected failure")

    def succeed(_: StreamingStageInvocation) -> dict[str, object]:
        healthy_started.set()
        return {"healthy_output": "done"}

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec(
                "failing",
                frozenset({"failing_input"}),
                frozenset({"failing_output"}),
                resource_group="shared",
            ),
            StreamingStageSpec(
                "healthy",
                frozenset({"healthy_input"}),
                frozenset({"healthy_output"}),
                resource_group="shared",
            ),
        ),
        edges=(
            StreamingEdgeSpec("failing_input", "failing"),
            StreamingEdgeSpec("healthy_input", "healthy"),
        ),
        output_artifacts=frozenset({"failing_output", "healthy_output"}),
        resource_groups=(StreamingResourceGroupSpec("shared"),),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "failing": LocalStageActor(fail, name="permit-failing"),
            "healthy": LocalStageActor(succeed, name="permit-healthy"),
        },
    )
    try:
        orchestrator.create_session("failed", final_sequence_id=0)
        orchestrator.create_session("healthy", final_sequence_id=0)
        orchestrator.push_input("failed", 0, "failing_input", 1)
        assert failing_started.wait(timeout=1)
        orchestrator.push_input("healthy", 0, "healthy_input", 2)
        assert not healthy_started.wait(timeout=0.05)

        failing_release.set()
        assert healthy_started.wait(timeout=1)
        orchestrator.barrier(timeout=1)
        assert orchestrator.status("failed") == StreamingSessionStatus.FAILED
        assert orchestrator.resource_group_usage("shared") == 0
    finally:
        failing_release.set()
        orchestrator.close()


def test_global_stage_limit_applies_across_sessions() -> None:
    actor = _ManualActor()
    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec(
                "stage",
                frozenset({"input"}),
                frozenset({"output"}),
                ordering=StageOrdering.NONE,
                max_in_flight_per_session=1,
                max_in_flight_global=1,
            ),
        ),
        edges=(StreamingEdgeSpec("input", "stage", capacity_per_session=2),),
        output_artifacts=frozenset({"output"}),
        output_capacity_per_session=2,
    )
    orchestrator = StreamingPipelineOrchestrator(spec, {"stage": actor})
    try:
        orchestrator.create_session("first", final_sequence_id=0)
        orchestrator.create_session("second", final_sequence_id=0)
        orchestrator.push_input("first", 0, "input", "first")
        orchestrator.push_input("second", 0, "input", "second")
        assert len(actor.invocations) == 1

        actor.futures[0].set_result({"output": actor.invocations[0].inputs["input"]})
        assert len(actor.invocations) == 2
        actor.futures[1].set_result({"output": actor.invocations[1].inputs["input"]})
        assert {
            orchestrator.poll_outputs("first")[0][2],
            orchestrator.poll_outputs("second")[0][2],
        } == {"first", "second"}
    finally:
        orchestrator.close()


def test_duplicate_resource_group_names_are_rejected() -> None:
    actor = LocalStageActor(lambda _: {"output": 1}, name="duplicate-resource")
    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec(
                "stage",
                frozenset({"input"}),
                frozenset({"output"}),
                resource_group="shared",
            ),
        ),
        edges=(StreamingEdgeSpec("input", "stage"),),
        output_artifacts=frozenset({"output"}),
        resource_groups=(StreamingResourceGroupSpec("shared"), StreamingResourceGroupSpec("shared")),
    )
    try:
        with pytest.raises(ValueError, match="must be unique"):
            StreamingPipelineOrchestrator(spec, {"stage": actor})
    finally:
        actor.close()


def test_output_artifact_slot_releases_after_internal_consumer_admission() -> None:
    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("first", frozenset({"input"}), frozenset({"shared"})),
            StreamingStageSpec("second", frozenset({"shared"}), frozenset({"final"})),
        ),
        edges=(
            StreamingEdgeSpec("input", "first"),
            StreamingEdgeSpec("shared", "second", source_stage="first"),
        ),
        output_artifacts=frozenset({"shared", "final"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "first": LocalStageActor(lambda _: {"shared": "value"}, name="output-first"),
            "second": LocalStageActor(lambda _: {"final": "done"}, name="output-second"),
        },
    )
    try:
        orchestrator.create_session("session", final_sequence_id=0)
        orchestrator.push_input("session", 0, "input", 1)
        orchestrator.barrier(timeout=1)
        assert orchestrator.artifact_stats("session").slot_count == 0
        assert orchestrator.poll_outputs("session") == [
            (0, "final", "done"),
            (0, "shared", "value"),
        ]
    finally:
        orchestrator.close()


def test_cancellation_drains_work_and_cleans_stages_in_reverse_topological_order() -> None:
    started = threading.Event()
    release = threading.Event()
    cleanup_calls: list[tuple[str, StreamingSessionCloseReason]] = []
    cancellation_errors: list[BaseException] = []

    def process(_: StreamingStageInvocation) -> dict[str, object]:
        started.set()
        release.wait(timeout=1)
        return {"intermediate": "ignored"}

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("first", frozenset({"input"}), frozenset({"intermediate"})),
            StreamingStageSpec("second", frozenset({"intermediate"}), frozenset({"output"})),
        ),
        edges=(
            StreamingEdgeSpec("input", "first"),
            StreamingEdgeSpec("intermediate", "second", source_stage="first"),
        ),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "first": LocalStageActor(
                process,
                name="cancel-first",
                session_closer=lambda _, reason: cleanup_calls.append(("first", reason)),
            ),
            "second": LocalStageActor(
                lambda _: {"output": "unexpected"},
                name="cancel-second",
                session_closer=lambda _, reason: cleanup_calls.append(("second", reason)),
            ),
        },
    )

    def cancel() -> None:
        try:
            orchestrator.cancel_session("session", timeout=1)
        except BaseException as exc:
            cancellation_errors.append(exc)

    cancel_thread = threading.Thread(target=cancel)
    try:
        orchestrator.create_session("session", final_sequence_id=0)
        orchestrator.push_input("session", 0, "input", "payload")
        assert started.wait(timeout=1)

        cancel_thread.start()
        deadline = time.monotonic() + 1
        while orchestrator.status("session") != StreamingSessionStatus.CANCELLED:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        with pytest.raises(RuntimeError, match="not accepting input"):
            orchestrator.push_input("session", 1, "input", "late")
        assert cancel_thread.is_alive()

        release.set()
        cancel_thread.join(timeout=1)
        assert not cancel_thread.is_alive()
        assert cancellation_errors == []
        assert cleanup_calls == [
            ("second", StreamingSessionCloseReason.CANCELLED),
            ("first", StreamingSessionCloseReason.CANCELLED),
        ]
        assert orchestrator.artifact_stats("session").slot_count == 0
        assert orchestrator.poll_outputs("session") == []
        assert all(orchestrator.actor_health(stage).pending_invocations == 0 for stage in ("first", "second"))
    finally:
        release.set()
        cancel_thread.join(timeout=1)
        orchestrator.close()


def test_backend_failure_poisons_every_active_session() -> None:
    class FailingBackendActor:
        def __init__(self) -> None:
            self.state = StreamingActorState.RUNNING
            self.future: Future[dict[str, object]] | None = None
            self.cleanup_sessions: list[str] = []

        def submit(self, invocation: StreamingStageInvocation) -> Future[dict[str, object]]:
            del invocation
            self.future = Future()
            return self.future

        def health(self) -> StreamingActorHealth:
            reason = "worker exited" if self.state == StreamingActorState.FAILED else None
            return StreamingActorHealth(self.state, int(self.future is not None and not self.future.done()), reason)

        def close_session(
            self,
            context: StreamingSessionContext,
            reason: StreamingSessionCloseReason,
            timeout: float = 5.0,
        ) -> None:
            del reason, timeout
            self.cleanup_sessions.append(context.session_id)

        def close(self) -> None:
            self.state = StreamingActorState.CLOSED

    actor = FailingBackendActor()
    spec = StreamingPipelineSpec(
        stages=(StreamingStageSpec("stage", frozenset({"input"}), frozenset({"output"})),),
        edges=(StreamingEdgeSpec("input", "stage"),),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(spec, {"stage": actor})
    try:
        orchestrator.create_session("first", final_sequence_id=0)
        orchestrator.create_session("second", final_sequence_id=0)
        orchestrator.push_input("first", 0, "input", "first")
        orchestrator.push_input("second", 0, "input", "second")
        assert actor.future is not None

        actor.state = StreamingActorState.FAILED
        actor.future.set_exception(RuntimeError("injected worker failure"))

        assert orchestrator.status("first") == StreamingSessionStatus.FAILED
        assert orchestrator.status("second") == StreamingSessionStatus.FAILED
        assert str(orchestrator.error("first")) == "injected worker failure"
        assert isinstance(orchestrator.error("second"), StreamingActorFailedError)
        assert orchestrator.wait_until_idle("first")
        assert orchestrator.wait_until_idle("second")

        orchestrator.close_session("first")
        orchestrator.close_session("second")
        assert set(actor.cleanup_sessions) == {"first", "second"}
    finally:
        orchestrator.close()


def test_result_diagnostics_classify_duplicate_stale_and_orphaned_callbacks() -> None:
    actor = _ManualActor()
    spec = StreamingPipelineSpec(
        stages=(StreamingStageSpec("stage", frozenset({"input"}), frozenset({"output"})),),
        edges=(StreamingEdgeSpec("input", "stage"),),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(spec, {"stage": actor})
    try:
        epoch = orchestrator.create_session("session", final_sequence_id=0)
        orchestrator.push_input("session", 0, "input", "value")
        key = actor.invocations[0].key
        actor.futures[0].set_result({"output": "value"})

        completed: Future[dict[str, object]] = Future()
        completed.set_result({"output": "ignored"})
        orchestrator._complete("session", key, completed)
        stale_key = StreamingTaskKey("session", epoch + 1, 0, "stage", "stale")
        orchestrator._complete("session", stale_key, completed)
        orphan_key = StreamingTaskKey("missing", epoch, 0, "stage", "orphan")
        orchestrator._complete("missing", orphan_key, completed)

        diagnostics = orchestrator.diagnostics()
        assert diagnostics.duplicate_results == 1
        assert diagnostics.stale_epoch_results == 1
        assert diagnostics.orphaned_results == 1
    finally:
        orchestrator.close()


def test_cleanup_failure_attempts_every_stage_and_can_be_retried() -> None:
    cleanup_calls: list[str] = []
    decode_attempts = 0

    def close_decode(_, __) -> None:
        nonlocal decode_attempts
        decode_attempts += 1
        cleanup_calls.append("decode")
        if decode_attempts == 1:
            raise RuntimeError("injected cleanup failure")

    def fail_decode(_: StreamingStageInvocation) -> dict[str, object]:
        raise RuntimeError("original stage failure")

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("encode", frozenset({"input"}), frozenset({"latent"})),
            StreamingStageSpec("decode", frozenset({"latent"}), frozenset({"output"})),
        ),
        edges=(
            StreamingEdgeSpec("input", "encode"),
            StreamingEdgeSpec("latent", "decode", source_stage="encode"),
        ),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "encode": LocalStageActor(
                lambda _: {"latent": 1},
                name="cleanup-encode",
                session_closer=lambda _, __: cleanup_calls.append("encode"),
            ),
            "decode": LocalStageActor(
                fail_decode,
                name="cleanup-decode",
                session_closer=close_decode,
            ),
        },
    )
    try:
        orchestrator.create_session("session", final_sequence_id=0)
        orchestrator.push_input("session", 0, "input", 1)
        assert orchestrator.wait_until_idle("session")
        original_error = orchestrator.error("session")
        assert str(original_error) == "original stage failure"

        assert cleanup_calls == ["decode", "encode"]
        assert str(orchestrator.cleanup_error("session")) == "injected cleanup failure"
        assert orchestrator.diagnostics().cleanup_failures == 1
        assert orchestrator.error("session") is original_error

        orchestrator.close_session("session")
        assert cleanup_calls == ["decode", "encode", "decode"]
    finally:
        orchestrator.close()


def test_shutdown_drains_active_sessions_before_closing_actors() -> None:
    started = threading.Event()
    release = threading.Event()
    cleanup_calls: list[StreamingSessionCloseReason] = []
    close_errors: list[BaseException] = []

    def process(_: StreamingStageInvocation) -> dict[str, object]:
        started.set()
        release.wait(timeout=1)
        return {"output": "ignored"}

    actor = LocalStageActor(
        process,
        name="shutdown-active",
        session_closer=lambda _, reason: cleanup_calls.append(reason),
    )
    spec = StreamingPipelineSpec(
        stages=(StreamingStageSpec("stage", frozenset({"input"}), frozenset({"output"})),),
        edges=(StreamingEdgeSpec("input", "stage"),),
        output_artifacts=frozenset({"output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(spec, {"stage": actor})

    def close() -> None:
        try:
            orchestrator.close(timeout=1)
        except BaseException as exc:
            close_errors.append(exc)

    close_thread = threading.Thread(target=close)
    orchestrator.create_session("session", final_sequence_id=0)
    orchestrator.push_input("session", 0, "input", "value")
    assert started.wait(timeout=1)
    close_thread.start()
    assert close_thread.is_alive()

    release.set()
    close_thread.join(timeout=1)
    assert not close_thread.is_alive()
    assert close_errors == []
    assert cleanup_calls == [StreamingSessionCloseReason.SHUTDOWN]
    assert actor.health().state == StreamingActorState.CLOSED


@pytest.mark.parametrize("failing_stage", ["encode", "denoise", "decode"])
def test_fault_at_each_stage_drains_tasks_and_releases_every_stage(failing_stage: str) -> None:
    cleanup_calls: list[str] = []

    def handler(stage_id: str, output_artifact: str):
        def process(invocation: StreamingStageInvocation) -> dict[str, object]:
            if stage_id == failing_stage:
                raise RuntimeError(f"{stage_id} injected failure")
            return {output_artifact: next(iter(invocation.inputs.values()))}

        return process

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("encode", frozenset({"input"}), frozenset({"condition"})),
            StreamingStageSpec("denoise", frozenset({"condition"}), frozenset({"latent"})),
            StreamingStageSpec("decode", frozenset({"latent"}), frozenset({"output"})),
        ),
        edges=(
            StreamingEdgeSpec("input", "encode"),
            StreamingEdgeSpec("condition", "denoise", source_stage="encode"),
            StreamingEdgeSpec("latent", "decode", source_stage="denoise"),
        ),
        output_artifacts=frozenset({"output"}),
    )
    actors = {
        stage_id: LocalStageActor(
            handler(stage_id, output_artifact),
            name=f"fault-{stage_id}",
            session_closer=lambda _, __, closed_stage=stage_id: cleanup_calls.append(closed_stage),
        )
        for stage_id, output_artifact in (
            ("encode", "condition"),
            ("denoise", "latent"),
            ("decode", "output"),
        )
    }
    orchestrator = StreamingPipelineOrchestrator(spec, actors)
    try:
        orchestrator.create_session("session", final_sequence_id=0)
        orchestrator.push_input("session", 0, "input", "value")
        assert orchestrator.wait_until_idle("session")
        assert orchestrator.status("session") == StreamingSessionStatus.FAILED
        assert str(orchestrator.error("session")) == f"{failing_stage} injected failure"
        assert orchestrator.artifact_stats("session").slot_count == 0
        assert all(orchestrator.actor_health(stage).pending_invocations == 0 for stage in actors)
        assert cleanup_calls == ["decode", "denoise", "encode"]

        orchestrator.close_session("session")
    finally:
        orchestrator.close()


def test_cleanup_detects_and_reports_an_injected_artifact_slot_leak() -> None:
    started = threading.Event()
    release = threading.Event()
    cancellation_errors: list[BaseException] = []

    def slow(_: StreamingStageInvocation) -> dict[str, object]:
        started.set()
        release.wait(timeout=1)
        return {"slow_output": "ignored"}

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("slow", frozenset({"slow_input"}), frozenset({"slow_output"})),
            StreamingStageSpec(
                "waiting",
                frozenset({"waiting_input", "missing_input"}),
                frozenset({"waiting_output"}),
            ),
        ),
        edges=(
            StreamingEdgeSpec("slow_input", "slow"),
            StreamingEdgeSpec("waiting_input", "waiting"),
            StreamingEdgeSpec("missing_input", "waiting"),
        ),
        output_artifacts=frozenset({"slow_output", "waiting_output"}),
    )
    orchestrator = StreamingPipelineOrchestrator(
        spec,
        {
            "slow": LocalStageActor(slow, name="leak-slow"),
            "waiting": LocalStageActor(lambda _: {"waiting_output": "unused"}, name="leak-waiting"),
        },
    )

    def cancel() -> None:
        try:
            orchestrator.cancel_session("session", timeout=1)
        except BaseException as exc:
            cancellation_errors.append(exc)

    cancel_thread = threading.Thread(target=cancel)
    try:
        orchestrator.create_session("session", final_sequence_id=0)
        orchestrator.push_input("session", 0, "waiting_input", "retained")
        with orchestrator._lock:
            leaked_slot = orchestrator._sessions["session"].artifacts[(0, "waiting_input")]
        orchestrator.push_input("session", 0, "slow_input", "value")
        assert started.wait(timeout=1)

        cancel_thread.start()
        deadline = time.monotonic() + 1
        while orchestrator.status("session") != StreamingSessionStatus.CANCELLED:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        with orchestrator._lock:
            orchestrator._sessions["session"].artifacts[(0, "waiting_input")] = leaked_slot

        release.set()
        cancel_thread.join(timeout=1)
        assert len(cancellation_errors) == 1
        assert isinstance(cancellation_errors[0], RuntimeError)
        assert orchestrator.diagnostics().slot_leaks == 1
        assert orchestrator.artifact_stats("session").slot_count == 0

        orchestrator.cancel_session("session")
    finally:
        release.set()
        cancel_thread.join(timeout=1)
        orchestrator.close()


def test_cancellation_drains_in_flight_work_at_every_stage() -> None:
    started = {stage: threading.Event() for stage in ("encode", "denoise", "decode")}
    release = {stage: threading.Event() for stage in ("encode", "denoise", "decode")}
    cancellation_errors: list[BaseException] = []

    def handler(stage_id: str, blocked_sequence: int, output_artifact: str):
        def process(invocation: StreamingStageInvocation) -> dict[str, object]:
            if invocation.key.sequence_id == blocked_sequence:
                started[stage_id].set()
                release[stage_id].wait(timeout=2)
            return {output_artifact: next(iter(invocation.inputs.values()))}

        return process

    spec = StreamingPipelineSpec(
        stages=(
            StreamingStageSpec("encode", frozenset({"input"}), frozenset({"condition"})),
            StreamingStageSpec("denoise", frozenset({"condition"}), frozenset({"latent"})),
            StreamingStageSpec("decode", frozenset({"latent"}), frozenset({"output"})),
        ),
        edges=(
            StreamingEdgeSpec("input", "encode", capacity_per_session=3),
            StreamingEdgeSpec("condition", "denoise", source_stage="encode", capacity_per_session=3),
            StreamingEdgeSpec("latent", "decode", source_stage="denoise", capacity_per_session=3),
        ),
        output_artifacts=frozenset({"output"}),
        output_capacity_per_session=3,
    )
    actors = {
        "encode": LocalStageActor(handler("encode", 2, "condition"), name="cancel-all-encode"),
        "denoise": LocalStageActor(handler("denoise", 1, "latent"), name="cancel-all-denoise"),
        "decode": LocalStageActor(handler("decode", 0, "output"), name="cancel-all-decode"),
    }
    orchestrator = StreamingPipelineOrchestrator(spec, actors)

    def cancel() -> None:
        try:
            orchestrator.cancel_session("session", timeout=2)
        except BaseException as exc:
            cancellation_errors.append(exc)

    cancel_thread = threading.Thread(target=cancel)
    try:
        orchestrator.create_session("session", final_sequence_id=2)
        for sequence_id in range(3):
            orchestrator.push_input("session", sequence_id, "input", sequence_id)
        assert all(event.wait(timeout=1) for event in started.values())

        cancel_thread.start()
        deadline = time.monotonic() + 1
        while orchestrator.status("session") != StreamingSessionStatus.CANCELLED:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert cancel_thread.is_alive()

        for event in release.values():
            event.set()
        cancel_thread.join(timeout=2)
        assert not cancel_thread.is_alive()
        assert cancellation_errors == []
        assert orchestrator.artifact_stats("session").slot_count == 0
        assert orchestrator.poll_outputs("session") == []
        assert all(orchestrator.actor_health(stage).pending_invocations == 0 for stage in actors)
    finally:
        for event in release.values():
            event.set()
        cancel_thread.join(timeout=2)
        orchestrator.close()
