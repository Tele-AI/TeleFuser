from __future__ import annotations

from collections import defaultdict

from telefuser.pipelines.lingbot_video import DEFAULT_NEGATIVE_PROMPT, DEFAULT_NEGATIVE_PROMPT_IMAGE
from tools.validation.benchmark_lingbot_video import _instrument_method, _resolve_negative_caption, _summary


def test_summary_reports_each_metric_count_and_aggregate_timings() -> None:
    summary = _summary({"base_total": [1.0, 3.0], "vae_decode": [0.5]})

    assert summary["base_total"] == {"count": 2, "total_seconds": 4.0, "mean_seconds": 2.0, "max_seconds": 3.0}
    assert summary["vae_decode"] == {"count": 1, "total_seconds": 0.5, "mean_seconds": 0.5, "max_seconds": 0.5}


def test_instrumented_method_records_into_the_active_benchmark_phase() -> None:
    class Owner:
        def run(self) -> str:
            return "ok"

    first_phase: dict[str, list[float]] = defaultdict(list)
    second_phase: dict[str, list[float]] = defaultdict(list)
    active_phase = first_phase
    owner = Owner()
    _instrument_method(lambda: active_phase, owner, "run", "operation")

    assert owner.run() == "ok"
    active_phase = second_phase
    assert owner.run() == "ok"

    assert len(first_phase["operation"]) == 1
    assert len(second_phase["operation"]) == 1


def test_benchmark_uses_source_negative_caption_when_not_overridden() -> None:
    assert _resolve_negative_caption(None, 1) == DEFAULT_NEGATIVE_PROMPT_IMAGE
    assert _resolve_negative_caption(None, 5) == DEFAULT_NEGATIVE_PROMPT
    assert _resolve_negative_caption("", 5) == ""
