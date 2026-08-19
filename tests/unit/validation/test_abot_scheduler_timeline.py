from __future__ import annotations

import json
from pathlib import Path

from tools.validation import render_abot_dispatch_timeline as dispatch_timeline
from tools.validation import trace_abot_scheduler_timeline as timeline


def test_three_user_timeline_proves_phase_alignment_changes_real_scheduler_batching(tmp_path: Path) -> None:
    """The CPU backend replaces compute only; ABot service selects the batches."""
    staggered_config = timeline.TimelineConfig(
        chunks_per_session=3,
        fps=12,
        frames_per_chunk=3,
        stagger_offsets_ms=(0.0, 260.0, 520.0),
        output_timeout_seconds=5.0,
    )

    staggered = timeline.run_scenario("staggered", staggered_config)
    aligned_config = timeline.TimelineConfig(
        chunks_per_session=1,
        fps=12,
        frames_per_chunk=3,
        stagger_offsets_ms=(0.0, 260.0, 520.0),
        output_timeout_seconds=5.0,
    )
    aligned = timeline.run_scenario("aligned", aligned_config)

    assert staggered["summary"]["batch_size_histogram"] == {"1": 9}
    assert staggered["summary"]["classification"] == "time_sliced_singletons"
    assert staggered["summary"]["serialized_scheduler_thread"] is True
    assert all(batch["batch_size"] == 1 for batch in staggered["batches"])

    assert aligned["summary"]["batch_size_histogram"] == {"3": 1}
    assert aligned["summary"]["first_batch_size"] == 3
    assert aligned["summary"]["classification"] == "coalesced_microbatching"
    assert all(batch["session_ids"] == ["user-1", "user-2", "user-3"] for batch in aligned["batches"])

    output = tmp_path / "aligned"
    timeline._write_result(output, aligned)
    payload = json.loads((output / "timeline.json").read_text())
    assert payload["backend"]["kind"] == "cpu_fake_pipeline_with_production_abot_service_scheduler"
    for name in ("timeline.json", "events.csv", "batches.csv", "chunks.csv", "summary.csv", "timeline.png"):
        assert (output / name).is_file()
    assert (output / "timeline.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_dispatch_timeline_keeps_public_trace_session_generations_distinct() -> None:
    assert dispatch_timeline._short_user("ts-00079-g01") == "u79g01"
    assert dispatch_timeline._short_user("ts-00079-g02") == "u79g02"
