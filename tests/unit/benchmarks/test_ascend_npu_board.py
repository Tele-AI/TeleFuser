# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for the Ascend NPU benchmark sanitizer and board renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.ascend_npu.render_board import (
    assert_sanitized_text,
    load_records_and_slots,
    render_all,
    validate_hardware_fields,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "leaked_snippet",
    [
        "tested on node01 host",
        "jump via cpu02",
        "host pm-ab12 result",
        "ssh 10.0.12.34",
        "login runner@cluster.local",
        "mounted /data_shared/models",
        "path /home/tester/weights",
    ],
)
def test_sanitizer_blocks_server_and_cluster_leaks(leaked_snippet: str) -> None:
    with pytest.raises(ValueError, match="Sanitization violation"):
        assert_sanitized_text(leaked_snippet, context="unit_test")


def test_hardware_validator_requires_explicit_910b2_or_910b4_and_card_count() -> None:
    valid_b2 = {
        "chip_model": "Ascend 910B2 (64GB)",
        "card_count": 4,
        "parallel_spec": "CFG2 x Ulysses2",
        "cpu_arch": "x86_64",
    }
    validate_hardware_fields(valid_b2, source="ok_b2")
    validate_hardware_fields(dict(valid_b2, chip_model="Ascend 910B4 (32GB)", card_count=2), source="ok_b4")

    with pytest.raises(ValueError, match="910B2 or 910B4"):
        validate_hardware_fields(dict(valid_b2, chip_model="Ascend 910B"), source="vague_chip")

    with pytest.raises(ValueError, match="card_count"):
        validate_hardware_fields(dict(valid_b2, card_count=0), source="zero_cards")


def test_all_benchmark_files_pass_sanitization() -> None:
    bench_dir = REPO_ROOT / "benchmarks" / "ascend_npu"
    for path in sorted(bench_dir.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            assert_sanitized_text(path.read_text(encoding="utf-8"), context=str(path.relative_to(REPO_ROOT)))


def test_checked_in_records_and_generated_docs_are_in_sync(tmp_path: Path) -> None:
    readme_md, docs_en_md, docs_zh_md = render_all(REPO_ROOT)
    for expected_hw in (
        "1× Ascend 910B2 (64GB)",
        "2× Ascend 910B2 (64GB)",
        "4× Ascend 910B2 (64GB)",
        "8× Ascend 910B2 (64GB)",
    ):
        assert expected_hw in readme_md
        assert expected_hw in docs_en_md
        assert expected_hw in docs_zh_md

    assert (REPO_ROOT / "benchmarks/ascend_npu/README.md").read_text(encoding="utf-8") == readme_md
    assert (REPO_ROOT / "docs/en/benchmark_ascend_npu.md").read_text(encoding="utf-8") == docs_en_md
    assert (REPO_ROOT / "docs/zh/benchmark_ascend_npu.md").read_text(encoding="utf-8") == docs_zh_md

    # Verify slot deduplication when a new verified record arrives
    slots_path = REPO_ROOT / "benchmarks" / "ascend_npu" / "planned_slots.json"
    empty_dir = tmp_path / "empty_records"
    empty_dir.mkdir()
    _, all_slots = load_records_and_slots(empty_dir, slots_path)

    rec_dir = tmp_path / "records"
    rec_dir.mkdir()
    sample = {
        "record_id": "sample_h3_x86",
        "recorded_date": "2026-11-26",
        "model": "MiniMax-H3",
        "task": "T2VA",
        "chip_model": "Ascend 910B2 (64GB)",
        "card_count": 1,
        "parallel_spec": "Offload (Single-Card)",
        "cpu_arch": "x86_64",
        "cann_version": "9.3.0",
        "workload": {"resolution": "768p", "num_frames": 120, "steps": 50},
        "telefuser": {"version_or_commit": "7a53d87", "warm_median_s": 120.0, "denoise_s": 100.0},
        "vllm_omni": {"version_or_commit": "v0.34.0", "warm_median_s": 135.0, "denoise_s": 112.0},
        "quality_gate": {"nan_or_inf": False, "psnr_db": 32.0, "audio_cosine": 0.998},
    }
    (rec_dir / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    _, remaining = load_records_and_slots(rec_dir, slots_path)
    assert len(remaining) == len(all_slots) - 1
