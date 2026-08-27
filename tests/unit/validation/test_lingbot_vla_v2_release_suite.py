from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

from tools.validation.run_lingbot_vla_v2_release_suite import (
    RELEASE_PROFILES,
    compare_actions,
    parse_profiles,
    render_service_module,
    tree_manifest,
)


def test_parse_profiles_preserves_declared_order() -> None:
    profiles = parse_profiles("bf16-graph,bnb-nf4")

    assert [profile.name for profile in profiles] == ["bf16-graph", "bnb-nf4"]


@pytest.mark.parametrize("value", ["", "unknown", "bf16-eager,bf16-eager"])
def test_parse_profiles_rejects_invalid_lists(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_profiles(value)


def test_render_service_module_freezes_profile_and_paths(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    processor_root = tmp_path / "processor"
    rendered = render_service_module(
        RELEASE_PROFILES["fused-fp8-graph"],
        model_root=model_root,
        qwen3vl_root=processor_root,
        max_image_bytes=1024,
        max_image_pixels=2048,
    )

    assert repr(str(model_root.resolve())) in rendered
    assert repr(str(processor_root.resolve())) in rendered
    assert "'quantization': 'fused-fp8-graph'" in rendered
    assert "'cuda_graph': True" in rendered
    assert "get_lingbot_vla_v2_pipeline(" in rendered
    assert "predict_lingbot_vla_v2_action(" in rendered
    assert "from examples" not in rendered
    assert "lingbot_vla_v2_release_fused_fp8_graph" in rendered
    compile(rendered, "service_profile.py", "exec")


def test_tree_manifest_hashes_paths_sizes_and_contents(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "nested" / "weights.bin").write_bytes(b"weights")

    metadata_only = tree_manifest(tmp_path, include_contents=False)
    full = tree_manifest(tmp_path, include_contents=True)

    assert metadata_only["file_count"] == 2
    assert metadata_only["total_bytes"] == 10
    assert metadata_only["hash_mode"] == "filename_and_size"
    assert full["hash_mode"] == "full_sha256"
    assert full["sha256"] != metadata_only["sha256"]
    assert len(full["sha256"]) == hashlib.sha256().digest_size * 2


def test_compare_actions_reports_exact_release_pass() -> None:
    actions = [[0.25] * 55 for _ in range(50)]

    report = compare_actions(
        actions,
        actions,
        min_cosine=0.995,
        max_relative_l2=0.10,
        max_absolute_error=0.5,
    )

    assert report["passed"] is True
    assert report["exact"] is True
    assert report["cosine_similarity"] == pytest.approx(1.0)


def test_compare_actions_fails_max_absolute_gate() -> None:
    reference = [[0.25] * 55 for _ in range(50)]
    candidate = [row.copy() for row in reference]
    candidate[0][0] = 1.0

    report = compare_actions(
        reference,
        candidate,
        min_cosine=0.0,
        max_relative_l2=1.0,
        max_absolute_error=0.5,
    )

    assert report["passed"] is False
    assert report["checks"]["max_absolute_error"] is False


def test_compare_actions_can_require_exact_quantized_replay() -> None:
    reference = [[0.25] * 55 for _ in range(50)]
    candidate = [row.copy() for row in reference]
    candidate[0][0] += 1e-6

    report = compare_actions(
        reference,
        candidate,
        min_cosine=0.995,
        max_relative_l2=0.10,
        max_absolute_error=0.5,
        require_exact=True,
    )

    assert report["checks"]["cosine"] is True
    assert report["checks"]["exact_replay"] is False
    assert report["passed"] is False


def test_compare_actions_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="50x55"):
        compare_actions(
            [[0.0]],
            [[0.0]],
            min_cosine=0.995,
            max_relative_l2=0.10,
            max_absolute_error=0.5,
        )
