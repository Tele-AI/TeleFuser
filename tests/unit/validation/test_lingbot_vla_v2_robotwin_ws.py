from __future__ import annotations

import argparse

import numpy as np
import pytest
from PIL import Image

from tools.validation import validate_lingbot_vla_v2_robotwin_ws as validator


def _metadata() -> dict:
    return {
        "protocol_version": validator.PROTOCOL_VERSION,
        "robot_profile": "robotwin",
        "action_type": validator.ACTION_TYPE,
        "action_horizon": 3,
        "action_dim": len(validator.ACTION_ORDER),
        "action_dtype": validator.ACTION_DTYPE,
        "action_order": list(validator.ACTION_ORDER),
        "policy_verified": False,
        "verification_status": "unverified_official_6b_base",
        "max_request_bytes": 16 * 1024 * 1024,
    }


def _response() -> dict:
    return {
        "action": np.arange(42, dtype=np.float32).reshape(3, 14),
        "seed": 7,
        "request_id": "request-1",
        "episode_id": "episode-1",
        "policy_verified": False,
        "verification_status": "unverified_official_6b_base",
        "server_timing": {
            "decode_ms": 0.1,
            "infer_ms": 2.0,
            "lock_wait_ms": 0.05,
            "pipeline_ms": 1.5,
            "action_mapping_ms": 0.2,
            "adapter_total_ms": 1.8,
        },
    }


def test_validate_metadata_accepts_contract_and_additive_fields() -> None:
    metadata = {**_metadata(), "future_field": "ignored"}

    summary = validator.validate_metadata(metadata)

    assert summary["action_shape"] == [3, 14]
    assert summary["action_type"] == "absolute_qpos"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_version", "2.0"),
        ("action_type", "delta_qpos"),
        ("action_dtype", "float64"),
        ("action_order", ["unknown"] * 14),
        ("action_horizon", 0),
    ],
)
def test_validate_metadata_rejects_contract_mismatch(field: str, value) -> None:
    metadata = {**_metadata(), field: value}

    with pytest.raises(validator.ValidationFailure, match=field):
        validator.validate_metadata(metadata)


def test_validate_action_response_returns_stable_float32_digest() -> None:
    summary = validator.validate_action_response(
        _response(),
        expected_horizon=3,
        request_id="request-1",
        episode_id="episode-1",
        seed=7,
    )

    assert summary["shape"] == [3, 14]
    assert summary["dtype"] == "float32"
    assert len(summary["sha256_float32_le"]) == 64
    assert summary["server_timing_ms"]["pipeline_ms"] == 1.5


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((2, 14), dtype=np.float32),
        np.zeros((3, 14), dtype=np.float64),
        np.full((3, 14), np.nan, dtype=np.float32),
    ],
)
def test_validate_action_response_rejects_invalid_action(actions: np.ndarray) -> None:
    response = {**_response(), "action": actions}

    with pytest.raises(validator.ValidationFailure, match="action"):
        validator.validate_action_response(
            response,
            expected_horizon=3,
            request_id="request-1",
            episode_id="episode-1",
            seed=7,
        )


def test_validate_reset_response_checks_trace_and_timings() -> None:
    summary = validator.validate_reset_response(
        {
            "action": None,
            "request_id": "reset-1",
            "episode_id": "episode-1",
            "server_timing": {"decode_ms": 0.1, "infer_ms": 0.2},
        },
        request_id="reset-1",
        episode_id="episode-1",
    )

    assert summary["server_timing_ms"] == {"decode_ms": 0.1, "infer_ms": 0.2}


def test_require_exact_replay_rejects_divergent_digests() -> None:
    records = [
        {"action": {"sha256_float32_le": "a"}},
        {"action": {"sha256_float32_le": "b"}},
    ]

    with pytest.raises(validator.ValidationFailure, match="replay diverged"):
        validator.require_exact_replay(records)


def test_parse_state_json_rejects_non_finite_state() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite numbers"):
        validator.parse_state_json("[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e999]")


def test_load_validation_image_bounds_longest_edge_without_upscaling(tmp_path) -> None:
    large_path = tmp_path / "large.png"
    small_path = tmp_path / "small.png"
    Image.new("RGB", (800, 400)).save(large_path)
    Image.new("RGB", (32, 16)).save(small_path)

    large, source_shape = validator.load_validation_image(large_path, max_image_edge=640)
    small, _ = validator.load_validation_image(small_path, max_image_edge=640)

    assert source_shape == [400, 800, 3]
    assert large.shape == (320, 640, 3)
    assert small.shape == (16, 32, 3)
