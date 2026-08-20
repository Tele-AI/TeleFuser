from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.validation.capture_lingbot_vla_v2_telefuser import TensorCapture, trace_predict_velocity
from tools.validation.run_lingbot_vla_v2_parity import compare_artifacts


def _arrays() -> dict[str, np.ndarray]:
    arrays = {
        "images": np.zeros((1, 3, 2, 2), dtype=np.float32),
        "img_masks": np.ones((1, 3), dtype=np.bool_),
        "image_grid_thw": np.ones((1, 3, 3), dtype=np.int64),
        "lang_tokens": np.arange(4, dtype=np.int64).reshape(1, 4),
        "lang_masks": np.ones((1, 4), dtype=np.bool_),
        "state": np.zeros((1, 55), dtype=np.float32),
        "initial_noise": np.ones((1, 2, 55), dtype=np.float32),
        "canonical_normalized_actions": np.zeros((2, 55), dtype=np.float32),
    }
    for step in range(2):
        suffix = f"{step:02d}"
        arrays[f"timestep_step_{suffix}"] = np.asarray([1.0 - 0.5 * step], dtype=np.float32)
        arrays[f"x_t_step_{suffix}"] = np.full((1, 2, 55), step, dtype=np.float32)
        arrays[f"velocity_step_{suffix}"] = np.full((1, 2, 55), step + 0.25, dtype=np.float32)
    return arrays


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "telefuser_regression",
        "checkpoint_manifest_sha256": "checkpoint",
        "processor_manifest_sha256": "processor",
        "norm_stats_sha256": "norm-stats",
        "input_sha256": "input",
        "seed": 7,
        "num_steps": 2,
        "torch_dtype": "bfloat16",
        "attention_backend": "eager",
        "moe_backend": "deterministic_torch_reference",
    }


def _write_artifact(
    root: Path,
    name: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object] | None = None,
) -> Path:
    path = root / f"{name}.npz"
    np.savez(path, **arrays)
    payload = dict(_metadata() if metadata is None else metadata)
    payload["arrays"] = {
        key: {
            "shape": list(array.shape),
            "original_dtype": str(array.dtype),
            "stored_dtype": str(array.dtype),
        }
        for key, array in arrays.items()
    }
    path.with_suffix(".json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def test_compare_artifacts_accepts_a_complete_strict_replay(tmp_path: Path) -> None:
    reference = _write_artifact(tmp_path, "reference", _arrays())
    candidate = _write_artifact(tmp_path, "candidate", _arrays())

    report = compare_artifacts(reference, candidate, rtol=0.0, atol=0.0)

    assert report["passed"] is True
    assert report["first_failed_step"] is None
    assert len(report["results"]) == 14


def test_compare_artifacts_requires_every_preprocessing_array(tmp_path: Path) -> None:
    arrays = _arrays()
    del arrays["image_grid_thw"]
    reference = _write_artifact(tmp_path, "reference", arrays)
    candidate = _write_artifact(tmp_path, "candidate", _arrays())

    with pytest.raises(ValueError, match="image_grid_thw"):
        compare_artifacts(reference, candidate, rtol=0.0, atol=0.0)


def test_compare_artifacts_requires_contiguous_sampling_steps(tmp_path: Path) -> None:
    arrays = _arrays()
    del arrays["velocity_step_01"]
    reference = _write_artifact(tmp_path, "reference", arrays)
    candidate = _write_artifact(tmp_path, "candidate", _arrays())

    with pytest.raises(ValueError, match="velocity steps"):
        compare_artifacts(reference, candidate, rtol=0.0, atol=0.0)


def test_compare_artifacts_reports_the_first_failed_step(tmp_path: Path) -> None:
    candidate_arrays = _arrays()
    candidate_arrays["velocity_step_01"][0, 0, 0] += 0.5
    reference = _write_artifact(tmp_path, "reference", _arrays())
    candidate = _write_artifact(tmp_path, "candidate", candidate_arrays)

    report = compare_artifacts(reference, candidate, rtol=0.0, atol=0.0)

    assert report["passed"] is False
    assert report["first_failed_step"] == 1
    failed = [item for item in report["results"] if not item["passed"]]
    assert [item["key"] for item in failed] == ["velocity_step_01"]
    assert failed[0]["mismatch_count"] == 1


def test_compare_artifacts_rejects_non_finite_values(tmp_path: Path) -> None:
    candidate_arrays = _arrays()
    candidate_arrays["x_t_step_00"][0, 0, 0] = np.nan
    reference = _write_artifact(tmp_path, "reference", _arrays())
    candidate = _write_artifact(tmp_path, "candidate", candidate_arrays)

    report = compare_artifacts(reference, candidate, rtol=0.0, atol=0.0)

    assert report["passed"] is False
    assert report["first_failed_step"] == 0


def test_compare_artifacts_rejects_different_artifact_identity(tmp_path: Path) -> None:
    candidate_metadata = _metadata()
    candidate_metadata["input_sha256"] = "different"
    reference = _write_artifact(tmp_path, "reference", _arrays())
    candidate = _write_artifact(tmp_path, "candidate", _arrays(), candidate_metadata)

    with pytest.raises(ValueError, match="input_sha256"):
        compare_artifacts(reference, candidate, rtol=0.0, atol=0.0)


def test_compare_artifacts_rejects_weak_checkpoint_hash_for_strict_profile(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["checkpoint_hash_mode"] = "filename_and_size"
    reference = _write_artifact(tmp_path, "reference", _arrays(), metadata)
    candidate = _write_artifact(tmp_path, "candidate", _arrays(), metadata)

    with pytest.raises(ValueError, match="Strict parity requires full_sha256"):
        compare_artifacts(
            reference,
            candidate,
            rtol=0.0,
            atol=0.0,
            require_full_checkpoint_hash=True,
        )


def test_compare_artifacts_rejects_different_moe_backends(tmp_path: Path) -> None:
    candidate_metadata = _metadata()
    candidate_metadata["moe_backend"] = "upstream_triton"
    reference = _write_artifact(tmp_path, "reference", _arrays())
    candidate = _write_artifact(tmp_path, "candidate", _arrays(), candidate_metadata)

    with pytest.raises(ValueError, match="moe_backend"):
        compare_artifacts(reference, candidate, rtol=0.0, atol=0.0)


def test_velocity_trace_snapshots_inputs_and_restores_the_model_instance() -> None:
    class _FlowModel:
        _use_compile_predict_velocity = True

        def predict_velocity(self, state, prefix_masks, cache, x_t, timestep, **kwargs):
            del state, prefix_masks, cache, timestep, kwargs
            return torch.ones_like(x_t)

    flow_model = _FlowModel()
    capture = TensorCapture()
    x_t = torch.zeros(1, 2, 3)

    with trace_predict_velocity(flow_model, capture) as trace:
        velocity = flow_model.predict_velocity(None, None, None, x_t, torch.ones(1))
        x_t.add_(velocity)

    assert trace.step == 1
    assert np.array_equal(capture.arrays["initial_noise"], np.zeros((1, 2, 3), dtype=np.float32))
    assert np.array_equal(capture.arrays["x_t_step_00"], np.zeros((1, 2, 3), dtype=np.float32))
    assert np.array_equal(capture.arrays["velocity_step_00"], np.ones((1, 2, 3), dtype=np.float32))
    assert "predict_velocity" not in vars(flow_model)
    assert flow_model._use_compile_predict_velocity is True


def test_tensor_capture_records_original_bfloat16_dtype() -> None:
    capture = TensorCapture()

    capture.add("value", torch.ones(2, dtype=torch.bfloat16))

    assert capture.arrays["value"].dtype == np.float32
    assert capture.array_metadata["value"] == {
        "shape": [2],
        "original_dtype": "bfloat16",
        "stored_dtype": "float32",
    }
