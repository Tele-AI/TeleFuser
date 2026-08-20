"""Compare layered LingBot-VLA v2 capture artifacts.

The same comparator is used for local TeleFuser regression artifacts and for
future upstream parity artifacts. Captures stay file based so implementations
with incompatible Python dependencies never need to share a process.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_REPOSITORY = "https://github.com/Robbyant/lingbot-vla-v2"
UPSTREAM_COMMIT = "be27333c9b5f2663b0ec33f069dd7dfd67fa32b5"
ARTIFACT_SCHEMA_VERSION = 1
PREPROCESSING_KEYS = (
    "images",
    "img_masks",
    "image_grid_thw",
    "lang_tokens",
    "lang_masks",
    "state",
)
STEP_LAYERS = ("timestep", "x_t", "velocity")
FINAL_ACTION_KEYS = ("canonical_normalized_actions", "actions")
IDENTITY_METADATA_KEYS = (
    "checkpoint_manifest_sha256",
    "processor_manifest_sha256",
    "norm_stats_sha256",
    "input_sha256",
    "seed",
    "num_steps",
    "torch_dtype",
    "attention_backend",
    "moe_backend",
)
_STEP_KEY = re.compile(r"^(timestep|x_t|velocity)_step_([0-9]+)$")


@dataclass(frozen=True)
class ArrayParity:
    layer: str
    key: str
    shape: tuple[int, ...]
    expected_dtype: str
    actual_dtype: str
    max_abs: float
    mean_abs: float
    mismatch_count: int
    rtol: float
    atol: float
    passed: bool


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _metadata_path(artifact_path: Path, explicit_path: Path | None) -> Path:
    return explicit_path if explicit_path is not None else artifact_path.with_suffix(".json")


def _load_metadata(artifact_path: Path, explicit_path: Path | None = None) -> dict[str, Any]:
    path = _metadata_path(artifact_path, explicit_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing LingBot-VLA v2 artifact metadata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact metadata must be a JSON object: {path}")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema in {path}: expected {ARTIFACT_SCHEMA_VERSION}, "
            f"got {payload.get('schema_version')!r}"
        )
    missing = [key for key in IDENTITY_METADATA_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Artifact metadata {path} is missing identity fields: {missing}")
    return payload


def _step_keys(arrays: dict[str, np.ndarray]) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {layer: {} for layer in STEP_LAYERS}
    for key in arrays:
        match = _STEP_KEY.fullmatch(key)
        if match is not None:
            layer, step_text = match.groups()
            step = int(step_text)
            if step in result[layer]:
                raise ValueError(f"Duplicate {layer} capture for step {step}")
            result[layer][step] = key
    return result


def _validate_contract(arrays: dict[str, np.ndarray], metadata: dict[str, Any], *, side: str) -> None:
    required = (*PREPROCESSING_KEYS, "initial_noise")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{side} artifact is missing required arrays: {missing}")

    if not any(key in arrays for key in FINAL_ACTION_KEYS):
        raise ValueError(f"{side} artifact is missing a final action array: {FINAL_ACTION_KEYS}")

    array_metadata = metadata.get("arrays")
    if not isinstance(array_metadata, dict):
        raise ValueError(f"{side} metadata must contain an arrays contract")
    missing_contracts = sorted(set(arrays) - set(array_metadata))
    if missing_contracts:
        raise ValueError(f"{side} metadata is missing array contracts: {missing_contracts}")
    for key, array in arrays.items():
        contract = array_metadata[key]
        if not isinstance(contract, dict):
            raise ValueError(f"{side} metadata contract for {key} must be an object")
        expected_contract = {
            "shape": list(array.shape),
            "stored_dtype": str(array.dtype),
        }
        mismatches = {
            field: {"metadata": contract.get(field), "artifact": value}
            for field, value in expected_contract.items()
            if contract.get(field) != value
        }
        if "original_dtype" not in contract:
            mismatches["original_dtype"] = {"metadata": None, "artifact": "required"}
        if mismatches:
            raise ValueError(f"{side} metadata contract for {key} does not match the artifact: {mismatches}")

    num_steps = metadata["num_steps"]
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps <= 0:
        raise ValueError(f"{side} metadata num_steps must be a positive integer, got {num_steps!r}")
    expected_steps = set(range(num_steps))
    for layer, keys in _step_keys(arrays).items():
        if set(keys) != expected_steps:
            raise ValueError(f"{side} artifact {layer} steps must be {sorted(expected_steps)}, got {sorted(keys)}")


def _compare_array(
    layer: str,
    key: str,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
    expected_original_dtype: str,
    actual_original_dtype: str,
) -> ArrayParity:
    expected_dtype = expected_original_dtype
    actual_dtype = actual_original_dtype
    shape_matches = expected.shape == actual.shape
    dtype_matches = expected.dtype == actual.dtype and expected_original_dtype == actual_original_dtype
    finite = True
    if np.issubdtype(expected.dtype, np.number) and not np.isfinite(expected).all():
        finite = False
    if np.issubdtype(actual.dtype, np.number) and not np.isfinite(actual).all():
        finite = False

    if not shape_matches or not finite:
        max_abs = float("inf")
        mean_abs = float("inf")
        mismatch_count = max(expected.size, actual.size)
        values_match = False
    elif expected.dtype == np.bool_ or np.issubdtype(expected.dtype, np.integer):
        difference = np.abs(expected.astype(np.int64) - actual.astype(np.int64))
        mismatch_count = int(np.count_nonzero(difference))
        max_abs = float(difference.max()) if difference.size else 0.0
        mean_abs = float(difference.mean()) if difference.size else 0.0
        values_match = mismatch_count == 0
    else:
        difference = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        close = np.isclose(expected, actual, rtol=rtol, atol=atol, equal_nan=False)
        mismatch_count = int(np.count_nonzero(~close))
        max_abs = float(difference.max()) if difference.size else 0.0
        mean_abs = float(difference.mean()) if difference.size else 0.0
        values_match = mismatch_count == 0

    return ArrayParity(
        layer=layer,
        key=key,
        shape=tuple(actual.shape),
        expected_dtype=expected_dtype,
        actual_dtype=actual_dtype,
        max_abs=max_abs,
        mean_abs=mean_abs,
        mismatch_count=mismatch_count,
        rtol=rtol,
        atol=atol,
        passed=shape_matches and dtype_matches and finite and values_match,
    )


def _action_key(arrays: dict[str, np.ndarray]) -> str:
    for key in FINAL_ACTION_KEYS:
        if key in arrays:
            return key
    raise ValueError(f"No final action key found: {FINAL_ACTION_KEYS}")


def _original_dtype(metadata: dict[str, Any], key: str) -> str:
    return str(metadata["arrays"][key]["original_dtype"])


def compare_artifacts(
    reference: Path,
    candidate: Path,
    *,
    rtol: float,
    atol: float,
    reference_metadata: Path | None = None,
    candidate_metadata: Path | None = None,
    require_full_checkpoint_hash: bool = False,
) -> dict[str, object]:
    expected = _load_npz(reference)
    actual = _load_npz(candidate)
    expected_metadata = _load_metadata(reference, reference_metadata)
    actual_metadata = _load_metadata(candidate, candidate_metadata)
    _validate_contract(expected, expected_metadata, side="reference")
    _validate_contract(actual, actual_metadata, side="candidate")

    if require_full_checkpoint_hash:
        invalid_hash_modes = {
            side: metadata.get("checkpoint_hash_mode")
            for side, metadata in (("reference", expected_metadata), ("candidate", actual_metadata))
            if metadata.get("checkpoint_hash_mode") != "full_sha256"
        }
        if invalid_hash_modes:
            raise ValueError(f"Strict parity requires full_sha256 checkpoint manifests: {invalid_hash_modes}")

    metadata_mismatches = {
        key: {"reference": expected_metadata[key], "candidate": actual_metadata[key]}
        for key in IDENTITY_METADATA_KEYS
        if expected_metadata[key] != actual_metadata[key]
    }
    if metadata_mismatches:
        raise ValueError(f"Artifact identity metadata does not match: {metadata_mismatches}")

    results: list[ArrayParity] = []
    for key in PREPROCESSING_KEYS:
        results.append(
            _compare_array(
                "preprocessing",
                key,
                expected[key],
                actual[key],
                rtol=0.0,
                atol=0.0,
                expected_original_dtype=_original_dtype(expected_metadata, key),
                actual_original_dtype=_original_dtype(actual_metadata, key),
            )
        )
    results.append(
        _compare_array(
            "noise",
            "initial_noise",
            expected["initial_noise"],
            actual["initial_noise"],
            rtol=0.0,
            atol=0.0,
            expected_original_dtype=_original_dtype(expected_metadata, "initial_noise"),
            actual_original_dtype=_original_dtype(actual_metadata, "initial_noise"),
        )
    )

    expected_steps = _step_keys(expected)
    actual_steps = _step_keys(actual)
    for step in range(expected_metadata["num_steps"]):
        for layer in STEP_LAYERS:
            expected_key = expected_steps[layer][step]
            actual_key = actual_steps[layer][step]
            layer_rtol = 0.0 if layer == "timestep" else rtol
            layer_atol = 0.0 if layer == "timestep" else atol
            results.append(
                _compare_array(
                    layer,
                    expected_key,
                    expected[expected_key],
                    actual[actual_key],
                    rtol=layer_rtol,
                    atol=layer_atol,
                    expected_original_dtype=_original_dtype(expected_metadata, expected_key),
                    actual_original_dtype=_original_dtype(actual_metadata, actual_key),
                )
            )

    expected_action_key = _action_key(expected)
    actual_action_key = _action_key(actual)
    results.append(
        _compare_array(
            "action",
            expected_action_key,
            expected[expected_action_key],
            actual[actual_action_key],
            rtol=rtol,
            atol=atol,
            expected_original_dtype=_original_dtype(expected_metadata, expected_action_key),
            actual_original_dtype=_original_dtype(actual_metadata, actual_action_key),
        )
    )

    expected_compared_keys = {
        *PREPROCESSING_KEYS,
        "initial_noise",
        expected_action_key,
        *(key for layer in expected_steps.values() for key in layer.values()),
    }
    actual_compared_keys = {
        *PREPROCESSING_KEYS,
        "initial_noise",
        actual_action_key,
        *(key for layer in actual_steps.values() for key in layer.values()),
    }
    unexpected_reference = sorted(set(expected) - expected_compared_keys)
    unexpected_candidate = sorted(set(actual) - actual_compared_keys)
    first_failed_step = next(
        (
            int(match.group(2))
            for item in results
            if not item.passed and (match := _STEP_KEY.fullmatch(item.key)) is not None
        ),
        None,
    )

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "reference": str(reference),
        "candidate": str(candidate),
        "reference_kind": expected_metadata.get("artifact_kind"),
        "candidate_kind": actual_metadata.get("artifact_kind"),
        "passed": all(item.passed for item in results),
        "first_failed_step": first_failed_step,
        "unexpected_reference_keys": unexpected_reference,
        "unexpected_candidate_keys": unexpected_candidate,
        "results": [asdict(item) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-metadata", type=Path, default=None)
    parser.add_argument("--candidate-metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--profile", choices=("strict", "portable"), default="strict")
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument("--atol", type=float, default=None)
    args = parser.parse_args()

    default_tolerance = 0.0 if args.profile == "strict" else 1e-3
    rtol = default_tolerance if args.rtol is None else args.rtol
    atol = default_tolerance if args.atol is None else args.atol
    if rtol < 0 or atol < 0:
        parser.error("rtol and atol must be non-negative")

    report = compare_artifacts(
        args.reference,
        args.candidate,
        rtol=rtol,
        atol=atol,
        reference_metadata=args.reference_metadata,
        candidate_metadata=args.candidate_metadata,
        require_full_checkpoint_hash=args.profile == "strict",
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
