"""Compare LingBot-VLA v2 BF16 and online-quantized action artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ACTION_KEY = "canonical_normalized_actions"
IDENTITY_KEYS = (
    "checkpoint_manifest_sha256",
    "processor_manifest_sha256",
    "norm_stats_sha256",
    "input_sha256",
    "seed",
    "num_steps",
    "attention_backend",
    "moe_backend",
)


def _load_action(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        if ACTION_KEY not in payload.files:
            raise ValueError(f"{path} does not contain {ACTION_KEY!r}")
        action = np.asarray(payload[ACTION_KEY], dtype=np.float64)
    if action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 2:
        raise ValueError(f"{path} action must have shape [H, D] or [1, H, D], got {action.shape}")
    return np.ascontiguousarray(action)


def _load_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"metadata must be a JSON object: {metadata_path}")
    return payload


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest()


def _cosine(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    if denominator == 0.0:
        return 1.0 if np.array_equal(reference_flat, candidate_flat) else 0.0
    return float(np.dot(reference_flat, candidate_flat) / denominator)


def action_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Return shape, finiteness, magnitude, L2, and cosine action facts."""
    if reference.shape != candidate.shape:
        raise ValueError(f"action shapes differ: reference={reference.shape}, candidate={candidate.shape}")
    reference_finite = bool(np.isfinite(reference).all())
    candidate_finite = bool(np.isfinite(candidate).all())
    if not reference_finite or not candidate_finite:
        return {
            "shape": list(candidate.shape),
            "reference_finite": reference_finite,
            "candidate_finite": candidate_finite,
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "relative_l2": math.inf,
            "cosine": 0.0,
            "min_step_cosine": 0.0,
            "mean_step_cosine": 0.0,
            "exact": False,
        }

    difference = candidate - reference
    reference_norm = float(np.linalg.norm(reference.reshape(-1)))
    step_cosines = [_cosine(expected, actual) for expected, actual in zip(reference, candidate, strict=True)]
    return {
        "shape": list(candidate.shape),
        "reference_finite": True,
        "candidate_finite": True,
        "max_abs": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / max(reference_norm, 1e-12)),
        "cosine": _cosine(reference, candidate),
        "min_step_cosine": min(step_cosines, default=1.0),
        "mean_step_cosine": float(np.mean(step_cosines)) if step_cosines else 1.0,
        "exact": bool(np.array_equal(reference, candidate)),
    }


def _validate_metadata_pair(reference: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict[str, Any]:
    if reference is None or candidate is None:
        return {"available": False}
    mismatches = {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in IDENTITY_KEYS
        if reference.get(key) != candidate.get(key)
    }
    if mismatches:
        raise ValueError(f"artifact identity metadata differs: {mismatches}")
    reference_quantization = reference.get("quantization", {"profile": "bf16", "enabled": False})
    candidate_quantization = candidate.get("quantization")
    if isinstance(reference_quantization, dict) and reference_quantization.get("profile") != "bf16":
        raise ValueError("reference artifact must use the BF16 profile")
    if not isinstance(candidate_quantization, dict) or candidate_quantization.get("profile") in (None, "bf16"):
        raise ValueError("candidate artifact must record a non-BF16 quantization profile")
    return {
        "available": True,
        "reference_quantization": reference_quantization,
        "candidate_quantization": candidate_quantization,
        "matched_identity": {key: reference.get(key) for key in IDENTITY_KEYS},
    }


def compare_quantized_actions(
    reference_path: Path,
    candidate_path: Path,
    *,
    candidate_replay_path: Path | None = None,
    min_cosine: float | None = None,
    max_relative_l2: float | None = None,
    max_abs: float | None = None,
    require_exact_replay: bool = False,
) -> dict[str, Any]:
    """Compare one quantized action against BF16 and an optional replay."""
    if min_cosine is not None and not -1.0 <= min_cosine <= 1.0:
        raise ValueError("min_cosine must be between -1 and 1")
    if max_relative_l2 is not None and max_relative_l2 < 0:
        raise ValueError("max_relative_l2 must be non-negative")
    if max_abs is not None and max_abs < 0:
        raise ValueError("max_abs must be non-negative")
    if require_exact_replay and candidate_replay_path is None:
        raise ValueError("require_exact_replay requires candidate_replay_path")

    reference = _load_action(reference_path)
    candidate = _load_action(candidate_path)
    metrics = action_error_metrics(reference, candidate)
    metadata = _validate_metadata_pair(_load_metadata(reference_path), _load_metadata(candidate_path))
    checks = {
        "shape_matches": reference.shape == candidate.shape,
        "finite": bool(metrics["reference_finite"] and metrics["candidate_finite"]),
    }
    if min_cosine is not None:
        checks["min_cosine"] = float(metrics["cosine"]) >= min_cosine
    if max_relative_l2 is not None:
        checks["max_relative_l2"] = float(metrics["relative_l2"]) <= max_relative_l2
    if max_abs is not None:
        checks["max_abs"] = float(metrics["max_abs"]) <= max_abs

    replay_report = None
    if candidate_replay_path is not None:
        replay = _load_action(candidate_replay_path)
        replay_metrics = action_error_metrics(candidate, replay)
        replay_report = {
            "path": str(candidate_replay_path.resolve()),
            "sha256_float64_le": _sha256(replay),
            "metrics_vs_candidate": replay_metrics,
        }
        if require_exact_replay:
            checks["exact_replay"] = bool(replay_metrics["exact"])

    thresholds_enabled = any(value is not None for value in (min_cosine, max_relative_l2, max_abs))
    return {
        "schema_version": 1,
        "comparison": "lingbot_vla_v2_bf16_vs_online_quantization",
        "passed": all(checks.values()),
        "mode": "thresholded" if thresholds_enabled or require_exact_replay else "report_only",
        "checks": checks,
        "thresholds": {
            "min_cosine": min_cosine,
            "max_relative_l2": max_relative_l2,
            "max_abs": max_abs,
            "require_exact_replay": require_exact_replay,
        },
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256_float64_le": _sha256(reference),
        },
        "candidate": {
            "path": str(candidate_path.resolve()),
            "sha256_float64_le": _sha256(candidate),
        },
        "metadata": metadata,
        "action_metrics": metrics,
        "candidate_replay": replay_report,
        "interpretation": (
            "This is a numerical quantization comparison against the BF16 TeleFuser baseline. "
            "It is not strict upstream parity and does not establish robot-control success."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--candidate-replay", type=Path)
    parser.add_argument("--min-cosine", type=float)
    parser.add_argument("--max-relative-l2", type=float)
    parser.add_argument("--max-abs", type=float)
    parser.add_argument("--require-exact-replay", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare_quantized_actions(
        args.reference,
        args.candidate,
        candidate_replay_path=args.candidate_replay,
        min_cosine=args.min_cosine,
        max_relative_l2=args.max_relative_l2,
        max_abs=args.max_abs,
        require_exact_replay=args.require_exact_replay,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "mode": report["mode"], "output": str(args.output)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
