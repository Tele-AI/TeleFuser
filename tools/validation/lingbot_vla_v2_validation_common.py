"""Shared numerical helpers for LingBot-VLA v2 validation tools."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

import numpy as np


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_latency(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize synchronized wall-clock latency samples in seconds."""
    if not values:
        raise ValueError("summary requires at least one value")
    total = sum(values)
    return {
        "count": len(values),
        "total_seconds": total,
        "mean_seconds": statistics.fmean(values),
        "stdev_seconds": statistics.pstdev(values),
        "min_seconds": min(values),
        "p50_seconds": percentile(values, 0.50),
        "p90_seconds": percentile(values, 0.90),
        "p95_seconds": percentile(values, 0.95),
        "p99_seconds": percentile(values, 0.99),
        "max_seconds": max(values),
        "throughput_requests_per_second": len(values) / total,
    }


def summarize_samples(values: Sequence[float]) -> dict[str, float | int] | None:
    """Summarize a possibly empty ordered sample."""
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def compare_windows(values: Sequence[float], fraction: float = 0.1) -> dict[str, float | int] | None:
    """Compare the first and last windows of an ordered measurement series."""
    if not values:
        return None
    window_count = max(1, math.ceil(len(values) * fraction))
    first_mean = statistics.fmean(values[:window_count])
    last_mean = statistics.fmean(values[-window_count:])
    delta = last_mean - first_mean
    return {
        "sample_count": len(values),
        "window_count": window_count,
        "first_mean": first_mean,
        "last_mean": last_mean,
        "delta": delta,
        "change_percent": delta / first_mean * 100.0 if first_mean else 0.0,
    }


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
