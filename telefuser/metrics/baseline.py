"""Performance baseline storage and regression checks.

The benchmark runners collect raw measurements, while this module owns the
stable, framework-independent representation used for comparison.  It has no
GPU or benchmark-runner dependencies, so baseline files can be generated and
checked in CPU CI as well as in the AIPerf environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BASELINE_SCHEMA_VERSION = 1
WORKLOAD_KINDS = frozenset({"t2i", "t2v", "vla", "realtime_stream"})


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metric values must be finite, got {value!r}")
    return number


def _number(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        return _finite(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True)
class DistributionSummary:
    """Descriptive statistics for one measured quantity."""

    count: int
    minimum: float
    maximum: float
    mean: float
    p50: float
    p95: float
    p99: float

    @classmethod
    def from_values(cls, values: Iterable[float | int]) -> DistributionSummary | None:
        numbers = [_finite(value) for value in values]
        values_float = [value for value in numbers if value is not None]
        if not values_float:
            return None
        return cls(
            count=len(values_float),
            minimum=min(values_float),
            maximum=max(values_float),
            mean=sum(values_float) / len(values_float),
            p50=_percentile(values_float, 50) or 0.0,
            p95=_percentile(values_float, 95) or 0.0,
            p99=_percentile(values_float, 99) or 0.0,
        )


@dataclass(frozen=True)
class OutputConsistency:
    """Consistency facts for generated output versus a fixed reference."""

    compared: int
    exact_matches: int
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    min_cosine_similarity: float | None = None

    @property
    def exact_match_rate(self) -> float | None:
        if not self.compared:
            return None
        return self.exact_matches / self.compared

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> OutputConsistency:
        compared = int(value.get("compared", value.get("count", 0)))
        exact_matches = int(value.get("exact_matches", value.get("exact", 0)))
        if compared < 0 or exact_matches < 0 or exact_matches > compared:
            raise ValueError("output consistency counts are invalid")
        return cls(
            compared=compared,
            exact_matches=exact_matches,
            max_abs_error=_number(value.get("max_abs_error"), name="max_abs_error"),
            mean_abs_error=_number(value.get("mean_abs_error"), name="mean_abs_error"),
            min_cosine_similarity=_number(
                value.get("min_cosine_similarity", value.get("cosine_similarity")),
                name="min_cosine_similarity",
            ),
        )


def compare_numeric_outputs(
    reference: Sequence[float | int],
    candidate: Sequence[float | int],
) -> OutputConsistency:
    """Compare two flattened numeric outputs without requiring NumPy or Torch."""

    if len(reference) != len(candidate):
        raise ValueError("reference and candidate outputs must have the same length")
    if not reference:
        return OutputConsistency(compared=0, exact_matches=0)
    reference_values = [_finite(value) for value in reference]
    candidate_values = [_finite(value) for value in candidate]
    if any(value is None for value in reference_values + candidate_values):
        raise ValueError("outputs must contain finite numeric values")
    reference_float = [value for value in reference_values if value is not None]
    candidate_float = [value for value in candidate_values if value is not None]
    errors = [abs(actual - expected) for expected, actual in zip(reference_float, candidate_float)]
    reference_norm = math.sqrt(sum(value * value for value in reference_float))
    candidate_norm = math.sqrt(sum(value * value for value in candidate_float))
    if reference_norm == 0 and candidate_norm == 0:
        cosine = 1.0
    elif reference_norm == 0 or candidate_norm == 0:
        cosine = 0.0
    else:
        cosine = sum(expected * actual for expected, actual in zip(reference_float, candidate_float)) / (
            reference_norm * candidate_norm
        )
    return OutputConsistency(
        compared=1,
        exact_matches=int(reference_float == candidate_float),
        max_abs_error=max(errors),
        mean_abs_error=sum(errors) / len(errors),
        min_cosine_similarity=cosine,
    )


def compare_output_hashes(reference: str, candidate: str) -> OutputConsistency:
    """Compare stable output digests, retaining the same consistency contract."""

    if not reference or not candidate:
        raise ValueError("output hashes must not be empty")
    exact = int(reference == candidate)
    return OutputConsistency(
        compared=1,
        exact_matches=exact,
        max_abs_error=0.0 if exact else 1.0,
        mean_abs_error=0.0 if exact else 1.0,
        min_cosine_similarity=1.0 if exact else 0.0,
    )


@dataclass(frozen=True)
class BenchmarkRecord:
    """One measured request or session exported by a benchmark adapter."""

    workload_id: str
    kind: str
    first_frame_latency_ms: float | None = None
    chunk_latencies_ms: tuple[float, ...] = ()
    steady_fps: float | None = None
    gpu_peak_memory_mb: float | None = None
    throughput_per_second: float | None = None
    output_consistency: OutputConsistency | None = None
    success: bool = True
    warmup: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise ValueError("workload_id must not be empty")
        if self.kind not in WORKLOAD_KINDS:
            raise ValueError(f"kind must be one of {sorted(WORKLOAD_KINDS)}, got {self.kind!r}")
        object.__setattr__(
            self,
            "first_frame_latency_ms",
            _finite(self.first_frame_latency_ms),
        )
        object.__setattr__(self, "steady_fps", _finite(self.steady_fps))
        object.__setattr__(self, "gpu_peak_memory_mb", _finite(self.gpu_peak_memory_mb))
        object.__setattr__(self, "throughput_per_second", _finite(self.throughput_per_second))
        object.__setattr__(
            self,
            "chunk_latencies_ms",
            tuple(_finite(value) for value in self.chunk_latencies_ms if value is not None),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        workload_id: str | None = None,
        kind: str | None = None,
    ) -> BenchmarkRecord:
        """Read a normalized AIPerf record, accepting common legacy aliases."""

        record_workload = workload_id or value.get("workload_id") or value.get("workload")
        record_kind = kind or value.get("kind") or value.get("workload_kind")
        if not record_workload or not record_kind:
            raise ValueError("each record requires workload_id and kind")

        first_frame = value.get("first_frame_latency_ms")
        if first_frame is None and value.get("first_frame_latency_s") is not None:
            first_frame = float(value["first_frame_latency_s"]) * 1000

        steady_fps = value.get("steady_fps")
        if steady_fps is None:
            steady_fps = value.get("stream_fps", value.get("target_compute_fps"))

        memory = value.get("gpu_peak_memory_mb")
        if memory is None:
            memory = value.get("peak_memory_mb", value.get("vla_peak_memory"))
        if memory is None and value.get("peak_memory_mib") is not None:
            memory = float(value["peak_memory_mib"]) * 1.048576

        throughput = value.get("throughput_per_second")
        if throughput is None:
            throughput = value.get(
                "throughput_rps",
                value.get("requests_per_second", value.get("request_throughput", value.get("throughput"))),
            )

        chunks = value.get("chunk_latencies_ms", value.get("chunk_compute_latencies_ms"))
        if chunks is None:
            chunks = _extract_chunk_latencies(value.get("chunk_measurements", ()))
        consistency = value.get("output_consistency")
        parsed_consistency = (
            consistency
            if isinstance(consistency, OutputConsistency)
            else OutputConsistency.from_mapping(consistency)
            if isinstance(consistency, Mapping)
            else None
        )
        return cls(
            workload_id=str(record_workload),
            kind=str(record_kind),
            first_frame_latency_ms=_number(first_frame, name="first_frame_latency_ms"),
            chunk_latencies_ms=tuple(chunks or ()),
            steady_fps=_number(steady_fps, name="steady_fps"),
            gpu_peak_memory_mb=_number(memory, name="gpu_peak_memory_mb"),
            throughput_per_second=_number(throughput, name="throughput_per_second"),
            output_consistency=parsed_consistency,
            success=bool(value.get("success", value.get("status", "ok") in {"ok", "success"})),
            warmup=bool(value.get("warmup", False)),
            metadata=dict(value.get("metadata", {})),
        )


def _extract_chunk_latencies(measurements: Any) -> tuple[float, ...]:
    values: list[float] = []
    if not isinstance(measurements, (list, tuple)):
        return ()
    for measurement in measurements:
        if isinstance(measurement, (int, float)):
            values.append(float(measurement))
            continue
        if not isinstance(measurement, Mapping):
            continue
        if measurement.get("warmup", measurement.get("is_warmup", False)):
            continue
        seconds = next(
            (
                measurement.get(name)
                for name in (
                    "compute_time_s",
                    "chunk_compute_s",
                    "compute_seconds",
                    "duration_s",
                    "latency_s",
                )
                if measurement.get(name) is not None
            ),
            None,
        )
        milliseconds = measurement.get("latency_ms", measurement.get("compute_time_ms"))
        if milliseconds is not None:
            values.append(float(milliseconds))
        elif seconds is not None:
            values.append(float(seconds) * 1000)
    return tuple(values)


@dataclass(frozen=True)
class WorkloadSummary:
    workload_id: str
    kind: str
    measured_records: int
    successful_records: int
    first_frame_latency_ms: DistributionSummary | None
    chunk_latency_ms: DistributionSummary | None
    steady_fps: DistributionSummary | None
    gpu_peak_memory_mb: DistributionSummary | None
    throughput_per_second: DistributionSummary | None
    output_consistency: OutputConsistency | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.measured_records <= 0:
            raise ValueError("measured_records must be positive")
        if not 0 <= self.successful_records <= self.measured_records:
            raise ValueError("successful_records must be between zero and measured_records")


@dataclass(frozen=True)
class BaselineFile:
    schema_version: int
    generated_at: str
    project_commit: str | None
    environment: Mapping[str, Any]
    workloads: Mapping[str, WorkloadSummary]

    @classmethod
    def from_records(
        cls,
        records: Iterable[BenchmarkRecord],
        *,
        project_commit: str | None = None,
        environment: Mapping[str, Any] | None = None,
    ) -> BaselineFile:
        grouped: dict[str, list[BenchmarkRecord]] = {}
        for record in records:
            if not record.warmup:
                grouped.setdefault(record.workload_id, []).append(record)
        if not grouped:
            raise ValueError("at least one measured record is required")
        workloads = {workload_id: summarize_records(items) for workload_id, items in grouped.items()}
        return cls(
            schema_version=BASELINE_SCHEMA_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            project_commit=project_commit,
            environment=dict(environment or {}),
            workloads=workloads,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> BaselineFile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
            raise ValueError(f"unsupported baseline schema: {payload.get('schema_version')!r}")
        workloads = {
            workload_id: _workload_summary_from_dict(summary)
            for workload_id, summary in payload.get("workloads", {}).items()
        }
        return cls(
            schema_version=int(payload["schema_version"]),
            generated_at=str(payload["generated_at"]),
            project_commit=payload.get("project_commit"),
            environment=dict(payload.get("environment", {})),
            workloads=workloads,
        )


def _distribution_from_dict(value: Mapping[str, Any] | None) -> DistributionSummary | None:
    if value is None:
        return None
    return DistributionSummary(
        count=int(value["count"]),
        minimum=float(value["minimum"]),
        maximum=float(value["maximum"]),
        mean=float(value["mean"]),
        p50=float(value["p50"]),
        p95=float(value["p95"]),
        p99=float(value["p99"]),
    )


def _workload_summary_from_dict(value: Mapping[str, Any]) -> WorkloadSummary:
    consistency = value.get("output_consistency")
    return WorkloadSummary(
        workload_id=str(value["workload_id"]),
        kind=str(value["kind"]),
        measured_records=int(value["measured_records"]),
        successful_records=int(value["successful_records"]),
        first_frame_latency_ms=_distribution_from_dict(value.get("first_frame_latency_ms")),
        chunk_latency_ms=_distribution_from_dict(value.get("chunk_latency_ms")),
        steady_fps=_distribution_from_dict(value.get("steady_fps")),
        gpu_peak_memory_mb=_distribution_from_dict(value.get("gpu_peak_memory_mb")),
        throughput_per_second=_distribution_from_dict(value.get("throughput_per_second")),
        output_consistency=OutputConsistency.from_mapping(consistency) if consistency else None,
        metadata=dict(value.get("metadata", {})),
    )


def summarize_records(records: Sequence[BenchmarkRecord]) -> WorkloadSummary:
    if not records:
        raise ValueError("at least one measured record is required")
    workload_id = records[0].workload_id
    kind = records[0].kind
    if any(record.workload_id != workload_id or record.kind != kind for record in records):
        raise ValueError("all records in one workload summary must share workload_id and kind")
    consistency_records = [record.output_consistency for record in records if record.output_consistency]
    consistency = None
    if consistency_records:
        consistency = OutputConsistency(
            compared=sum(item.compared for item in consistency_records),
            exact_matches=sum(item.exact_matches for item in consistency_records),
            max_abs_error=max(
                (item.max_abs_error for item in consistency_records if item.max_abs_error is not None),
                default=None,
            ),
            mean_abs_error=(
                sum(item.mean_abs_error for item in consistency_records if item.mean_abs_error is not None)
                / len([item for item in consistency_records if item.mean_abs_error is not None])
                if any(item.mean_abs_error is not None for item in consistency_records)
                else None
            ),
            min_cosine_similarity=min(
                (item.min_cosine_similarity for item in consistency_records if item.min_cosine_similarity is not None),
                default=None,
            ),
        )
    return WorkloadSummary(
        workload_id=workload_id,
        kind=kind,
        measured_records=len(records),
        successful_records=sum(record.success for record in records),
        first_frame_latency_ms=DistributionSummary.from_values(
            record.first_frame_latency_ms for record in records if record.first_frame_latency_ms is not None
        ),
        chunk_latency_ms=DistributionSummary.from_values(
            latency for record in records for latency in record.chunk_latencies_ms
        ),
        steady_fps=DistributionSummary.from_values(
            record.steady_fps for record in records if record.steady_fps is not None
        ),
        gpu_peak_memory_mb=DistributionSummary.from_values(
            record.gpu_peak_memory_mb for record in records if record.gpu_peak_memory_mb is not None
        ),
        throughput_per_second=DistributionSummary.from_values(
            record.throughput_per_second for record in records if record.throughput_per_second is not None
        ),
        output_consistency=consistency,
        metadata=dict(records[0].metadata),
    )


@dataclass(frozen=True)
class RegressionRule:
    relative_tolerance: float = 0.05
    absolute_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.relative_tolerance < 0 or self.absolute_tolerance < 0:
            raise ValueError("regression tolerances must be non-negative")


@dataclass(frozen=True)
class RegressionIssue:
    workload_id: str
    metric: str
    baseline: float | None
    candidate: float | None
    message: str


@dataclass(frozen=True)
class RegressionReport:
    passed: bool
    issues: tuple[RegressionIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "issues": [asdict(issue) for issue in self.issues]}


_DISTRIBUTIONS = (
    ("first_frame_latency_ms.p95", "first_frame_latency_ms", "p95", "higher"),
    ("chunk_latency_ms.p50", "chunk_latency_ms", "p50", "higher"),
    ("chunk_latency_ms.p95", "chunk_latency_ms", "p95", "higher"),
    ("chunk_latency_ms.p99", "chunk_latency_ms", "p99", "higher"),
    ("steady_fps.p50", "steady_fps", "p50", "lower"),
    ("gpu_peak_memory_mb.maximum", "gpu_peak_memory_mb", "maximum", "higher"),
    ("throughput_per_second.p50", "throughput_per_second", "p50", "lower"),
)


def compare_workload(
    baseline: WorkloadSummary,
    candidate: WorkloadSummary,
    *,
    rule: RegressionRule = RegressionRule(),
    consistency_cosine_floor: float = 0.9999,
    consistency_max_abs_error: float = 1e-4,
) -> RegressionReport:
    issues: list[RegressionIssue] = []
    if baseline.kind != candidate.kind:
        issues.append(
            RegressionIssue(baseline.workload_id, "kind", baseline.kind, candidate.kind, "workload kind changed")
        )
        return RegressionReport(False, tuple(issues))
    baseline_success_rate = baseline.successful_records / baseline.measured_records
    candidate_success_rate = candidate.successful_records / candidate.measured_records
    success_delta = max(rule.absolute_tolerance, baseline_success_rate * rule.relative_tolerance)
    if candidate_success_rate < baseline_success_rate - success_delta:
        issues.append(
            RegressionIssue(
                baseline.workload_id,
                "success_rate",
                baseline_success_rate,
                candidate_success_rate,
                f"candidate exceeds {rule.relative_tolerance:.1%} tolerance",
            )
        )
    for metric_name, summary_name, field_name, direction in _DISTRIBUTIONS:
        expected_summary = getattr(baseline, summary_name)
        actual_summary = getattr(candidate, summary_name)
        expected = getattr(expected_summary, field_name) if expected_summary else None
        actual = getattr(actual_summary, field_name) if actual_summary else None
        if expected is None:
            continue
        if actual is None:
            issues.append(
                RegressionIssue(baseline.workload_id, metric_name, expected, None, "candidate metric is missing")
            )
            continue
        delta = max(rule.absolute_tolerance, abs(expected) * rule.relative_tolerance)
        regressed = actual > expected + delta if direction == "higher" else actual < expected - delta
        if regressed:
            issues.append(
                RegressionIssue(
                    baseline.workload_id,
                    metric_name,
                    expected,
                    actual,
                    f"candidate exceeds {rule.relative_tolerance:.1%} tolerance",
                )
            )
    expected_consistency = baseline.output_consistency
    actual_consistency = candidate.output_consistency
    if expected_consistency and not actual_consistency:
        issues.append(
            RegressionIssue(baseline.workload_id, "output_consistency", 1.0, None, "candidate consistency is missing")
        )
    elif expected_consistency and actual_consistency:
        if actual_consistency.exact_match_rate is not None and actual_consistency.exact_match_rate < 1.0:
            issues.append(
                RegressionIssue(
                    baseline.workload_id,
                    "output_consistency.exact_match_rate",
                    1.0,
                    actual_consistency.exact_match_rate,
                    "output is not exactly reproducible",
                )
            )
        if (
            actual_consistency.min_cosine_similarity is not None
            and actual_consistency.min_cosine_similarity < consistency_cosine_floor
        ):
            issues.append(
                RegressionIssue(
                    baseline.workload_id,
                    "output_consistency.min_cosine_similarity",
                    consistency_cosine_floor,
                    actual_consistency.min_cosine_similarity,
                    "cosine similarity is below floor",
                )
            )
        if (
            actual_consistency.max_abs_error is not None
            and actual_consistency.max_abs_error > consistency_max_abs_error
        ):
            issues.append(
                RegressionIssue(
                    baseline.workload_id,
                    "output_consistency.max_abs_error",
                    consistency_max_abs_error,
                    actual_consistency.max_abs_error,
                    "absolute error is above tolerance",
                )
            )
    return RegressionReport(not issues, tuple(issues))


def compare_baselines(
    baseline: BaselineFile,
    candidate: BaselineFile,
    *,
    rule: RegressionRule = RegressionRule(),
) -> RegressionReport:
    issues: list[RegressionIssue] = []
    for workload_id, expected in baseline.workloads.items():
        actual = candidate.workloads.get(workload_id)
        if actual is None:
            issues.append(RegressionIssue(workload_id, "workload", 1.0, None, "candidate workload is missing"))
            continue
        issues.extend(compare_workload(expected, actual, rule=rule).issues)
    return RegressionReport(not issues, tuple(issues))


def _load_records(path: str | Path) -> list[BenchmarkRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_records = payload if isinstance(payload, list) else payload.get("records", [payload])
    return [BenchmarkRecord.from_mapping(item) for item in raw_records]


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build and compare TeleFuser performance baselines")
    subparsers = parser.add_subparsers(dest="command", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--input", required=True, type=Path, help="JSON array/object of normalized records")
    summarize.add_argument("--output", required=True, type=Path)
    summarize.add_argument("--project-commit")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--relative-tolerance", type=float, default=0.05)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "summarize":
        BaselineFile.from_records(_load_records(args.input), project_commit=args.project_commit).write(args.output)
        return 0
    report = compare_baselines(
        BaselineFile.read(args.baseline),
        BaselineFile.read(args.candidate),
        rule=RegressionRule(relative_tolerance=args.relative_tolerance),
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
