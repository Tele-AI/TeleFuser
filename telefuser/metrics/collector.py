"""
Core metric types for TeleFuser metrics module.

Provides Prometheus-compatible metric types: Counter, Gauge, Histogram, Summary.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricLabel:
    """Label for metric dimensions."""

    name: str
    value: str

    def __hash__(self) -> int:
        return hash((self.name, self.value))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetricLabel):
            return False
        return self.name == other.name and self.value == other.value

    def to_prometheus(self) -> str:
        """Export label in Prometheus format."""
        return f'{self.name}="{self.value}"'


Labels = dict[str, str] | list[MetricLabel] | None


def normalize_labels(labels: Labels) -> tuple[MetricLabel, ...]:
    """Normalize labels to a consistent format."""
    if labels is None:
        return ()

    if isinstance(labels, dict):
        return tuple(MetricLabel(k, v) for k, v in sorted(labels.items()))

    return tuple(sorted(labels, key=lambda x: x.name))


class Metric(ABC):
    """Base class for all metrics."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: Labels = None,
    ) -> None:
        self.name = name
        self.description = description
        self._labels = normalize_labels(labels)
        self._created_at = time.time()

    @property
    def labels(self) -> tuple[MetricLabel, ...]:
        """Get metric labels."""
        return self._labels

    def _labels_to_prometheus(self) -> str:
        """Convert labels to Prometheus format string."""
        if not self._labels:
            return ""
        return "{" + ", ".join(label.to_prometheus() for label in self._labels) + "}"

    @abstractmethod
    def to_prometheus(self) -> list[str]:
        """Export metric in Prometheus format.

        Returns:
            List of lines for Prometheus exposition format.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric to initial state."""
        pass


class Counter(Metric):
    """Monotonically increasing counter.

    Use for cumulative values like total requests, tasks completed, etc.
    Counter can only go up (and be reset).
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Labels = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        """Increment counter by amount.

        Args:
            amount: Amount to increment by. Must be non-negative.

        Raises:
            ValueError: If amount is negative.
        """
        if amount < 0:
            raise ValueError("Counter can only be incremented by non-negative values")
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement counter. Not allowed for Counter type.

        Raises:
            RuntimeError: Always, as counters cannot be decremented.
        """
        raise RuntimeError("Counter cannot be decremented")

    def set(self, value: float) -> None:
        """Set counter value. Only allowed for testing.

        Raises:
            RuntimeError: Always, as counters cannot be set directly.
        """
        raise RuntimeError("Counter cannot be set directly, use inc() instead")

    @property
    def value(self) -> float:
        """Get current counter value."""
        return self._value

    def to_prometheus(self) -> list[str]:
        """Export counter in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
            f"{self.name}{self._labels_to_prometheus()} {self._value}",
        ]
        return lines

    def reset(self) -> None:
        """Reset counter to zero."""
        self._value = 0.0


class Gauge(Metric):
    """Point-in-time value that can go up or down.

    Use for current state like queue size, memory usage, temperature, etc.
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Labels = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge by amount."""
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge by amount."""
        self._value -= amount

    def set(self, value: float) -> None:
        """Set gauge to specific value."""
        self._value = value

    def set_to_current_time(self) -> None:
        """Set gauge to current unix timestamp."""
        self._value = time.time()

    @property
    def value(self) -> float:
        """Get current gauge value."""
        return self._value

    def to_prometheus(self) -> list[str]:
        """Export gauge in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
            f"{self.name}{self._labels_to_prometheus()} {self._value}",
        ]
        return lines

    def reset(self) -> None:
        """Reset gauge to zero."""
        self._value = 0.0


@dataclass
class HistogramBucket:
    """A single bucket in a histogram."""

    upper_bound: float
    count: int = 0

    def __lt__(self, other: "HistogramBucket") -> bool:
        return self.upper_bound < other.upper_bound


# Default Prometheus histogram buckets
DEFAULT_BUCKETS: list[float] = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]


class Histogram(Metric):
    """Distribution of values with configurable buckets.

    Use for latency measurements, request sizes, etc.
    Automatically tracks count and sum of observations.
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Labels = None,
        buckets: list[float] | None = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._buckets: list[HistogramBucket] = []
        self._count: int = 0
        self._sum: float = 0.0

        # Initialize buckets
        bucket_values = buckets or DEFAULT_BUCKETS
        # Ensure +Inf bucket is included
        if bucket_values[-1] != float("inf"):
            bucket_values = list(bucket_values) + [float("inf")]

        for bound in sorted(bucket_values):
            self._buckets.append(HistogramBucket(upper_bound=bound))

    def observe(self, value: float) -> None:
        """Observe a value and add it to appropriate buckets.

        Args:
            value: The observed value.
        """
        self._count += 1
        self._sum += value

        for bucket in self._buckets:
            if value <= bucket.upper_bound:
                bucket.count += 1

    @property
    def count(self) -> int:
        """Get total number of observations."""
        return self._count

    @property
    def sum(self) -> float:
        """Get sum of all observations."""
        return self._sum

    @property
    def average(self) -> float:
        """Get average of observations."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count

    def to_prometheus(self) -> list[str]:
        """Export histogram in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} histogram",
        ]

        label_str = self._labels_to_prometheus()

        # Bucket counts are already cumulative because observe() increments every
        # upper bound containing the value.
        for bucket in self._buckets:
            le = "+Inf" if bucket.upper_bound == float("inf") else str(bucket.upper_bound)
            bucket_label = f'{{le="{le}"}}'
            if label_str:
                bucket_label = "{" + f'le="{le}", ' + label_str[1:]
            lines.append(f"{self.name}_bucket{bucket_label} {bucket.count}")

        # Sum and count
        lines.append(f"{self.name}_sum{label_str} {self._sum}")
        lines.append(f"{self.name}_count{label_str} {self._count}")

        return lines

    def reset(self) -> None:
        """Reset histogram to initial state."""
        for bucket in self._buckets:
            bucket.count = 0
        self._count = 0
        self._sum = 0.0


@dataclass
class Quantile:
    """A quantile for summary calculation."""

    quantile: float
    value: float = 0.0


# Default summary quantiles
DEFAULT_QUANTILES: list[float] = [0.5, 0.9, 0.95, 0.99]


class Summary(Metric):
    """Summary metric with configurable quantiles.

    Similar to histogram but calculates quantiles over a sliding time window.
    Note: This is a simplified implementation that tracks all observations.
    For production use with large data, consider using a streaming algorithm.
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Labels = None,
        quantiles: list[float] | None = None,
        max_age_seconds: float | None = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._quantiles = quantiles or DEFAULT_QUANTILES
        self._max_age = max_age_seconds
        self._observations: list[tuple[float, float]] = []  # (timestamp, value)
        self._count: int = 0
        self._sum: float = 0.0

    def observe(self, value: float) -> None:
        """Observe a value.

        Args:
            value: The observed value.
        """
        self._count += 1
        self._sum += value
        self._observations.append((time.time(), value))

        # Clean old observations if max_age is set
        if self._max_age is not None:
            cutoff = time.time() - self._max_age
            self._observations = [(t, v) for t, v in self._observations if t > cutoff]

    def _calculate_quantile(self, q: float) -> float:
        """Calculate the quantile value."""
        if not self._observations:
            return 0.0

        values = sorted(v for _, v in self._observations)
        n = len(values)
        idx = int(q * (n - 1))
        return values[idx]

    @property
    def count(self) -> int:
        """Get total number of observations."""
        return self._count

    @property
    def sum(self) -> float:
        """Get sum of all observations."""
        return self._sum

    def to_prometheus(self) -> list[str]:
        """Export summary in Prometheus format."""
        lines = [
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} summary",
        ]

        label_str = self._labels_to_prometheus()

        # Quantiles
        for q in self._quantiles:
            value = self._calculate_quantile(q)
            quantile_label = f'{{quantile="{q}"}}'
            if label_str:
                quantile_label = "{" + f'quantile="{q}", ' + label_str[1:]
            lines.append(f"{self.name}{quantile_label} {value}")

        # Sum and count
        lines.append(f"{self.name}_sum{label_str} {self._sum}")
        lines.append(f"{self.name}_count{label_str} {self._count}")

        return lines

    def reset(self) -> None:
        """Reset summary to initial state."""
        self._observations.clear()
        self._count = 0
        self._sum = 0.0


def create_metric(
    metric_type: str,
    name: str,
    description: str,
    labels: Labels = None,
    **kwargs: Any,
) -> Metric:
    """Factory function to create a metric by type.

    Args:
        metric_type: Type of metric ('counter', 'gauge', 'histogram', 'summary').
        name: Metric name.
        description: Metric description.
        labels: Optional labels.
        **kwargs: Additional arguments for specific metric types.

    Returns:
        A Metric instance.

    Raises:
        ValueError: If metric type is unknown.
    """
    metric_types = {
        "counter": Counter,
        "gauge": Gauge,
        "histogram": Histogram,
        "summary": Summary,
    }

    metric_cls = metric_types.get(metric_type.lower())
    if metric_cls is None:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return metric_cls(name=name, description=description, labels=labels, **kwargs)
