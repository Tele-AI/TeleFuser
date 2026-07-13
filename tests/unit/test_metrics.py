"""
Unit tests for TeleFuser metrics module.
"""

from __future__ import annotations

import time

import pytest

from telefuser.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricAlreadyRegisteredError,
    MetricNotFoundError,
    MetricRegistry,
    MetricsConfig,
    StageMetricContext,
    Summary,
    create_metric,
    get_metrics_registry,
    reset_global_registry,
)
from telefuser.metrics.collector import MetricLabel, normalize_labels


class TestMetricLabel:
    """Tests for MetricLabel class."""

    def test_label_creation(self) -> None:
        """Test creating a metric label."""
        label = MetricLabel("method", "GET")
        assert label.name == "method"
        assert label.value == "GET"

    def test_label_hash_and_equality(self) -> None:
        """Test label hashing and equality."""
        label1 = MetricLabel("method", "GET")
        label2 = MetricLabel("method", "GET")
        label3 = MetricLabel("method", "POST")

        assert label1 == label2
        assert label1 != label3
        assert hash(label1) == hash(label2)

    def test_label_to_prometheus(self) -> None:
        """Test Prometheus format output."""
        label = MetricLabel("status", "200")
        assert label.to_prometheus() == 'status="200"'


class TestNormalizeLabels:
    """Tests for label normalization."""

    def test_normalize_none(self) -> None:
        """Test normalizing None labels."""
        assert normalize_labels(None) == ()

    def test_normalize_dict(self) -> None:
        """Test normalizing dict labels."""
        labels = {"method": "GET", "status": "200"}
        result = normalize_labels(labels)
        assert len(result) == 2
        assert MetricLabel("method", "GET") in result
        assert MetricLabel("status", "200") in result

    def test_normalize_list(self) -> None:
        """Test normalizing list labels."""
        labels = [MetricLabel("method", "GET")]
        result = normalize_labels(labels)
        assert result == (MetricLabel("method", "GET"),)


class TestCounter:
    """Tests for Counter metric type."""

    def test_counter_creation(self) -> None:
        """Test creating a counter."""
        counter = Counter("requests_total", "Total requests")
        assert counter.name == "requests_total"
        assert counter.description == "Total requests"
        assert counter.value == 0.0

    def test_counter_increment(self) -> None:
        """Test incrementing a counter."""
        counter = Counter("requests_total", "Total requests")
        counter.inc()
        assert counter.value == 1.0

        counter.inc(5)
        assert counter.value == 6.0

    def test_counter_negative_increment_fails(self) -> None:
        """Test that negative increments are not allowed."""
        counter = Counter("requests_total", "Total requests")
        with pytest.raises(ValueError, match="non-negative"):
            counter.inc(-1)

    def test_counter_decrement_fails(self) -> None:
        """Test that decrement is not allowed for counters."""
        counter = Counter("requests_total", "Total requests")
        with pytest.raises(RuntimeError, match="cannot be decremented"):
            counter.dec()

    def test_counter_set_fails(self) -> None:
        """Test that set is not allowed for counters."""
        counter = Counter("requests_total", "Total requests")
        with pytest.raises(RuntimeError, match="cannot be set"):
            counter.set(10)

    def test_counter_reset(self) -> None:
        """Test resetting a counter."""
        counter = Counter("requests_total", "Total requests")
        counter.inc(10)
        assert counter.value == 10.0
        counter.reset()
        assert counter.value == 0.0

    def test_counter_prometheus_output(self) -> None:
        """Test Prometheus format output."""
        counter = Counter("requests_total", "Total requests")
        counter.inc(5)
        lines = counter.to_prometheus()

        assert "# HELP requests_total Total requests" in lines
        assert "# TYPE requests_total counter" in lines
        assert "requests_total 5.0" in lines


class TestGauge:
    """Tests for Gauge metric type."""

    def test_gauge_creation(self) -> None:
        """Test creating a gauge."""
        gauge = Gauge("memory_used", "Memory used in bytes")
        assert gauge.name == "memory_used"
        assert gauge.value == 0.0

    def test_gauge_set(self) -> None:
        """Test setting a gauge value."""
        gauge = Gauge("memory_used", "Memory used in bytes")
        gauge.set(100)
        assert gauge.value == 100.0

    def test_gauge_increment_and_decrement(self) -> None:
        """Test incrementing and decrementing a gauge."""
        gauge = Gauge("connections", "Active connections")
        gauge.inc()
        assert gauge.value == 1.0

        gauge.inc(5)
        assert gauge.value == 6.0

        gauge.dec(2)
        assert gauge.value == 4.0

    def test_gauge_set_to_current_time(self) -> None:
        """Test setting gauge to current time."""
        gauge = Gauge("last_update", "Last update timestamp")
        before = time.time()
        gauge.set_to_current_time()
        after = time.time()

        assert before <= gauge.value <= after

    def test_gauge_reset(self) -> None:
        """Test resetting a gauge."""
        gauge = Gauge("memory_used", "Memory used in bytes")
        gauge.set(100)
        gauge.reset()
        assert gauge.value == 0.0

    def test_gauge_prometheus_output(self) -> None:
        """Test Prometheus format output."""
        gauge = Gauge("memory_used", "Memory used in bytes")
        gauge.set(1024)
        lines = gauge.to_prometheus()

        assert "# HELP memory_used Memory used in bytes" in lines
        assert "# TYPE memory_used gauge" in lines
        assert "memory_used 1024" in lines


class TestHistogram:
    """Tests for Histogram metric type."""

    def test_histogram_creation(self) -> None:
        """Test creating a histogram."""
        hist = Histogram("request_duration", "Request duration in seconds")
        assert hist.name == "request_duration"
        assert hist.count == 0
        assert hist.sum == 0.0

    def test_histogram_observe(self) -> None:
        """Test observing values in a histogram."""
        hist = Histogram("request_duration", "Request duration in seconds")
        hist.observe(0.1)
        hist.observe(0.5)
        hist.observe(1.0)

        assert hist.count == 3
        assert hist.sum == 1.6
        assert abs(hist.average - 0.53333) < 0.01

    def test_histogram_custom_buckets(self) -> None:
        """Test histogram with custom buckets."""
        hist = Histogram(
            "request_duration",
            "Request duration",
            buckets=[0.1, 0.5, 1.0],
        )
        hist.observe(0.2)
        hist.observe(0.6)

        lines = hist.to_prometheus()
        assert any('le="0.1"' in line for line in lines)
        assert any('le="0.5"' in line for line in lines)

    def test_histogram_reset(self) -> None:
        """Test resetting a histogram."""
        hist = Histogram("request_duration", "Request duration")
        hist.observe(0.5)
        hist.reset()
        assert hist.count == 0
        assert hist.sum == 0.0

    def test_histogram_prometheus_output(self) -> None:
        """Test Prometheus format output."""
        hist = Histogram("request_duration", "Request duration", buckets=[0.1, 1.0])
        hist.observe(0.05)
        hist.observe(0.5)

        lines = hist.to_prometheus()

        assert "# HELP request_duration Request duration" in lines
        assert "# TYPE request_duration histogram" in lines
        assert "request_duration_bucket" in "".join(lines)
        assert "request_duration_sum" in "".join(lines)
        assert "request_duration_count" in "".join(lines)
        assert 'request_duration_bucket{le="0.1"} 1' in lines
        assert 'request_duration_bucket{le="1.0"} 2' in lines
        assert 'request_duration_bucket{le="+Inf"} 2' in lines


class TestSummary:
    """Tests for Summary metric type."""

    def test_summary_creation(self) -> None:
        """Test creating a summary."""
        summary = Summary("response_time", "Response time in seconds")
        assert summary.name == "response_time"
        assert summary.count == 0

    def test_summary_observe(self) -> None:
        """Test observing values in a summary."""
        summary = Summary("response_time", "Response time")
        for i in range(100):
            summary.observe(i / 100.0)

        assert summary.count == 100
        assert abs(summary.sum - 49.5) < 0.1

    def test_summary_quantiles(self) -> None:
        """Test quantile calculation."""
        summary = Summary("response_time", "Response time", quantiles=[0.5, 0.9, 0.99])
        for i in range(100):
            summary.observe(i / 100.0)

        # Median should be around 0.5
        lines = summary.to_prometheus()
        assert 'quantile="0.5"' in "".join(lines)

    def test_summary_reset(self) -> None:
        """Test resetting a summary."""
        summary = Summary("response_time", "Response time")
        summary.observe(0.5)
        summary.reset()
        assert summary.count == 0


class TestCreateMetric:
    """Tests for the create_metric factory function."""

    def test_create_counter(self) -> None:
        """Test creating a counter via factory."""
        metric = create_metric("counter", "requests", "Total requests")
        assert isinstance(metric, Counter)
        assert metric.name == "requests"

    def test_create_gauge(self) -> None:
        """Test creating a gauge via factory."""
        metric = create_metric("gauge", "memory", "Memory used")
        assert isinstance(metric, Gauge)

    def test_create_histogram(self) -> None:
        """Test creating a histogram via factory."""
        metric = create_metric("histogram", "duration", "Duration", buckets=[0.1, 1.0])
        assert isinstance(metric, Histogram)

    def test_create_summary(self) -> None:
        """Test creating a summary via factory."""
        metric = create_metric("summary", "latency", "Latency")
        assert isinstance(metric, Summary)

    def test_create_unknown_type_fails(self) -> None:
        """Test that unknown types raise an error."""
        with pytest.raises(ValueError, match="Unknown metric type"):
            create_metric("unknown", "test", "Test")


class TestMetricRegistry:
    """Tests for MetricRegistry."""

    def setup_method(self) -> None:
        """Reset the global registry before each test."""
        reset_global_registry()

    def test_registry_creation(self) -> None:
        """Test creating a registry."""
        registry = MetricRegistry(namespace="test")
        assert registry.namespace == "test"

    def test_register_counter(self) -> None:
        """Test registering a counter."""
        registry = MetricRegistry()
        counter = registry.counter("requests", "Total requests")

        assert counter.name == "telefuser_requests"
        assert "telefuser_requests" in registry.list_metrics()

    def test_register_gauge(self) -> None:
        """Test registering a gauge."""
        registry = MetricRegistry()
        gauge = registry.gauge("memory", "Memory used")

        assert gauge.name == "telefuser_memory"

    def test_register_histogram(self) -> None:
        """Test registering a histogram."""
        registry = MetricRegistry()
        hist = registry.histogram("duration", "Duration")

        assert hist.name == "telefuser_duration"

    def test_duplicate_registration_fails(self) -> None:
        """Test that duplicate registration raises error."""
        registry = MetricRegistry()
        registry.counter("requests", "Total requests")

        with pytest.raises(MetricAlreadyRegisteredError):
            registry.register(Counter("telefuser_requests", "Duplicate"))

    def test_get_metric(self) -> None:
        """Test getting a metric by name."""
        registry = MetricRegistry()
        registry.counter("requests", "Total requests")

        metric = registry.get_metric("requests")
        assert metric.name == "telefuser_requests"

    def test_get_nonexistent_metric_fails(self) -> None:
        """Test that getting nonexistent metric raises error."""
        registry = MetricRegistry()
        with pytest.raises(MetricNotFoundError):
            registry.get_metric("nonexistent")

    def test_register_stage(self) -> None:
        """Test registering a stage."""
        registry = MetricRegistry()
        context = registry.register_stage("text_encoding")

        assert isinstance(context, StageMetricContext)
        assert "text_encoding" in registry.list_stages()

    def test_unregister_stage(self) -> None:
        """Test unregistering a stage."""
        registry = MetricRegistry()
        registry.register_stage("text_encoding")
        registry.unregister_stage("text_encoding")

        assert "text_encoding" not in registry.list_stages()

    def test_prometheus_export(self) -> None:
        """Test Prometheus format export."""
        registry = MetricRegistry()
        counter = registry.counter("requests", "Total requests")
        counter.inc(5)

        output = registry.get_prometheus_format()
        assert "# HELP telefuser_requests Total requests" in output
        assert "telefuser_requests 5.0" in output

    def test_reset_all(self) -> None:
        """Test resetting all metrics."""
        registry = MetricRegistry()
        counter = registry.counter("requests", "Total requests")
        counter.inc(10)

        registry.reset_all()
        assert counter.value == 0.0

    def test_clear(self) -> None:
        """Test clearing all metrics."""
        registry = MetricRegistry()
        registry.counter("requests", "Total requests")
        registry.register_stage("text_encoding")

        registry.clear()
        assert len(registry.list_metrics()) == 0
        assert len(registry.list_stages()) == 0


class TestStageMetricContext:
    """Tests for StageMetricContext."""

    def setup_method(self) -> None:
        """Reset the global registry before each test."""
        reset_global_registry()

    def test_context_creation(self) -> None:
        """Test creating a stage metric context."""
        registry = MetricRegistry()
        context = registry.register_stage("my_stage")

        assert context.stage_name == "my_stage"
        assert context.duration is not None
        assert context.total is not None
        assert context.errors is not None

    def test_record_execution(self) -> None:
        """Test recording execution."""
        registry = MetricRegistry()
        context = registry.register_stage("my_stage")

        context.record_execution(0.5, success=True)
        context.record_execution(0.3, success=False)

        assert context.total.value == 2.0
        assert context.errors.value == 1.0
        assert context.duration.count == 2


class TestMetricsConfig:
    """Tests for MetricsConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = MetricsConfig()

        assert config.enabled is True
        assert config.enable_gpu_metrics is True
        assert config.enable_stage_metrics is True
        assert config.gpu_metrics_interval == 5.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = MetricsConfig(
            enabled=False,
            enable_gpu_metrics=False,
            histogram_buckets=[0.1, 0.5, 1.0],
        )

        assert config.enabled is False
        assert config.enable_gpu_metrics is False
        assert config.histogram_buckets == [0.1, 0.5, 1.0]

    def test_invalid_interval(self) -> None:
        """Test that invalid interval raises error."""
        with pytest.raises(ValueError):
            MetricsConfig(gpu_metrics_interval=0)

    def test_empty_buckets(self) -> None:
        """Test that empty buckets raises error."""
        with pytest.raises(ValueError):
            MetricsConfig(histogram_buckets=[])


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def setup_method(self) -> None:
        """Reset the global registry before each test."""
        reset_global_registry()

    def test_get_global_registry(self) -> None:
        """Test getting the global registry."""
        registry1 = get_metrics_registry()
        registry2 = get_metrics_registry()

        assert registry1 is registry2

    def test_reset_global_registry(self) -> None:
        """Test resetting the global registry."""
        registry = get_metrics_registry()
        registry.counter("test", "Test metric")

        reset_global_registry()
        registry2 = get_metrics_registry()

        # After reset, should be a different instance or empty
        assert len(registry2.list_metrics()) == 0
