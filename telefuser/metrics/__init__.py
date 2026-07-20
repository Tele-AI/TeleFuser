"""
TeleFuser Metrics Module

Provides comprehensive metrics collection for service monitoring.

Features:
- Prometheus-compatible metric types (Counter, Gauge, Histogram, Summary)
- Stage-level metrics with automatic tracking
- Service-level metrics (tasks, queue, GPU)
- GPU metrics collection (NVIDIA via pynvml)
- Thread-safe registry

Example Usage:
    from telefuser.metrics import get_metrics_registry, enable_stage_metrics

    # Get the global registry
    registry = get_metrics_registry()

    # Create a counter
    counter = registry.counter("my_counter", "A simple counter")
    counter.inc()

    # Enable metrics for a stage
    stage = MyStage("my_stage", config)
    enable_stage_metrics(stage)

    # Export in Prometheus format
    print(registry.get_prometheus_format())
"""

from __future__ import annotations

from .collector import (
    Counter,
    Gauge,
    Histogram,
    Labels,
    Metric,
    MetricLabel,
    Summary,
    create_metric,
)
from .config import MetricsConfig, create_metrics_config, default_metrics_config
from .exporters import MetricsExporter, PrometheusExporter, get_exporter
from .registry import (
    MetricAlreadyRegisteredError,
    MetricNotFoundError,
    MetricRegistry,
    StageMetricContext,
    get_metrics_registry,
    reset_global_registry,
)
from .runtime import (
    RuntimeMeasurement,
    canonical_cuda_devices,
    collect_runtime_environment,
    finish_runtime_measurement,
    start_runtime_measurement,
    visible_cuda_devices,
)
from .service_metrics import (
    ServiceMetrics,
    get_service_metrics,
    reset_service_metrics,
)
from .stage_metrics import (
    StageMetricsManager,
    disable_stage_metrics,
    enable_stage_metrics,
    with_metrics,
    with_metrics_async,
)

__all__ = [
    # Metric types
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",
    "Metric",
    "MetricLabel",
    "Labels",
    "create_metric",
    # Registry
    "MetricRegistry",
    "StageMetricContext",
    "get_metrics_registry",
    "reset_global_registry",
    "MetricAlreadyRegisteredError",
    "MetricNotFoundError",
    # Stage metrics
    "StageMetricsManager",
    "enable_stage_metrics",
    "disable_stage_metrics",
    "with_metrics",
    "with_metrics_async",
    # Service metrics
    "ServiceMetrics",
    "get_service_metrics",
    "reset_service_metrics",
    # Runtime benchmark facts
    "RuntimeMeasurement",
    "canonical_cuda_devices",
    "visible_cuda_devices",
    "start_runtime_measurement",
    "finish_runtime_measurement",
    "collect_runtime_environment",
    # Configuration
    "MetricsConfig",
    "default_metrics_config",
    "create_metrics_config",
    # Exporters
    "MetricsExporter",
    "PrometheusExporter",
    "get_exporter",
]
