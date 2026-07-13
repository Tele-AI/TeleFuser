from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from telefuser.service.core.pipeline_runner import PipelineRunResult
from telefuser.service.core.pipeline_service import PipelineService
from telefuser.service_types import PipelineRunStatus


@pytest.mark.asyncio
async def test_pipeline_service_exports_measured_runtime_and_peak_memory() -> None:
    service = PipelineService.__new__(PipelineService)
    service.is_running = True
    service.pipeline = object()
    service._runner = SimpleNamespace(
        run=AsyncMock(
            return_value=PipelineRunResult(
                status=PipelineRunStatus.SUCCESS,
                output_path="result.mp4",
            )
        )
    )

    with (
        patch("telefuser.service.core.pipeline_service.current_platform.reset_peak_memory_stats") as reset_peak,
        patch(
            "telefuser.service.core.pipeline_service.current_platform.max_memory_allocated",
            return_value=2 * 1024 * 1024 * 1024,
        ),
    ):
        result = await service.run_task_with_stop_event(
            {"task_id": "task-1"},
            threading.Event(),
        )

    reset_peak.assert_called_once_with()
    assert result["inference_time_s"] >= 0
    assert result["peak_memory_mb"] == 2048.0


@pytest.mark.asyncio
async def test_pipeline_service_prefers_pipeline_reported_runtime_metrics() -> None:
    service = PipelineService.__new__(PipelineService)
    service.is_running = True
    service.pipeline = object()
    service._runner = SimpleNamespace(
        run=AsyncMock(
            return_value=PipelineRunResult(
                status=PipelineRunStatus.SUCCESS,
                output_path="result.mp4",
                raw={
                    "metrics": {
                        "inference_time_s": 12.5,
                        "peak_memory_gb": 3.0,
                    }
                },
            )
        )
    )

    with patch("telefuser.service.core.pipeline_service.current_platform.reset_peak_memory_stats"):
        result = await service.run_task_with_stop_event(
            {"task_id": "task-1"},
            threading.Event(),
        )

    assert result["inference_time_s"] == 12.5
    assert result["peak_memory_mb"] == 3072.0
