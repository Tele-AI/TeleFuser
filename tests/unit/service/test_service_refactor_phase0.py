from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from telefuser.entrypoints.cli.main import main
from telefuser.service.api.api_server import ApiServer
from telefuser.service.api.routers.service import ServiceRoutes
from telefuser.service.core.config import ServerConfig
from telefuser.service.core.file_service import FileService
from telefuser.service.core.task_manager import TaskManager, TaskStatus as CoreTaskStatus
from telefuser.service.security.security_validator import SecurityLevel
from telefuser.service_types import MediaType, TaskStatus


def test_task_status_uses_shared_service_enum() -> None:
    assert CoreTaskStatus is TaskStatus
    assert TaskStatus.STREAMING.value == "streaming"


def test_file_service_rejects_unsafe_output_paths(tmp_path: Path) -> None:
    files = FileService(tmp_path)

    with pytest.raises(ValueError, match="Absolute paths"):
        files.get_output_path("/tmp/escape.mp4", media_type=MediaType.VIDEO)

    with pytest.raises(ValueError, match="escapes"):
        files.get_output_path("../escape.mp4", media_type=MediaType.VIDEO)


def test_file_service_allows_image_and_video_download_roots(tmp_path: Path) -> None:
    files = FileService(tmp_path)
    image = files.output_image_dir / "result.png"
    video = files.output_video_dir / "result.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")

    assert files.resolve_output_file("result.png") == image
    assert files.resolve_output_file("result.mp4") == video


def test_api_server_initializes_file_service_with_configured_max_file_size(tmp_path: Path) -> None:
    config = ServerConfig(max_file_size=2 * 1024 * 1024)
    server = ApiServer(task_manager=TaskManager(), config=config, enable_openai_api=False)
    inference_service = Mock()

    server.initialize_services(tmp_path, inference_service)

    assert server.file_service is not None
    assert server.file_service.max_file_size == config.max_file_size


def test_health_and_readiness_are_separate() -> None:
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    routes = ServiceRoutes(server)

    health = asyncio.run(routes.health_check())
    ready = asyncio.run(routes.readiness_check())

    assert health["status"] == "healthy"
    assert health["ready"] is False
    assert ready.status_code == 503
    assert ready.body


def test_readiness_passes_when_pipeline_is_running() -> None:
    class RunningPipeline:
        is_running = True

    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.inference_service = RunningPipeline()
    routes = ServiceRoutes(server)

    health = asyncio.run(routes.health_check())
    ready = asyncio.run(routes.readiness_check())

    assert ready.status_code == 200
    assert health["ready"] is True
    assert health["pipeline_ready"] is True


def test_cli_serve_forwards_security_and_skip_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_run_server(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("telefuser.service.main.run_server", fake_run_server)

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "pipeline.py",
            "--skip-validation",
            "--security-level",
            "basic",
        ],
    )

    assert result.exit_code == 0
    assert captured["security_level"] == "basic"
    assert captured["skip_validation"] is True


def test_cli_stream_serve_does_not_force_skip_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_assert_safe(self, pipe_path):
        captured["validated"] = pipe_path

    def fake_run_stream_server(pipe_path, port, host, **kwargs):
        captured.update({"pipe_path": pipe_path, "port": port, "host": host, **kwargs})

    monkeypatch.setattr(
        "telefuser.service.security.security_validator.PipelineSecurityValidator.assert_safe",
        fake_assert_safe,
    )
    monkeypatch.setattr("telefuser.service.main.run_stream_server", fake_run_stream_server)

    result = CliRunner().invoke(
        main,
        [
            "stream-serve",
            "stream_pipeline.py",
            "--security-level",
            "none",
        ],
    )

    assert result.exit_code == 0
    assert captured["validated"] == "stream_pipeline.py"
    assert captured["security_level"] == "none"
    assert captured["skip_validation"] is False


def test_run_server_security_level_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    from telefuser.service import main as service_main

    class FakeContainer:
        def __init__(self, config):
            self.config = config

        def initialize_all(self, **kwargs):
            return True

        def get_api_app(self, enable_rate_limit=True):
            return Mock()

        async def cleanup(self):
            return None

    captured = {}

    def fake_create(config=None, cache_dir=None):
        captured["config"] = config
        return FakeContainer(config)

    monkeypatch.setattr(service_main.ServiceContainer, "create", fake_create)
    monkeypatch.setattr(service_main.uvicorn, "run", lambda *args, **kwargs: None)

    service_main.run_server(
        pipe_path="pipeline.py",
        task="t2v",
        port=9999,
        host="127.0.0.1",
        security_level="basic",
        skip_validation=True,
    )

    assert captured["config"].security_level is SecurityLevel.BASIC
