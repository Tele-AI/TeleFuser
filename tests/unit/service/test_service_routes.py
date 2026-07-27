from __future__ import annotations

import asyncio
import threading
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from telefuser.entrypoints.cli.main import main
from telefuser.service.api.api_server import ApiServer
from telefuser.service.api.routers.service import ServiceRoutes
from telefuser.service.core.config import ServerConfig
from telefuser.service.core.file_service import FileService
from telefuser.service.core.pipeline_service import PipelineService
from telefuser.service.core.stream_pipeline_service import StreamPipelineService
from telefuser.service.core.task_manager import TaskManager
from telefuser.service.core.task_manager import TaskStatus as CoreTaskStatus
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


def test_service_status_ignores_mock_pool_status_attribute() -> None:
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.inference_service = Mock()
    routes = ServiceRoutes(server)

    status = asyncio.run(routes.get_status())

    assert status["execution_mode"] == "serial_single_pipeline"
    assert "pool" not in status


def test_service_status_and_readiness_use_pipeline_pool_status_list() -> None:
    class PoolPipeline:
        is_running = True

        def pool_status(self) -> list[dict]:
            return [{"id": 0, "device_ids": ["0"], "status": "idle"}]

    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.inference_service = PoolPipeline()
    routes = ServiceRoutes(server)

    status = asyncio.run(routes.get_status())
    health = asyncio.run(routes.health_check())
    ready = asyncio.run(routes.readiness_check())

    assert status["execution_mode"] == "concurrent_pipeline_pool"
    assert status["pool"] == [{"id": 0, "device_ids": ["0"], "status": "idle"}]
    assert health["ready"] is True
    assert ready.status_code == 200


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


def test_pipeline_service_uses_injected_config() -> None:
    config = ServerConfig(
        security_level=SecurityLevel.BASIC,
        max_ppl_file_size=4096,
        allow_unsafe_pipelines=True,
        strict_validation=False,
    )

    service = PipelineService(config=config)

    assert service.security_level is SecurityLevel.BASIC
    assert service.security_validator.max_file_size == 4096
    assert service.validation_config.allow_unsafe_pipelines is True
    assert service.validation_config.strict_validation is False


def test_pipeline_service_task_timeout_uses_injected_config() -> None:
    class Status:
        value = "completed"

    class Result:
        status = Status()
        output_path = "output.mp4"
        message = "ok"
        raw = {"ok": True}

    class Runner:
        def __init__(self) -> None:
            self.timeout_s = None

        async def run(self, **kwargs: object) -> Result:
            self.timeout_s = kwargs["timeout_s"]
            return Result()

    config = ServerConfig(task_timeout=60, security_level=SecurityLevel.NONE)
    service = PipelineService(config=config)
    runner = Runner()
    service.is_running = True
    service.pipeline = object()
    service._runner = runner

    result = asyncio.run(service.run_task_with_stop_event({"task_id": "task-1"}, threading.Event()))

    assert runner.timeout_s == 60.0
    assert result["status"] == "completed"


def test_stream_pipeline_service_uses_injected_config() -> None:
    config = ServerConfig(
        security_level=SecurityLevel.NONE,
        max_ppl_file_size=8192,
        allow_unsafe_pipelines=True,
        strict_validation=False,
    )

    service = StreamPipelineService(config=config)

    assert service.security_level is SecurityLevel.NONE
    assert service.security_validator.max_file_size == 8192
    assert service.validation_config.allow_unsafe_pipelines is True
    assert service.validation_config.strict_validation is False


def test_stream_pipeline_service_passes_gpu_num_to_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeBidirectionalService:
        def start(self) -> None:
            captured["started"] = True

        def stop(self) -> None:
            pass

        def create_session(self, config: dict) -> str:
            return "session"

        def push_chunk(self, session_id: str, chunk: dict) -> None:
            pass

        async def pull_chunks(self, session_id: str):
            if False:
                yield {}

        def close_session(self, session_id: str) -> None:
            pass

    def get_service(gpu_num: int) -> FakeBidirectionalService:
        captured["gpu_num"] = gpu_num
        return FakeBidirectionalService()

    module = types.SimpleNamespace(get_service=get_service)
    monkeypatch.setattr(
        "telefuser.service.core.stream_pipeline_service.load_pipeline_module",
        lambda *args, **kwargs: (module, "test_stream_module"),
    )
    service = StreamPipelineService(config=ServerConfig(security_level=SecurityLevel.NONE))

    assert service.start_service("stream_pipeline.py", gpu_num=3, skip_validation=True)
    assert captured == {"gpu_num": 3, "started": True}
