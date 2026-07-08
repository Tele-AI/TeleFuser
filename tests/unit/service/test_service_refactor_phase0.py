from __future__ import annotations

import asyncio
import sys
import threading
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from telefuser.entrypoints.cli.main import main
from telefuser.service.api.api_server import ApiServer
from telefuser.service.api.routers.service import ServiceRoutes
from telefuser.service.api.routers.stream import StreamRoutes
from telefuser.service.core.config import ServerConfig
from telefuser.service.core.container import ServiceContainer
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


def test_stream_route_profile_exposes_only_stream_service_routes() -> None:
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=True, route_profile="stream")
    paths = {route.path for route in server.app.routes}

    assert "/v1/service/health" in paths
    assert "/v1/stream/sessions/{session_id}/status" in paths
    assert "/v1/tasks/create" not in paths
    assert "/v1/tasks/form" not in paths
    assert "/v1/files/download/{file_path:path}" not in paths
    assert "/v1/images/generations" not in paths
    assert "/v1/videos" not in paths


def test_request_response_route_profile_excludes_stream_routes() -> None:
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False, route_profile="request_response")
    paths = {route.path for route in server.app.routes}

    assert "/v1/service/health" in paths
    assert "/v1/tasks/create" in paths
    assert "/v1/files/download/{file_path:path}" in paths
    assert "/v1/stream/sessions/{session_id}/status" not in paths


def test_container_stream_app_uses_stream_route_profile() -> None:
    container = ServiceContainer.create(config=ServerConfig())
    container.stream_pipeline_service = Mock()
    container.stream_pipeline_service.is_running = True

    app = container.get_api_app()
    paths = {route.path for route in app.routes}

    assert "/v1/service/health" in paths
    assert "/v1/stream/sessions/{session_id}/status" in paths
    assert "/v1/tasks/create" not in paths
    assert "/v1/files/download/{file_path:path}" not in paths


def test_stream_session_close_logs_pipeline_close_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class RunningStreamService:
        is_running = True

        def close_session(self, session_id: str) -> None:
            raise RuntimeError("pipeline close failed")

    class WebRTCSessionManager:
        async def close_session(self, session_id: str, **kwargs: object) -> bool:
            return True

    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.stream_service = RunningStreamService()
    server._webrtc_routes = Mock(_session_manager=WebRTCSessionManager())
    warnings: list[str] = []
    monkeypatch.setattr("telefuser.service.api.routers.stream.logger.warning", warnings.append)

    result = asyncio.run(StreamRoutes(server).close_session("session-123"))

    assert result == {"session_id": "session-123", "status": "closed"}
    assert warnings == ["Failed to close pipeline stream session session-123: pipeline close failed"]


def test_stream_session_alias_closes_webrtc_without_pipeline_notify() -> None:
    class RunningStreamService:
        is_running = True

        def __init__(self) -> None:
            self.closed_sessions: list[str] = []

        def close_session(self, session_id: str) -> None:
            self.closed_sessions.append(session_id)

    class WebRTCSessionManager:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def close_session(self, session_id: str, *, reason: str, notify_pipeline: bool) -> bool:
            self.calls.append(
                {
                    "session_id": session_id,
                    "reason": reason,
                    "notify_pipeline": notify_pipeline,
                }
            )
            return True

    stream_service = RunningStreamService()
    webrtc_manager = WebRTCSessionManager()
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.stream_service = stream_service
    server._webrtc_routes = Mock(_session_manager=webrtc_manager)

    result = asyncio.run(StreamRoutes(server).close_session("session-123"))

    assert result == {"session_id": "session-123", "status": "closed"}
    assert stream_service.closed_sessions == ["session-123"]
    assert webrtc_manager.calls == [
        {
            "session_id": "session-123",
            "reason": "stream_session_delete",
            "notify_pipeline": False,
        }
    ]


def test_webrtc_delete_uses_session_manager_pipeline_owner() -> None:
    from telefuser.service.api.routers.webrtc import WebRTCRoutes

    class RunningStreamService:
        stream_mode = "bidirectional"

        def __init__(self) -> None:
            self.closed_sessions: list[str] = []

        def close_session(self, session_id: str) -> None:
            self.closed_sessions.append(session_id)

    class WebRTCSessionManager:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def close_session(self, session_id: str, *, reason: str) -> bool:
            self.calls.append({"session_id": session_id, "reason": reason})
            return True

    stream_service = RunningStreamService()
    webrtc_manager = WebRTCSessionManager()
    routes = WebRTCRoutes.__new__(WebRTCRoutes)
    routes.api = Mock(stream_service=stream_service)
    routes._session_manager = webrtc_manager

    result = asyncio.run(routes.close_session("session-123"))

    assert result == {"session_id": "session-123", "status": "closed"}
    assert stream_service.closed_sessions == []
    assert webrtc_manager.calls == [{"session_id": "session-123", "reason": "webrtc_session_delete"}]


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


def test_webrtc_routes_use_api_server_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeRTCIceServer:
        def __init__(self, urls: str, username: str | None = None, credential: str | None = None) -> None:
            self.urls = urls
            self.username = username
            self.credential = credential

    class FakeRTCConfiguration:
        def __init__(self, iceServers: list[FakeRTCIceServer]) -> None:
            self.iceServers = iceServers

    class FakeWebRTCSessionManager:
        def __init__(self, max_sessions: int, configuration: FakeRTCConfiguration) -> None:
            captured["max_sessions"] = max_sessions
            captured["configuration"] = configuration

    monkeypatch.setitem(
        sys.modules,
        "aiortc",
        types.SimpleNamespace(RTCConfiguration=FakeRTCConfiguration, RTCIceServer=FakeRTCIceServer),
    )
    monkeypatch.setitem(
        sys.modules,
        "telefuser.service.webrtc.session_manager",
        types.SimpleNamespace(WebRTCSessionManager=FakeWebRTCSessionManager),
    )

    config = ServerConfig(
        webrtc_max_sessions=3,
        stun_servers=["stun:local:3478"],
        turn_server="turn:local:3478",
        turn_username="user",
        turn_credential="secret",
    )
    server = ApiServer(
        task_manager=TaskManager(),
        enable_openai_api=False,
        config=config,
        route_profile="request_response",
    )

    from telefuser.service.api.routers.webrtc import WebRTCRoutes

    WebRTCRoutes(server)

    assert captured["max_sessions"] == 3
    ice_servers = captured["configuration"].iceServers
    assert [server.urls for server in ice_servers] == ["stun:local:3478", "turn:local:3478"]
    assert ice_servers[1].username == "user"
    assert ice_servers[1].credential == "secret"
