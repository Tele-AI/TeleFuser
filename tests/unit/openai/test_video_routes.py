"""
Integration tests for OpenAI video routes.

Uses real FastAPI TestClient with mocked services.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException

from telefuser.service.api.openai.video_routes import VideoRoutes, create_router
from telefuser.service.api.task_application_service import TaskApplicationService
from telefuser.service.core.task_manager import TaskStatus

from ._asgi_test_client import ASGITestClient as TestClient


@pytest.fixture
def client(tmp_path):
    """Create test client with mocked services."""
    # Create mock message with proper string attributes
    mock_message = MagicMock()
    mock_message.prompt = "test prompt"
    mock_message.resolution = "720p"
    mock_message.target_video_length = 5
    mock_message.model = "wan-video"

    task_manager = MagicMock()
    task_manager.create_task = MagicMock(return_value="vid_123")
    task_manager.get_task_status = MagicMock(
        return_value={
            "task_id": "vid_123",
            "status": TaskStatus.COMPLETED.value,
            "output_path": str(tmp_path / "video.mp4"),
            "peak_memory_mb": 2048.0,
            "inference_time_s": 12.5,
        }
    )
    task_manager.get_task = MagicMock(
        return_value=MagicMock(
            task_id="vid_123",
            output_path=str(tmp_path / "video.mp4"),
            status=TaskStatus.COMPLETED,
            message=mock_message,
        )
    )
    task_manager.get_all_tasks = MagicMock(return_value={})

    (tmp_path / "video.mp4").write_bytes(b"fake_video")

    server = MagicMock()
    server.task_manager = task_manager
    server.ensure_task_processor_running = AsyncMock()
    server.get_supported_tasks.return_value = ("t2v", "i2v", "vc")
    server.get_task_contract.side_effect = lambda task: {
        "t2v": {
            "required_inputs": [],
            "media_type": "video",
            "parameters": {
                "target_video_length": {"default": 8},
                "resolution": {"default": "480p"},
            },
        },
        "i2v": {"required_inputs": ["first_image_path"], "media_type": "video"},
        "vc": {"required_inputs": ["ref_video_path"], "media_type": "video"},
    }.get(task)
    server.file_service = MagicMock()
    server.file_service.output_video_dir = tmp_path
    server.file_service.input_video_dir = tmp_path / "input"
    server.file_service.input_video_dir.mkdir(exist_ok=True)
    server.task_app_service = TaskApplicationService(server)

    app = FastAPI()
    app.state.task_manager = task_manager
    app.state.server = server
    app.include_router(create_router(server))

    with TestClient(app) as client:
        yield client


class TestVideoCreate:
    """Tests for POST /v1/videos."""

    def test_create_video_success(self, client):
        """Create video returns queued status."""
        response = client.post(
            "/v1/videos",
            json={
                "prompt": "a cat playing",
                "seconds": 5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "vid_123"
        assert data["status"] == "queued"

    def test_create_video_uses_contract_defaults_for_omitted_fields(self, client):
        response = client.post(
            "/v1/videos",
            json={
                "prompt": "a cat playing",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["seconds"] == "8"
        task_request = client.app.state.task_manager.create_task.call_args.args[0]
        assert task_request.target_video_length == 8
        assert task_request.resolution == "480p"

    def test_create_video_validation_error(self, client):
        """Missing prompt returns 422."""
        response = client.post("/v1/videos", json={"seconds": 5})

        assert response.status_code == 422

    def test_create_video_with_video_reference_prefers_video_conditioning(self, client):
        response = client.post(
            "/v1/videos",
            json={
                "prompt": "continue this clip",
                "input_reference": "/tmp/reference.mp4",
            },
        )

        assert response.status_code == 200
        task_request = client.app.state.task_manager.create_task.call_args.args[0]
        assert task_request.task == "vc"
        assert task_request.ref_video_path == "/tmp/reference.mp4"


class TestVideoRetrieve:
    """Tests for GET /v1/videos/{id}."""

    def test_get_video_success(self, client):
        """Get existing video status."""
        response = client.get("/v1/videos/vid_123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "vid_123"
        assert data["peak_memory_mb"] == 2048.0
        assert data["inference_time_s"] == 12.5

    def test_delete_completed_video_reports_completed_status(self, client):
        """Delete on a terminal task reports the real terminal status."""
        response = client.delete("/v1/videos/vid_123")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "vid_123"
        assert data["status"] == "completed"
        client.app.state.task_manager.cancel_task.assert_not_called()


class TestVideoContent:
    """Tests for GET /v1/videos/{id}/content."""

    def test_download_video_success(self, client):
        """Download completed video."""
        routes = VideoRoutes(client.app.state.server)
        response = asyncio.run(routes.get_video_content("vid_123"))

        assert response.path
        assert response.path.endswith("video.mp4")
