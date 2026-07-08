"""
Integration tests for OpenAI API with ApiServer.

Tests actual route registration without over-mocking.
"""

from unittest.mock import MagicMock

import pytest

from telefuser.service.api.api_server import ApiServer
from telefuser.service.core.task_manager import TaskStatus

from ._asgi_test_client import ASGITestClient as TestClient


@pytest.fixture
def api_server(tmp_path):
    """Create ApiServer with OpenAI routes enabled."""
    task_manager = MagicMock()
    task_manager.create_task = MagicMock(return_value="task_123")
    task_manager.get_task_status = MagicMock(
        return_value={
            "task_id": "task_123",
            "status": TaskStatus.COMPLETED.value,
            "output_path": str(tmp_path / "output.png"),
        }
    )
    task_manager.get_task = MagicMock(
        return_value=MagicMock(
            task_id="task_123",
            output_path=str(tmp_path / "output.png"),
            status=TaskStatus.COMPLETED,
        )
    )
    task_manager.get_all_tasks = MagicMock(return_value={})

    (tmp_path / "output.png").write_bytes(b"fake_image")

    server = ApiServer(
        enable_openai_api=True,
        enable_rate_limit=False,
        task_manager=task_manager,
    )
    server.file_service = MagicMock()
    server.file_service.output_image_dir = tmp_path
    server.file_service.output_video_dir = tmp_path
    server.file_service.input_image_dir = tmp_path / "input"
    server.file_service.input_image_dir.mkdir(exist_ok=True)

    return server


class TestRouteRegistration:
    """Tests that OpenAI routes are properly registered."""

    def test_openai_routes_exist(self, api_server):
        """OpenAI routes are in the OpenAPI schema."""
        openapi = api_server.app.openapi()
        paths = openapi.get("paths", {})

        assert any("/v1/images" in p for p in paths.keys())
        assert any("/v1/videos" in p for p in paths.keys())

    def test_native_routes_preserved(self, api_server):
        """Native TeleFuser routes still exist."""
        openapi = api_server.app.openapi()
        paths = list(openapi.get("paths", {}).keys())

        assert any("/v1/tasks" in p for p in paths)


class TestEndpointResponses:
    """Tests for actual endpoint responses."""

    def test_images_endpoint(self, api_server):
        """Image generation endpoint works."""
        with TestClient(api_server.app) as client:
            response = client.post(
                "/v1/images/generations",
                json={
                    "prompt": "a cat",
                    "size": "512x512",
                },
            )

            assert response.status_code == 200
            assert "data" in response.json()

    def test_videos_endpoint(self, api_server):
        """Video creation endpoint works."""
        with TestClient(api_server.app) as client:
            response = client.post(
                "/v1/videos",
                json={
                    "prompt": "a cat playing",
                    "seconds": 4,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
