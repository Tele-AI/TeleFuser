"""
Integration tests for OpenAI image routes.

Uses real FastAPI TestClient with mocked services.
Avoids testing implementation details like internal helper methods.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from telefuser.service.api.openai.image_routes import create_router
from telefuser.service.core.task_manager import TaskStatus


@pytest.fixture
def client(tmp_path):
    """Create test client with mocked services."""
    # Setup mock task manager
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

    # Create output file
    (tmp_path / "output.png").write_bytes(b"fake_image")

    # Setup mock API server
    server = MagicMock()
    server.task_manager = task_manager
    server.ensure_task_processor_running = AsyncMock()
    server.get_supported_tasks.return_value = ("t2i", "i2i")
    server.get_task_contract.side_effect = lambda task: {
        "t2i": {
            "required_inputs": [],
            "media_type": "image",
            "parameters": {"resolution": {"default": "768x768"}},
        },
        "i2i": {"required_inputs": ["first_image_path"], "media_type": "image"},
    }.get(task)
    server.file_service = MagicMock()
    server.file_service.output_image_dir = tmp_path
    server.file_service.input_image_dir = tmp_path / "input"
    server.file_service.input_image_dir.mkdir(exist_ok=True)

    # Create app with routes
    app = FastAPI()
    app.state.task_manager = task_manager
    app.include_router(create_router(server))

    with TestClient(app) as client:
        yield client


class TestImageGenerations:
    """Tests for POST /v1/images/generations."""

    def test_generate_image_success(self, client):
        """Successful image generation returns image data."""
        response = client.post(
            "/v1/images/generations",
            json={
                "prompt": "a cat",
                "size": "1024x1024",
                "response_format": "url",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) == 1
        assert "url" in data["data"][0]

    def test_generate_image_validation_error(self, client):
        """Invalid parameters return 422."""
        response = client.post(
            "/v1/images/generations",
            json={
                "size": "1024x1024",  # missing prompt
            },
        )

        assert response.status_code == 422

    def test_edit_image_prefers_reference_task(self, client):
        response = client.post(
            "/v1/images/edits",
            data={
                "prompt": "make it blue",
                "image_url": "https://example.com/image.png",
            },
        )

        assert response.status_code == 200
        task_request = client.app.state.task_manager.create_task.call_args.args[0]
        assert task_request.task == "i2i"

    def test_generate_image_uses_contract_default_resolution(self, client):
        response = client.post(
            "/v1/images/generations",
            json={
                "prompt": "a cat",
            },
        )

        assert response.status_code == 200
        task_request = client.app.state.task_manager.create_task.call_args.args[0]
        assert task_request.resolution == "768x768"


class TestImageContent:
    """Tests for GET /v1/images/{id}/content."""

    def test_download_image_success(self, client):
        """Download existing image."""
        response = client.get("/v1/images/task_123/content")

        assert response.status_code == 200
        assert response.content == b"fake_image"

    def test_download_image_not_found(self, tmp_path):
        """Download non-existent image returns 404."""
        # Create client with task manager returning None for this task
        task_manager = MagicMock()
        task_manager.get_task = MagicMock(return_value=None)

        server = MagicMock()
        server.task_manager = task_manager
        server.file_service = MagicMock()
        server.file_service.output_image_dir = tmp_path

        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(create_router(server))

        with TestClient(app) as client:
            response = client.get("/v1/images/nonexistent/content")
            assert response.status_code == 404
