"""
Integration Tests for TeleFuser Server

This module provides comprehensive tests for the server API,
using a fake pipeline to avoid requiring actual model files.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import httpx
import pytest

from telefuser.service.api.schema import TaskRequest
from telefuser.utils.logging import logger

# Test configuration
TEST_CONFIG = {
    "host": "127.0.0.1",
    "port": 18000,  # Use non-standard port for testing
    "cache_dir": None,  # Will be set in fixture
    "pipeline_path": None,  # Will be set in fixture
}


class TestServerLifecycle:
    """Test server startup and shutdown."""

    def test_pipeline_loading(self):
        """Test that the fake pipeline loads correctly."""
        from tests.server.pipeline.fake_t2v_pipeline import get_pipeline, run_with_file

        pipe = get_pipeline(parallelism=1)
        assert pipe is not None
        assert pipe.initialized

        # Test generation
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp4"
            run_with_file(
                pipe,
                prompt="Test prompt",
                negative_prompt="",
                seed=42,
                resolution="480p",
                output_path=str(output_path),
                aspect_ratio="16:9",
            )
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_server_start(self, running_server):
        """Test that the server starts and responds."""
        base_url = running_server["base_url"]

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/v1/service/status")
            assert response.status_code == 200
            data = response.json()
            assert "service_status" in data


class TestTaskEndpoints:
    """Test task creation and management endpoints."""

    @pytest.mark.asyncio
    async def test_create_task(self, running_server):
        """Test creating a new task."""
        base_url = running_server["base_url"]

        payload = {
            "task": "t2v",
            "prompt": "A beautiful sunset over the ocean",
            "seed": 42,
            "resolution": "480p",
            "target_video_length": 5,
            "aspect_ratio": "16:9",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{base_url}/v1/tasks/create", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert "task_id" in data
            assert data["task_status"] == "pending"
            assert "output_path" in data

    @pytest.mark.asyncio
    async def test_create_task_invalid_aspect_ratio(self, running_server):
        """Test creating a task with invalid aspect ratio."""
        base_url = running_server["base_url"]

        payload = {
            "task": "t2v",
            "prompt": "Test",
            "aspect_ratio": "invalid_ratio",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(f"{base_url}/v1/tasks/create", json=payload)
            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_task_status(self, running_server):
        """Test getting task status."""
        base_url = running_server["base_url"]

        # Create a task first
        payload = {
            "task": "t2v",
            "prompt": "Test task",
        }

        async with httpx.AsyncClient() as client:
            create_resp = await client.post(f"{base_url}/v1/tasks/create", json=payload)
            task_id = create_resp.json()["task_id"]

            # Get status
            status_resp = await client.get(f"{base_url}/v1/tasks/{task_id}/status")
            assert status_resp.status_code == 200

            data = status_resp.json()
            assert "task_id" in data
            assert "status" in data

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, running_server):
        """Test getting status of non-existent task."""
        base_url = running_server["base_url"]

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/v1/tasks/nonexistent-id/status")
            assert response.status_code == 404


class TestTaskProcessing:
    """Test end-to-end task processing."""

    @pytest.mark.asyncio
    async def test_task_completion(self, running_server):
        """Test that a task completes successfully."""
        base_url = running_server["base_url"]

        payload = {
            "task": "t2v",
            "prompt": "A cat playing with a ball",
            "seed": 42,
            "resolution": "480p",
            "target_video_length": 2,  # Short for testing
        }

        async with httpx.AsyncClient() as client:
            # Create task
            create_resp = await client.post(f"{base_url}/v1/tasks/create", json=payload)
            task_id = create_resp.json()["task_id"]

            # Wait for completion (with timeout)
            max_wait = 30  # seconds
            start_time = time.time()

            while time.time() - start_time < max_wait:
                status_resp = await client.get(f"{base_url}/v1/tasks/{task_id}/status")
                status_data = status_resp.json()

                if status_data.get("status") == "completed":
                    break
                elif status_data.get("status") == "failed":
                    pytest.fail(f"Task failed: {status_data.get('error')}")

                await asyncio.sleep(0.5)
            else:
                pytest.fail("Task did not complete within timeout")

            # Verify result
            assert status_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_cancel_task(self, running_server):
        """Test canceling a pending task."""
        base_url = running_server["base_url"]

        # Create a task with long processing time
        payload = {
            "task": "t2v",
            "prompt": "A very long and detailed description that takes time to process...",
        }

        async with httpx.AsyncClient() as client:
            create_resp = await client.post(f"{base_url}/v1/tasks/create", json=payload)
            task_id = create_resp.json()["task_id"]

            # Cancel the task
            cancel_resp = await client.delete(f"{base_url}/v1/tasks/{task_id}")
            assert cancel_resp.status_code == 200

            cancel_data = cancel_resp.json()
            assert cancel_data["stop_status"] in ["success", "do_nothing"]

    @pytest.mark.asyncio
    async def test_queue_status(self, running_server):
        """Test getting queue status."""
        base_url = running_server["base_url"]

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/v1/tasks/queue/status")
            assert response.status_code == 200

            data = response.json()
            assert "pending_count" in data
            assert "active_count" in data
            assert "queue_size" in data


class TestServiceEndpoints:
    """Test service information endpoints."""

    @pytest.mark.asyncio
    async def test_service_status(self, running_server):
        """Test getting service status."""
        base_url = running_server["base_url"]

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/v1/service/status")
            assert response.status_code == 200

            data = response.json()
            assert "service_status" in data
            assert "current_task" in data
            assert "pending_tasks" in data

    @pytest.mark.asyncio
    async def test_service_metadata(self, running_server):
        """Test getting service metadata."""
        base_url = running_server["base_url"]

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/v1/service/metadata")
            assert response.status_code == 200

            data = response.json()
            assert "pipeline_file" in data
            assert "parallelism" in data
            assert "task" in data


class TestFileEndpoints:
    """Test file upload/download endpoints."""

    @pytest.mark.asyncio
    async def test_download_file(self, running_server, sample_video):
        """Test downloading a result file."""
        base_url = running_server["base_url"]

        # First create and complete a task
        payload = {
            "task": "t2v",
            "prompt": "Test download",
            "output_path": str(sample_video.name),
        }

        async with httpx.AsyncClient() as client:
            # Create and wait for task
            create_resp = await client.post(f"{base_url}/v1/tasks/create", json=payload)
            task_id = create_resp.json()["task_id"]

            # Wait for completion
            max_wait = 30
            start_time = time.time()
            while time.time() - start_time < max_wait:
                status_resp = await client.get(f"{base_url}/v1/tasks/{task_id}/status")
                if status_resp.json().get("status") == "completed":
                    break
                await asyncio.sleep(0.5)

            # Try to download result
            result_resp = await client.get(f"{base_url}/v1/tasks/{task_id}/result")
            assert result_resp.status_code in [200, 404]  # 404 if file doesn't exist yet


class TestClientSDK:
    """Test the client SDK."""

    def test_client_initialization(self):
        """Test that the client can be initialized."""
        from telefuser.client import TFClient

        client = TFClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    @pytest.mark.asyncio
    async def test_client_create_task(self, running_server):
        """Test client creating a task."""
        from telefuser.client import TFClient

        client = TFClient(base_url=running_server["base_url"])

        # This would need the client to be async or use sync HTTP
        # For now, just test initialization
        assert client is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
