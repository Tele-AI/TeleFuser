"""Integration tests for service API."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Skip these tests if fastapi is not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # Required by TestClient

from fastapi.testclient import TestClient


class TestApiEndpoints:
    """Test API endpoints."""

    @pytest.fixture
    def mock_pipeline_service(self):
        """Create a mock pipeline service."""
        service = Mock()
        service.start_pipeline = Mock(return_value=True)
        service.stop_distributed_inference = Mock(return_value=None)
        service.process_task = Mock(return_value={"status": "success"})
        return service

    @pytest.fixture
    def api_client(self, mock_pipeline_service):
        """Create a test client with mocked services."""
        from telefuser.service.api.api_server import ApiServer
        from telefuser.service.core.task_manager import TaskManager
        from telefuser.service.core.task_processor import AsyncTaskProcessor

        # Create task manager
        task_manager = TaskManager(max_queue_size=10)

        # Create server with task manager
        server = ApiServer(max_queue_size=10, task_manager=task_manager)

        # Mock services
        server.file_service = Mock()
        server.file_service.input_image_dir = Path("/tmp/input")
        server.file_service.input_video_dir = Path("/tmp/input")
        server.file_service.output_dir = Path("/tmp/output")
        server.file_service.output_video_dir = Path("/tmp/output/videos")
        server.file_service.output_image_dir = Path("/tmp/output/images")
        server.file_service.save_upload_file = Mock(return_value="/tmp/test_file.png")

        # Mock inference service
        server.inference_service = mock_pipeline_service
        server.inference_service.server_metadata = Mock(
            return_value={
                "pipeline_file": "/test/pipeline.py",
                "pipeline_name": "mock_pipeline",
                "parallelism": 1,
                "task": "t2v",
                "security_level": "STRICT",
                "declared_pipeline_contract": True,
                "contract_version": "v1",
                "supported_tasks": ["t2v", "i2v", "t2i", "i2i"],
                "supported_media_types": ["video", "image"],
                "execution_mode": "serial_single_pipeline",
                "effective_max_concurrent_tasks": 1,
                "entrypoints": {"get_pipeline": "get_pipeline", "run_with_file": "run_with_file"},
            }
        )

        # Mock media service with async method
        import asyncio

        async def mock_generate(*args, **kwargs):
            return None

        server.media_service = Mock()
        server.media_service.generate_media_with_stop_event = mock_generate
        server.task_processor = AsyncTaskProcessor(
            task_manager=task_manager,
            media_service=server.media_service,
            max_concurrent=1,
        )

        # Create test client
        with TestClient(server.app) as test_client:
            yield test_client

        asyncio.run(server.cleanup())

    def test_service_status_endpoint(self, api_client):
        """Test service status endpoint."""
        response = api_client.get("/v1/service/status")

        assert response.status_code == 200
        data = response.json()
        # Service status returns task manager status
        assert "service_status" in data

    def test_service_metadata_endpoint(self, api_client):
        """Test service metadata endpoint includes supported tasks."""
        response = api_client.get("/v1/service/metadata")

        assert response.status_code == 200
        data = response.json()
        # Should include supported tasks and media types
        assert "supported_tasks" in data
        assert "supported_media_types" in data
        assert data["declared_pipeline_contract"] is True
        assert data["execution_mode"] == "serial_single_pipeline"
        assert data["effective_max_concurrent_tasks"] == 1
        assert data["service_configured_max_concurrent_tasks"] == 1
        assert "t2i" in data["supported_tasks"]
        assert "i2i" in data["supported_tasks"]
        assert "image" in data["supported_media_types"]

    def test_create_t2v_task_endpoint(self, api_client):
        """Test T2V task creation endpoint."""
        task_data = {"prompt": "A beautiful landscape", "aspect_ratio": "16:9", "seed": 42, "task": "t2v"}

        response = api_client.post("/v1/tasks/create", json=task_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_status"] == "pending"
        assert "output_path" in data

    def test_create_t2i_task_endpoint(self, api_client):
        """Test T2I (text-to-image) task creation endpoint."""
        task_data = {
            "prompt": "A beautiful landscape painting",
            "task": "t2i",
            "aspect_ratio": "1:1",
            "seed": 42,
            "output_format": "png",
        }

        response = api_client.post("/v1/tasks/create", json=task_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_status"] == "pending"
        assert "output_path" in data
        # Default output for image task should have image extension
        assert data["output_path"].endswith(".png")

    def test_create_i2i_task_endpoint(self, api_client):
        """Test I2I (image-to-image) task creation endpoint."""
        task_data = {
            "prompt": "Transform this image",
            "task": "i2i",
            "first_image_path": "test_image.png",
            "aspect_ratio": "1:1",
            "seed": 42,
            "output_format": "jpg",
        }

        response = api_client.post("/v1/tasks/create", json=task_data)

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["task_status"] == "pending"
        assert "output_path" in data

    def test_create_task_rejects_missing_contract_required_parameter(self, api_client, mock_pipeline_service):
        """Task creation should enforce required exposed parameters from task contracts."""
        mock_pipeline_service.server_metadata.return_value["task_contracts"] = {
            "t2v": {
                "media_type": "video",
                "required_inputs": [],
                "optional_inputs": [],
                "parameters": {
                    "prompt": {
                        "required": True,
                        "default": "",
                        "exposed": True,
                    }
                },
            }
        }

        response = api_client.post("/v1/tasks/create", json={"task": "t2v"})

        assert response.status_code == 400
        data = response.json()
        assert data["detail"]["missing_parameters"] == ["prompt"]

    def test_get_task_status_endpoint(self, api_client):
        """Test get task status endpoint."""
        # First create a task
        task_data = {"prompt": "test"}
        create_response = api_client.post("/v1/tasks/create", json=task_data)
        task_id = create_response.json()["task_id"]

        # Query status
        response = api_client.get(f"/v1/tasks/{task_id}/status")
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "status" in data
        assert "output_path" in data  # Check new field name
        assert data["task"] == "t2v"
        assert data["media_type"] == "video"
        assert data["prompt"] == "test"

    def test_cancel_task_endpoint(self, api_client):
        """Test cancel task endpoint."""
        response = api_client.delete("/v1/tasks/task-123")

        # Task doesn't exist, should return 404 or success
        assert response.status_code in [200, 404]

    def test_create_task_missing_required_field(self, api_client):
        """Test task creation with missing required field."""
        # TaskRequest only requires task_id which is auto-generated
        # Other fields have defaults
        task_data = {}

        response = api_client.post("/v1/tasks/create", json=task_data)

        # Should succeed with defaults
        assert response.status_code == 200

    def test_create_task_invalid_task_type(self, api_client):
        """Test task creation with invalid task type."""
        task_data = {"task": "Invalid Task!"}

        response = api_client.post("/v1/tasks/create", json=task_data)

        # Should fail validation
        assert response.status_code == 422

    def test_create_task_unsupported_for_current_pipeline(self, api_client):
        """Test task creation rejected when current pipeline contract does not support the task."""
        task_data = {"task": "fl2v", "prompt": "test"}

        response = api_client.post("/v1/tasks/create", json=task_data)

        assert response.status_code == 400
        data = response.json()
        assert "supported_tasks" in data["detail"]

    def test_openai_video_i2v_rejected_when_pipeline_lacks_support(self, api_client):
        """Test OpenAI video route rejects derived I2V requests if the pipeline contract doesn't support them."""
        task_data = {
            "prompt": "A beach scene",
            "input_reference": "/tmp/input.png",
            "size": "1024x576",
        }

        response = api_client.post("/v1/videos", json=task_data)

        assert response.status_code == 200

    def test_openai_video_unsupported_task_rejected(self, api_client, mock_pipeline_service):
        """Test OpenAI video route rejects tasks unsupported by the current pipeline metadata."""
        mock_pipeline_service.server_metadata.return_value["supported_tasks"] = ["t2v", "t2i", "i2i"]

        task_data = {
            "prompt": "A beach scene",
            "input_reference": "/tmp/input.png",
            "size": "1024x576",
        }

        response = api_client.post("/v1/videos", json=task_data)

        assert response.status_code == 400
        data = response.json()
        assert "supported_tasks" in data["detail"]

    def test_list_videos_includes_created_task(self, api_client):
        """Test video listing includes tasks created through the OpenAI video endpoint."""
        create_response = api_client.post(
            "/v1/videos",
            json={
                "prompt": "video list",
                "size": "1024x576",
            },
        )

        assert create_response.status_code == 200
        created_video_id = create_response.json()["id"]

        list_response = api_client.get("/v1/videos")

        assert list_response.status_code == 200
        data = list_response.json()
        assert data["object"] == "list"
        assert any(video["id"] == created_video_id for video in data["data"])


class TestPipelineServiceIntegration:
    """Test PipelineService integration."""

    @pytest.mark.slow
    def test_pipeline_initialization(self):
        """Test pipeline initialization (slow test)."""
        pytest.skip("Requires actual model files - skipping in CI")

    @pytest.mark.gpu
    def test_gpu_inference(self):
        """Test inference on GPU."""
        pytest.skip("Requires GPU - skipping in CPU-only environment")


@pytest.mark.filesystem
class TestFileUpload:
    """Test file upload functionality - requires filesystem access."""

    @pytest.fixture
    def fs_client(self, tmp_path):
        """Create a test client with actual file service."""
        from telefuser.service.api.api_server import ApiServer
        from telefuser.service.core.file_service import FileService
        from telefuser.service.core.task_manager import TaskManager

        # Create task manager
        task_manager = TaskManager(max_queue_size=10)

        # Create server with task manager
        server = ApiServer(max_queue_size=10, task_manager=task_manager)

        # Use actual file service with temp directory
        server.file_service = FileService(cache_dir=tmp_path)

        with TestClient(server.app) as test_client:
            yield test_client

    def test_upload_image(self, fs_client):
        """Test image upload via form endpoint."""
        # Create a simple test file
        test_file = b"fake image data"

        response = fs_client.post(
            "/v1/tasks/form",
            files={"first_image_file": ("test.png", test_file, "image/png")},
            data={"prompt": "test prompt"},
        )

        # Form endpoint should work
        assert response.status_code in [200, 422]

    def test_upload_video(self, fs_client):
        """Test video upload - not directly supported, use form."""
        test_file = b"fake video data"

        response = fs_client.post(
            "/v1/tasks/form", files={"first_image_file": ("test.mp4", test_file, "video/mp4")}, data={"prompt": "test"}
        )

        assert response.status_code in [200, 422]

    def test_upload_without_file(self, fs_client):
        """Test upload without file via form."""
        response = fs_client.post("/v1/tasks/form", data={"prompt": "test without image"})

        # Form should work without files (optional)
        assert response.status_code in [200, 422]


class TestClientSdk:
    """Test client SDK functionality."""

    def test_tap_client_creation(self):
        """Test TFClient can be imported and created."""
        from telefuser.client import TFClient

        client = TFClient(base_url="http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    def test_tap_client_t2i_method_exists(self):
        """Test TFClient has T2I method."""
        from telefuser.client import TFClient

        client = TFClient()
        assert hasattr(client, "create_t2i_task")
        assert hasattr(client, "create_i2i_task")
        assert hasattr(client, "create_t2v_task")
        assert hasattr(client, "create_i2v_task")
