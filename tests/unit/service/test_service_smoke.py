from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock

from telefuser.service.api.api_server import ApiServer
from telefuser.service.api.routers.files import FileRoutes
from telefuser.service.core.file_service import FileService
from telefuser.service.core.task_manager import TaskManager
from telefuser.service_types import MediaType
from tests.unit.openai._asgi_test_client import ASGITestClient


def _make_smoke_server(tmp_path: Path) -> ApiServer:
    task_manager = TaskManager(max_queue_size=10)
    server = ApiServer(
        task_manager=task_manager,
        enable_rate_limit=False,
        enable_openai_api=True,
    )
    server.file_service = FileService(tmp_path)
    server.inference_service = Mock()
    server.inference_service.server_metadata.return_value = {
        "pipeline_file": "/test/pipeline.py",
        "pipeline_name": "smoke_pipeline",
        "parallelism": 1,
        "task": "t2v",
        "declared_pipeline_contract": True,
        "supported_tasks": ["t2v", "t2i"],
        "supported_media_types": ["video", "image"],
        "execution_mode": "serial_single_pipeline",
        "effective_max_concurrent_tasks": 1,
        "entrypoints": {"get_pipeline": "get_pipeline", "run_with_file": "run_with_file"},
        "task_contracts": {
            "t2v": {"required_inputs": [], "media_type": "video"},
            "t2i": {"required_inputs": [], "media_type": "image"},
        },
    }
    return server


def test_task_create_status_and_artifact_id_download_smoke(tmp_path: Path) -> None:
    server = _make_smoke_server(tmp_path)

    with ASGITestClient(server.app) as client:
        create_response = client.post(
            "/v1/tasks/create",
            json={"task": "t2i", "prompt": "a cat", "output_format": "png"},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        output_path = server.file_service.get_output_path("result.png", media_type=MediaType.IMAGE, task_id=task_id)
        output_path.write_bytes(b"image")
        server.task_manager.complete_task(task_id, output_path=str(output_path))

        status_response = client.get(f"/v1/tasks/{task_id}/status")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "completed"

        artifact_id = server.file_service.artifact_id_for_path(output_path)
        download_response = asyncio.run(FileRoutes(server).download_file(artifact_id))
        assert download_response.headers["content-length"] == "5"
        assert download_response.headers["content-disposition"] == 'attachment; filename="result.png"'


def test_openai_video_retrieve_includes_artifact_metadata_smoke(tmp_path: Path) -> None:
    server = _make_smoke_server(tmp_path)

    with ASGITestClient(server.app) as client:
        create_response = client.post(
            "/v1/videos",
            json={"prompt": "a cat playing", "seconds": 4, "size": "1024x576"},
        )
        assert create_response.status_code == 200
        video_id = create_response.json()["id"]

        output_path = server.file_service.get_output_path("clip.mp4", media_type=MediaType.VIDEO, task_id=video_id)
        output_path.write_bytes(b"video")
        server.task_manager.complete_task(video_id, output_path=str(output_path))

        retrieve_response = client.get(f"/v1/videos/{video_id}")
        assert retrieve_response.status_code == 200
        data = retrieve_response.json()
        assert data["status"] == "completed"
        assert data["url"].endswith(f"/v1/videos/{video_id}/content")
        assert data["artifact_id"] == f"local:tasks/{video_id}/outputs/videos/clip.mp4"
        assert data["artifact_metadata"]["backend"] == "local"
        assert data["artifact_metadata"]["size_bytes"] == 5
