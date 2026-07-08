from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from telefuser.service.api.api_server import ApiServer
from telefuser.service.api.schema import TaskRequest
from telefuser.service.core.file_service import FileService
from telefuser.service.core.task_manager import TaskManager


def test_task_application_service_applies_defaults_and_creates_task(tmp_path: Path) -> None:
    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    server.file_service = FileService(tmp_path)
    server.validate_task_supported = lambda task: None
    server.get_task_contract = lambda task: {
        "parameters": {
            "resolution": {"default": "480p"},
            "seed": {"default": 7},
        }
    }
    message = TaskRequest(task="t2v")

    response = asyncio.run(
        server.task_app_service.submit(
            message,
            explicit_fields={"task"},
            ensure_processing=False,
        )
    )

    assert response.task_id == message.task_id
    assert response.task_status.value == "pending"
    assert message.resolution == "480p"
    assert message.seed == 7
    assert task_manager.get_task(message.task_id).message is message


def test_task_application_service_rejects_unsafe_output_path(tmp_path: Path) -> None:
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.file_service = FileService(tmp_path)
    server.validate_task_supported = lambda task: None
    server.get_task_contract = lambda task: None
    message = TaskRequest(task="t2v", output_path="../escape.mp4")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server.task_app_service.submit(
                message,
                explicit_fields={"task", "output_path"},
                ensure_processing=False,
            )
        )

    assert exc_info.value.status_code == 400


def test_task_application_service_runs_route_specific_input_validation() -> None:
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.validate_task_supported = lambda task: None
    server.get_task_contract = lambda task: None
    message = TaskRequest(task="i2v")

    def validate_inputs(message: TaskRequest, contract: dict | None) -> None:
        raise HTTPException(status_code=400, detail="missing input")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server.task_app_service.submit(
                message,
                explicit_fields={"task"},
                validate_inputs=validate_inputs,
                ensure_processing=False,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "missing input"
    assert server.task_manager.get_task(message.task_id) is None


def test_task_application_service_waits_for_completed_task() -> None:
    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    message = TaskRequest(task="t2v")
    task_id = task_manager.create_task(message)
    task_manager.complete_task(task_id, output_path="out.mp4")

    status = asyncio.run(
        server.task_app_service.wait_for_completion(
            task_id,
            timeout=1,
            poll_interval=0,
        )
    )

    assert status["status"] == "completed"
    assert status["output_path"] == "out.mp4"


def test_task_application_service_wait_maps_failed_task() -> None:
    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    message = TaskRequest(task="t2v")
    task_id = task_manager.create_task(message)
    task_manager.fail_task(task_id, "pipeline failed")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server.task_app_service.wait_for_completion(
                task_id,
                timeout=1,
                poll_interval=0,
            )
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Generation failed: pipeline failed"


def test_task_application_service_wait_times_out_pending_task() -> None:
    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    message = TaskRequest(task="t2v")
    task_id = task_manager.create_task(message)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            server.task_app_service.wait_for_completion(
                task_id,
                timeout=0,
                poll_interval=0,
            )
        )

    assert exc_info.value.status_code == 504


def test_task_application_service_returns_output_response(tmp_path: Path) -> None:
    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    server.file_service = FileService(tmp_path)
    message = TaskRequest(task="t2i")
    task_id = task_manager.create_task(message)
    output_path = server.file_service.get_output_path("result.png", media_type="image", task_id=task_id)
    output_path.write_bytes(b"image")
    task_manager.complete_task(task_id, output_path=str(output_path))

    response = server.task_app_service.get_output_response(task_id, media_type="image")

    assert response.headers["content-length"] == "5"


def test_task_application_service_rejects_unready_required_output(tmp_path: Path) -> None:
    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    server.file_service = FileService(tmp_path)
    message = TaskRequest(task="t2v")
    task_id = task_manager.create_task(message)

    with pytest.raises(HTTPException) as exc_info:
        server.task_app_service.get_output_response(
            task_id,
            media_type="video",
            require_completed=True,
        )

    assert exc_info.value.status_code == 400
    assert "is not ready" in exc_info.value.detail


def test_task_application_service_cancel_accepts_active_task() -> None:
    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    message = TaskRequest(task="t2v")
    task_id = task_manager.create_task(message)

    result = server.task_app_service.cancel_task(task_id)

    assert result == {"result": "accepted", "task_status": "cancelled"}
    assert task_manager.get_task_status(task_id)["status"] == "cancelled"


def test_task_application_service_cancel_reports_terminal_task() -> None:
    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    message = TaskRequest(task="t2v")
    task_id = task_manager.create_task(message)
    task_manager.complete_task(task_id)

    result = server.task_app_service.cancel_task(task_id)

    assert result == {"result": "already_terminal", "task_status": "completed"}


def test_task_application_service_cancel_reports_missing_task() -> None:
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)

    result = server.task_app_service.cancel_task("missing")

    assert result == {"result": "not_found", "task_status": None}
