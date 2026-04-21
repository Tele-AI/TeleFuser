from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, UploadFile

from telefuser.service.api.routers.tasks import TaskRoutes


def _make_routes(supported_tasks: tuple[str, ...] = ()) -> TaskRoutes:
    api_server = Mock()
    api_server.get_supported_tasks.return_value = supported_tasks
    api_server.validate_task_supported = Mock()
    api_server.get_task_contract.side_effect = lambda task: {
        "t2v": {"required_inputs": []},
        "t2i": {"required_inputs": []},
        "i2v": {"required_inputs": ["first_image_path"]},
        "i2i": {"required_inputs": ["first_image_path"]},
        "fl2v": {"required_inputs": ["first_image_path", "last_image_path"]},
        "vc": {"required_inputs": ["ref_video_path"]},
        "vsr": {"required_inputs": ["ref_video_path"]},
    }.get(task)
    return TaskRoutes(api_server)


def test_resolve_form_task_prefers_supported_pipeline_task_order() -> None:
    routes = _make_routes(("t2i", "i2i", "i2v"))

    task = routes._resolve_form_task(
        requested_task="",
        first_image_path="/tmp/input.png",
        last_image_path="",
        ref_video_path="",
    )

    assert task == "i2i"


def test_resolve_form_task_defaults_to_video_flow_without_contract() -> None:
    routes = _make_routes()

    task = routes._resolve_form_task(
        requested_task="",
        first_image_path="/tmp/input.png",
        last_image_path="",
        ref_video_path="",
    )

    assert task == "i2v"


def test_validate_task_inputs_rejects_missing_required_image() -> None:
    routes = _make_routes()

    with pytest.raises(HTTPException) as exc:
        routes._validate_task_inputs(
            "i2i",
            first_image_path="",
            last_image_path="",
            ref_video_path="",
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["missing_inputs"] == ["first_image_path"]


def test_resolve_form_task_uses_contract_required_inputs() -> None:
    routes = _make_routes(("vc", "vsr", "t2v"))

    task = routes._resolve_form_task(
        requested_task="",
        first_image_path="",
        last_image_path="",
        ref_video_path="/tmp/input.mp4",
    )

    assert task == "vc"


def test_resolve_form_task_rejects_unsupported_requested_task_first() -> None:
    routes = _make_routes(("t2v",))
    routes.api.validate_task_supported.side_effect = HTTPException(
        status_code=400,
        detail={"supported_tasks": ["t2v"]},
    )

    with pytest.raises(HTTPException) as exc:
        routes._resolve_form_task(
            requested_task="fl2v",
            first_image_path="",
            last_image_path="",
            ref_video_path="",
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["supported_tasks"] == ["t2v"]


def test_is_video_upload_detects_video_content_type() -> None:
    routes = _make_routes()
    upload = UploadFile(filename="clip.mp4", file=open(__file__, "rb"), headers={"content-type": "video/mp4"})
    try:
        assert routes._is_video_upload(upload) is True
    finally:
        upload.file.close()


def test_is_video_upload_detects_video_extension_without_content_type() -> None:
    routes = _make_routes()
    upload = UploadFile(filename="clip.webm", file=open(__file__, "rb"))
    try:
        assert routes._is_video_upload(upload) is True
    finally:
        upload.file.close()
