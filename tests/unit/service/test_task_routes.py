from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import FormData

from telefuser.service.api.routers.tasks import TaskRoutes
from telefuser.service.api.schema import TaskResponse
from telefuser.service.api.task_application_service import TaskApplicationService
from telefuser.service.core.file_service import FileService


class _AsyncUpload:
    def __init__(self, filename: str, content_type: str, chunks: list[bytes]) -> None:
        self.filename = filename
        self.content_type = content_type
        self._chunks = list(chunks)

    async def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


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


def _make_form_routes(tmp_path: Path, supported_tasks: tuple[str, ...]) -> TaskRoutes:
    routes = _make_routes(supported_tasks)
    routes.api.file_service = FileService(tmp_path)
    routes.api.task_app_service = TaskApplicationService(routes.api)
    routes.api.task_app_service.submit = AsyncMock(
        return_value=TaskResponse(task_id="task-123", task_status="pending", output_path="output.mp4")
    )
    return routes


def _make_form_request(data: dict[str, object]) -> Mock:
    request = Mock()
    request.form = AsyncMock(return_value=FormData(list(data.items())))
    return request


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


def test_create_task_form_uses_file_service_upload_for_image(tmp_path: Path) -> None:
    routes = _make_form_routes(tmp_path, ("i2i",))
    upload = _AsyncUpload("cat.png", "image/png", [b"cat"])

    asyncio.run(
        routes.create_task_form(request=_make_form_request({"prompt": "make it blue"}), first_image_file=upload)
    )

    task_request = routes.api.task_app_service.submit.call_args.args[0]
    input_path = Path(task_request.first_image_path)
    assert task_request.task == "i2i"
    assert input_path.parent == routes.api.file_service.input_image_dir
    assert input_path.read_bytes() == b"cat"


def test_create_task_form_uses_file_service_upload_for_video(tmp_path: Path) -> None:
    routes = _make_form_routes(tmp_path, ("vc",))
    upload = _AsyncUpload("clip.mp4", "video/mp4", [b"video"])

    asyncio.run(routes.create_task_form(request=_make_form_request({"prompt": "continue"}), first_image_file=upload))

    task_request = routes.api.task_app_service.submit.call_args.args[0]
    input_path = Path(task_request.ref_video_path)
    assert task_request.task == "vc"
    assert input_path.parent == routes.api.file_service.input_video_dir
    assert input_path.read_bytes() == b"video"


def test_create_task_form_forwards_dynamic_form_parameters(tmp_path: Path) -> None:
    routes = _make_form_routes(tmp_path, ("i2i",))
    upload = _AsyncUpload("cat.png", "image/png", [b"cat"])

    asyncio.run(
        routes.create_task_form(
            request=_make_form_request(
                {
                    "prompt": "make it blue",
                    "resolution": "480p",
                    "seed": "123",
                    "cfg_scale": "5.5",
                    "enable_feature": "true",
                    "extra_options": '{"mode": "fast"}',
                }
            ),
            first_image_file=upload,
        )
    )

    task_request = routes.api.task_app_service.submit.call_args.args[0]
    assert task_request.task == "i2i"
    assert task_request.prompt == "make it blue"
    assert task_request.resolution == "480p"
    assert task_request.seed == 123
    assert task_request.cfg_scale == 5.5
    assert task_request.enable_feature is True
    assert task_request.extra_options == {"mode": "fast"}

    explicit_fields = routes.api.task_app_service.submit.call_args.kwargs["explicit_fields"]
    assert {"resolution", "cfg_scale", "enable_feature", "extra_options"}.issubset(explicit_fields)
