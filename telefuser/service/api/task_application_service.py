"""Canonical task submission service for HTTP-facing APIs."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from telefuser.service.core.pipeline_contract import infer_media_type_for_task
from telefuser.service_types import MediaType, TaskStatus

from .schema import TaskRequest, TaskResponse
from .task_contract_runtime import apply_task_contract_defaults, validate_required_task_parameters

if TYPE_CHECKING:
    from .api_server import ApiServer


class TaskApplicationService:
    """Create tasks through a single validated submission path."""

    def __init__(self, api_server: ApiServer) -> None:
        self.api = api_server

    async def submit(
        self,
        message: TaskRequest,
        *,
        explicit_fields: set[str],
        validate_inputs: Callable[[TaskRequest, dict[str, Any] | None], None] | None = None,
        ensure_processing: bool = True,
    ) -> TaskResponse:
        """Apply contract defaults, validate paths/required params, enqueue, and start processing."""
        try:
            self.api.validate_task_supported(message.task)
            contract = self.api.get_task_contract(message.task)
            apply_task_contract_defaults(message, task_contract=contract, explicit_fields=explicit_fields)
            self.validate_output_path(message)
            validate_required_task_parameters(message, task_contract=contract)
            if validate_inputs is not None:
                validate_inputs(message, contract)

            task_id = self.api.task_manager.create_task(message)
            message.task_id = task_id

            if ensure_processing:
                await self.api.ensure_task_processor_running()

            return TaskResponse(task_id=task_id, task_status=TaskStatus.PENDING, output_path=message.output_path)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    def validate_output_path(self, message: TaskRequest) -> None:
        file_service = self.api.file_service
        if file_service is None:
            return
        if getattr(type(file_service), "get_output_path", None) is None:
            return

        resolver = getattr(file_service, "get_output_path", None)
        if not callable(resolver):
            return

        try:
            resolver(message.output_path, media_type=infer_media_type_for_task(message.task), task_id=message.task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    async def wait_for_completion(
        self,
        task_id: str,
        *,
        timeout: float,
        poll_interval: float = 0.5,
    ) -> dict[str, Any]:
        """Wait until a task reaches a terminal state and return completed task status."""
        deadline = time.monotonic() + timeout

        while True:
            status = self.api.task_manager.get_task_status(task_id)
            if not status:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

            task_status = status.get("status")
            if task_status == TaskStatus.COMPLETED.value:
                return status
            if task_status == TaskStatus.FAILED.value:
                error_msg = status.get("error", "Unknown error")
                raise HTTPException(status_code=500, detail=f"Generation failed: {error_msg}")
            if task_status == TaskStatus.CANCELLED.value:
                raise HTTPException(status_code=400, detail="Generation was cancelled")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(status_code=504, detail=f"Generation timeout after {timeout} seconds")

            await asyncio.sleep(min(poll_interval, remaining))

    def get_output_response(
        self,
        task_id: str,
        *,
        media_type: MediaType | str,
        require_completed: bool = False,
    ) -> Response:
        """Return a validated streaming response for a task output artifact."""
        task_manager = getattr(self.api, "task_manager", None)
        if not task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        file_service = getattr(self.api, "file_service", None)
        if not file_service:
            raise HTTPException(status_code=503, detail="File service not initialized")

        media_label = "Image" if str(media_type) == MediaType.IMAGE.value else "Video"
        task_info = task_manager.get_task(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail=f"{media_label} {task_id} not found")

        task_status = task_manager.get_task_status(task_id) or {}
        if require_completed and task_status.get("status") != TaskStatus.COMPLETED.value:
            raise HTTPException(
                status_code=400,
                detail=f"{media_label} {task_id} is not ready (status: {task_status.get('status')})",
            )

        output_path = task_info.output_path or task_status.get("output_path")
        if not output_path:
            raise HTTPException(status_code=404, detail=f"{media_label} {task_id} has no output path")

        path = self.resolve_output_file(file_service, output_path, media_type=media_type)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"{media_label} file not found: {path}")

        stream_response = self.stream_file_response(path)
        if stream_response is not None:
            return stream_response

        response_media_type = "image/png" if str(media_type) == MediaType.IMAGE.value else "video/mp4"
        return FileResponse(path=str(path), media_type=response_media_type, filename=path.name)

    def resolve_task_output_path(self, output_path: str, *, media_type: MediaType | str) -> Path:
        """Resolve a task output path through the configured file service."""
        file_service = getattr(self.api, "file_service", None)
        if not file_service:
            return Path(output_path)
        return self.resolve_output_file(file_service, output_path, media_type=media_type)

    def get_output_metadata(
        self,
        task_id: str,
        *,
        output_path: str | None = None,
        media_type: MediaType | str,
    ) -> dict[str, Any] | None:
        """Return artifact metadata for a task output when the file service supports it."""
        file_service = getattr(self.api, "file_service", None)
        if not file_service:
            return None

        if output_path is None:
            task_info = self.api.task_manager.get_task(task_id)
            task_status = self.api.task_manager.get_task_status(task_id) or {}
            output_path = getattr(task_info, "output_path", None) or task_status.get("output_path")
        if not output_path:
            return None

        metadata_builder = self.get_declared_method(file_service, "artifact_metadata")
        if not callable(metadata_builder):
            return None

        try:
            return metadata_builder(output_path, task_id=task_id, media_type=media_type)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access to this file is not allowed")

    def get_base_url(self) -> str:
        """Return the configured HTTP base URL for local content routes."""
        server_config = getattr(self.api, "server_config", None)
        if server_config:
            host = getattr(server_config, "host", "localhost")
            port = getattr(server_config, "port", 8000)
            return f"http://{host}:{port}"
        return "http://localhost:8000"

    def get_openai_content_url(self, task_id: str, *, media_type: MediaType | str) -> str:
        """Return the OpenAI-compatible content URL for a task output."""
        resource = "images" if str(media_type) == MediaType.IMAGE.value else "videos"
        return f"{self.get_base_url()}/v1/{resource}/{task_id}/content"

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a task and distinguish accepted, terminal, and missing outcomes."""
        task_manager = getattr(self.api, "task_manager", None)
        if not task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        task_status = task_manager.get_task_status(task_id)
        if not task_status:
            return {"result": "not_found", "task_status": None}

        current_status = task_status.get("status")
        if current_status in {
            TaskStatus.COMPLETED.value,
            TaskStatus.FAILED.value,
            TaskStatus.CANCELLED.value,
        }:
            return {"result": "already_terminal", "task_status": current_status}

        if task_manager.cancel_task(task_id):
            return {"result": "accepted", "task_status": TaskStatus.CANCELLED.value}

        task_status = task_manager.get_task_status(task_id)
        if not task_status:
            return {"result": "not_found", "task_status": None}
        return {"result": "already_terminal", "task_status": task_status.get("status")}

    async def save_upload_file(
        self,
        file: Any,
        *,
        media_type: MediaType | str,
        prefix: str = "upload",
        fallback_filename: str = "upload.bin",
    ) -> str:
        """Save an upload through FileService, with a bounded fallback for lightweight mocks."""
        file_service = getattr(self.api, "file_service", None)
        if not file_service:
            raise HTTPException(status_code=503, detail="File service not initialized")

        writer = self.get_declared_method(file_service, "save_upload_file")
        if callable(writer):
            path = await writer(
                file,
                media_type=media_type,
                prefix=prefix,
                fallback_filename=fallback_filename,
            )
            return str(path)

        input_dir = self.input_dir_for_media(file_service, media_type)
        suffix = Path(getattr(file, "filename", None) or fallback_filename).suffix
        file_path = input_dir / f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        await self.write_upload_stream(
            file,
            file_path,
            max_file_size=getattr(file_service, "max_file_size", None),
        )
        return str(file_path)

    def input_dir_for_media(self, file_service: Any, media_type: MediaType | str) -> Path:
        """Return a legacy input directory for fallback upload writes."""
        if str(media_type) == MediaType.IMAGE.value:
            input_dir = getattr(file_service, "input_image_dir", None)
        else:
            input_dir = getattr(file_service, "input_video_dir", None) or getattr(file_service, "input_image_dir", None)

        if not input_dir:
            input_dir = Path("/tmp/telefuser/inputs")
            input_dir.mkdir(parents=True, exist_ok=True)
        return Path(input_dir)

    async def write_upload_stream(
        self,
        file: Any,
        file_path: Path,
        *,
        max_file_size: int | None = None,
    ) -> None:
        """Write upload content through a .part file without reading it all into memory."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = file_path.with_name(f"{file_path.name}.{uuid.uuid4().hex}.part")
        bytes_written = 0
        try:
            with open(part_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if max_file_size is not None and bytes_written > max_file_size:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large: {bytes_written} bytes, max: {max_file_size} bytes",
                        )
                    buffer.write(chunk)
            part_path.replace(file_path)
        except Exception:
            part_path.unlink(missing_ok=True)
            raise

    def resolve_output_file(self, file_service: Any, output_path: str, *, media_type: MediaType | str) -> Path:
        """Resolve an output path through FileService when available."""
        resolver = self.get_declared_method(file_service, "resolve_output_file")
        if callable(resolver):
            try:
                return resolver(output_path)
            except ValueError:
                raise HTTPException(status_code=403, detail="Access to this file is not allowed")

        path = Path(output_path)
        if not path.is_absolute():
            output_attr = "output_image_dir" if str(media_type) == MediaType.IMAGE.value else "output_video_dir"
            output_dir = getattr(file_service, output_attr, None)
            if output_dir:
                path = Path(output_dir) / path
        return path

    def stream_file_response(self, path: Path) -> StreamingResponse | None:
        responder = self.get_declared_method(self.api, "_stream_file_response")
        if callable(responder):
            return responder(path, filename=path.name)
        return None

    @staticmethod
    def get_declared_method(obj: Any, name: str) -> Any | None:
        if getattr(type(obj), name, None) is None:
            return None
        return getattr(obj, name, None)
