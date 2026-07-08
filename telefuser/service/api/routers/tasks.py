"""
Task Routes for TeleFuser API

Provides route handlers for task-related endpoints.
Routes are defined as class methods for better testability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import ValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile

from telefuser.service.core.pipeline_contract import default_task_contract, validate_task_name_format
from telefuser.service_types import MediaType, StopTaskStatus
from telefuser.utils.logging import logger

from ..schema import StopTaskResponse, TaskRequest, TaskResponse
from ..task_contract_runtime import match_task_candidates

if TYPE_CHECKING:
    from ..api_server import ApiServer

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

_VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def create_router(api_server: ApiServer) -> APIRouter:
    """Create a new router with fresh routes for the given ApiServer instance."""
    new_router = APIRouter(prefix="/v1/tasks", tags=["tasks"])
    routes = TaskRoutes(api_server)

    @new_router.post("/create", response_model=TaskResponse)
    async def create_task(message: TaskRequest) -> TaskResponse:
        """Create a new generation task."""
        return await routes.create_task(message)

    @new_router.post("/form", response_model=TaskResponse)
    async def create_task_form(
        request: Request,
        first_image_file: UploadFile | None = File(default=None, description="First frame image file"),
        last_image_file: UploadFile | None = File(default=None, description="Last frame image file"),
    ) -> TaskResponse:
        """Create task with file upload support."""
        return await routes.create_task_form(
            request=request,
            first_image_file=first_image_file,
            last_image_file=last_image_file,
        )

    @new_router.get("/queue/status", response_model=dict)
    async def get_queue_status() -> dict:
        """Get current queue status."""
        return await routes.get_queue_status()

    @new_router.get("/{task_id}/status")
    async def get_task_status(task_id: str) -> dict | None:
        """Get the current status of a specific task."""
        return await routes.get_task_status(task_id)

    @new_router.delete("/{task_id}", response_model=StopTaskResponse)
    async def stop_task(task_id: str) -> StopTaskResponse:
        """Cancel a running or pending task."""
        return await routes.stop_task(task_id)

    return new_router


class TaskRoutes:
    """Task route handlers with dependency injection support."""

    def __init__(self, api_server: ApiServer) -> None:
        self.api = api_server

    async def create_task(self, message: TaskRequest) -> TaskResponse:
        """Create a new generation task."""

        async def check_image_path(image_name: str) -> None:
            if hasattr(message, image_name) and getattr(message, image_name).startswith("http"):
                if not await self.api._validate_image_url(getattr(message, image_name)):
                    raise HTTPException(status_code=400, detail=f"{image_name} URL is not accessible")

        try:
            await check_image_path("first_image_path")
            await check_image_path("last_image_path")
            return await self.api.task_app_service.submit(
                message,
                explicit_fields=set(getattr(message, "model_fields_set", set())),
                validate_inputs=self._validate_message_inputs,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def list_tasks(self) -> dict:
        """List all tasks."""
        return self.api.task_manager.get_all_tasks()

    async def get_queue_status(self) -> dict:
        """Get current queue status."""
        service_status = self.api.task_manager.get_service_status()
        return {
            "is_processing": self.api.task_manager.is_processing(),
            "current_task": service_status.get("current_task"),
            "pending_count": self.api.task_manager.get_pending_task_count(),
            "active_count": self.api.task_manager.get_active_task_count(),
            "queue_size": self.api.max_queue_size,
            "queue_available": self.api.max_queue_size - self.api.task_manager.get_active_task_count(),
        }

    async def get_task_status(self, task_id: str) -> dict | None:
        """Get status of a specific task."""
        status = self.api.task_manager.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        return status

    async def stop_task(self, task_id: str) -> StopTaskResponse:
        """Stop/cancel a task and clean up resources."""
        try:
            cancel_result = self.api.task_app_service.cancel_task(task_id)
            if cancel_result["result"] == "accepted":
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Task {task_id} cancellation accepted.")
                return StopTaskResponse(stop_status=StopTaskStatus.SUCCESS, reason="Task cancellation accepted.")
            if cancel_result["result"] == "already_terminal":
                return StopTaskResponse(
                    stop_status=StopTaskStatus.DO_NOTHING,
                    reason=f"Task already terminal: {cancel_result['task_status']}.",
                )
            return StopTaskResponse(stop_status=StopTaskStatus.DO_NOTHING, reason="Task not found.")
        except Exception as e:
            logger.error(f"Error occurred while stopping task {task_id}: {str(e)}")
            return StopTaskResponse(stop_status=StopTaskStatus.ERROR, reason=str(e))

    async def create_task_form(
        self,
        request: Request,
        first_image_file: UploadFile | None = None,
        last_image_file: UploadFile | None = None,
    ) -> TaskResponse:
        """Create task with file upload support and dynamic form parameters."""
        assert self.api.file_service is not None, "File service is not initialized"

        async def save_file_async(file: UploadFile, media_type: MediaType) -> str:
            if not file or not file.filename:
                return ""

            return await self.api.task_app_service.save_upload_file(
                file,
                media_type=media_type,
                prefix="input",
                fallback_filename="input.mp4" if media_type == MediaType.VIDEO else "input.png",
            )

        try:
            task_payload = await self._collect_form_task_payload(request)

            first_image_path = ""
            ref_video_path = ""
            if first_image_file and first_image_file.filename:
                if self._is_video_upload(first_image_file):
                    ref_video_path = await save_file_async(first_image_file, MediaType.VIDEO)
                else:
                    first_image_path = await save_file_async(first_image_file, MediaType.IMAGE)

            last_image_path = ""
            if last_image_file and last_image_file.filename:
                if self._is_video_upload(last_image_file):
                    raise HTTPException(status_code=400, detail="last_image_file must be an image upload")
                last_image_path = await save_file_async(last_image_file, MediaType.IMAGE)

            if first_image_path:
                task_payload["first_image_path"] = first_image_path
            if last_image_path:
                task_payload["last_image_path"] = last_image_path
            if ref_video_path:
                task_payload["ref_video_path"] = ref_video_path

            resolved_task = self._resolve_form_task(
                requested_task=str(task_payload.get("task") or ""),
                first_image_path=first_image_path,
                last_image_path=last_image_path,
                ref_video_path=ref_video_path,
            )
            task_payload["task"] = resolved_task

            message = TaskRequest(**task_payload)
            return await self.api.task_app_service.submit(
                message,
                explicit_fields=set(task_payload),
                validate_inputs=self._validate_message_inputs,
            )
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to create form task: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _collect_form_task_payload(self, request: Request) -> dict[str, Any]:
        """Collect non-file multipart fields into a TaskRequest-compatible payload."""
        form = await request.form()
        task_payload: dict[str, Any] = {}
        file_field_names = {"first_image_file", "last_image_file"}

        for key, value in form.multi_items():
            if key in file_field_names or isinstance(value, (UploadFile, StarletteUploadFile)):
                continue
            if value in (None, ""):
                continue

            coerced_value = self._coerce_form_value(value)
            if key in task_payload:
                existing_value = task_payload[key]
                if isinstance(existing_value, list):
                    existing_value.append(coerced_value)
                else:
                    task_payload[key] = [existing_value, coerced_value]
            else:
                task_payload[key] = coerced_value

        return task_payload

    def _coerce_form_value(self, value: Any) -> Any:
        """Coerce multipart string values into basic JSON-compatible scalar types."""
        if not isinstance(value, str):
            return value

        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _resolve_form_task(
        self,
        *,
        requested_task: str,
        first_image_path: str,
        last_image_path: str,
        ref_video_path: str,
    ) -> str:
        """Resolve the task for form-based submissions.

        If the caller specifies a task, validate its required inputs.
        Otherwise infer a task from uploaded inputs and the currently loaded pipeline capabilities.
        """
        if requested_task.strip():
            task = validate_task_name_format(requested_task)
            self.api.validate_task_supported(task)
            self._validate_task_inputs(
                task,
                first_image_path=first_image_path,
                last_image_path=last_image_path,
                ref_video_path=ref_video_path,
            )
            return task

        candidate_tasks = self._get_candidate_tasks_for_inputs(
            first_image_path=first_image_path,
            last_image_path=last_image_path,
            ref_video_path=ref_video_path,
        )
        supported_tasks = self.api.get_supported_tasks()

        if candidate_tasks:
            return candidate_tasks[0]

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Could not infer a compatible task for the uploaded inputs",
                "candidate_tasks": list(candidate_tasks),
                "supported_tasks": list(supported_tasks),
            },
        )

    def _get_candidate_tasks_for_inputs(
        self,
        *,
        first_image_path: str,
        last_image_path: str,
        ref_video_path: str,
    ) -> tuple[str, ...]:
        """Return compatible task candidates based on provided form inputs."""
        available_inputs = self._collect_available_inputs(
            first_image_path=first_image_path,
            last_image_path=last_image_path,
            ref_video_path=ref_video_path,
        )
        supported_tasks = self.api.get_supported_tasks()
        if supported_tasks:
            return tuple(
                match_task_candidates(
                    supported_tasks,
                    get_task_contract=self.api.get_task_contract,
                    available_inputs=available_inputs,
                )
            )

        fallback_tasks = ("t2v", "t2i", "i2v", "i2i", "fl2v", "vc", "vsr")
        return tuple(
            match_task_candidates(
                fallback_tasks,
                get_task_contract=lambda task: default_task_contract(task).to_metadata(),
                available_inputs=available_inputs,
            )
        )

    def _validate_task_inputs(
        self,
        task: str,
        *,
        first_image_path: str,
        last_image_path: str,
        ref_video_path: str,
    ) -> None:
        """Validate minimum required inputs for a task."""
        available_inputs = self._collect_available_inputs(
            first_image_path=first_image_path,
            last_image_path=last_image_path,
            ref_video_path=ref_video_path,
        )
        contract = self._get_task_contract(task)
        required_inputs = tuple(contract.get("required_inputs", ())) if contract else tuple()
        missing_inputs = [input_name for input_name in required_inputs if input_name not in available_inputs]

        if missing_inputs:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Task '{task}' is missing required inputs",
                    "required_inputs": list(required_inputs),
                    "missing_inputs": missing_inputs,
                },
            )

        if contract is None:
            if task in {"i2i", "i2v"} and not first_image_path:
                raise HTTPException(status_code=400, detail=f"Task '{task}' requires first_image_path")
            if task == "fl2v" and (not first_image_path or not last_image_path):
                raise HTTPException(
                    status_code=400,
                    detail="Task 'fl2v' requires both first_image_path and last_image_path",
                )
            if task in {"vc", "vsr"} and not ref_video_path:
                raise HTTPException(status_code=400, detail=f"Task '{task}' requires ref_video_path")

    def _validate_message_inputs(self, message: TaskRequest, contract: dict[str, Any] | None) -> None:
        self._validate_task_inputs(
            message.task,
            first_image_path=message.first_image_path,
            last_image_path=message.last_image_path,
            ref_video_path=message.ref_video_path,
        )

    def _collect_available_inputs(
        self,
        *,
        first_image_path: str,
        last_image_path: str,
        ref_video_path: str,
    ) -> set[str]:
        """Collect provided input names for task inference and validation."""
        available_inputs: set[str] = set()
        if first_image_path:
            available_inputs.add("first_image_path")
        if last_image_path:
            available_inputs.add("last_image_path")
        if ref_video_path:
            available_inputs.add("ref_video_path")
        return available_inputs

    def _get_task_contract(self, task: str) -> dict[str, Any] | None:
        """Get task-level contract metadata from the active pipeline, if available."""
        return self.api.get_task_contract(task)

    def _is_video_upload(self, file: UploadFile) -> bool:
        """Whether an uploaded form file should be treated as video input."""
        content_type = (file.content_type or "").lower()
        suffix = Path(file.filename or "").suffix.lower()
        return content_type.startswith("video/") or suffix in _VIDEO_FILE_EXTENSIONS


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup routes with ApiServer instance."""
    return create_router(api_server)
