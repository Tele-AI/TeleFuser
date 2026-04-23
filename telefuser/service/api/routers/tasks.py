"""
Task Routes for TeleFuser API

Provides route handlers for task-related endpoints.
Routes are defined as class methods for better testability.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from telefuser.service.core.pipeline_contract import default_task_contract, validate_task_name_format
from telefuser.service_types import AspectRatio, OutputFormat, StopTaskStatus, TaskStatus
from telefuser.utils.logging import logger

from ..schema import StopTaskResponse, TaskRequest, TaskResponse
from ..task_contract_runtime import (
    apply_task_contract_defaults,
    match_task_candidates,
    validate_required_task_parameters,
)

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
        first_image_file: UploadFile | None = File(default=None, description="First frame image file"),
        last_image_file: UploadFile | None = File(default=None, description="Last frame image file"),
        prompt: str = Form(default="", description="Generation prompt"),
        task: str = Form(default="", description="Optional explicit task name"),
        output_path: str = Form(default="", description="Custom output path"),
        negative_prompt: str = Form(default="", description="Negative prompt"),
        target_video_length: int = Form(default=5, description="Video length in seconds (for video tasks)"),
        seed: int = Form(default=42, description="Random seed"),
        aspect_ratio: AspectRatio = Form(
            default=AspectRatio.RATIO_16_9, description="Aspect ratio (16:9, 9:16, 4:3, etc.)"
        ),
        output_format: OutputFormat = Form(
            default=OutputFormat.PNG, description="Output format (png, jpg, webp for images)"
        ),
    ) -> TaskResponse:
        """Create task with file upload support."""
        return await routes.create_task_form(
            first_image_file=first_image_file,
            last_image_file=last_image_file,
            prompt=prompt,
            task=task,
            output_path=output_path,
            negative_prompt=negative_prompt,
            target_video_length=target_video_length,
            seed=seed,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
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
            self.api.validate_task_supported(message.task)
            contract = self._get_task_contract(message.task)
            apply_task_contract_defaults(
                message,
                task_contract=contract,
                explicit_fields=set(getattr(message, "model_fields_set", set())),
            )
            validate_required_task_parameters(message, task_contract=contract)
            await check_image_path("first_image_path")
            await check_image_path("last_image_path")
            self._validate_task_inputs(
                message.task,
                first_image_path=message.first_image_path,
                last_image_path=message.last_image_path,
                ref_video_path=message.ref_video_path,
            )
            task_id = self.api.task_manager.create_task(message)
            message.task_id = task_id
            await self.api.ensure_task_processor_running()
            return TaskResponse(task_id=task_id, task_status=TaskStatus.PENDING, output_path=message.output_path)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
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
            if self.api.task_manager.cancel_task(task_id):
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Task {task_id} stopped successfully.")
                return StopTaskResponse(stop_status=StopTaskStatus.SUCCESS, reason="Task stopped successfully.")
            else:
                return StopTaskResponse(
                    stop_status=StopTaskStatus.DO_NOTHING, reason="Task not found or already completed."
                )
        except Exception as e:
            logger.error(f"Error occurred while stopping task {task_id}: {str(e)}")
            return StopTaskResponse(stop_status=StopTaskStatus.ERROR, reason=str(e))

    async def create_task_form(
        self,
        first_image_file: UploadFile | None = None,
        last_image_file: UploadFile | None = None,
        prompt: str | None = None,
        task: str = "",
        output_path: str | None = None,
        negative_prompt: str | None = None,
        target_video_length: int | None = None,
        seed: int | None = None,
        aspect_ratio: AspectRatio = AspectRatio.RATIO_16_9,
        output_format: OutputFormat = OutputFormat.PNG,
    ) -> TaskResponse:
        """Create task with file upload support."""
        assert self.api.file_service is not None, "File service is not initialized"

        async def save_file_async(file: UploadFile, target_dir: Path) -> str:
            if not file or not file.filename:
                return ""

            file_extension = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = target_dir / unique_filename

            content = await file.read()
            await asyncio.to_thread(self._write_file_sync, file_path, content)

            return str(file_path)

        first_image_path = ""
        ref_video_path = ""
        if first_image_file and first_image_file.filename:
            if self._is_video_upload(first_image_file):
                ref_video_path = await save_file_async(first_image_file, self.api.file_service.input_video_dir)
            else:
                first_image_path = await save_file_async(first_image_file, self.api.file_service.input_image_dir)

        last_image_path = ""
        if last_image_file and last_image_file.filename:
            if self._is_video_upload(last_image_file):
                raise HTTPException(status_code=400, detail="last_image_file must be an image upload")
            last_image_path = await save_file_async(last_image_file, self.api.file_service.input_image_dir)

        try:
            resolved_task = self._resolve_form_task(
                requested_task=task,
                first_image_path=first_image_path,
                last_image_path=last_image_path,
                ref_video_path=ref_video_path,
            )

            task_payload: dict[str, Any] = {"task": resolved_task}
            if prompt not in (None, ""):
                task_payload["prompt"] = prompt
            if negative_prompt not in (None, ""):
                task_payload["negative_prompt"] = negative_prompt
            if first_image_path:
                task_payload["first_image_path"] = first_image_path
            if last_image_path:
                task_payload["last_image_path"] = last_image_path
            if ref_video_path:
                task_payload["ref_video_path"] = ref_video_path
            if output_path not in (None, ""):
                task_payload["output_path"] = output_path
            if target_video_length is not None:
                task_payload["target_video_length"] = target_video_length
            if seed is not None:
                task_payload["seed"] = seed
            if aspect_ratio not in (None, ""):
                task_payload["aspect_ratio"] = aspect_ratio
            if output_format not in (None, ""):
                task_payload["output_format"] = output_format

            message = TaskRequest(**task_payload)
            self.api.validate_task_supported(message.task)
            contract = self.api.get_task_contract(message.task)
            apply_task_contract_defaults(message, task_contract=contract, explicit_fields=set(task_payload))
            validate_required_task_parameters(message, task_contract=contract)
            self._validate_task_inputs(
                message.task,
                first_image_path=message.first_image_path,
                last_image_path=message.last_image_path,
                ref_video_path=message.ref_video_path,
            )
            task_id = self.api.task_manager.create_task(message)
            message.task_id = task_id
            await self.api.ensure_task_processor_running()

            return TaskResponse(task_id=task_id, task_status=TaskStatus.PENDING, output_path=message.output_path)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        except HTTPException:
            raise
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to create form task: {e}")
            raise HTTPException(status_code=500, detail=str(e))

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

    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        """Write file synchronously."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as buffer:
            buffer.write(content)


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup routes with ApiServer instance."""
    return create_router(api_server)
