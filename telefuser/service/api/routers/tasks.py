"""
Task Routes for TeleFuser API

Provides route handlers for task-related endpoints.
Routes are defined as class methods for better testability.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from telefuser.service_types import AspectRatio, OutputFormat, StopTaskStatus, TaskStatus
from telefuser.utils.logging import logger

from ..schema import StopTaskResponse, TaskRequest, TaskResponse

if TYPE_CHECKING:
    from ..api_server import ApiServer

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


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
            await check_image_path("first_image_path")
            await check_image_path("last_image_path")
            task_id = self.api.task_manager.create_task(message)
            message.task_id = task_id
            self.api._ensure_processing_thread_running()
            return TaskResponse(task_id=task_id, task_status=TaskStatus.PENDING, output_path=message.output_path)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
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
        prompt: str = "",
        output_path: str = "",
        negative_prompt: str = "",
        target_video_length: int = 5,
        seed: int = 42,
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
        if first_image_file and first_image_file.filename:
            first_image_path = await save_file_async(first_image_file, self.api.file_service.input_image_dir)

        last_image_path = ""
        if last_image_file and last_image_file.filename:
            last_image_path = await save_file_async(last_image_file, self.api.file_service.input_image_dir)

        message = TaskRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            first_image_path=first_image_path,
            last_image_path=last_image_path,
            output_path=output_path,
            target_video_length=target_video_length,
            seed=seed,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
        )

        try:
            task_id = self.api.task_manager.create_task(message)
            message.task_id = task_id
            self.api._ensure_processing_thread_running()

            return TaskResponse(task_id=task_id, task_status=TaskStatus.PENDING, output_path=message.output_path)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to create form task: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        """Write file synchronously."""
        with open(file_path, "wb") as buffer:
            buffer.write(content)


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup routes with ApiServer instance."""
    return create_router(api_server)
