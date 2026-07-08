"""
OpenAI Compatible Video API Routes

Provides OpenAI-compatible REST API endpoints for video generation:
- POST /v1/videos - Create video generation task (async)
- GET /v1/videos - List video generation tasks
- GET /v1/videos/{video_id} - Get video generation status
- DELETE /v1/videos/{video_id} - Cancel/delete video generation
- GET /v1/videos/{video_id}/content - Download generated video

Reference: OpenAI Video API (https://platform.openai.com/docs/api-reference/videos)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from telefuser.service.api.task_contract_runtime import (
    map_contract_fields,
    match_task_candidates,
)
from telefuser.service.core.pipeline_contract import is_video_task
from telefuser.service.core.task_manager import TaskManager, TaskStatus
from telefuser.service_types import MediaType
from telefuser.utils.logging import logger

from .adapter import OpenAIRequestAdapter, OpenAIResponseAdapter, is_probable_video_reference
from .protocol import ErrorResponse, VideoGenerationsRequest, VideoListResponse, VideoResponse

if TYPE_CHECKING:
    from ..api_server import ApiServer


def create_router(api_server: ApiServer) -> APIRouter:
    """Create a new router with fresh routes for the given ApiServer instance."""
    router = APIRouter(prefix="/v1/videos", tags=["videos"])
    routes = VideoRoutes(api_server)

    @router.post("", response_model=VideoResponse)
    async def create_video(
        request: Request,
        prompt: str | None = Form(None),
        input_reference: UploadFile | None = File(None, description="Input image/video file"),
        reference_url: str | None = Form(None, description="URL of input reference"),
        model: str | None = Form(None),
        seconds: int | None = Form(4),
        size: str | None = Form(None),
        seed: int | None = Form(1024),
        negative_prompt: str | None = Form(None),
        output_path: str | None = Form(None),
    ) -> VideoResponse:
        """Create a video generation task (OpenAI compatible, async)."""
        content_type = request.headers.get("content-type", "").lower()

        if "application/json" in content_type:
            body = await request.json()
            try:
                req = VideoGenerationsRequest(**body)
            except Exception as e:
                raise HTTPException(status_code=422, detail=str(e))
            explicit_source_fields = set(getattr(req, "model_fields_set", set()))
        else:
            input_path = None
            if input_reference and input_reference.filename:
                input_path = await routes._save_uploaded_file(input_reference, "input")
            elif reference_url:
                input_path = reference_url

            explicit_source_fields = {"prompt"}
            if input_path:
                explicit_source_fields.add("input_reference")
            if model is not None:
                explicit_source_fields.add("model")
            if seconds is not None:
                explicit_source_fields.add("seconds")
            if size not in (None, ""):
                explicit_source_fields.add("size")
            if seed is not None:
                explicit_source_fields.add("seed")
            if negative_prompt not in (None, ""):
                explicit_source_fields.add("negative_prompt")
            if output_path not in (None, ""):
                explicit_source_fields.add("output_path")

            req = VideoGenerationsRequest(
                prompt=prompt or "",
                input_reference=input_path,
                model=model,
                seconds=seconds,
                size=size or "1024x576",
                seed=seed,
                negative_prompt=negative_prompt,
                output_path=output_path,
            )

        return await routes.create_video(req, explicit_source_fields=explicit_source_fields)

    @router.get("", response_model=VideoListResponse)
    async def list_videos(
        after: str | None = Query(None, description="Cursor for pagination"),
        limit: int | None = Query(20, ge=1, le=100, description="Number of results"),
        order: str | None = Query("desc", description="Sort order: asc or desc"),
    ) -> VideoListResponse:
        """List video generation tasks."""
        return await routes.list_videos(after=after, limit=limit, order=order)

    @router.get("/{video_id}", response_model=VideoResponse)
    async def retrieve_video(video_id: str) -> VideoResponse:
        """Get video generation status by ID."""
        return await routes.retrieve_video(video_id)

    @router.delete("/{video_id}", response_model=VideoResponse)
    async def delete_video(video_id: str) -> VideoResponse:
        """Cancel or delete a video generation task."""
        return await routes.delete_video(video_id)

    @router.get("/{video_id}/content")
    async def get_video_content(video_id: str) -> Response:
        """Download a generated video by its ID."""
        return await routes.get_video_content(video_id)

    return router


class VideoRoutes:
    """Video API route handlers with dependency injection support."""

    def __init__(self, api_server: ApiServer) -> None:
        """Initialize with ApiServer instance."""
        self.api = api_server
        self.task_manager: TaskManager | None = getattr(api_server, "task_manager", None)

    async def create_video(
        self,
        request: VideoGenerationsRequest,
        explicit_source_fields: set[str] | None = None,
    ) -> VideoResponse:
        """Handle video creation request (async - returns queued status immediately)."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        try:
            task_type = self._resolve_video_task(request)
            task_request = OpenAIRequestAdapter.to_task_request(request, task_type=task_type)
            task_response = await self.api.task_app_service.submit(
                task_request,
                explicit_fields=self._get_video_explicit_fields(
                    task_type=task_type,
                    source_fields=explicit_source_fields or set(getattr(request, "model_fields_set", set())),
                ),
                ensure_processing=False,
            )
            task_id = task_response.task_id
            logger.info(f"Created video generation task: {task_id}")

            await self._ensure_processing()

            response = OpenAIResponseAdapter.to_video_response(
                task_id=task_id,
                status=TaskStatus.PENDING.value,
                prompt=request.prompt,
                size=OpenAIRequestAdapter.resolution_to_size(task_request.resolution, media_type="video"),
                seconds=task_request.target_video_length,
                model=getattr(task_request, "model", None) or request.model or "wan-video",
                progress=0,
            )

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Video creation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Video creation failed: {str(e)}")

    async def list_videos(self, after: str | None = None, limit: int = 20, order: str = "desc") -> VideoListResponse:
        """List video generation tasks with pagination."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        try:
            all_tasks = self.task_manager.get_all_tasks()

            # Filter to video tasks only
            video_tasks = []
            for task_id, task_data in all_tasks.items():
                task_type = task_data.get("task", "")
                if is_video_task(task_type):
                    video_tasks.append((task_id, task_data))

            # Sort by start_time
            reverse = order.lower() == "desc"
            video_tasks.sort(key=lambda x: x[1].get("start_time") or "", reverse=reverse)

            # Apply cursor pagination
            if after:
                found = False
                filtered_tasks = []
                for task_id, task_data in video_tasks:
                    if found:
                        filtered_tasks.append((task_id, task_data))
                    if task_id == after:
                        found = True
                video_tasks = filtered_tasks

            video_tasks = video_tasks[:limit]

            # Convert to VideoResponse objects
            videos: List[VideoResponse] = []
            for task_id, task_data in video_tasks:
                task_info = self.task_manager.get_task(task_id)
                message = task_info.message if task_info else None
                status = task_data.get("status", TaskStatus.PENDING.value)
                video = OpenAIResponseAdapter.to_video_response_from_task(
                    task_id=task_id,
                    task_status=task_data,
                    message=message,
                    url=self._content_url_for_completed_video(task_id, status),
                    artifact_metadata=self._artifact_metadata_for_video(task_id, task_data.get("output_path")),
                )
                videos.append(video)

            has_more = len(all_tasks) > len(videos)
            first_id = videos[0].id if videos else None
            last_id = videos[-1].id if videos else None

            return OpenAIResponseAdapter.to_video_list_response(
                videos=videos, has_more=has_more, first_id=first_id, last_id=last_id
            )

        except Exception as e:
            logger.exception(f"List videos failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

    async def retrieve_video(self, video_id: str) -> VideoResponse:
        """Get video generation status."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        task_status = self.task_manager.get_task_status(video_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

        task_info = self.task_manager.get_task(video_id)
        message = task_info.message if task_info else None

        status = task_status.get("status", TaskStatus.PENDING.value)
        return OpenAIResponseAdapter.to_video_response_from_task(
            task_id=video_id,
            task_status=task_status,
            message=message,
            url=self._content_url_for_completed_video(video_id, status),
            artifact_metadata=self._artifact_metadata_for_video(video_id, task_status.get("output_path")),
        )

    async def delete_video(self, video_id: str) -> VideoResponse:
        """Cancel or delete a video generation task."""
        cancel_result = self.api.task_app_service.cancel_task(video_id)
        if cancel_result["result"] == "not_found":
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

        task_status = cancel_result["task_status"] or TaskStatus.CANCELLED.value
        video_status = OpenAIResponseAdapter.task_status_to_video_status(task_status)

        return VideoResponse(
            id=video_id,
            status=video_status,
            model="wan-video",
        )

    async def get_video_content(self, video_id: str) -> Response:
        """Download generated video."""
        return self.api.task_app_service.get_output_response(
            video_id,
            media_type=MediaType.VIDEO,
            require_completed=True,
        )

    async def _ensure_processing(self) -> None:
        """Ensure the task processor is running."""
        await self.api.ensure_task_processor_running()

    def _content_url_for_completed_video(self, video_id: str, status: str) -> str | None:
        """Return a video content URL only when the output is ready."""
        if status != TaskStatus.COMPLETED.value:
            return None
        return self.api.task_app_service.get_openai_content_url(video_id, media_type=MediaType.VIDEO)

    def _artifact_metadata_for_video(self, video_id: str, output_path: str | None) -> dict | None:
        """Return artifact metadata for a video output when it exists."""
        if not output_path:
            return None
        return self.api.task_app_service.get_output_metadata(
            video_id,
            output_path=output_path,
            media_type=MediaType.VIDEO,
        )

    async def _save_uploaded_file(self, file: UploadFile, prefix: str = "upload") -> str:
        """Save an uploaded file to disk."""
        return await self.api.task_app_service.save_upload_file(
            file,
            media_type=MediaType.VIDEO,
            prefix=prefix,
            fallback_filename="input.mp4",
        )

    def _resolve_video_task(self, request: VideoGenerationsRequest) -> str:
        """Resolve the best video task supported by the current pipeline."""
        reference_path = request.input_reference or request.reference_url or ""
        available_inputs: set[str] = set()
        fallback_task = "t2v"
        if reference_path:
            if is_probable_video_reference(reference_path):
                available_inputs.add("ref_video_path")
                fallback_task = "vc"
            else:
                available_inputs.add("first_image_path")
                fallback_task = "i2v"

        candidates = match_task_candidates(
            self.api.get_supported_tasks(),
            get_task_contract=self.api.get_task_contract,
            available_inputs=available_inputs,
            media_type="video",
        )
        if candidates:
            return candidates[0]
        return fallback_task

    def _get_video_explicit_fields(self, *, task_type: str, source_fields: set[str]) -> set[str]:
        field_mapping = {
            "prompt": "prompt",
            "model": "model",
            "seconds": "target_video_length",
            "size": "resolution",
            "seed": "seed",
            "negative_prompt": "negative_prompt",
            "output_path": "output_path",
        }
        explicit_fields = map_contract_fields(source_fields, field_mapping)
        if "input_reference" in source_fields or "reference_url" in source_fields:
            if task_type in {"vc", "vsr"}:
                explicit_fields.add("ref_video_path")
            else:
                explicit_fields.add("first_image_path")
        return explicit_fields


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup video routes."""
    return create_router(api_server)
