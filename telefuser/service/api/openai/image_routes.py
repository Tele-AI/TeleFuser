"""
OpenAI Compatible Image API Routes

Provides OpenAI-compatible REST API endpoints for image generation:
- POST /v1/images/generations - Generate images from text prompts
- POST /v1/images/edits - Edit images (I2I)
- GET /v1/images/{image_id}/content - Download generated image

Reference: https://platform.openai.com/docs/api-reference/images
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from telefuser.service.api.schema import TaskRequest
from telefuser.service.api.task_contract_runtime import (
    apply_task_contract_defaults,
    map_contract_fields,
    match_task_candidates,
    validate_required_task_parameters,
)
from telefuser.service.core.task_manager import TaskManager, TaskStatus
from telefuser.utils.logging import logger

from .adapter import OpenAIRequestAdapter, OpenAIResponseAdapter
from .protocol import ErrorResponse, ImageEditRequest, ImageGenerationsRequest, ImageResponse

if TYPE_CHECKING:
    from ..api_server import ApiServer


def create_router(api_server: ApiServer) -> APIRouter:
    """Create a new router with fresh routes for the given ApiServer instance."""
    router = APIRouter(prefix="/v1/images", tags=["images"])
    routes = ImageRoutes(api_server)

    @router.post("/generations", response_model=ImageResponse)
    async def create_image_generation(request: ImageGenerationsRequest) -> ImageResponse:
        """Generate an image from a text prompt (OpenAI compatible)."""
        return await routes.create_image_generation(request)

    @router.post("/edits", response_model=ImageResponse)
    async def create_image_edit(
        request: Request,
        image: UploadFile | None = File(None, description="The image to edit"),
        image_url: str | None = Form(None, description="URL of the image to edit"),
        prompt: str = Form(..., description="A text description of the desired image(s)"),
        mask: UploadFile | None = File(None, description="An additional image for masking"),
        model: str | None = Form(None, description="The model to use for image editing"),
        n: int | None = Form(1, description="The number of images to generate"),
        size: str | None = Form("1024x1024", description="The size of the generated images"),
        response_format: str | None = Form("url", description="The format of the response"),
        seed: int | None = Form(42, description="Random seed"),
        negative_prompt: str | None = Form(None, description="Negative prompt"),
    ) -> ImageResponse:
        """Edit an image based on a prompt (OpenAI compatible)."""
        return await routes.create_image_edit(
            request=request,
            image=image,
            image_url=image_url,
            prompt=prompt,
            mask=mask,
            model=model,
            n=n,
            size=size,
            response_format=response_format,
            seed=seed,
            negative_prompt=negative_prompt,
        )

    @router.get("/{image_id}/content")
    async def get_image_content(image_id: str) -> FileResponse:
        """Download a generated image by its ID."""
        return await routes.get_image_content(image_id)

    return router


class ImageRoutes:
    """Image API route handlers with dependency injection support."""

    DEFAULT_TIMEOUT = 300.0  # Default timeout for sync generation (seconds)
    POLL_INTERVAL = 0.5  # Polling interval for task status checks (seconds)

    def __init__(self, api_server: ApiServer) -> None:
        """Initialize with ApiServer instance."""
        self.api = api_server
        self.task_manager: TaskManager | None = getattr(api_server, "task_manager", None)
        self.media_service = getattr(api_server, "media_service", None)

    @property
    def file_service(self) -> Any | None:
        """Return the current file service from the API server.

        The ApiServer installs routes before initialize_services() wires the
        file service, so route handlers must not keep the initial None value.
        """
        return getattr(self.api, "file_service", None)

    async def create_image_generation(self, request: ImageGenerationsRequest) -> ImageResponse:
        """Handle image generation request with synchronous waiting."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        if request.n and request.n > 1:
            logger.warning(f"Multiple image generation (n={request.n}) not supported, using n=1")

        try:
            task_type = self._resolve_image_task(has_reference=False)
            task_request = OpenAIRequestAdapter.to_task_request(request, task_type=task_type)
            self.api.validate_task_supported(task_request.task)
            contract = self.api.get_task_contract(task_request.task)
            apply_task_contract_defaults(
                task_request,
                task_contract=contract,
                explicit_fields=self._get_image_generation_explicit_fields(request),
            )
            validate_required_task_parameters(task_request, task_contract=contract)
            task_id = self.task_manager.create_task(task_request)
            logger.info(f"Created image generation task: {task_id}")

            await self._ensure_processing()

            result = await self._wait_for_task_completion(task_id=task_id, timeout=self.DEFAULT_TIMEOUT)

            if result is None:
                self.task_manager.cancel_task(task_id)
                raise HTTPException(status_code=500, detail="Image generation failed or was cancelled")

            output_path = result.get("output_path")
            if not output_path:
                raise HTTPException(status_code=500, detail="Generation completed but no output path found")

            peak_memory_mb = result.get("peak_memory_mb")
            inference_time_s = result.get("inference_time_s")
            base_url = self._get_base_url()

            response = OpenAIResponseAdapter.to_image_response(
                output_path=output_path,
                prompt=request.prompt,
                response_format=request.response_format or "url",
                base_url=base_url,
                task_id=task_id,
                peak_memory_mb=peak_memory_mb,
                inference_time_s=inference_time_s,
            )

            logger.info(f"Image generation completed: {task_id}")
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Image generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

    async def create_image_edit(
        self,
        request: Request,
        image: UploadFile | None,
        image_url: str | None,
        prompt: str,
        mask: UploadFile | None,
        model: str | None,
        n: int | None,
        size: str | None,
        response_format: str | None,
        seed: int | None,
        negative_prompt: str | None,
    ) -> ImageResponse:
        """Handle image edit request."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        if not image and not image_url:
            raise HTTPException(status_code=422, detail="Either 'image' file or 'image_url' must be provided")

        try:
            image_path = ""
            if image and image.filename:
                image_path = await self._save_uploaded_file(image, "input_image")
            elif image_url:
                image_path = image_url

            mask_path = ""
            if mask and mask.filename:
                mask_path = await self._save_uploaded_file(mask, "mask")

            edit_request = ImageEditRequest(
                prompt=prompt,
                image=image_path if not image_url else None,
                image_url=image_url if image_url else None,
                mask=mask_path if mask_path else None,
                model=model,
                n=n,
                size=size,
                response_format=response_format,
                seed=seed,
                negative_prompt=negative_prompt,
            )

            task_type = self._resolve_image_task(has_reference=True)
            task_request = OpenAIRequestAdapter.to_task_request(edit_request, task_type=task_type)
            self.api.validate_task_supported(task_request.task)
            contract = self.api.get_task_contract(task_request.task)
            apply_task_contract_defaults(
                task_request,
                task_contract=contract,
                explicit_fields=self._get_image_edit_explicit_fields(
                    prompt=prompt,
                    image_path=image_path,
                    image_url=image_url,
                    mask_path=mask_path,
                    model=model,
                    n=n,
                    size=size,
                    seed=seed,
                    negative_prompt=negative_prompt,
                ),
            )
            validate_required_task_parameters(task_request, task_contract=contract)
            task_id = self.task_manager.create_task(task_request)
            logger.info(f"Created image edit task: {task_id}")

            await self._ensure_processing()

            result = await self._wait_for_task_completion(task_id=task_id, timeout=self.DEFAULT_TIMEOUT)

            if result is None:
                self.task_manager.cancel_task(task_id)
                raise HTTPException(status_code=500, detail="Image edit failed")

            output_path = result.get("output_path")
            if not output_path:
                raise HTTPException(status_code=500, detail="No output from generation")

            base_url = self._get_base_url()
            response = OpenAIResponseAdapter.to_image_response(
                output_path=output_path,
                prompt=prompt,
                response_format=response_format or "url",
                base_url=base_url,
                task_id=task_id,
            )

            logger.info(f"Image edit completed: {task_id}")
            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Image edit failed: {e}")
            raise HTTPException(status_code=500, detail=f"Image edit failed: {str(e)}")

    async def get_image_content(self, image_id: str) -> FileResponse:
        """Retrieve generated image by ID."""
        if not self.task_manager:
            raise HTTPException(status_code=503, detail="Task manager not initialized")

        if not self.file_service:
            raise HTTPException(status_code=503, detail="File service not initialized")

        task_info = self.task_manager.get_task(image_id)
        if not task_info:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

        output_path = task_info.output_path
        if not output_path:
            raise HTTPException(status_code=404, detail=f"Image {image_id} has no output path")

        path = Path(output_path)
        if not path.is_absolute() and not path.exists():
            output_dir = getattr(self.file_service, "output_image_dir", None)
            if output_dir:
                path = output_dir / path

        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {path}")

        suffix = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/png")

        return FileResponse(path=str(path), media_type=media_type, filename=path.name)

    async def _ensure_processing(self) -> None:
        """Ensure the task processor is running."""
        await self.api.ensure_task_processor_running()

    def _get_base_url(self) -> str:
        """Get the base URL for constructing image URLs."""
        server_config = getattr(self.api, "server_config", None)
        if server_config:
            host = getattr(server_config, "host", "localhost")
            port = getattr(server_config, "port", 8000)
            return f"http://{host}:{port}"
        return "http://localhost:8000"

    async def _wait_for_task_completion(self, task_id: str, timeout: float) -> dict[str, Any] | None:
        """Wait for a task to complete with timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.task_manager.get_task_status(task_id)

            if not status:
                raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

            task_status = status.get("status")

            if task_status == TaskStatus.COMPLETED.value:
                return {
                    "output_path": status.get("output_path"),
                    "error": status.get("error"),
                }
            elif task_status == TaskStatus.FAILED.value:
                error_msg = status.get("error", "Unknown error")
                raise HTTPException(status_code=500, detail=f"Generation failed: {error_msg}")
            elif task_status == TaskStatus.CANCELLED.value:
                raise HTTPException(status_code=400, detail="Generation was cancelled")

            await asyncio.sleep(self.POLL_INTERVAL)

        raise HTTPException(status_code=504, detail=f"Generation timeout after {timeout} seconds")

    async def _save_uploaded_file(self, file: UploadFile, prefix: str = "upload") -> str:
        """Save an uploaded file to disk."""
        import uuid

        if not self.file_service:
            raise HTTPException(status_code=503, detail="File service not initialized")

        input_dir = getattr(self.file_service, "input_image_dir", None)
        if not input_dir:
            input_dir = Path("/tmp/telefuser/inputs")
            input_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(file.filename or "image.png").suffix
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        file_path = input_dir / filename

        content = await file.read()
        await asyncio.to_thread(self._write_file_sync, file_path, content)

        return str(file_path)

    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        """Write file synchronously."""
        file_path.write_bytes(content)

    def _resolve_image_task(self, *, has_reference: bool) -> str:
        """Resolve the best image task supported by the current pipeline."""
        available_inputs = {"first_image_path"} if has_reference else set()
        candidates = match_task_candidates(
            self.api.get_supported_tasks(),
            get_task_contract=self.api.get_task_contract,
            available_inputs=available_inputs,
            media_type="image",
        )
        if candidates:
            return candidates[0]
        return "i2i" if has_reference else "t2i"

    def _get_image_generation_explicit_fields(self, request: ImageGenerationsRequest) -> set[str]:
        return map_contract_fields(
            set(getattr(request, "model_fields_set", set())),
            {
                "prompt": "prompt",
                "model": "model",
                "n": "n",
                "quality": "quality",
                "size": "resolution",
                "style": "style",
                "seed": "seed",
                "negative_prompt": "negative_prompt",
            },
        )

    def _get_image_edit_explicit_fields(
        self,
        *,
        prompt: str,
        image_path: str,
        image_url: str | None,
        mask_path: str,
        model: str | None,
        n: int | None,
        size: str | None,
        seed: int | None,
        negative_prompt: str | None,
    ) -> set[str]:
        explicit_fields = {"prompt", "first_image_path"}
        if mask_path:
            explicit_fields.add("mask")
        if model is not None:
            explicit_fields.add("model")
        if n is not None:
            explicit_fields.add("n")
        if size not in (None, ""):
            explicit_fields.add("resolution")
        if seed is not None:
            explicit_fields.add("seed")
        if negative_prompt not in (None, ""):
            explicit_fields.add("negative_prompt")
        if image_url or image_path:
            explicit_fields.add("first_image_path")
        return explicit_fields


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup image routes."""
    return create_router(api_server)
