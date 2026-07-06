"""API Server for TeleFuser Service."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from telefuser.utils.logging import logger

from ..core.file_service import FileService
from ..core.pipeline_service import PipelineService
from ..core.stream_pipeline_service import StreamPipelineService
from ..core.task_manager import TaskManager, TaskStatus
from ..core.task_processor import AsyncTaskProcessor
from ..core.task_service import MediaGenerationService
from . import routers


class ApiServer:
    """Main API server coordinating all services.

    Manages FastAPI app lifecycle, route setup, task processing loop,
    and OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        max_queue_size: int = 10,
        max_concurrent_tasks: int = 1,
        configured_max_concurrent_tasks: int | None = None,
        app: FastAPI | None = None,
        task_manager: TaskManager | None = None,
        enable_rate_limit: bool = True,
        enable_logging: bool = False,
        enable_openai_api: bool = True,
    ) -> None:
        self.app = app or FastAPI(
            title="TeleFuser API",
            description="API for video and image generation using TeleFuser framework. "
            "OpenAI compatible endpoints available at /v1/images and /v1/videos.",
            version="1.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )
        self.file_service: FileService | None = None
        self.inference_service: PipelineService | None = None
        self.stream_service: StreamPipelineService | None = None
        self._webrtc_routes: object | None = None
        self.media_service: MediaGenerationService | None = None
        self.cache_service: Any | None = None
        self.max_queue_size = max_queue_size
        self.max_concurrent_tasks = max_concurrent_tasks
        self.configured_max_concurrent_tasks = configured_max_concurrent_tasks or max_concurrent_tasks
        self._task_manager = task_manager
        self.enable_openai_api = enable_openai_api

        self.task_processor: AsyncTaskProcessor | None = None
        self._task_processor_lock = asyncio.Lock()

        self._setup_routes()

        from .middleware import setup_middleware

        setup_middleware(self.app, enable_rate_limit=enable_rate_limit, enable_logging=enable_logging)

    @property
    def task_manager(self):
        """Get task manager (supports dependency injection)."""
        if self._task_manager is not None:
            return self._task_manager

    def _setup_routes(self) -> None:
        """Setup all API routes."""
        tasks_router = routers.tasks.setup_routes(self)
        files_router = routers.files.setup_routes(self)
        service_router = routers.service.setup_routes(self)

        self.app.include_router(tasks_router)
        self.app.include_router(files_router)
        self.app.include_router(service_router)

        stream_router = routers.setup_stream_routes(self)
        self.app.include_router(stream_router)

        if routers.setup_webrtc_routes is not None:
            try:
                webrtc_router = routers.setup_webrtc_routes(self)
                self.app.include_router(webrtc_router)
                logger.info("WebRTC routes enabled at /v1/stream/webrtc")
            except Exception as e:
                logger.info(f"WebRTC routes not available: {e}")

        if self.enable_openai_api:
            try:
                from .openai import image_routes, video_routes

                image_router = image_routes.setup_routes(self)
                video_router = video_routes.setup_routes(self)

                self.app.include_router(image_router)
                self.app.include_router(video_router)

                logger.info("OpenAI compatible API enabled at /v1/images and /v1/videos")
            except Exception as e:
                logger.warning(f"Failed to setup OpenAI routes: {e}")

    def _write_file_sync(self, file_path: Path, content: bytes) -> None:
        with open(file_path, "wb") as buffer:
            buffer.write(content)

    def _stream_file_response(self, file_path: Path, filename: str | None = None) -> StreamingResponse:
        """Stream file response with proper security checks."""
        assert self.file_service is not None, "File service is not initialized"

        try:
            resolved_path = file_path.resolve()

            # Security: ensure file is within allowed directories
            if not str(resolved_path).startswith(str(self.file_service.output_video_dir.resolve())):
                raise HTTPException(status_code=403, detail="Access to this file is not allowed")

            if not resolved_path.exists() or not resolved_path.is_file():
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

            file_size = resolved_path.stat().st_size
            actual_filename = filename or resolved_path.name

            mime_type = "application/octet-stream"
            if actual_filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                mime_type = "video/mp4"
            elif actual_filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                mime_type = "image/jpeg"

            headers = {
                "Content-Disposition": f'attachment; filename="{actual_filename}"',
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }

            def file_stream_generator(file_path: str, chunk_size: int = 1024 * 1024):
                with open(file_path, "rb") as file:
                    while chunk := file.read(chunk_size):
                        yield chunk

            return StreamingResponse(
                file_stream_generator(str(resolved_path)),
                media_type=mime_type,
                headers=headers,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error occurred while processing file stream response: {e}")
            raise HTTPException(status_code=500, detail="File transfer failed")

    async def _validate_image_url(self, image_url: str) -> bool:
        """Validate image URL is accessible."""
        from ..core.config import server_config

        if not image_url or not image_url.startswith("http"):
            return True

        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return False

            timeout = httpx.Timeout(connect=5.0, read=5.0)
            verify = server_config.ssl_cert_path if server_config.ssl_cert_path else server_config.verify_ssl
            async with httpx.AsyncClient(verify=verify, timeout=timeout) as client:
                response = await client.head(image_url, follow_redirects=True)
                return response.status_code < 400
        except Exception as e:
            logger.warning(f"URL validation failed for {image_url}: {str(e)}")
            return False

    async def ensure_task_processor_running(self) -> None:
        """Ensure the async task processor is running."""
        if self.task_processor is None:
            logger.warning("Task processor is not initialized; task will remain pending until services are ready")
            return

        if self.task_processor.is_running:
            return

        async with self._task_processor_lock:
            if self.task_processor is None:
                logger.warning("Task processor is not initialized; task will remain pending until services are ready")
                return
            if self.task_processor.is_running:
                return
            await self.task_processor.start()

    def get_supported_tasks(self) -> tuple[str, ...]:
        """Get tasks supported by the loaded pipeline contract, if available."""
        if self.inference_service is None:
            return tuple()

        supports = getattr(self.inference_service, "supported_tasks", None)
        if callable(supports):
            try:
                tasks = supports()
                if isinstance(tasks, (list, tuple, set, frozenset)):
                    return tuple(str(task) for task in tasks)
            except Exception as e:
                logger.warning(f"Failed to query supported tasks from inference service: {e}")

        metadata_fn = getattr(self.inference_service, "server_metadata", None)
        if callable(metadata_fn):
            try:
                metadata = metadata_fn()
                tasks = metadata.get("supported_tasks") or []
                return tuple(str(task) for task in tasks)
            except Exception as e:
                logger.warning(f"Failed to query supported tasks from inference service metadata: {e}")

        return tuple()

    def validate_task_supported(self, task: str) -> None:
        """Validate that the current pipeline supports the requested task."""
        supported_tasks = self.get_supported_tasks()
        if supported_tasks and task not in supported_tasks:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Task '{task}' is not supported by the current pipeline",
                    "supported_tasks": list(supported_tasks),
                },
            )

    def get_task_contract(self, task: str) -> dict[str, Any] | None:
        """Get task-level contract metadata for the current pipeline, if available."""
        if self.inference_service is None:
            return None

        getter = getattr(self.inference_service, "get_task_contract", None)
        if callable(getter):
            try:
                contract = getter(task)
                if isinstance(contract, dict):
                    return contract
                if contract is not None and hasattr(contract, "to_metadata"):
                    metadata = contract.to_metadata()
                    if isinstance(metadata, dict):
                        return metadata
            except Exception as e:
                logger.warning(f"Failed to query task contract from inference service: {e}")

        metadata_fn = getattr(self.inference_service, "server_metadata", None)
        if callable(metadata_fn):
            try:
                metadata = metadata_fn()
                task_contracts = metadata.get("task_contracts") or {}
                contract = task_contracts.get(task)
                if isinstance(contract, dict):
                    return contract
            except Exception as e:
                logger.warning(f"Failed to query task contract from inference service metadata: {e}")
        return None

    def initialize_services(
        self,
        cache_dir: Path,
        inference_service: PipelineService,
        cache_service: Any | None = None,
        cache_adapter: Any | None = None,
    ) -> None:
        """Initialize file and media services."""
        self.file_service = FileService(cache_dir)
        self.inference_service = inference_service
        self.cache_service = cache_service
        self.cache_adapter = cache_adapter
        self.media_service = MediaGenerationService(
            self.file_service,
            inference_service,
            cache_service=cache_service,
            cache_adapter=cache_adapter,
        )
        self.task_processor = AsyncTaskProcessor(
            task_manager=self.task_manager,
            media_service=self.media_service,
            max_concurrent=self.max_concurrent_tasks,
        )

    def initialize_stream_service(self, stream_service: StreamPipelineService) -> None:
        """Initialize stream pipeline service for stream-mode endpoints."""
        self.stream_service = stream_service
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def cleanup(self) -> None:
        """Cleanup resources and stop processing workers."""
        if self.task_processor is not None:
            await self.task_processor.stop()

        if self.stream_service is not None:
            await self.stream_service.aclose()

        if self._webrtc_routes is not None:
            await self._webrtc_routes.cleanup()

        if self.file_service:
            cleanup = getattr(self.file_service, "cleanup", None)
            if cleanup is not None:
                result = cleanup()
                if inspect.isawaitable(result):
                    await result

        if getattr(self, "cache_service", None) is not None:
            try:
                self.cache_service.shutdown()
            except Exception as exc:
                logger.warning(f"cache service shutdown failed: {exc}")

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app
