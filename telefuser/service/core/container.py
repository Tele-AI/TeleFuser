"""
Dependency Injection Container for TeleFuser Service

Provides centralized service management and dependency resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from telefuser.service_types import TaskType
from telefuser.utils.logging import logger

from .config import ServerConfig, server_config
from .file_service import FileService
from .pipeline_service import PipelineService
from .stream_pipeline_service import StreamPipelineService
from .task_manager import TaskManager
from .task_service import MediaGenerationService


@dataclass
class ServiceContainer:
    """Container for managing service dependencies.

    Usage:
        container = ServiceContainer.create()
        async with container:
            app = container.get_api_app()
    """

    config: ServerConfig
    task_manager: TaskManager
    file_service: FileService | None = None
    pipeline_service: PipelineService | None = None
    stream_pipeline_service: StreamPipelineService | None = None
    media_service: MediaGenerationService | None = None
    _cache_dir: Path | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        config: ServerConfig | None = None,
        cache_dir: Path | None = None,
    ) -> ServiceContainer:
        """Create a new service container with all dependencies."""
        config = config or server_config

        if config.max_concurrent_tasks != config.effective_max_concurrent_tasks:
            logger.warning(
                "Configured max_concurrent_tasks=%s but effective task concurrency is fixed to %s "
                "for a single ppl instance. Use max_queue_size to control queue admission.",
                config.max_concurrent_tasks,
                config.effective_max_concurrent_tasks,
            )

        task_manager = TaskManager(
            max_queue_size=config.max_queue_size,
            cleanup_keep_count=config.cleanup_keep_count,
            cancel_timeout=config.cancel_timeout,
            processing_lock_timeout=config.processing_lock_timeout,
        )

        return cls(
            config=config,
            task_manager=task_manager,
            _cache_dir=Path(cache_dir) if cache_dir else None,
        )

    def initialize_pipeline(
        self,
        pipe_path: str,
        parallelism: int = 1,
        task: TaskType | str = TaskType.T2V,
        skip_validation: bool = False,
    ) -> bool:
        """Initialize the pipeline service."""
        self.pipeline_service = PipelineService(security_level=self.config.security_level)

        return self.pipeline_service.start_pipeline(
            ppl_file=pipe_path,
            parallelism=parallelism,
            task=task,
            skip_validation=skip_validation,
        )

    def initialize_file_service(self, cache_dir: Path | None = None) -> FileService:
        """Initialize file service."""
        cache_dir = cache_dir or self._cache_dir or Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.file_service = FileService(
            cache_dir=cache_dir,
            max_file_size=getattr(self.config, "max_file_size", None),
        )
        return self.file_service

    def initialize_media_service(self) -> MediaGenerationService:
        """Initialize media generation service (requires file_service and pipeline_service)."""
        if not self.file_service:
            raise RuntimeError("FileService must be initialized before MediaGenerationService")
        if not self.pipeline_service:
            raise RuntimeError("PipelineService must be initialized before MediaGenerationService")

        self.media_service = MediaGenerationService(
            file_service=self.file_service,
            inference_service=self.pipeline_service,
        )
        return self.media_service

    def initialize_all(
        self,
        pipe_path: str,
        parallelism: int = 1,
        task: TaskType | str = TaskType.T2V,
        cache_dir: Path | None = None,
        skip_validation: bool = False,
    ) -> bool:
        """Initialize all services at once."""
        if cache_dir:
            self._cache_dir = Path(cache_dir)

        self.pipeline_service = PipelineService(security_level=self.config.security_level)
        if not self.pipeline_service.start_pipeline(
            ppl_file=pipe_path,
            parallelism=parallelism,
            task=task,
            skip_validation=skip_validation,
        ):
            return False

        self.initialize_file_service()
        self.initialize_media_service()

        return True

    def initialize_stream_service(
        self,
        pipe_path: str,
        skip_validation: bool = False,
    ) -> bool:
        """Initialize stream pipeline service (alternative to initialize_all)."""
        self.stream_pipeline_service = StreamPipelineService(
            security_level=self.config.security_level,
        )
        return self.stream_pipeline_service.start_service(
            ppl_file=pipe_path,
            skip_validation=skip_validation,
        )

    def get_api_app(self, enable_rate_limit: bool = True) -> FastAPI:
        """Get FastAPI application with all services initialized."""
        from ..api.api_server import ApiServer

        api_server = ApiServer(
            max_queue_size=self.config.max_queue_size,
            max_concurrent_tasks=self.config.effective_max_concurrent_tasks,
            configured_max_concurrent_tasks=self.config.max_concurrent_tasks,
            task_manager=self.task_manager,
            enable_rate_limit=enable_rate_limit,
            enable_logging=False,
        )

        if self.file_service and self.pipeline_service:
            api_server.initialize_services(self.file_service.cache_dir, self.pipeline_service)

        if self.stream_pipeline_service:
            api_server.initialize_stream_service(self.stream_pipeline_service)

        return api_server.get_app()

    async def __aenter__(self) -> ServiceContainer:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - cleanup all services."""
        await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup all services."""
        if self.file_service:
            await self.file_service.cleanup()
            self.file_service = None

        if self.pipeline_service:
            await self.pipeline_service.aclose()
            self.pipeline_service = None

        if self.stream_pipeline_service:
            await self.stream_pipeline_service.aclose()
            self.stream_pipeline_service = None

        self.media_service = None


def create_container(
    config: ServerConfig | None = None,
    cache_dir: Path | None = None,
) -> ServiceContainer:
    """Factory function to create a service container."""
    return ServiceContainer.create(config, cache_dir)
