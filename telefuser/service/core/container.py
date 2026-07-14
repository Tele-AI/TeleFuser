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

from ..api.api_server import ApiServer
from .config import ServerConfig, server_config
from .file_service import FileService
from .pipeline_pool import PipelinePool
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
    cache_service: Any | None = None
    cache_adapter: Any | None = None  # cacheseek.adapters.telefuser.TeleFuserCacheAdapter
    _cache_dir: Path | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        config: ServerConfig | None = None,
        cache_dir: Path | None = None,
    ) -> ServiceContainer:
        """Create a new service container with all dependencies."""
        config = config or server_config

        if config.num_replicas == 1 and config.max_concurrent_tasks != config.effective_max_concurrent_tasks:
            logger.warning(
                "Configured max_concurrent_tasks=%s but a single ppl instance executes serially "
                "(effective=%s). Use --num-replicas>1 for concurrency, or max_queue_size for admission control.",
                config.max_concurrent_tasks,
                config.effective_max_concurrent_tasks,
            )

        task_manager = TaskManager(
            max_queue_size=config.max_queue_size,
            cleanup_keep_count=config.cleanup_keep_count,
            cancel_timeout=config.cancel_timeout,
            processing_lock_timeout=config.processing_lock_timeout,
            max_concurrent_processing=config.num_replicas,
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
        self.pipeline_service = PipelineService(security_level=self.config.security_level, config=self.config)

        return self.pipeline_service.start_pipeline(
            ppl_file=pipe_path,
            parallelism=parallelism,
            task=task,
            skip_validation=skip_validation,
        )

    def initialize_file_service(self, cache_dir: Path | None = None) -> FileService:
        """Initialize file service."""
        if self.config.artifact_storage_backend != "local":
            raise RuntimeError(
                "Only the local artifact backend is implemented. "
                f"Configured backend: {self.config.artifact_storage_backend}"
            )

        cache_dir = cache_dir or self._cache_dir or Path(self.config.effective_artifact_local_root)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.file_service = FileService(
            cache_dir=cache_dir,
            max_file_size=getattr(self.config, "max_file_size", None),
            verify_ssl=getattr(self.config, "verify_ssl", True),
            ssl_cert_path=getattr(self.config, "ssl_cert_path", None),
            artifact_retention_seconds=self.config.artifact_retention_seconds,
            artifact_tmp_retention_seconds=self.config.artifact_tmp_retention_seconds,
            artifact_persistence_mode=self.config.artifact_persistence_mode,
            artifact_preserve_failed_outputs=self.config.artifact_preserve_failed_outputs,
            artifact_max_total_bytes=self.config.artifact_max_total_bytes,
            artifact_max_task_bytes=self.config.artifact_max_task_bytes,
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
            cache_service=self.cache_service,
            cache_adapter=self.cache_adapter,
        )
        return self.media_service

    def _load_pipeline_cache_config(self, pipe_path: str) -> Any | None:
        """Load CACHE_CONFIG for deciding whether CacheSeek should be imported."""
        module = getattr(self.pipeline_service, "_module", None)
        if module is not None and hasattr(module, "CACHE_CONFIG"):
            return getattr(module, "CACHE_CONFIG")

        try:
            from telefuser.utils.utils import import_function_from_file

            return import_function_from_file(pipe_path, "CACHE_CONFIG")
        except AttributeError:
            return None
        except Exception as exc:
            logger.warning(f"Failed to load CACHE_CONFIG for latent cache enable check, ignored: {exc}")
            return None

    @staticmethod
    def _cache_config_enable_value(cache_config: Any | None) -> bool:
        if isinstance(cache_config, dict):
            return bool(cache_config.get("enable_latent_cache", False))
        return bool(getattr(cache_config, "enable_latent_cache", False))

    def _resolve_enable_latent_cache(self, pipe_path: str) -> bool:
        override = getattr(self.config, "enable_latent_cache", None)
        if override is not None:
            return bool(override)
        return self._cache_config_enable_value(self._load_pipeline_cache_config(pipe_path))

    def initialize_cache_service(self, pipe_path: str) -> Any | None:
        """Initialize optional latent cache service when enabled."""
        enable_override = getattr(self.config, "enable_latent_cache", None)
        if not self._resolve_enable_latent_cache(pipe_path):
            return None

        # Lazy import to avoid pulling cacheseek deps when disabled.
        try:
            from cacheseek.adapters.telefuser.cache_factory import CacheServiceFactory
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Latent cache is enabled, but CacheSeek is not installed. "
                "Install CacheSeek into the TeleFuser environment first. "
                "CacheSeek is not yet published to public PyPI; install from a checkout that contains "
                "the TeleFuser adapter support, for example: "
                "python -m pip install /path/to/CacheSeek. "
                "If the matching commit or branch has been pushed to GitHub, you can also use: "
                "python -m pip install 'cacheseek @ git+https://github.com/Tele-AI/CacheSeek.git@<commit-or-branch>'. "
                "Once cacheseek is published to your pip package index, python -m pip install cacheseek is also valid."
            ) from exc

        try:
            result = CacheServiceFactory.create_cache_service(
                ppl_file=pipe_path,
                enable_latent_cache=enable_override,
                cache_mode=getattr(self.config, "cache_mode", None),
            )
        except Exception as exc:
            raise RuntimeError(f"Latent cache failed to initialize via CacheSeek: {exc}") from exc

        if result is None:
            self.cache_service = None
            self.cache_adapter = None
            raise RuntimeError("Latent cache failed to initialize via CacheSeek: factory returned None")

        self.cache_service, self.cache_adapter = result
        return self.cache_service

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

        if self.config.num_replicas > 1:
            replica_device_ids = self.config.resolve_replica_device_ids(parallelism)
            parallelism_per_replica = len(replica_device_ids[0])

            pool = PipelinePool(
                num_replicas=self.config.num_replicas,
                replica_device_ids=replica_device_ids,
                security_level_name=self.config.security_level.name,
                config=self.config,
                task_manager=self.task_manager,
            )
            if not pool.start_all(
                ppl_file=pipe_path,
                parallelism_per_replica=parallelism_per_replica,
                task=task if isinstance(task, str) else task.value,
                skip_validation=skip_validation,
            ):
                return False
            self.pipeline_service = pool  # duck-type compatible
        else:
            self.pipeline_service = PipelineService(security_level=self.config.security_level, config=self.config)
            if not self.pipeline_service.start_pipeline(
                ppl_file=pipe_path,
                parallelism=parallelism,
                task=task,
                skip_validation=skip_validation,
            ):
                return False

        self.initialize_file_service()
        self.initialize_cache_service(pipe_path=pipe_path)
        self.initialize_media_service()

        return True

    def initialize_stream_service(
        self,
        pipe_path: str,
        skip_validation: bool = False,
        gpu_num: int = 1,
    ) -> bool:
        """Initialize stream pipeline service (alternative to initialize_all)."""
        self.stream_pipeline_service = StreamPipelineService(
            security_level=self.config.security_level,
            config=self.config,
        )
        return self.stream_pipeline_service.start_service(
            ppl_file=pipe_path,
            gpu_num=gpu_num,
            skip_validation=skip_validation,
        )

    def get_api_app(self, enable_rate_limit: bool = True) -> FastAPI:
        """Get FastAPI application with all services initialized."""
        route_profile = "stream" if self.stream_pipeline_service and not self.pipeline_service else "request_response"
        api_server = ApiServer(
            max_queue_size=self.config.max_queue_size,
            max_concurrent_tasks=self.config.effective_max_concurrent_tasks,
            configured_max_concurrent_tasks=self.config.max_concurrent_tasks,
            task_manager=self.task_manager,
            enable_rate_limit=enable_rate_limit,
            enable_logging=False,
            config=self.config,
            route_profile=route_profile,
        )

        if self.file_service and self.pipeline_service:
            api_server.initialize_services(
                self.file_service.cache_dir,
                self.pipeline_service,
                cache_service=self.cache_service,
                cache_adapter=self.cache_adapter,  # forward adapter to api_server
                file_service=self.file_service,
            )

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

        if self.cache_service is not None:
            try:
                self.cache_service.shutdown()
            except Exception as exc:
                logger.warning(f"cache service shutdown failed: {exc}")
            self.cache_service = None
        self.cache_adapter = None
        self.media_service = None


def create_container(
    config: ServerConfig | None = None,
    cache_dir: Path | None = None,
) -> ServiceContainer:
    """Factory function to create a service container."""
    return ServiceContainer.create(config, cache_dir)
