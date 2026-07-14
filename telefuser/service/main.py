from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import uvicorn

from telefuser._logo import TELEFUSER_LOGO
from telefuser.service_types import TaskType
from telefuser.utils.logging import logger

from .core.config import server_config
from .core.container import ServiceContainer


def _run(
    label: str,
    container: ServiceContainer,
    enable_rate_limit: bool,
) -> None:
    """Shared server lifecycle: print logo, build app, run uvicorn, cleanup."""
    try:
        print(TELEFUSER_LOGO)
        logger.info(f"Starting TeleFuser {label}...")

        config = container.config
        if not config.validate():
            raise RuntimeError("Invalid server configuration")

        app = container.get_api_app(enable_rate_limit=enable_rate_limit)
        logger.info(f"Starting {label} on {config.host}:{config.port}")
        uvicorn.run(app, host=config.host, port=config.port, log_level="warning")

    except KeyboardInterrupt:
        logger.info(f"{label.capitalize()} interrupted by user")
    except Exception as e:
        logger.error(f"{label.capitalize()} failed: {e}")
        sys.exit(1)
    finally:
        try:
            asyncio.run(container.cleanup())
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def run_server(
    pipe_path: str,
    task: TaskType | str,
    port: int,
    host: str,
    cache_dir: str = "",
    parallelism: int = 1,
    enable_rate_limit: bool = True,
    num_replicas: int = 1,
    enable_latent_cache: bool | None = None,
    cache_mode: str | None = None,
    security_level: str | None = None,
    skip_validation: bool = False,
) -> None:
    """Run the TeleFuser server with dependency injection container."""
    server_config.host = host
    server_config.port = port
    server_config.num_replicas = num_replicas
    server_config.enable_latent_cache = enable_latent_cache
    server_config.cache_mode = cache_mode
    if security_level is not None:
        from .security.security_validator import SecurityLevel

        server_config.security_level = SecurityLevel[security_level.upper()]

    container = ServiceContainer.create(
        config=server_config,
        cache_dir=Path(cache_dir) if cache_dir else None,
    )

    if not container.initialize_all(
        pipe_path=pipe_path,
        parallelism=parallelism,
        task=task,
        cache_dir=Path(cache_dir) if cache_dir else None,
        skip_validation=skip_validation,
    ):
        raise RuntimeError("Failed to initialize services")

    logger.info("All services initialized successfully")
    _run("server", container, enable_rate_limit)


def run_stream_server(
    pipe_path: str,
    port: int,
    host: str,
    enable_rate_limit: bool = True,
    skip_validation: bool = False,
    security_level: str | None = None,
    gpu_num: int = 1,
) -> None:
    """Run the TeleFuser stream server.

    Unlike run_server (request-response), this loads a stream pipeline
    that exposes get_service() and serves via WebRTC or WebSocket.
    """
    server_config.host = host
    server_config.port = port
    if security_level is not None:
        from .security.security_validator import SecurityLevel

        server_config.security_level = SecurityLevel[security_level.upper()]

    container = ServiceContainer.create(config=server_config)

    if not container.initialize_stream_service(
        pipe_path=pipe_path,
        gpu_num=gpu_num,
        skip_validation=skip_validation,
    ):
        raise RuntimeError("Failed to initialize stream service")

    logger.info("Stream service initialized successfully")
    _run("stream server", container, enable_rate_limit)
