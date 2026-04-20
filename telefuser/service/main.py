from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

from telefuser.core.config import TELEFUSER_LOGO
from telefuser.service_types import TaskType
from telefuser.utils.logging import logger

from .core.config import server_config
from .core.container import ServiceContainer


def run_server(
    pipe_path: str,
    task: TaskType | str,
    port: int,
    host: str,
    cache_dir: str = "",
    parallelism: int = 1,
    enable_rate_limit: bool = True,
) -> None:
    """Run the TeleFuser server with dependency injection container."""
    container = None

    try:
        print(TELEFUSER_LOGO)
        logger.info("Starting TeleFuser server...")

        # Update server config
        server_config.host = host
        server_config.port = port

        if not server_config.validate():
            raise RuntimeError("Invalid server configuration")

        # Create service container
        container = ServiceContainer.create(
            config=server_config,
            cache_dir=Path(cache_dir) if cache_dir else None,
        )

        # Initialize all services
        if not container.initialize_all(
            pipe_path=pipe_path,
            parallelism=parallelism,
            task=task,
            cache_dir=Path(cache_dir) if cache_dir else None,
        ):
            raise RuntimeError("Failed to initialize services")

        logger.info("All services initialized successfully")

        # Get FastAPI app
        app = container.get_api_app(enable_rate_limit=enable_rate_limit)

        logger.info(f"Starting server on {server_config.host}:{server_config.port}")
        uvicorn.run(app, host=server_config.host, port=server_config.port, log_level="warning")

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if container:
            import asyncio

            try:
                asyncio.run(container.cleanup())
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
