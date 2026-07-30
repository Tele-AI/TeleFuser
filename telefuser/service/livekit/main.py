"""Entrypoint helpers for ``telefuser stream-serve``."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import uvicorn

from telefuser._logo import TELEFUSER_LOGO
from telefuser.utils.logging import logger

from .app import create_livekit_app
from .config import LiveKitServeConfig
from .runtime import LiveKitServeRuntime


def run_stream_server(
    *,
    pipe_path: str,
    host: str | None = None,
    port: int | None = None,
    livekit_url: str | None = None,
    livekit_api_key: str | None = None,
    livekit_api_secret: str | None = None,
    num_workers: int | None = None,
    max_sessions_per_worker: int | None = None,
    worker_gpu_map: str | None = None,
    queue_size: int | None = None,
    control_idle_timeout: float | None = None,
    session_timeout: int | None = None,
    token_ttl: int | None = None,
    controller_timeout: int | None = None,
    room_empty_timeout: int | None = None,
    worker_mode: str | None = None,
    skip_validation: bool = False,
    security_level: str | None = None,
) -> None:
    """Run the LiveKit-backed streaming HTTP API."""
    config_kwargs: dict[str, Any] = _drop_none(
        {
            "host": host,
            "port": port,
            "livekit_url": livekit_url,
            "livekit_api_key": livekit_api_key,
            "livekit_api_secret": livekit_api_secret,
            "num_workers": num_workers,
            "max_sessions_per_worker": max_sessions_per_worker,
            "worker_gpu_map": worker_gpu_map,
            "queue_size": queue_size,
            "control_idle_timeout": control_idle_timeout,
            "session_timeout": session_timeout,
            "token_ttl": token_ttl,
            "controller_timeout": controller_timeout,
            "room_empty_timeout": room_empty_timeout,
            "worker_mode": worker_mode,
        }
    )
    config = LiveKitServeConfig(**config_kwargs)
    config.require_livekit_credentials()

    runtime = LiveKitServeRuntime(
        config=config,
        pipeline_file=pipe_path,
        skip_validation=skip_validation,
        security_level=security_level,
    )
    app = create_livekit_app(runtime)

    try:
        print(TELEFUSER_LOGO)
        logger.info(
            "Starting TeleFuser LiveKit server on %s:%s, workers=%s, pipeline=%s, "
            "skip_validation=%s, security_level=%s",
            config.host,
            config.port,
            config.num_workers,
            pipe_path,
            skip_validation,
            security_level,
        )
        uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    except KeyboardInterrupt:
        logger.info("LiveKit server interrupted by user")
    except Exception as exc:
        logger.error(f"LiveKit server failed: {exc}")
        sys.exit(1)
    finally:
        try:
            asyncio.run(runtime.aclose())
        except Exception as exc:
            logger.warning(f"Error during LiveKit server cleanup: {exc}")


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}
