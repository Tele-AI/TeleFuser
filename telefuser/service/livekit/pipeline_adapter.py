"""Adapter from LiveKit workers to TeleFuser stream pipeline services."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from telefuser.service.core.config import ServerConfig
from telefuser.service.core.stream_pipeline_service import StreamPipelineService
from telefuser.service.security.security_validator import SecurityLevel


class LiveKitPipelineAdapter:
    """Thin wrapper around ``StreamPipelineService`` for LiveKit workers."""

    def __init__(self, *, security_level: SecurityLevel | None = None, config: ServerConfig | None = None) -> None:
        self.stream_service = StreamPipelineService(security_level=security_level, config=config)

    def start(self, pipeline_file: str, *, skip_validation: bool = False, gpu_num: int = 1) -> None:
        """Load and start a stream pipeline."""
        if not self.stream_service.start_service(pipeline_file, skip_validation=skip_validation, gpu_num=gpu_num):
            raise RuntimeError(f"Failed to start LiveKit stream pipeline: {pipeline_file}")

    @property
    def stream_mode(self) -> str | None:
        """Return the detected TeleFuser stream interaction mode."""
        return self.stream_service.stream_mode

    async def aclose(self) -> None:
        """Stop the wrapped stream service."""
        await self.stream_service.aclose()

    def create_session(self, config: dict) -> str:
        return self.stream_service.create_session(config)

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self.stream_service.push_chunk(session_id, chunk)

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        async for chunk in self.stream_service.pull_chunks(session_id):
            yield chunk

    async def stream_task(self, config: dict) -> AsyncGenerator[dict, None]:
        """Yield chunks from a server-push service."""
        async for chunk in self.stream_service.stream_task(config):
            yield chunk

    def close_session(self, session_id: str) -> None:
        self.stream_service.close_session(session_id)
