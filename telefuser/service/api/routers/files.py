"""
File Routes for TeleFuser API

Handles file upload and download operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from telefuser.utils.logging import logger

if TYPE_CHECKING:
    from ..api_server import ApiServer

router = APIRouter(prefix="/v1/files", tags=["files"])


class FileRoutes:
    """File route handlers with dependency injection."""

    def __init__(self, api_server: ApiServer) -> None:
        self.api = api_server

    async def download_file(self, file_path: str) -> StreamingResponse:
        """Download a file by path. Supports both video and image files."""
        assert self.api.file_service is not None, "File service is not initialized"

        try:
            full_path = self.api.file_service.resolve_output_file(file_path)
            return self.api._stream_file_response(full_path)
        except ValueError:
            raise HTTPException(status_code=403, detail="Access to this file is not allowed")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error occurred while processing file download request: {e}")
            raise HTTPException(status_code=500, detail="File download failed")


def create_router(api_server: ApiServer) -> APIRouter:
    """Create a new router with fresh routes for the given ApiServer instance."""
    new_router = APIRouter(prefix="/v1/files", tags=["files"])
    routes = FileRoutes(api_server)

    @new_router.get("/download/{file_path:path}")
    async def download_file(file_path: str) -> StreamingResponse:
        return await routes.download_file(file_path)

    return new_router


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup file routes."""
    return create_router(api_server)
