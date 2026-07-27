"""
API Routers for TeleFuser Service

This module contains FastAPI route handlers organized by resource type.
Each router can be tested independently.
"""

from __future__ import annotations

from .files import router as files_router
from .service import router as service_router
from .tasks import router as tasks_router

__all__ = ["tasks_router", "files_router", "service_router"]
