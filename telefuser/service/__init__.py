"""
TeleFuser Service Module

This module provides server-side functionality for the TeleFuser framework.
It is organized into the following submodules:

- api: HTTP API layer (FastAPI routes, middleware, schema)
- core: Core business logic (task management, pipeline service, config)
- media: Media processing utilities (image, video, audio)
- security: Security validation and related tools

Client code has been moved to telefuser.client (see P3.2).
"""

from __future__ import annotations

from telefuser.service.api.schema import StopTaskResponse, TaskRequest, TaskResponse
from telefuser.service.core.config import SecurityLevel, ServerConfig, server_config
from telefuser.service.core.task_manager import TaskManager, TaskStatus
from telefuser.service_types import AspectRatio, OutputFormat, StopTaskStatus, TaskType

__all__ = [
    # API models
    "TaskRequest",
    "TaskResponse",
    "StopTaskResponse",
    "TaskType",
    "AspectRatio",
    "OutputFormat",
    "StopTaskStatus",
    # Config
    "ServerConfig",
    "server_config",
    "SecurityLevel",
    # Task management
    "TaskManager",
    "TaskStatus",
]
