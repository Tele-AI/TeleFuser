"""TeleFuser Client — thin facade over the standalone ``tf_client`` module.

The real implementation lives in ``tf_client.py``, which is a self-contained
single-file SDK (no internal ``telefuser`` dependency). Drop that file into
any project and ``from tf_client import TFClient`` works as-is.

Within the ``telefuser`` package, prefer:

    from telefuser.client import TFClient

Helpers that are not part of the core surface (multi-server utilities, task
type / aspect-ratio / status string constants) are still exported by the
``tf_client`` module — import them from there directly when needed:

    from telefuser.client.tf_client import send_and_monitor_task, TASK_VC
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tf_client import (
    TFClient,
    TaskCreationError,
    TaskFailedError,
    TaskTimeoutError,
    TeleFuserError,
)

if TYPE_CHECKING:
    from .vla import AsyncVLAClient, VLAClientError

__all__ = [
    "TFClient",
    "AsyncVLAClient",
    "TeleFuserError",
    "TaskCreationError",
    "TaskFailedError",
    "TaskTimeoutError",
    "VLAClientError",
]


def __getattr__(name: str) -> Any:
    """Load the optional VLA client without changing the existing HTTP client import path."""
    if name in {"AsyncVLAClient", "VLAClientError"}:
        from . import vla

        return getattr(vla, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
