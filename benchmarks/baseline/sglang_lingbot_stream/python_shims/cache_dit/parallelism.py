from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ParallelismBackend:
    AUTO = "auto"


@dataclass
class ParallelismConfig:
    backend: str
    ulysses_size: int | None = None
    ring_size: int | None = None
    tp_size: int | None = None
    extra: dict[str, Any] | None = None
