from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class BlockAdapter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class DBCacheConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


class ForwardPattern:
    Pattern_2 = "Pattern_2"


class ParamsModifier:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


@dataclass
class TaylorSeerCalibratorConfig:
    order: int = 1


def steps_mask(*, total_steps: int, **_: Any) -> list[int]:
    return [1] * total_steps


def enable_cache(*_: Any, **__: Any) -> None:
    raise RuntimeError("cache_dit is not installed; SGLang cache-dit acceleration is unavailable")


def refresh_context(*_: Any, **__: Any) -> None:
    raise RuntimeError("cache_dit is not installed; SGLang cache-dit acceleration is unavailable")
