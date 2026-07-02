"""Utility functions for async operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

try:
    import ray
except ImportError:
    ray = None

P = ParamSpec("P")
R = TypeVar("R")


def auto_async_call(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> Callable[[], R | Any]:
    """Execute function locally or remotely via Ray."""
    is_ray: bool = kwargs.pop("is_ray", False)
    result = func.remote(*args, **kwargs) if is_ray else func(*args, **kwargs)

    def wait() -> R | Any:
        if is_ray:
            if ray is None:
                raise ImportError("Ray execution requires installing the optional 'ray' dependency.")
            return ray.get(result)
        if isinstance(result, Callable):
            return result()
        return result

    return wait
