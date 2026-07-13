from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class Dict(dict):
    """Minimal addict.Dict compatibility used by SGLang-Diffusion config parsing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        items = dict(*args, **kwargs)
        for key, value in items.items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, Dict):
            return value
        if isinstance(value, Mapping):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._wrap(item) for item in value)
        return value
