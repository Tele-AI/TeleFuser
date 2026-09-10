"""Policy protocol for model-independent VLA serving."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import ModelActionChunk, VLACapabilities, VLARequest


@runtime_checkable
class VLAPolicy(Protocol):
    """Adapt one model pipeline to the semantic VLA action contract."""

    def capabilities(self) -> VLACapabilities: ...

    def predict(self, request: VLARequest) -> ModelActionChunk: ...

    def reset(self, episode_id: str) -> None: ...
