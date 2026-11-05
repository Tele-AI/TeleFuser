"""Explicit component registry for VLA policies and embodiments."""

from __future__ import annotations

from typing import TypeVar

from .embodiment import EmbodimentAdapter
from .policy import VLAPolicy

T = TypeVar("T")


class VLARegistry:
    """Resolve already-loaded policy and embodiment adapters by stable IDs."""

    def __init__(self) -> None:
        self._policies: dict[str, VLAPolicy] = {}
        self._embodiments: dict[str, EmbodimentAdapter] = {}

    def register_policy(self, model_id: str, policy: VLAPolicy, *, replace: bool = False) -> None:
        """Register one loaded policy adapter."""
        if policy.capabilities().model_id != model_id:
            raise ValueError("registered model_id must match policy capabilities")
        self._register(self._policies, model_id, policy, replace=replace)

    def register_embodiment(
        self,
        embodiment_id: str,
        embodiment: EmbodimentAdapter,
        *,
        replace: bool = False,
    ) -> None:
        """Register one embodiment adapter."""
        if embodiment.embodiment_id != embodiment_id:
            raise ValueError("registered embodiment_id must match the adapter")
        self._register(self._embodiments, embodiment_id, embodiment, replace=replace)

    def get_policy(self, model_id: str) -> VLAPolicy:
        """Return a registered policy or fail with a stable lookup error."""
        try:
            return self._policies[model_id]
        except KeyError as error:
            raise KeyError(f"unknown VLA model_id: {model_id!r}") from error

    def get_embodiment(self, embodiment_id: str) -> EmbodimentAdapter:
        """Return a registered embodiment or fail with a stable lookup error."""
        try:
            return self._embodiments[embodiment_id]
        except KeyError as error:
            raise KeyError(f"unknown VLA embodiment_id: {embodiment_id!r}") from error

    def model_ids(self) -> tuple[str, ...]:
        """List registered model IDs in deterministic order."""
        return tuple(sorted(self._policies))

    def embodiment_ids(self) -> tuple[str, ...]:
        """List registered embodiment IDs in deterministic order."""
        return tuple(sorted(self._embodiments))

    @staticmethod
    def _register(registry: dict[str, T], component_id: str, component: T, *, replace: bool) -> None:
        if not isinstance(component_id, str) or not component_id:
            raise ValueError("component ID must be a non-empty string")
        if component_id in registry and not replace:
            raise ValueError(f"VLA component is already registered: {component_id!r}")
        registry[component_id] = component
