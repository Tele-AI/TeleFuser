"""Registry for model loader configurations.

Provides a decentralized registration mechanism where each model file
registers its own hash-based detection configs. Supports autodiscovery:
any .py file under the ``telefuser`` package tree that calls
``register_model_config`` is automatically imported on first use.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import torch.nn as nn

from telefuser.utils.logging import logger

_MARKER = "register_model_config"


class ModelRegistry:
    """Singleton registry for model loader configurations.

    Model files call ``register_model_config()`` at module level to
    register their hash-based detection entries.  On first access via
    ``get_configs()``, ``autodiscover()`` scans the ``telefuser``
    package tree and imports every module that contains the marker,
    ensuring all registrations are populated regardless of where the
    model file lives.
    """

    _instance: ModelRegistry | None = None

    def __init__(self) -> None:
        self._configs: list[tuple[str | None, str | None, list[str], list[type[nn.Module]], str]] = []
        self._discovered = False

    @classmethod
    def instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        keys_hash: str | None,
        keys_hash_with_shape: str | None,
        model_names: list[str],
        model_classes: list[type[nn.Module]],
        model_resource: str,
    ) -> None:
        self._configs.append((keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource))

    def get_configs(self) -> list[tuple[str | None, str | None, list[str], list[type[nn.Module]], str]]:
        if not self._discovered:
            self.autodiscover()
        return list(self._configs)

    def autodiscover(self) -> None:
        """Scan the ``telefuser`` package tree for modules that call
        ``register_model_config`` and import them to trigger registration.
        """
        if self._discovered:
            return
        self._discovered = True

        import telefuser

        pkg_root = Path(telefuser.__file__).parent
        for dirpath, _dirnames, filenames in os.walk(pkg_root):
            for fname in filenames:
                if not fname.endswith(".py") or fname.startswith("_"):
                    continue
                filepath = Path(dirpath) / fname
                try:
                    text = filepath.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if _MARKER not in text:
                    continue
                rel = filepath.relative_to(pkg_root)
                module_name = "telefuser." + str(rel.with_suffix("")).replace(os.sep, ".")
                try:
                    importlib.import_module(module_name)
                except Exception:
                    logger.warning(f"Failed to autodiscover model registry in {module_name}")


def register_model_config(
    keys_hash: str | None,
    keys_hash_with_shape: str | None,
    model_names: list[str],
    model_classes: list[type[nn.Module]],
    model_resource: str,
) -> None:
    """Register a model loader configuration for hash-based detection.

    Called at module level in each model file. Example::

        register_model_config(None, "abc123...", ["my_dit"], [MyDiT], "official")

    ``keys_hash_with_shape`` can be None when a model shares checkpoint hashes
    with another registered model (e.g., WanModelWithMemory shares hashes with
    WanModel). In that case, only ``keys_hash_dict`` (key-only hash) is populated,
    avoiding collision in ``keys_hash_with_shape_dict``.
    """
    ModelRegistry.instance().register(keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource)
