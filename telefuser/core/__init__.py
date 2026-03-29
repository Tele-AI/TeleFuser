"""Core abstractions for TeleFuser framework.

This module provides base classes for pipelines, models, and stages.
For configuration classes, import directly from telefuser.core.config.
For model weight utilities, import directly from telefuser.utils.model_weight.
For module management, import directly from telefuser.core.module_manager.

Direct imports are preferred to avoid unnecessary module loading and potential
circular dependencies. The following patterns are recommended:

    # Base classes (from this module)
    from telefuser.core import BaseModel, BasePipeline, BaseStage

    # Configuration classes
    from telefuser.core.config import ModelRuntimeConfig, ParallelConfig

    # Model weight utilities
    from telefuser.utils.model_weight import load_state_dict, hash_state_dict_keys

    # Module management
    from telefuser.core.module_manager import ModuleManager
"""

from __future__ import annotations

from .base_model import BaseModel
from .base_pipeline import BasePipeline
from .base_stage import BaseStage, with_model_offload

__all__ = [
    # Base classes
    "BaseModel",
    "BasePipeline",
    "BaseStage",
    # Decorators
    "with_model_offload",
]
