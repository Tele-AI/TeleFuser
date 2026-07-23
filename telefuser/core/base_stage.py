"""Base stage for pipeline execution with model offload support."""

from __future__ import annotations

import functools
from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import torch

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger

from .config import FeatureCacheConfig, ModelRuntimeConfig, WeightOffloadType

if TYPE_CHECKING:
    from telefuser.metrics import StageMetricContext

F = TypeVar("F", bound=Callable[..., Any])


def with_model_offload(model_names: list[str]) -> Callable[[F], F]:
    """Decorator for automatic model load/unload around stage execution.

    Optimized: Skip wrapper overhead when models are already on device (NO_CPU_OFFLOAD mode).
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: BaseStage, *args: Any, **kwargs: Any) -> Any:
            offload_type = self.model_runtime_config.offload_config.offload_type

            # Fast path: Models already on device, no offloading needed
            # Skip all overhead when in NO_CPU_OFFLOAD mode and models are loaded
            if self.onload_models_flag and offload_type == WeightOffloadType.NO_CPU_OFFLOAD:
                return func(self, *args, **kwargs)

            rank = 0
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()

            pin_memory = self.model_runtime_config.offload_config.pin_cpu_memory

            # Load models to device if not already loaded or if offloading is enabled
            models_to_move = []
            for model_name in model_names:
                model = getattr(self, model_name)
                if model is not None and not self.onload_models_flag:
                    models_to_move.append((model_name, model))

            # Batch log for models that need to be moved (only log once, not per model)
            if models_to_move:
                logger.info(f"onload {model_names} to {self.device} with rank {rank}")
                for model_name, model in models_to_move:
                    if hasattr(model, "onload_device"):
                        model.onload_device(self.device, non_blocking=pin_memory)
                    else:
                        model.to(self.device)
                self.onload_models_flag = True

            try:
                return func(self, *args, **kwargs)
            finally:
                # Offload models after execution if offloading is enabled
                if offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                    logger.info(f"offload {model_names} with rank {rank}")
                    for model_name in model_names:
                        model = getattr(self, model_name)
                        if model is not None:
                            if hasattr(model, "offload_device"):
                                model.offload_device(pin_memory=pin_memory)
                            else:
                                model.cpu()
                    current_platform.empty_cache()
                    self.onload_models_flag = False  # Reset flag after offload

        return cast(F, wrapper)

    return decorator


class BaseStage(ABC):
    """Base class for pipeline stages with model lifecycle management."""

    def __init__(self, name: str, model_runtime_config: ModelRuntimeConfig) -> None:
        self.name = name
        self.model_runtime_config = model_runtime_config
        self.torch_dtype = model_runtime_config.torch_dtype
        self.device_type = model_runtime_config.device_type or current_platform.device_type
        self.device = torch.device(type=self.device_type, index=model_runtime_config.device_id)
        self.model_names: list[str] = []
        self.onload_models_flag = False
        self._metrics_hook: StageMetricContext | None = None

    def enable_metrics(self, registry: Any | None = None) -> None:
        """Enable metrics collection for this stage.

        Args:
            registry: Optional metrics registry. If not provided, uses the global registry.
        """
        try:
            from telefuser.metrics import get_metrics_registry

            reg = registry or get_metrics_registry()
            self._metrics_hook = reg.register_stage(self.name)
        except ImportError:
            logger.warning("Metrics module not available, metrics will not be collected")
            self._metrics_hook = None

    def disable_metrics(self) -> None:
        """Disable metrics collection for this stage."""
        self._metrics_hook = None

    @property
    def metrics_hook(self) -> StageMetricContext | None:
        """Get the metrics hook for this stage."""
        return self._metrics_hook

    def onload_models(self) -> None:
        """Load models to device."""
        pin_memory = self.model_runtime_config.offload_config.pin_cpu_memory
        for model_name in self.model_names:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                if hasattr(model, "onload_device"):
                    model.onload_device(self.device, non_blocking=pin_memory)
                else:
                    model.to(self.device)

    def offload_models(self) -> None:
        """Offload models from device."""
        pin_memory = self.model_runtime_config.offload_config.pin_cpu_memory
        for model_name in self.model_names:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                if hasattr(model, "offload_device"):
                    model.offload_device(pin_memory=pin_memory)
                else:
                    model.cpu()

    def parallel_models(self) -> None:
        """Apply parallelization to models. Override in subclass."""
        pass

    def setup_feature_cache(
        self,
        model: Any,
        cache_config: FeatureCacheConfig,
        num_inference_steps: int,
        init_step: int = 0,
    ) -> None:
        """Set up feature caching for a model.

        Args:
            model: The model to set up caching for (must have set_ada_taylor_cache method).
            cache_config: Feature cache configuration.
            num_inference_steps: Total number of inference steps.
            init_step: Initial step for caching (used when switching models mid-generation).
        """
        if cache_config.enabled:
            model.set_ada_taylor_cache(
                num_inference_steps=num_inference_steps,
                model_type=cache_config.model_type,
                n_derivatives=cache_config.n_derivatives,
                taylor_threshold=cache_config.taylor_threshold,
                init_step=init_step,
            )
        elif hasattr(model, "_feature_cache") and model._feature_cache is None:
            model.reset_feature_cache()
