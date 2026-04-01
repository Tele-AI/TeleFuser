"""Feature Cache module for diffusion transformers.

This module provides feature caching strategies for accelerating diffusion inference:

- AdaTaylorCache: Adaptive skip with Taylor series approximation
- AdaTaylorCacheCalibrator: Collect residual data for parameter calibration
- NoOpCache: No caching, always compute

Usage:
    cache = create_feature_cache("ada_taylor", model_type="Wan2.1-T2V-1.3B", num_inference_steps=50)

    # In forward loop:
    if cache.should_compute(cond_flag):
        output = forward_blocks(x, ...)
        cache.update(output, x, cond_flag)
    else:
        output = cache.approximate(x, cond_flag)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseFeatureCache(ABC):
    """Abstract base class for feature caching strategies.

    Feature caches intercept DiT block forward passes to enable skipping
    expensive computations when possible.

    Core interface:
    - should_compute: Check if real computation is needed (auto-increments step)
    - update: Store computation result for future approximation
    - approximate: Get approximated output from cache
    - reset: Clear all cached state
    """

    @abstractmethod
    def should_compute(self, is_cond: bool) -> bool:
        """Check if real computation is needed for this step.

        This method automatically increments the internal step counter
        for the corresponding path (cond or uncond).

        Args:
            is_cond: True for conditional path, False for unconditional path

        Returns:
            True if real computation is needed, False if can use approximation
        """
        pass

    @abstractmethod
    def update(self, output: torch.Tensor, ori_input: torch.Tensor, is_cond: bool) -> None:
        """Update cache with newly computed output.

        Args:
            output: Output tensor from DiT blocks
            ori_input: Original input tensor to DiT blocks
            is_cond: True for conditional path, False for unconditional path
        """
        pass

    @abstractmethod
    def approximate(self, input: torch.Tensor, is_cond: bool) -> torch.Tensor:
        """Get approximated output using cached data.

        Args:
            input: Current input tensor to DiT blocks
            is_cond: True for conditional path, False for unconditional path

        Returns:
            Approximated output tensor
        """
        pass

    def reset(self) -> None:
        """Reset all cached state. Override if needed."""
        pass


class NoOpCache(BaseFeatureCache):
    """No-operation cache that always computes.

    Used when feature caching is disabled.
    """

    def should_compute(self, is_cond: bool) -> bool:
        """Always returns True to compute normally."""
        return True

    def update(self, output: torch.Tensor, ori_input: torch.Tensor, is_cond: bool) -> None:
        """Does nothing."""
        pass

    def approximate(self, input: torch.Tensor, is_cond: bool) -> torch.Tensor:
        """Never called since should_compute always returns True."""
        return input


def create_feature_cache(
    cache_type: str = "none",
    **kwargs,
) -> BaseFeatureCache:
    """Factory function to create feature cache instances.

    Args:
        cache_type: Type of cache to create:
            - "none": No caching (default)
            - "ada_taylor": AdaTaylorCache for inference acceleration
            - "calibrator": AdaTaylorCacheCalibrator for parameter collection
        **kwargs: Additional arguments passed to the cache constructor

    Returns:
        Feature cache instance

    Examples:
        # No caching
        cache = create_feature_cache("none")

        # AdaTaylorCache for inference
        cache = create_feature_cache(
            "ada_taylor",
            model_type="Wan2.1-T2V-1.3B",
            num_inference_steps=50,
        )

        # Calibrator for parameter collection
        cache = create_feature_cache(
            "calibrator",
            num_inference_steps=50,
            sigma_shift=8.0,
            model_name="MyModel",
        )
    """
    if cache_type == "none":
        return NoOpCache()
    elif cache_type == "ada_taylor":
        from .ada_taylor_cache import AdaTaylorCache

        return AdaTaylorCache(**kwargs)
    elif cache_type == "calibrator":
        from .ada_taylor_cache import AdaTaylorCacheCalibrator

        return AdaTaylorCacheCalibrator(**kwargs)
    else:
        raise ValueError(f"Unknown cache_type: {cache_type}. Expected: none, ada_taylor, calibrator")