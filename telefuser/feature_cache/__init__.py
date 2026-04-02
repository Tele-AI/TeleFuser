"""Feature Cache module for diffusion transformers.

This module provides feature caching strategies for accelerating diffusion inference:

- AdaTaylorCache: Adaptive skip with Taylor series approximation
- AdaTaylorCacheCalibrator: Collect residual data for parameter calibration
- NoOpCache: No caching, always compute

Usage:
    from telefuser.feature_cache import create_feature_cache

    cache = create_feature_cache("ada_taylor", model_type="Wan2.1-T2V-1.3B", num_inference_steps=50)

    # In forward loop:
    if cache.should_compute(cond_flag):
        output = forward_blocks(x, ...)
        cache.update(output, x, cond_flag)
    else:
        output = cache.approximate(x, cond_flag)
"""

from __future__ import annotations

from .ada_taylor_cache import (
    AdaTaylorCache,
    AdaTaylorCacheCalibrator,
    AdaTaylorCacheConfig,
    AdaTaylorCacheState,
    load_cache_params,
    nearest_interp,
)
from .base import BaseFeatureCache, NoOpCache, create_feature_cache

__all__ = [
    # Base interface
    "BaseFeatureCache",
    "NoOpCache",
    "create_feature_cache",
    # AdaTaylorCache
    "AdaTaylorCache",
    "AdaTaylorCacheConfig",
    "AdaTaylorCacheState",
    "AdaTaylorCacheCalibrator",
    # Utils
    "load_cache_params",
    "nearest_interp",
]
