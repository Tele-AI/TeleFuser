"""Deprecated cache package namespace.

The latent cache implementation moved to the external `cacheseek` package.
TeleFuser service code imports CacheSeek directly and no longer supports the
legacy `telefuser.service.cache` facade.
"""

_BACKEND = "cacheseek"

__all__ = ["_BACKEND"]
