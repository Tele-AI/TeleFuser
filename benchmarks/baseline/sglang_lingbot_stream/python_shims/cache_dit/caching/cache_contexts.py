from __future__ import annotations


class _CachedContextManager:
    def similarity(self, *_args: object, **_kwargs: object) -> bool:
        return False


class _CacheManager:
    CachedContextManager = _CachedContextManager


cache_manager = _CacheManager()
