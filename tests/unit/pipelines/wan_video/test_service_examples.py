from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wan22_nocache_service_imports_without_cacheseek() -> None:
    root = Path(__file__).resolve().parents[4]
    module = _load_module(root / "examples/wan_video/wan22_14b_text_to_video_service_nocache.py")

    assert module.CACHE_CONFIG["enable_latent_cache"] is False


def test_wan22_cache_service_imports_without_cacheseek() -> None:
    root = Path(__file__).resolve().parents[4]
    module = _load_module(root / "examples/wan_video/wan22_14b_text_to_video_service.py")

    assert module.CACHE_CONFIG["enable_latent_cache"] is True
