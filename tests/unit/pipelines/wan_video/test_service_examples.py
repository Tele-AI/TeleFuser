from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image


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


def test_wan21_i2v_benchmark_service_uses_fixed_workload() -> None:
    root = Path(__file__).resolve().parents[4]
    module = _load_module(root / "examples/wan_video/wan21_14b_image_to_video_480p_service.py")
    pipeline = MagicMock(return_value=[Image.new("RGB", (8, 8))])

    module.run(pipeline, Image.new("RGB", (8, 8)), "test prompt", seed=7)

    kwargs = pipeline.call_args.kwargs
    assert (kwargs["width"], kwargs["height"]) == (832, 480)
    assert kwargs["num_frames"] == 81
    assert kwargs["num_inference_steps"] == 40
    assert kwargs["seed"] == 7


def test_wan21_i2v_benchmark_service_writes_requested_output(tmp_path) -> None:
    root = Path(__file__).resolve().parents[4]
    module = _load_module(root / "examples/wan_video/wan21_14b_image_to_video_480p_service.py")
    image_path = tmp_path / "input.png"
    output_path = tmp_path / "output.mp4"
    Image.new("RGB", (8, 8)).save(image_path)

    with (
        patch.object(module, "run", return_value=[Image.new("RGB", (8, 8))]),
        patch.object(module, "save_video") as save_video,
    ):
        result = module.run_with_file(
            MagicMock(),
            first_image_path=str(image_path),
            prompt="test",
            negative_prompt="",
            seed=42,
            output_path=str(output_path),
        )

    assert result == {"output_path": str(output_path)}
    save_video.assert_called_once()
