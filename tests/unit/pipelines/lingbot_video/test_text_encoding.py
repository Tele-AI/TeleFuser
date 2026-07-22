from __future__ import annotations

from types import SimpleNamespace

import torch

from telefuser.core.config import ModelRuntimeConfig
from telefuser.pipelines.lingbot_video.text_encoding import LingBotVideoTextEncodingStage


class _VisionEncoder:
    config = SimpleNamespace(vision_config=SimpleNamespace(patch_size=14))


def test_prepare_ti2v_vlm_image_matches_source_smart_resize() -> None:
    stage = LingBotVideoTextEncodingStage(
        "text",
        _VisionEncoder(),
        processor=object(),
        model_runtime_config=ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
    )

    image = stage.prepare_ti2v_vlm_image(torch.zeros(1, 3, 1, 192, 320))

    assert image.size == (308, 196)
