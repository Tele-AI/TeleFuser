from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from telefuser.pipelines.lingbot_video.data import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NEGATIVE_PROMPT_IMAGE,
    LingBotVideoRequest,
    default_negative_caption,
    load_lingbot_video_model_config,
    num_frames_from_duration,
    parse_lingbot_video_prompt,
    preprocess_ti2v_image,
)


def test_load_dense_transformer_config() -> None:
    config_path = Path("/hhb-data/aigc/model_zoo/lingbot/lingbot-video-dense-1.3b/transformer")
    if not config_path.is_dir():
        return

    config = load_lingbot_video_model_config(config_path)

    assert config.num_layers == 24
    assert config.hidden_size == 2048
    assert config.patch_size == (1, 2, 2)


def test_preprocess_ti2v_image_preserves_official_uint8_interpolation() -> None:
    image = torch.tensor([[[[0.0, 10.0], [20.0, 30.0]]]]).repeat(1, 3, 1, 1)

    result = preprocess_ti2v_image(image, height=3, width=3)
    expected = F.interpolate(image.to(torch.uint8), size=(3, 3), mode="bilinear", align_corners=False).float() / 255.0

    assert torch.equal(result, expected.unsqueeze(2))


def test_rewriter_prompt_envelope_matches_upstream_caption_and_duration_contract() -> None:
    caption, duration = parse_lingbot_video_prompt({"caption": {"scene": "a test scene"}, "duration": 5, "fps": 24})

    assert caption == '{"scene":"a test scene"}'
    assert duration == 5.0
    assert num_frames_from_duration(duration) == 121


def test_raw_prompt_sample_excludes_runtime_metadata() -> None:
    caption, duration = parse_lingbot_video_prompt({"scene": "test", "duration": 2, "fps": 24})

    assert caption == '{"scene":"test"}'
    assert duration == 2.0
    assert num_frames_from_duration(duration) == 49


def test_request_requires_dit_patch_aligned_spatial_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible by 16"):
        LingBotVideoRequest(caption="{}", height=480, width=856, num_frames=1)


def test_default_negative_captions_match_mode_contract() -> None:
    assert default_negative_caption(1) == DEFAULT_NEGATIVE_PROMPT_IMAGE
    assert default_negative_caption(5) == DEFAULT_NEGATIVE_PROMPT
    assert "temporal_and_motion_stability" not in DEFAULT_NEGATIVE_PROMPT_IMAGE
    assert "temporal_and_motion_stability" in DEFAULT_NEGATIVE_PROMPT
