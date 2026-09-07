from __future__ import annotations

import base64
import io

import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from telefuser.pipelines.lingbot_vla_v2.pipeline import LingBotVlaV2CanonicalActionChunk
from telefuser.pipelines.lingbot_vla_v2.robot_profile import ROBOTWIN_CAMERA_KEYS
from telefuser.pipelines.lingbot_vla_v2.service import (
    LingBotVlaV2ActionRequest,
    predict_lingbot_vla_v2_action,
)


def _encoded_image(*, data_url: bool = False) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}" if data_url else encoded


def _payload() -> dict:
    image = _encoded_image()
    return {
        "task": "pick up the red block",
        "state": [0.0] * 14,
        "camera_high": image,
        "camera_left_wrist": image,
        "camera_right_wrist": image,
        "seed": 7,
    }


class _Pipeline:
    def __init__(self) -> None:
        self.observations = []
        self.seeds = []

    def __call__(self, observation, seed=None) -> LingBotVlaV2CanonicalActionChunk:
        self.observations.append(observation)
        self.seeds.append(seed)
        return LingBotVlaV2CanonicalActionChunk(
            canonical_normalized_actions=torch.zeros(2, 55),
            horizon=2,
            action_dim=55,
        )


def test_request_adapter_returns_normalized_action_contract() -> None:
    pipeline = _Pipeline()
    request = LingBotVlaV2ActionRequest.model_validate(_payload())

    response = predict_lingbot_vla_v2_action(pipeline, request, max_image_bytes=1024 * 1024)

    assert response.horizon == 2
    assert response.action_dim == 55
    assert response.checkpoint_variant == "base"
    assert response.policy_verified is False
    assert response.verification_status == "unverified_official_6b_base"
    assert len(response.canonical_normalized_actions[0]) == 55
    assert pipeline.seeds == [7]
    assert tuple(pipeline.observations[0].images) == ROBOTWIN_CAMERA_KEYS
    assert all(image.mode == "RGB" for image in pipeline.observations[0].images.values())


def test_request_adapter_accepts_image_data_urls() -> None:
    pipeline = _Pipeline()
    payload = _payload()
    payload["camera_high"] = _encoded_image(data_url=True)

    predict_lingbot_vla_v2_action(
        pipeline,
        LingBotVlaV2ActionRequest.model_validate(payload),
        max_image_bytes=1024 * 1024,
    )

    assert len(pipeline.observations) == 1


def test_request_adapter_rejects_invalid_image() -> None:
    pipeline = _Pipeline()
    request = LingBotVlaV2ActionRequest.model_validate({**_payload(), "camera_high": "not-base64"})

    with pytest.raises(ValueError, match="image must be valid base64"):
        predict_lingbot_vla_v2_action(pipeline, request, max_image_bytes=1024 * 1024)

    assert pipeline.observations == []


def test_request_adapter_rejects_non_positive_image_limit() -> None:
    pipeline = _Pipeline()
    request = LingBotVlaV2ActionRequest.model_validate(_payload())

    with pytest.raises(ValueError, match="max_image_bytes must be positive"):
        predict_lingbot_vla_v2_action(pipeline, request, max_image_bytes=0)

    assert pipeline.observations == []


def test_request_adapter_rejects_image_over_pixel_limit() -> None:
    pipeline = _Pipeline()
    request = LingBotVlaV2ActionRequest.model_validate(_payload())

    with pytest.raises(ValueError, match="decoded image must not exceed 63 pixels"):
        predict_lingbot_vla_v2_action(
            pipeline,
            request,
            max_image_bytes=1024 * 1024,
            max_image_pixels=63,
        )

    assert pipeline.observations == []


def test_request_adapter_rejects_non_positive_pixel_limit() -> None:
    pipeline = _Pipeline()
    request = LingBotVlaV2ActionRequest.model_validate(_payload())

    with pytest.raises(ValueError, match="max_image_pixels must be positive"):
        predict_lingbot_vla_v2_action(
            pipeline,
            request,
            max_image_bytes=1024 * 1024,
            max_image_pixels=0,
        )

    assert pipeline.observations == []


@pytest.mark.parametrize(
    "payload",
    (
        {**_payload(), "state": [0.0] * 13},
        {**_payload(), "state": [0.0] * 13 + [float("inf")]},
        {**_payload(), "output_path": "/tmp/action"},
    ),
)
def test_request_adapter_rejects_invalid_request(payload: dict) -> None:
    with pytest.raises(ValidationError):
        LingBotVlaV2ActionRequest.model_validate(payload)
