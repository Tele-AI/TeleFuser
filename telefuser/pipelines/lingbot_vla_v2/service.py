"""Structured request adapter for LingBot-VLA v2 action inference."""

from __future__ import annotations

import base64
import binascii
import io
import math
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data import LingBotVlaV2Observation
from .pipeline import LingBotVlaV2CanonicalActionChunk
from .robot_profile import ROBOTWIN_CAMERA_KEYS

DEFAULT_MAX_IMAGE_PIXELS = 16 * 1024 * 1024
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024


class LingBotVlaV2ActionRequest(BaseModel):
    """One RobotWin observation encoded for the HTTP boundary."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1)
    state: list[float] = Field(min_length=14, max_length=14)
    camera_high: str = Field(min_length=1)
    camera_left_wrist: str = Field(min_length=1)
    camera_right_wrist: str = Field(min_length=1)
    seed: int | None = None

    @field_validator("task")
    @classmethod
    def validate_task(cls, value: str) -> str:
        """Reject whitespace-only instructions."""
        value = value.strip()
        if not value:
            raise ValueError("task must be a non-empty string")
        return value

    @field_validator("state")
    @classmethod
    def validate_state(cls, value: list[float]) -> list[float]:
        """Reject non-finite robot state values."""
        if not all(math.isfinite(item) for item in value):
            raise ValueError("state must contain only finite values")
        return value


class LingBotVlaV2ActionResponse(BaseModel):
    """Normalized canonical action chunk returned by the base checkpoint."""

    canonical_normalized_actions: list[list[float]]
    horizon: int
    action_dim: int
    checkpoint_variant: str
    policy_verified: bool
    verification_status: str


class _Pipeline(Protocol):
    def __call__(
        self,
        observation: LingBotVlaV2Observation,
        seed: int | None = None,
        stop_event: Any | None = None,
    ) -> LingBotVlaV2CanonicalActionChunk: ...


def _decode_image(
    value: str,
    *,
    max_image_bytes: int,
    max_image_pixels: int,
) -> Image.Image:
    if max_image_bytes <= 0:
        raise ValueError("max_image_bytes must be positive")
    if max_image_pixels <= 0:
        raise ValueError("max_image_pixels must be positive")
    payload = value.strip()
    if payload.startswith("data:"):
        header, separator, payload = payload.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("image data URLs must use base64 encoding")
    max_encoded_length = 4 * ((max_image_bytes + 2) // 3)
    if len(payload) > max_encoded_length:
        raise ValueError(f"decoded image must not exceed {max_image_bytes} bytes")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image must be valid base64") from error
    if not decoded or len(decoded) > max_image_bytes:
        raise ValueError(f"decoded image must contain 1 to {max_image_bytes} bytes")
    try:
        with Image.open(io.BytesIO(decoded)) as image:
            pixel_count = image.width * image.height
            if pixel_count > max_image_pixels:
                raise ValueError(f"decoded image must not exceed {max_image_pixels} pixels")
            return image.convert("RGB").copy()
    except Image.DecompressionBombError as error:
        raise ValueError(f"decoded image must not exceed {max_image_pixels} pixels") from error
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("decoded payload must be a supported image") from error


def predict_lingbot_vla_v2_action(
    pipeline: _Pipeline,
    request: LingBotVlaV2ActionRequest,
    *,
    max_image_bytes: int,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    stop_event: Any | None = None,
) -> LingBotVlaV2ActionResponse:
    """Decode one request and return the canonical normalized action chunk."""
    images = _decode_request_images(
        request,
        max_image_bytes=max_image_bytes,
        max_image_pixels=max_image_pixels,
    )
    if stop_event is not None and stop_event.is_set():
        raise RuntimeError("LingBot-VLA v2 inference cancelled")
    observation = LingBotVlaV2Observation(task=request.task, state=request.state, images=images)
    if stop_event is None:
        chunk = pipeline(observation, seed=request.seed)
    else:
        chunk = pipeline(observation, seed=request.seed, stop_event=stop_event)
    return LingBotVlaV2ActionResponse(
        canonical_normalized_actions=chunk.canonical_normalized_actions.tolist(),
        horizon=chunk.horizon,
        action_dim=chunk.action_dim,
        checkpoint_variant=chunk.checkpoint_variant,
        policy_verified=chunk.policy_verified,
        verification_status=chunk.verification_status,
    )


def _decode_request_images(
    request: LingBotVlaV2ActionRequest,
    *,
    max_image_bytes: int,
    max_image_pixels: int,
) -> dict[str, Image.Image]:
    encoded_images = (request.camera_high, request.camera_left_wrist, request.camera_right_wrist)
    return {
        key: _decode_image(
            value,
            max_image_bytes=max_image_bytes,
            max_image_pixels=max_image_pixels,
        )
        for key, value in zip(ROBOTWIN_CAMERA_KEYS, encoded_images, strict=True)
    }


def validate_lingbot_vla_v2_action_request(
    request: LingBotVlaV2ActionRequest,
    *,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> None:
    """Validate all encoded camera images without invoking the policy."""
    _decode_request_images(
        request,
        max_image_bytes=max_image_bytes,
        max_image_pixels=max_image_pixels,
    )
