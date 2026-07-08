"""
OpenAI API Protocol Models

Pydantic models for OpenAI-compatible API requests and responses.
References:
    - https://platform.openai.com/docs/api-reference/images
    - https://platform.openai.com/docs/api-reference/videos
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


class ImageGenerationsRequest(BaseModel):
    """Request model for POST /v1/images/generations"""

    prompt: str = Field(
        ...,
        description="A text description of the desired image(s).",
        min_length=1,
        max_length=4000,
    )
    model: str | None = Field(
        None,
        description="The model to use for image generation.",
        examples=["dall-e-3", "qwen-image"],
    )
    n: int | None = Field(
        1,
        description="The number of images to generate. Must be between 1 and 10.",
        ge=1,
        le=10,
    )
    quality: Literal["standard", "hd", "auto"] | None = Field(
        "auto",
        description="The quality of the image that will be generated.",
    )
    response_format: Literal["url", "b64_json"] | None = Field(
        "url",
        description="The format in which the generated images are returned.",
    )
    size: str | None = Field(
        "1024x1024",
        description="The size of the generated images. Format: WIDTHxHEIGHT",
        examples=["1024x1024", "1024x768", "768x1024", "512x512"],
    )
    style: Literal["vivid", "natural"] | None = Field(
        "vivid",
        description="The style of the generated images.",
    )
    user: str | None = Field(None, description="A unique identifier representing your end-user.")

    # TeleFuser extensions
    seed: int | None = Field(42, description="Random seed for reproducible generation.", ge=0)
    negative_prompt: str | None = Field(None, description="Text describing what to avoid.")

    @field_validator("size")
    @classmethod
    def validate_size(cls: type[ImageGenerationsRequest], v: str | None) -> str | None:
        """Validate size format (WIDTHxHEIGHT)."""
        if v is None:
            return v
        try:
            width, height = v.lower().split("x")
            width_val = int(width)
            height_val = int(height)
            if width_val <= 0 or height_val <= 0:
                raise ValueError("Width and height must be positive integers")
            if width_val > 8192 or height_val > 8192:
                raise ValueError("Width and height must not exceed 8192")
        except ValueError as e:
            if "x" not in str(v).lower():
                raise ValueError(f"Size must be in format 'WIDTHxHEIGHT', got: {v}")
            raise ValueError(f"Invalid size format: {v}. Error: {e}")
        return v


class ImageEditRequest(BaseModel):
    """Request model for POST /v1/images/edits"""

    prompt: str = Field(
        ...,
        description="A text description of the desired image(s).",
        min_length=1,
        max_length=4000,
    )
    image: str | None = Field(None, description="The image to edit.")
    image_url: str | None = Field(None, description="URL of the image to edit.")
    mask: str | None = Field(None, description="An additional image for masking.")
    model: str | None = Field(None, description="The model to use for image editing.")
    n: int | None = Field(1, description="The number of images to generate.", ge=1, le=10)
    size: str | None = Field("1024x1024", description="The size of the generated images.")
    response_format: Literal["url", "b64_json"] | None = Field("url", description="Response format.")
    user: str | None = Field(None, description="A unique identifier representing your end-user.")

    # TeleFuser extensions
    seed: int | None = Field(42, ge=0)
    negative_prompt: str | None = None

    @field_validator("size")
    @classmethod
    def validate_size(cls: type[ImageEditRequest], v: str | None) -> str | None:
        """Validate size format."""
        if v is None:
            return v
        try:
            width, height = v.lower().split("x")
            if int(width) <= 0 or int(height) <= 0:
                raise ValueError("Dimensions must be positive")
        except ValueError:
            raise ValueError(f"Size must be in format 'WIDTHxHEIGHT', got: {v}")
        return v


class ImageResponseData(BaseModel):
    """Data model for a single image in the response."""

    b64_json: str | None = Field(None, description="The base64-encoded JSON of the generated image.")
    url: str | None = Field(None, description="The URL of the generated image.")
    revised_prompt: str | None = Field(
        None, description="The prompt that was used to generate the image, if any revision was made."
    )
    file_path: str | None = Field(None, description="Local file path of the generated image (TeleFuser ext).")
    artifact_id: str | None = Field(None, description="Stable output artifact id (TeleFuser ext).")
    artifact_metadata: Dict[str, Any] | None = Field(None, description="Output artifact metadata (TeleFuser ext).")


class ImageResponse(BaseModel):
    """Response model for image generation endpoints."""

    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp of when the image was created.",
    )
    data: List[ImageResponseData] = Field(..., description="The list of generated images.")

    # TeleFuser extensions
    peak_memory_mb: float | None = Field(None, description="Peak memory usage in MB during generation.")
    inference_time_s: float | None = Field(None, description="Inference time in seconds.")


class VideoGenerationsRequest(BaseModel):
    """Request model for POST /v1/videos"""

    prompt: str = Field(
        ...,
        description="A text description of the desired video.",
        min_length=1,
        max_length=4000,
    )
    input_reference: str | None = Field(
        None, description="Path to input image/video for conditioning (I2V, V2V tasks)."
    )
    reference_url: str | None = Field(None, description="URL of input reference.")
    model: str | None = Field(
        None, description="The model to use for video generation.", examples=["wan-video", "svd", "cogvideo"]
    )
    seconds: int | None = Field(4, description="Duration of the generated video in seconds.", ge=1, le=60)
    size: str | None = Field("", description="The size of the generated video (WIDTHxHEIGHT or resolution name).")
    seed: int | None = Field(1024, description="Random seed for reproducible generation.", ge=0)
    negative_prompt: str | None = Field(None, description="Text describing what to avoid.")
    output_path: str | None = Field(None, description="Custom output path for the generated video.")

    @field_validator("seconds")
    @classmethod
    def validate_seconds(cls: type[VideoGenerationsRequest], v: int | None) -> int | None:
        """Validate seconds is reasonable."""
        if v is not None and v > 60:
            raise ValueError("Video length cannot exceed 60 seconds")
        return v


class VideoResponse(BaseModel):
    """Response model for video generation endpoints."""

    id: str = Field(..., description="The unique identifier of the video.")
    object: Literal["video"] = Field("video", description="The object type, which is always 'video'.")
    model: str = Field("sora-2", description="The model used to generate the video.")
    status: Literal["queued", "generating", "completed", "failed", "cancelled", "deleted"] = Field(
        "queued", description="The status of the video generation."
    )
    progress: int = Field(0, description="The progress percentage (0-100).", ge=0, le=100)
    created_at: int = Field(
        default_factory=lambda: int(time.time()),
        description="The Unix timestamp of when the video was created.",
    )
    size: str = Field("", description="The size of the generated video.")
    seconds: str = Field("4", description="The duration of the video in seconds.")
    quality: str = Field("standard", description="The quality of the generated video.")
    url: str | None = Field(None, description="The URL of the generated video.")
    remixed_from_video_id: str | None = Field(None, description="The ID of the source video if this is a remix.")
    completed_at: int | None = Field(None, description="The Unix timestamp of when the video generation completed.")
    expires_at: int | None = Field(None, description="The Unix timestamp when the video will be deleted.")
    error: Dict[str, Any] | None = Field(None, description="Error information if generation failed.")
    file_path: str | None = Field(None, description="Local file path of the generated video.")
    artifact_id: str | None = Field(None, description="Stable output artifact id (TeleFuser ext).")
    artifact_metadata: Dict[str, Any] | None = Field(None, description="Output artifact metadata (TeleFuser ext).")

    # TeleFuser extensions
    peak_memory_mb: float | None = Field(None, description="Peak memory usage in MB during generation.")
    inference_time_s: float | None = Field(None, description="Inference time in seconds.")


class VideoListResponse(BaseModel):
    """Response model for listing videos."""

    data: List[VideoResponse] = Field(..., description="The list of videos.")
    object: Literal["list"] = Field("list", description="The object type, which is always 'list'.")
    has_more: bool | None = Field(None, description="Whether there are more results available.")
    first_id: str | None = Field(None, description="The ID of the first video in the list.")
    last_id: str | None = Field(None, description="The ID of the last video in the list.")


class VideoRemixRequest(BaseModel):
    """Request model for remixing a video."""

    prompt: str = Field(
        ...,
        description="A text description of the desired modifications.",
        min_length=1,
        max_length=4000,
    )


class ErrorDetail(BaseModel):
    """Detailed error information."""

    loc: List[str] | None = Field(None, description="The location of the error (e.g., ['body', 'prompt']).")
    msg: str = Field(..., description="A human-readable error message.")
    type: str = Field(..., description="The type of error.")


class ErrorResponse(BaseModel):
    """Standard error response format (OpenAI compatible)."""

    error: Dict[str, Any] = Field(..., description="Error information.")

    @classmethod
    def from_exception(
        cls: type[ErrorResponse],
        message: str,
        type: str = "invalid_request_error",
        code: str | None = None,
        param: str | None = None,
    ) -> ErrorResponse:
        """Create an error response from an exception."""
        error_dict: Dict[str, Any] = {
            "message": message,
            "type": type,
        }
        if code:
            error_dict["code"] = code
        if param:
            error_dict["param"] = param
        return cls(error=error_dict)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    import uuid

    return f"req_{uuid.uuid4().hex[:24]}"


def validate_image_size(size: str) -> tuple[int, int]:
    """Validate and parse image size string.

    Args:
        size: Size string in format "WIDTHxHEIGHT"

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size format is invalid
    """
    try:
        width_str, height_str = size.lower().split("x")
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")
        return width, height
    except ValueError as e:
        if "positive" in str(e).lower():
            raise ValueError(f"Invalid size '{size}': {e}") from e
        raise ValueError(f"Invalid size format '{size}'. Expected 'WIDTHxHEIGHT'.") from e
