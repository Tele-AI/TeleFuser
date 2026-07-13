"""
OpenAI API Adapter

Adapters to convert between OpenAI API format and TeleFuser internal format.
Allows exposing OpenAI-compatible API while using TeleFuser's task-based system internally.
"""

from __future__ import annotations

from typing import Any, Dict, List

from telefuser.service.api.openai.protocol import (
    ImageEditRequest,
    ImageGenerationsRequest,
    ImageResponse,
    ImageResponseData,
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
    validate_image_size,
)
from telefuser.service.api.schema import TaskRequest
from telefuser.service_types import AspectRatio
from telefuser.utils.logging import logger

VIDEO_REFERENCE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def create_extended_task_request(base_fields: dict[str, Any], extra_fields: dict[str, Any]) -> TaskRequest:
    """Create a TaskRequest with extra fields stored via object.__setattr__.

    Since TaskRequest has a fixed schema, extra OpenAI-specific parameters
    are attached as dynamic attributes.
    """
    task_req = TaskRequest(**base_fields)
    for key, value in extra_fields.items():
        if value is not None:
            setattr(task_req, key, value)
    return task_req


# Size to resolution mapping for videos
VIDEO_SIZE_TO_RESOLUTION = {
    "1024x1024": "720p",
    "1024x576": "720p",
    "1280x720": "720p",
    "1920x1080": "1080p",
    "1080x1920": "1080p",
    "256x256": "480p",
    "512x512": "480p",
    "768x768": "720p",
    "768x1024": "720p",
    "480p": "480p",
    "720p": "720p",
    "1080p": "1080p",
    "2160p": "2160p",
    "4k": "2160p",
}

# Aspect ratio mapping
ASPECT_RATIOS = {
    "1024x1024": "1:1",
    "1024x576": "16:9",
    "576x1024": "9:16",
    "1024x768": "4:3",
    "768x1024": "3:4",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1920x1080": "16:9",
    "1080x1920": "9:16",
}


class OpenAIRequestAdapter:
    """Adapter to convert OpenAI API requests to TeleFuser TaskRequest."""

    @staticmethod
    def to_task_request(
        openai_req: ImageGenerationsRequest | ImageEditRequest | VideoGenerationsRequest,
        task_type: str | None = None,
    ) -> TaskRequest:
        """Convert an OpenAI request to TeleFuser TaskRequest."""
        if isinstance(openai_req, ImageGenerationsRequest):
            return OpenAIRequestAdapter._image_to_task(openai_req, task_type)
        elif isinstance(openai_req, ImageEditRequest):
            return OpenAIRequestAdapter._image_edit_to_task(openai_req, task_type)
        elif isinstance(openai_req, VideoGenerationsRequest):
            return OpenAIRequestAdapter._video_to_task(openai_req, task_type)
        else:
            raise ValueError(f"Unsupported request type: {type(openai_req)}")

    @staticmethod
    def _image_to_task(req: ImageGenerationsRequest, task_type: str | None = None) -> TaskRequest:
        """Convert ImageGenerationsRequest to TaskRequest."""
        task = task_type or "t2i"
        aspect_ratio = ASPECT_RATIOS.get(req.size, "1:1")

        base_fields = {
            "task": task,
            "prompt": req.prompt,
            "resolution": req.size or "1024x1024",
            "seed": req.seed or 42,
            "negative_prompt": req.negative_prompt or "",
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }

        extra_fields = {
            "model": req.model,
            "n": req.n,
            "quality": req.quality,
            "style": req.style,
        }

        return create_extended_task_request(base_fields, extra_fields)

    @staticmethod
    def _image_edit_to_task(req: ImageEditRequest, task_type: str | None = None) -> TaskRequest:
        """Convert ImageEditRequest to TaskRequest."""
        task = task_type or "i2i"
        first_image_path = req.image_url or req.image or ""
        aspect_ratio = ASPECT_RATIOS.get(req.size, "1:1")

        base_fields = {
            "task": task,
            "prompt": req.prompt,
            "first_image_path": first_image_path,
            "resolution": req.size or "1024x1024",
            "seed": req.seed or 42,
            "negative_prompt": req.negative_prompt or "",
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
        }

        extra_fields = {
            "model": req.model,
            "n": req.n,
            "mask": req.mask,
        }

        return create_extended_task_request(base_fields, extra_fields)

    @staticmethod
    def _video_to_task(req: VideoGenerationsRequest, task_type: str | None = None) -> TaskRequest:
        """Convert VideoGenerationsRequest to TaskRequest."""
        if task_type:
            task = task_type
        elif req.input_reference or req.reference_url:
            task = "i2v"
        else:
            task = "t2v"

        resolution = OpenAIRequestAdapter.size_to_resolution(req.size or "1024x576", media_type="video")
        target_video_length = req.seconds or 4
        aspect_ratio = ASPECT_RATIOS.get(req.size, "16:9")
        ref_path = req.input_reference or req.reference_url or ""
        extra_fields: dict[str, Any] = {"model": req.model}
        if req.size:
            extra_fields["openai_video_size"] = req.size
            try:
                width, height = validate_image_size(req.size)
            except ValueError:
                pass
            else:
                extra_fields["width"] = width
                extra_fields["height"] = height
                inferred_aspect_ratio = infer_aspect_ratio(width, height)
                if inferred_aspect_ratio in AspectRatio.values():
                    aspect_ratio = inferred_aspect_ratio

        base_fields = {
            "task": task,
            "prompt": req.prompt,
            "resolution": resolution,
            "target_video_length": target_video_length,
            "seed": req.seed or 42,
            "negative_prompt": req.negative_prompt or "",
            "aspect_ratio": aspect_ratio,
            "output_path": req.output_path or "",
        }

        if task in {"vc", "vsr"}:
            base_fields["ref_video_path"] = ref_path
        else:
            base_fields["first_image_path"] = ref_path

        return create_extended_task_request(base_fields, extra_fields)

    @staticmethod
    def size_to_resolution(size: str, media_type: str = "image") -> str:
        """Convert OpenAI size format to TeleFuser resolution format.

        OpenAI uses format like "1024x1024" for both images and videos.
        TeleFuser uses "1024x1024" for images but "720p", "1080p" for videos.
        """
        if not size:
            return "1024x1024" if media_type == "image" else "720p"

        if size in ["480p", "720p", "1080p", "2160p", "4k"]:
            return "2160p" if size == "4k" else size

        if media_type == "video":
            resolution = VIDEO_SIZE_TO_RESOLUTION.get(size)
            if resolution:
                return resolution

            # Infer from dimensions
            try:
                width, height = validate_image_size(size)
                if height >= 2160:
                    return "2160p"
                elif height >= 1080:
                    return "1080p"
                elif height >= 720:
                    return "720p"
                else:
                    return "480p"
            except ValueError:
                logger.warning(f"Could not parse video size '{size}', defaulting to 720p")
                return "720p"
        else:
            return size

    @staticmethod
    def resolution_to_size(resolution: str, media_type: str = "image") -> str:
        """Convert TeleFuser resolution to OpenAI size format."""
        if not resolution:
            return "1024x1024"

        if "x" in resolution.lower():
            return resolution

        resolution_to_size_map = {
            "480p": "854x480",
            "720p": "1280x720",
            "1080p": "1920x1080",
            "2160p": "3840x2160",
        }

        return resolution_to_size_map.get(resolution, "1024x1024")


class OpenAIResponseAdapter:
    """Adapter to convert TeleFuser responses to OpenAI API format."""

    @staticmethod
    def to_image_response(
        output_path: str,
        prompt: str,
        response_format: str = "url",
        base_url: str = "",
        task_id: str = "",
        peak_memory_mb: float | None = None,
        inference_time_s: float | None = None,
    ) -> ImageResponse:
        """Convert TeleFuser output to OpenAI ImageResponse."""
        data_list: List[ImageResponseData] = []

        if response_format == "b64_json":
            try:
                import base64

                with open(output_path, "rb") as f:
                    image_data = f.read()
                b64_string = base64.b64encode(image_data).decode("utf-8")
                data_list.append(
                    ImageResponseData(
                        b64_json=b64_string,
                        revised_prompt=prompt,
                        file_path=output_path,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to encode image to base64: {e}")
                raise
        else:
            if base_url and task_id:
                url = f"{base_url}/v1/images/{task_id}/content"
            else:
                url = f"file://{output_path}"

            data_list.append(
                ImageResponseData(
                    url=url,
                    revised_prompt=prompt,
                    file_path=output_path,
                )
            )

        return ImageResponse(
            data=data_list,
            peak_memory_mb=peak_memory_mb,
            inference_time_s=inference_time_s,
        )

    @staticmethod
    def to_video_response(
        task_id: str,
        status: str,
        prompt: str,
        size: str = "",
        seconds: int = 4,
        model: str = "",
        output_path: str | None = None,
        url: str | None = None,
        progress: int = 0,
        peak_memory_mb: float | None = None,
        inference_time_s: float | None = None,
        error: Dict[str, Any] | None = None,
    ) -> VideoResponse:
        """Convert TeleFuser task info to OpenAI VideoResponse."""
        status_map = {
            "pending": "queued",
            "processing": "generating",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        openai_status = status_map.get(status, "queued")

        return VideoResponse(
            id=task_id,
            model=model or "wan-video",
            status=openai_status,
            progress=progress,
            size=size,
            seconds=str(seconds),
            url=url,
            file_path=output_path,
            peak_memory_mb=peak_memory_mb,
            inference_time_s=inference_time_s,
            error=error,
        )

    @staticmethod
    def to_video_list_response(
        videos: List[VideoResponse],
        has_more: bool = False,
        first_id: str | None = None,
        last_id: str | None = None,
    ) -> VideoListResponse:
        """Convert list of video responses to VideoListResponse."""
        return VideoListResponse(
            data=videos,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    @staticmethod
    def task_status_to_video_status(task_status: str) -> str:
        """Convert TeleFuser task status to OpenAI video status."""
        status_map = {
            "pending": "queued",
            "processing": "generating",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
        }
        return status_map.get(task_status, "queued")

    @staticmethod
    def video_status_to_task_status(video_status: str) -> str:
        """Convert OpenAI video status to TeleFuser task status."""
        status_map = {
            "queued": "pending",
            "generating": "processing",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
            "deleted": "cancelled",
        }
        return status_map.get(video_status, "pending")


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    import base64

    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


def decode_base64_to_image(b64_string: str, output_path: str) -> None:
    """Decode base64 string to image file."""
    import base64

    image_data = base64.b64decode(b64_string)
    with open(output_path, "wb") as f:
        f.write(image_data)


def infer_aspect_ratio(width: int, height: int) -> str:
    """Infer aspect ratio from dimensions."""
    from math import gcd

    if width == 0 or height == 0:
        return "1:1"

    common = gcd(width, height)
    w = width // common
    h = height // common

    common_ratios = {
        (1, 1): "1:1",
        (16, 9): "16:9",
        (9, 16): "9:16",
        (4, 3): "4:3",
        (3, 4): "3:4",
        (2, 3): "2:3",
        (3, 2): "3:2",
        (21, 9): "21:9",
    }

    return common_ratios.get((w, h), f"{w}:{h}")


def calculate_num_frames(seconds: int, fps: int | None = None) -> int:
    """Calculate number of frames from duration and FPS."""
    fps = fps or 24
    return seconds * fps


def calculate_video_duration(num_frames: int, fps: int | None = None) -> int:
    """Calculate video duration from frames and FPS."""
    fps = fps or 24
    return num_frames // fps


def is_probable_video_reference(path_or_url: str) -> bool:
    """Best-effort detection for whether a reference path points to a video asset."""
    from pathlib import Path

    return Path(path_or_url).suffix.lower() in VIDEO_REFERENCE_EXTENSIONS
