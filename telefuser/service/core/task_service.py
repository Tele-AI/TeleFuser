"""Task service for media generation orchestration."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

from telefuser.service_types import MediaType, PipelineRunStatus, TaskStatus, TaskType
from telefuser.utils.logging import logger

from ..api.schema import TaskRequest, TaskResponse
from ..media.media_base import AudioHandler, ImageHandler, VideoHandler
from .file_service import FileService
from .pipeline_contract import infer_media_type_for_task
from .pipeline_service import PipelineService

# Media handlers
_image_handler = ImageHandler()
_video_handler = VideoHandler()
_audio_handler = AudioHandler()


def _build_cache_task_request(task_data: dict) -> SimpleNamespace:
    """Build a minimal task_request stub for the cache layer.

    Splatting ``task_data`` directly would crash because ``TaskRequest`` is
    ``extra="allow"`` and may contain keys that are not valid Python
    identifiers. The cache layer only reads ``task_id`` / ``task`` /
    ``prompt`` via ``getattr``, so we whitelist those.
    """
    return SimpleNamespace(
        task_id=task_data.get("task_id"),
        task=task_data.get("task"),
        prompt=task_data.get("prompt") or "",
    )


class MediaGenerationService:
    """Service for media generation (video and image).

    Orchestrates file downloads, path resolution, and pipeline execution.
    """

    def __init__(
        self,
        file_service: FileService,
        inference_service: PipelineService,
        cache_service: Any | None = None,
    ) -> None:
        self.file_service = file_service
        self.inference_service = inference_service
        self.cache_service = cache_service

    async def generate_media_with_stop_event(
        self, message: TaskRequest, stop_event: threading.Event
    ) -> TaskResponse | None:
        """Generate media (video or image) with stop event support."""
        task_data = message.model_dump(mode="json")
        if stop_event.is_set():
            logger.info(f"Task {message.task_id} cancelled before processing")
            return None

        async def update_video_path(video_name: str, message: TaskRequest, task_data: dict) -> dict:
            if video_name in message.model_fields_set and getattr(message, video_name):
                video_path = getattr(message, video_name)
                if video_path.startswith("http"):
                    video_path = await self.file_service.download_video(video_path)
                    task_data[video_name] = str(video_path)
                elif _video_handler.is_base64(video_path):
                    video_path = _video_handler.save(video_path, str(self.file_service.input_video_dir))
                    task_data[video_name] = str(video_path)
                else:
                    task_data[video_name] = video_path
            return task_data

        async def update_image_path(image_name: str, message: TaskRequest, task_data: dict) -> dict:
            if image_name in message.model_fields_set and getattr(message, image_name):
                image_path = getattr(message, image_name)
                if image_path.startswith("http"):
                    image_path = await self.file_service.download_image(image_path)
                    task_data[image_name] = str(image_path)
                elif _image_handler.is_base64(image_path):
                    image_path = _image_handler.save(image_path, str(self.file_service.input_image_dir))
                    task_data[image_name] = str(image_path)
                else:
                    task_data[image_name] = image_path
            return task_data

        async def update_audio_path(audio_name: str, message: TaskRequest, task_data: dict) -> dict:
            if audio_name in message.model_fields_set and getattr(message, audio_name):
                audio_path = getattr(message, audio_name)
                if audio_path.startswith("http"):
                    audio_path = await self.file_service.download_audio(audio_path)
                    task_data[audio_name] = str(audio_path)
                elif _audio_handler.is_base64(audio_path):
                    audio_path = _audio_handler.save(audio_path, str(self.file_service.input_audio_dir))
                    task_data[audio_name] = str(audio_path)
                else:
                    task_data[audio_name] = audio_path
            return task_data

        task_data = await update_image_path("first_image_path", message, task_data)
        task_data = await update_image_path("last_image_path", message, task_data)
        task_data = await update_video_path("ref_video_path", message, task_data)
        task_data = await update_audio_path("audio_path", message, task_data)

        # Determine media type and set appropriate output path
        media_type = infer_media_type_for_task(message.task)
        actual_save_path = self.file_service.get_output_path(message.output_path, media_type=media_type)
        task_data["output_path"] = str(actual_save_path)

        # Cache lookup (best-effort: degrade silently on any failure)
        cache_task_request: SimpleNamespace | None = None
        if self.cache_service is not None:
            try:
                cache_task_request = _build_cache_task_request(task_data)
                latent_data = await self.cache_service.build_latent_data(cache_task_request, task_data)
                if latent_data is not None:
                    task_data["latent_data"] = latent_data
            except Exception as exc:
                logger.warning(f"[task_service] cache lookup failed, ignored: {exc}")

        result = await self.inference_service.run_task_with_stop_event(
            task_data,
            stop_event,
            output_root=str(self.file_service.output_dir),
        )

        if result is None:
            if stop_event.is_set():
                logger.info(f"Task {message.task_id} cancelled during processing")
                return None
            raise RuntimeError("Task processing timeout")

        if result.get("status") == PipelineRunStatus.SUCCESS:
            # Cache writeback (best-effort: degrade silently on any failure)
            if self.cache_service is not None:
                try:
                    raw = result.get("raw")
                    latent_payload = raw.get("latent_payload") if isinstance(raw, dict) else None
                    if latent_payload:
                        if cache_task_request is None:
                            cache_task_request = _build_cache_task_request(task_data)
                        await self.cache_service.save_latent_payload(cache_task_request, latent_payload)
                except Exception as exc:
                    logger.warning(f"[task_service] cache writeback failed, ignored: {exc}")

            output_path = result.get("output_path") or task_data.get("output_path") or message.output_path
            return TaskResponse(
                task_id=message.task_id,
                task_status=TaskStatus.COMPLETED,
                output_path=str(output_path),
                peak_memory_mb=result.get("peak_memory_mb"),
                inference_time_s=result.get("inference_time_s"),
            )
        else:
            raise RuntimeError(result.get("message") or "Inference failed")
