from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from telefuser.service_types import AspectRatio, OutputFormat, StopTaskStatus, TaskStatus, TaskType

from ..core.pipeline_contract import infer_media_type_for_task, validate_task_name_format
from .utils import generate_task_id


class TaskRequest(BaseModel):
    """Request model for media generation tasks."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(default_factory=generate_task_id, description="Task ID (auto-generated)")
    task: str = Field("t2v", description="Task type: t2v, i2v, fl2v, vc, t2i, i2i, s2v, vsr, or custom")
    prompt: str = Field("", description="Generation prompt")
    negative_prompt: str = Field("", description="Negative prompt")
    first_image_path: str = Field("", description="Base64 encoded image or URL")
    last_image_path: str = Field("", description="Base64 encoded image or URL")
    ref_video_path: str = Field("", description="ref video for video continue")
    output_path: str = Field("", description="Output file path (optional, defaults based on task type)")
    target_video_length: int = Field(5, description="Target video length (seconds), for video tasks")
    resolution: str = Field(
        "720p", description="Target resolution, e.g., 720p, 1080p, 480p for video; 1024x1024, 1024x768 for image"
    )
    seed: int = Field(42, description="Random seed")
    aspect_ratio: AspectRatio = Field(
        AspectRatio.RATIO_16_9, description="Allowed values: 16:9, 9:16, 4:3, 3:4, 1:1, 2:3, 3:2"
    )
    output_format: OutputFormat = Field(OutputFormat.PNG, description="Output image format (for t2i, i2i tasks)")

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls: type[TaskRequest], v: str) -> str:
        allowed = ["16:9", "9:16", "4:3", "3:4", "1:1", "2:3", "3:2"]
        if v not in allowed:
            raise ValueError(f"Invalid aspect ratio. Allowed values are: {', '.join(allowed)}")
        return v

    @field_validator("task")
    @classmethod
    def validate_task(cls: type[TaskRequest], v: str) -> str:
        return validate_task_name_format(v)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        if not self.output_path:
            # Set default output path based on task type
            if self.task in ("t2i", "i2i"):
                self.output_path = f"{self.task_id}.{self.output_format}"
            else:
                self.output_path = f"{self.task_id}.mp4"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class TaskStatusMessage(BaseModel):
    """Status message for a task."""

    task_id: str = Field(..., description="Task ID")


class TaskResponse(BaseModel):
    """Response model for task creation."""

    task_id: str
    task_status: TaskStatus
    output_path: str
    peak_memory_mb: float | None = Field(default=None, ge=0, description="Peak accelerator memory used by the task.")
    inference_time_s: float | None = Field(default=None, ge=0, description="Task inference duration in seconds.")


class StopTaskResponse(BaseModel):
    """Response model for task stop/cancel."""

    stop_status: StopTaskStatus
    reason: str
