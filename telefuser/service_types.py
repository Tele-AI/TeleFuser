"""Shared service type definitions."""

from __future__ import annotations

from enum import Enum


class _StringEnum(str, Enum):
    """String enum with helpers for CLI/API choices."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]


class TaskType(_StringEnum):
    """Supported media generation task types."""

    T2V = "t2v"
    I2V = "i2v"
    FL2V = "fl2v"
    VC = "vc"
    T2I = "t2i"
    I2I = "i2i"


class AspectRatio(_StringEnum):
    """Supported media aspect ratios."""

    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
    RATIO_1_1 = "1:1"
    RATIO_2_3 = "2:3"
    RATIO_3_2 = "3:2"


class OutputFormat(_StringEnum):
    """Supported image output formats."""

    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    WEBP = "webp"


class TaskStatus(_StringEnum):
    """Task lifecycle status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StopTaskStatus(_StringEnum):
    """Stop task operation result."""

    SUCCESS = "success"
    DO_NOTHING = "do_nothing"
    ERROR = "error"


class MediaType(_StringEnum):
    """Generated media type."""

    IMAGE = "image"
    VIDEO = "video"


class PipelineRunStatus(_StringEnum):
    """Pipeline runner execution status."""

    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
