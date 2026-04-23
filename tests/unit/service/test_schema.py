"""Tests for service schema models."""

import pytest
from pydantic import ValidationError

# Skip if pydantic not available
pytest.importorskip("pydantic")

from telefuser.service.api.schema import (
    AspectRatio,
    OutputFormat,
    StopTaskResponse,
    StopTaskStatus,
    TaskRequest,
    TaskResponse,
    TaskStatus,
    TaskStatusMessage,
    TaskType,
)


class TestTaskRequest:
    """Test TaskRequest model."""

    def test_default_values(self):
        """Test default field values."""
        request = TaskRequest()

        assert request.task == TaskType.T2V
        assert request.prompt == ""
        assert request.negative_prompt == ""
        assert request.target_video_length == 5
        assert request.resolution == "720p"
        assert request.seed == 42
        assert request.aspect_ratio == AspectRatio.RATIO_16_9
        assert request.output_format == OutputFormat.PNG
        # output_path should be auto-generated
        assert request.output_path.endswith(".mp4")

    def test_custom_values(self):
        """Test with custom values."""
        request = TaskRequest(
            task="i2v",
            prompt="A beautiful scene",
            negative_prompt="blurry",
            target_video_length=10,
            resolution="1080p",
            seed=123,
            aspect_ratio="9:16",
        )

        assert request.task == TaskType.I2V
        assert request.prompt == "A beautiful scene"
        assert request.negative_prompt == "blurry"
        assert request.target_video_length == 10
        assert request.resolution == "1080p"
        assert request.seed == 123
        assert request.aspect_ratio == AspectRatio.RATIO_9_16

    def test_valid_aspect_ratios(self):
        """Test valid aspect ratio values."""
        valid_ratios = ["16:9", "9:16", "4:3", "3:4", "1:1", "2:3", "3:2"]

        for ratio in valid_ratios:
            request = TaskRequest(aspect_ratio=ratio)
            assert request.aspect_ratio == AspectRatio(ratio)

    def test_invalid_aspect_ratio(self):
        """Test invalid aspect ratio raises error."""
        with pytest.raises(ValidationError):
            TaskRequest(aspect_ratio="invalid")

    def test_valid_tasks(self):
        """Test valid task values."""
        valid_tasks = ["t2v", "i2v", "fl2v", "vc", "t2i", "i2i", "s2v", "vsr", "custom_task"]

        for task in valid_tasks:
            request = TaskRequest(task=task)
            assert request.task == task

    def test_invalid_task(self):
        """Test invalid task raises error."""
        with pytest.raises(ValidationError):
            TaskRequest(task="Invalid Task!")

    def test_get_method(self):
        """Test get method for attribute access."""
        request = TaskRequest(prompt="test prompt")

        assert request.get("prompt") == "test prompt"
        assert request.get("nonexistent", "default") == "default"

    def test_task_id_auto_generated(self):
        """Test that task_id is auto-generated."""
        request1 = TaskRequest()
        request2 = TaskRequest()

        # Should have different auto-generated IDs
        assert request1.task_id != request2.task_id
        assert len(request1.task_id) > 0

    def test_output_path_default(self):
        """Test output_path defaults to task_id.mp4."""
        request = TaskRequest()

        assert request.output_path == f"{request.task_id}.mp4"

    def test_output_path_custom(self):
        """Test custom output_path."""
        request = TaskRequest(output_path="custom/path/video.mp4")

        assert request.output_path == "custom/path/video.mp4"

    def test_valid_output_formats(self):
        """Test valid output format values."""
        valid_formats = ["png", "jpg", "jpeg", "webp"]

        for output_format in valid_formats:
            request = TaskRequest(task="t2i", output_format=output_format)
            assert request.output_format == OutputFormat(output_format)

    def test_invalid_output_format(self):
        """Test invalid output format raises error."""
        with pytest.raises(ValidationError):
            TaskRequest(output_format="bmp")

    def test_json_dump_uses_string_values(self):
        """Test enum fields serialize to API-compatible strings."""
        request = TaskRequest(task=TaskType.T2I, aspect_ratio=AspectRatio.RATIO_1_1, output_format=OutputFormat.WEBP)

        data = request.model_dump(mode="json")

        assert data["task"] == "t2i"
        assert data["aspect_ratio"] == "1:1"
        assert data["output_format"] == "webp"

    def test_extra_contract_fields_are_preserved(self):
        """TaskRequest should preserve extra contract-managed fields for runner execution."""
        request = TaskRequest(task="t2v", num_inference_steps=8)

        assert request.num_inference_steps == 8
        assert request.model_dump()["num_inference_steps"] == 8


class TestTaskStatusMessage:
    """Test TaskStatusMessage model."""

    def test_creation(self):
        """Test basic creation."""
        msg = TaskStatusMessage(task_id="task-123")

        assert msg.task_id == "task-123"

    def test_task_id_required(self):
        """Test that task_id is required."""
        with pytest.raises(ValidationError):
            TaskStatusMessage()


class TestTaskResponse:
    """Test TaskResponse model."""

    def test_creation(self):
        """Test basic creation."""
        response = TaskResponse(
            task_id="task-123",
            task_status="completed",
            output_path="output.mp4",
        )

        assert response.task_id == "task-123"
        assert response.task_status == TaskStatus.COMPLETED
        assert response.output_path == "output.mp4"

    def test_all_fields_required(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            TaskResponse(task_id="task-123")


class TestStopTaskResponse:
    """Test StopTaskResponse model."""

    def test_creation(self):
        """Test basic creation."""
        response = StopTaskResponse(
            stop_status="success",
            reason="User requested",
        )

        assert response.stop_status == StopTaskStatus.SUCCESS
        assert response.reason == "User requested"

    def test_all_fields_required(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            StopTaskResponse(stop_status="success")
