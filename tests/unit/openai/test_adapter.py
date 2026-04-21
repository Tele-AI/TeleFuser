"""
Unit tests for OpenAI adapter.

Tests conversion between OpenAI format and TeleFuser format.
Avoids over-mocking by testing actual conversion logic.
"""

import pytest

from telefuser.service.api.openai.adapter import (
    OpenAIRequestAdapter,
    OpenAIResponseAdapter,
    calculate_num_frames,
    encode_image_to_base64,
    infer_aspect_ratio,
    is_probable_video_reference,
)
from telefuser.service.api.openai.protocol import (
    ImageEditRequest,
    ImageGenerationsRequest,
    VideoGenerationsRequest,
)


class TestRequestAdapter:
    """Tests for OpenAIRequestAdapter."""

    def test_image_to_task(self):
        """Convert image request to TaskRequest."""
        openai_req = ImageGenerationsRequest(
            prompt="a cat",
            size="1024x768",
            seed=42,
        )
        task_req = OpenAIRequestAdapter.to_task_request(openai_req)

        assert task_req.task == "t2i"
        assert task_req.prompt == "a cat"
        assert task_req.resolution == "1024x768"
        assert task_req.seed == 42
        assert task_req.aspect_ratio == "4:3"  # inferred from size

    def test_image_edit_to_task(self):
        """Convert image edit request."""
        openai_req = ImageEditRequest(
            prompt="make it blue",
            image_url="https://example.com/image.png",
            size="512x512",
        )
        task_req = OpenAIRequestAdapter.to_task_request(openai_req)

        assert task_req.task == "i2i"
        assert task_req.first_image_path == "https://example.com/image.png"

    def test_video_to_task_text_to_video(self):
        """Convert T2V request."""
        openai_req = VideoGenerationsRequest(
            prompt="a cat playing",
            seconds=5,
            size="1024x576",
        )
        task_req = OpenAIRequestAdapter.to_task_request(openai_req)

        assert task_req.task == "t2v"
        assert task_req.target_video_length == 5
        assert task_req.aspect_ratio == "16:9"

    def test_video_to_task_image_to_video(self):
        """Convert I2V request (auto-detected from input)."""
        openai_req = VideoGenerationsRequest(
            prompt="animate this",
            input_reference="/path/to/image.png",
        )
        task_req = OpenAIRequestAdapter.to_task_request(openai_req)

        assert task_req.task == "i2v"
        assert task_req.first_image_path == "/path/to/image.png"

    def test_video_to_task_video_conditioning(self):
        """Convert explicit video-conditioned request."""
        openai_req = VideoGenerationsRequest(
            prompt="continue this clip",
            input_reference="/path/to/input.mp4",
        )
        task_req = OpenAIRequestAdapter.to_task_request(openai_req, task_type="vc")

        assert task_req.task == "vc"
        assert task_req.ref_video_path == "/path/to/input.mp4"


class TestSizeConversion:
    """Tests for size/resolution conversion."""

    @pytest.mark.parametrize(
        "size,media_type,expected",
        [
            ("1024x1024", "image", "1024x1024"),
            ("1024x576", "video", "720p"),
            ("1920x1080", "video", "1080p"),
            ("720p", "video", "720p"),
        ],
    )
    def test_size_to_resolution(self, size, media_type, expected):
        """Size format conversion."""
        result = OpenAIRequestAdapter.size_to_resolution(size, media_type)
        assert result == expected


class TestResponseAdapter:
    """Tests for OpenAIResponseAdapter."""

    def test_task_to_video_status_mapping(self):
        """Status mapping between formats."""
        assert OpenAIResponseAdapter.task_status_to_video_status("pending") == "queued"
        assert OpenAIResponseAdapter.task_status_to_video_status("processing") == "generating"
        assert OpenAIResponseAdapter.task_status_to_video_status("completed") == "completed"

    def test_video_to_task_status_mapping(self):
        """Reverse status mapping."""
        assert OpenAIResponseAdapter.video_status_to_task_status("queued") == "pending"
        assert OpenAIResponseAdapter.video_status_to_task_status("generating") == "processing"


class TestHelperFunctions:
    """Tests for utility functions."""

    @pytest.mark.parametrize(
        "w,h,expected",
        [
            (1024, 1024, "1:1"),
            (1920, 1080, "16:9"),
            (1080, 1920, "9:16"),
        ],
    )
    def test_infer_aspect_ratio(self, w, h, expected):
        """Aspect ratio inference."""
        assert infer_aspect_ratio(w, h) == expected

    def test_calculate_num_frames(self):
        """Frame calculation from duration."""
        assert calculate_num_frames(5, 24) == 120
        assert calculate_num_frames(10, 30) == 300

    def test_encode_image_to_base64(self, tmp_path):
        """Base64 encoding."""
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake_image_data")

        import base64

        result = encode_image_to_base64(str(image_path))
        assert result == base64.b64encode(b"fake_image_data").decode()

    def test_is_probable_video_reference(self):
        assert is_probable_video_reference("/tmp/video.mp4") is True
        assert is_probable_video_reference("https://example.com/frame.png") is False
