from __future__ import annotations

from pathlib import Path

from telefuser.service.core.file_service import FileService


def test_file_service_creates_all_media_directories(tmp_path: Path) -> None:
    """FileService eagerly creates every upload and output directory."""

    service = FileService(cache_dir=tmp_path)

    assert service.input_image_dir.is_dir()
    assert service.input_video_dir.is_dir()
    assert service.input_audio_dir.is_dir()
    assert service.output_video_dir.is_dir()
    assert service.output_image_dir.is_dir()
