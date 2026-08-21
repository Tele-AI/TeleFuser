"""Tests for LTX-2.5 validation capture runtime controls."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch

from tools.validation.ltx25_capture_utils import deterministic_audio_kernels, mp4_container_metadata


def test_deterministic_audio_kernels_restores_runtime_state() -> None:
    """The capture-only mode must not leak process-wide CUDA settings."""
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    try:
        with deterministic_audio_kernels(True):
            assert not torch.backends.cudnn.benchmark
            assert torch.backends.cudnn.deterministic
            assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic_algorithms)
    assert torch.backends.cudnn.benchmark is benchmark
    assert torch.backends.cudnn.deterministic is deterministic
    assert torch.are_deterministic_algorithms_enabled() is deterministic_algorithms


def test_mp4_container_metadata_records_video_stream(tmp_path: Path) -> None:
    """Container provenance includes the fields needed to replay a Golden output."""
    path = tmp_path / "decoded.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = 2
        stream.height = 2
        stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=48000)
        audio_stream.layout = "stereo"
        frame = av.VideoFrame.from_ndarray(np.zeros((2, 2, 3), dtype=np.uint8), format="rgb24")
        audio_frame = av.AudioFrame.from_ndarray(np.zeros((2, 1024), dtype=np.float32), format="fltp", layout="stereo")
        audio_frame.sample_rate = 48000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in audio_stream.encode(audio_frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)

    metadata = mp4_container_metadata(path)

    assert metadata["path"] == "decoded.mp4"
    assert metadata["size_bytes"] > 0
    assert len(metadata["sha256"]) == 64
    video_stream = next(stream for stream in metadata["streams"] if stream["type"] == "video")
    audio_stream = next(stream for stream in metadata["streams"] if stream["type"] == "audio")
    assert video_stream["width"] == 2
    assert video_stream["height"] == 2
    assert audio_stream["sample_rate"] == 48000
    assert audio_stream["channels"] == 2
