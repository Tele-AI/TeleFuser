"""Shared runtime controls for LTX-2.5 validation captures."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


@contextmanager
def deterministic_audio_kernels(enabled: bool) -> Iterator[None]:
    """Stabilize CUDA convolution selection while capturing decoded audio."""
    if not enabled:
        yield
        return
    benchmark = torch.backends.cudnn.benchmark
    deterministic = torch.backends.cudnn.deterministic
    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic_algorithms)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


def mp4_container_metadata(path: Path) -> dict[str, Any]:
    """Return JSON-safe provenance for an encoded MP4 container and its streams."""
    import av

    def optional_string(value: object | None) -> str | None:
        return None if value is None else str(value)

    def stream_metadata(stream: Any) -> dict[str, Any]:
        codec = stream.codec_context
        result: dict[str, Any] = {
            "type": stream.type,
            "codec": codec.name,
            "time_base": optional_string(stream.time_base),
            "duration": stream.duration,
            "frames": stream.frames,
            "bit_rate": codec.bit_rate,
            "metadata": dict(stream.metadata),
        }
        if stream.type == "video":
            result.update(
                {
                    "width": codec.width,
                    "height": codec.height,
                    "pixel_format": optional_string(codec.format),
                    "frame_rate": optional_string(stream.average_rate),
                }
            )
        elif stream.type == "audio":
            result.update(
                {
                    "sample_rate": codec.sample_rate,
                    "channels": codec.channels,
                    "layout": optional_string(codec.layout),
                    "sample_format": optional_string(codec.format),
                }
            )
        return result

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with av.open(str(path)) as container:
        return {
            "path": path.name,
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
            "format": container.format.name,
            "duration": container.duration,
            "start_time": container.start_time,
            "bit_rate": container.bit_rate,
            "metadata": dict(container.metadata),
            "streams": [stream_metadata(stream) for stream in container.streams],
        }
