"""Utilities for adapting TeleFuser chunks to LiveKit media publishing."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import av
import cv2
import numpy as np
from PIL import Image

from .data_protocol import strip_media_fields


class MediaDecodeError(ValueError):
    """Raised when chunk media payloads cannot be decoded."""


@dataclass(frozen=True)
class AudioPayload:
    """Decoded PCM16 audio carried by one stream chunk."""

    pcm: bytes
    sample_rate: int
    channels: int


def frame_to_rgb(source: object) -> np.ndarray:
    """Convert one native pipeline frame to contiguous RGB24 pixels."""
    if isinstance(source, av.VideoFrame):
        frame = source.to_ndarray(format="rgb24")
    elif isinstance(source, Image.Image):
        frame = np.asarray(source.convert("RGB"))
    elif isinstance(source, np.ndarray):
        frame = source
    else:
        raise MediaDecodeError(f"Unsupported native frame type: {type(source).__name__}")

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise MediaDecodeError(f"Native frame must have shape HxWx3, got {frame.shape}")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise MediaDecodeError("Native frame dimensions must be positive")
    if frame.dtype != np.uint8:
        raise MediaDecodeError(f"Native frame dtype must be uint8, got {frame.dtype}")
    return np.ascontiguousarray(frame)


def decode_jpeg_frames(frames_b64: list[str]) -> list[np.ndarray]:
    """Decode base64 JPEG frames into RGB numpy arrays."""
    frames: list[np.ndarray] = []
    for frame_b64 in frames_b64:
        raw = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise MediaDecodeError("Chunk contains an undecodable JPEG frame")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def split_chunk_media(chunk: dict) -> tuple[list[np.ndarray], AudioPayload | None, dict]:
    """Split a TeleFuser stream chunk into decoded video, audio bytes, and metadata."""
    has_top_level_frames = "frames" in chunk or "frames_b64" in chunk
    data = chunk if has_top_level_frames or not isinstance(chunk.get("data"), dict) else chunk["data"]

    raw_frames = data.get("frames")
    if isinstance(raw_frames, (list, tuple)):
        frames = [frame_to_rgb(frame) for frame in raw_frames]
    else:
        frames = decode_jpeg_frames(list(data.get("frames_b64") or []))

    audio_b64 = data.get("audio_b64")
    audio = None
    if audio_b64:
        pcm = base64.b64decode(audio_b64)
        sample_rate = int(data.get("audio_sample_rate") or 48_000)
        channels = int(data.get("audio_channels") or 1)
        if sample_rate <= 0 or channels <= 0:
            raise MediaDecodeError("Audio sample rate and channel count must be positive")
        bytes_per_sample = channels * 2
        if len(pcm) % bytes_per_sample:
            raise MediaDecodeError("PCM16 audio byte length must align with its channel count")
        audio = AudioPayload(pcm=pcm, sample_rate=sample_rate, channels=channels)
    metadata = strip_media_fields(chunk)
    return frames, audio, metadata
