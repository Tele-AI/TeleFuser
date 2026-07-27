from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest
from PIL import Image

from telefuser.service.livekit.media_bridge import MediaDecodeError, frame_to_rgb, split_chunk_media


def test_split_chunk_media_accepts_native_pil_frames() -> None:
    image = Image.new("RGB", (4, 3), color=(10, 20, 30))

    frames, audio, metadata = split_chunk_media(
        {
            "type": "chunk",
            "index": 2,
            "frames": [image],
            "stream_progress": {"completed_chunks": 3},
        }
    )

    assert audio is None
    assert len(frames) == 1
    assert frames[0].shape == (3, 4, 3)
    assert frames[0][0, 0].tolist() == [10, 20, 30]
    assert metadata == {
        "type": "chunk",
        "index": 2,
        "stream_progress": {"completed_chunks": 3},
    }


def test_split_chunk_media_keeps_base64_jpeg_compatibility() -> None:
    bgr = np.zeros((3, 4, 3), dtype=np.uint8)
    bgr[:, :] = (30, 20, 10)
    ok, encoded = cv2.imencode(".jpg", bgr)
    assert ok

    frames, _, metadata = split_chunk_media(
        {
            "data": {
                "frames_b64": [base64.b64encode(encoded.tobytes()).decode()],
                "fps": 16,
            },
            "index": 1,
        }
    )

    assert frames[0].shape == (3, 4, 3)
    assert metadata == {"data": {"fps": 16}, "index": 1}


def test_split_chunk_media_treats_numeric_frames_as_status_metadata() -> None:
    frames, audio, metadata = split_chunk_media(
        {
            "type": "status",
            "stage": "chunk_decoded",
            "index": 0,
            "frames": 13,
        }
    )

    assert frames == []
    assert audio is None
    assert metadata == {
        "type": "status",
        "stage": "chunk_decoded",
        "index": 0,
        "frames": 13,
    }


def test_split_chunk_media_decodes_pcm16_audio_format() -> None:
    pcm = np.zeros(960 * 2, dtype=np.int16).tobytes()

    frames, audio, metadata = split_chunk_media(
        {
            "type": "chunk",
            "audio_b64": base64.b64encode(pcm).decode(),
            "audio_sample_rate": 48_000,
            "audio_channels": 2,
        }
    )

    assert frames == []
    assert audio is not None
    assert audio.pcm == pcm
    assert audio.sample_rate == 48_000
    assert audio.channels == 2
    assert metadata == {"type": "chunk"}


def test_frame_to_rgb_rejects_wrong_pixel_shape() -> None:
    with pytest.raises(MediaDecodeError, match="HxWx3"):
        frame_to_rgb(np.zeros((3, 4), dtype=np.uint8))
