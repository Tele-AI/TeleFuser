from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("aiortc")

from aiortc.codecs import h264, vpx

from telefuser.service.webrtc.chunk_router import ChunkRouter
from telefuser.service.webrtc.session_manager import WebRTCSessionManager, _BidirectionalSession


def test_video_quality_configuration_prefers_h264_and_raises_bitrate() -> None:
    with (
        patch.object(h264, "DEFAULT_BITRATE", 1_000_000),
        patch.object(h264, "MAX_BITRATE", 3_000_000),
        patch.object(vpx, "DEFAULT_BITRATE", 500_000),
        patch.object(vpx, "MAX_BITRATE", 1_500_000),
    ):
        manager = WebRTCSessionManager(video_codec="H264", video_bitrate=8_000_000)

        assert h264.DEFAULT_BITRATE == h264.MAX_BITRATE == 8_000_000
        assert vpx.DEFAULT_BITRATE == vpx.MAX_BITRATE == 8_000_000

        transceiver = MagicMock(kind="video")
        peer_connection = MagicMock()
        peer_connection.getTransceivers.return_value = [transceiver]
        manager._set_video_codec_preferences(peer_connection)

        codecs = transceiver.setCodecPreferences.call_args.args[0]
        assert codecs[0].mimeType == "video/H264"


def test_chunk_router_prefers_raw_frames_over_jpeg_transport() -> None:
    video_track = MagicMock()
    image = Image.fromarray(np.full((8, 12, 3), [17, 113, 241], dtype=np.uint8))
    router = ChunkRouter(
        generator=MagicMock(),
        video_track=video_track,
        audio_track=None,
        data_channel_send=None,
        session_id="quality-test",
    )

    router._route_chunk({"frames": [image], "frames_b64": ["not-used"]})

    frame = video_track.push_frame.call_args.args[0]
    np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), np.asarray(image))


def test_chunk_router_preserves_numeric_frame_count_as_metadata() -> None:
    data_channel_send = MagicMock()
    router = ChunkRouter(
        generator=MagicMock(),
        video_track=MagicMock(),
        audio_track=None,
        data_channel_send=data_channel_send,
        session_id="status-test",
    )

    router._route_chunk({"type": "status", "stage": "chunk_sent", "frames": 13})

    message = json.loads(data_channel_send.call_args.args[0])
    assert message["data"]["frames"] == 13


def test_close_session_runs_pipeline_callback_outside_event_loop() -> None:
    callback_threads: list[int] = []

    class PeerConnection:
        async def close(self) -> None:
            pass

    def on_close(session_id: str) -> None:
        callback_threads.append(threading.get_ident())

    async def close_session() -> tuple[bool, int]:
        manager = WebRTCSessionManager()
        manager._sessions["session-123"] = _BidirectionalSession(
            pc=PeerConnection(),
            on_close=on_close,
            session_id="session-123",
        )
        event_loop_thread = threading.get_ident()
        closed = await manager.close_session("session-123")
        return closed, event_loop_thread

    closed, event_loop_thread = asyncio.run(close_session())

    assert closed is True
    assert len(callback_threads) == 1
    assert callback_threads[0] != event_loop_thread
