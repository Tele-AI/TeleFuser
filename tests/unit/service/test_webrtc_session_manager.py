from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import av
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("aiortc")

from aiortc.codecs import h264, vpx

from telefuser.service.webrtc.chunk_router import ChunkRouter
from telefuser.service.webrtc.session_manager import WebRTCSessionManager, _BidirectionalSession
from telefuser.service.webrtc.track import FrameGeneratorTrack


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


def test_frame_track_discards_oldest_frames_when_client_falls_behind() -> None:
    track = FrameGeneratorTrack(fps=2, max_buffer_seconds=1.0)
    frames = [
        av.VideoFrame.from_ndarray(np.full((2, 2, 3), color, dtype=np.uint8), format="rgb24") for color in (10, 20, 30)
    ]

    track.push_frames(frames[:2])
    track.push_frame(frames[2])

    queued = [track._queue.get_nowait(), track._queue.get_nowait()]
    assert [frame.to_ndarray(format="rgb24")[0, 0, 0] for frame in queued] == [20, 30]
    assert track.dropped_frames == 1


def test_frame_track_holds_last_frame_after_output_finishes() -> None:
    async def receive_frames() -> tuple[int, int]:
        track = FrameGeneratorTrack(fps=120)
        source = av.VideoFrame.from_ndarray(np.full((2, 2, 3), 77, dtype=np.uint8), format="rgb24")
        track.push_frame(source)

        first = await track.recv()
        track.signal_done()
        held = await track.recv()
        return first.to_ndarray(format="rgb24")[0, 0, 0], held.to_ndarray(format="rgb24")[0, 0, 0]

    assert asyncio.run(receive_frames()) == (77, 77)


def test_chunk_router_counts_only_video_chunks() -> None:
    async def output():
        yield {"type": "status", "stage": "ready"}
        yield {"type": "chunk", "frames": [Image.new("RGB", (2, 2))]}

    data_channel_send = MagicMock()
    router = ChunkRouter(
        generator=output(),
        video_track=MagicMock(),
        audio_track=None,
        data_channel_send=data_channel_send,
        session_id="count-test",
    )

    asyncio.run(router.run())

    done = json.loads(data_channel_send.call_args.args[0])
    assert done["type"] == "done"
    assert done["total_chunks"] == 1


def test_output_completion_schedules_transport_cleanup() -> None:
    manager = WebRTCSessionManager(terminal_grace_seconds=0, close_on_output_complete=True)
    close_session = MagicMock()

    async def close(*args, **kwargs):
        close_session(*args, **kwargs)
        return True

    manager.close_session = close

    async def run() -> None:
        manager._schedule_output_complete("completed-session")
        await asyncio.sleep(0)

    asyncio.run(run())

    close_session.assert_called_once_with("completed-session", reason="output_complete")


def test_output_completion_keeps_session_open_by_default() -> None:
    manager = WebRTCSessionManager(terminal_grace_seconds=0)
    close_session = MagicMock()

    async def run() -> None:
        manager.close_session = close_session
        manager._schedule_output_complete("completed-session")
        await asyncio.sleep(0)

    asyncio.run(run())

    close_session.assert_not_called()


def test_missing_data_channel_closes_bidirectional_session_after_timeout() -> None:
    manager = WebRTCSessionManager(data_channel_timeout_seconds=0.001)
    manager._sessions["missing-channel"] = _BidirectionalSession(
        pc=MagicMock(),
        session_id="missing-channel",
    )
    manager.close_session = AsyncMock(return_value=True)

    asyncio.run(manager._close_if_data_channel_missing("missing-channel"))

    manager.close_session.assert_awaited_once_with("missing-channel", reason="data_channel_timeout")


def test_disconnect_grace_does_not_close_a_recovered_session() -> None:
    manager = WebRTCSessionManager(disconnected_grace_seconds=0)
    peer_connection = MagicMock()
    peer_connection.connectionState = "connected"
    manager._sessions["recovered"] = _BidirectionalSession(pc=peer_connection, session_id="recovered")
    manager.close_session = AsyncMock(return_value=True)

    asyncio.run(manager._close_after_disconnect_grace("recovered"))

    manager.close_session.assert_not_awaited()


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
