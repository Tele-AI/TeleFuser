from __future__ import annotations

import asyncio
import sys
import types

import numpy as np
import pytest

from telefuser.service.livekit.room_client import LiveKitRoomClient


def test_livekit_room_client_uses_sdk_room_publish_and_video_source(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeVideoFrame:
        def __init__(self, width: int, height: int, buffer_type: str, data: bytes) -> None:
            captured["frame"] = {
                "width": width,
                "height": height,
                "buffer_type": buffer_type,
                "data": data,
            }

    class FakeVideoSource:
        def __init__(self, width: int, height: int) -> None:
            captured["source"] = (width, height)
            self.frames = []

        def capture_frame(self, frame) -> None:
            self.frames.append(frame)
            captured["captured_frame"] = frame

        async def aclose(self) -> None:
            captured["source_closed"] = True

    class FakeLocalVideoTrack:
        @staticmethod
        def create_video_track(name: str, source: FakeVideoSource):
            captured["track"] = (name, source)
            return object()

    class FakeAudioFrame:
        def __init__(
            self,
            data: bytes,
            sample_rate: int,
            channels: int,
            samples_per_channel: int,
        ) -> None:
            captured["audio_frame"] = (data, sample_rate, channels, samples_per_channel)

    class FakeAudioSource:
        def __init__(self, sample_rate: int, channels: int) -> None:
            captured["audio_source"] = (sample_rate, channels)

        async def capture_frame(self, frame) -> None:
            captured["captured_audio_frame"] = frame

        async def aclose(self) -> None:
            captured["audio_source_closed"] = True

    class FakeLocalAudioTrack:
        @staticmethod
        def create_audio_track(name: str, source: FakeAudioSource):
            captured["audio_track"] = (name, source)
            return object()

    class FakePublication:
        sid = "track-sid"

    class FakeLocalParticipant:
        async def publish_track(self, track, options):
            captured["published_track"] = (track, options)
            return FakePublication()

        async def publish_data(self, data: bytes, *, topic: str, reliable: bool) -> None:
            captured["published_data"] = (data, topic, reliable)

        async def unpublish_track(self, sid: str) -> None:
            captured["unpublished_track"] = sid

    class FakeRoom:
        def __init__(self) -> None:
            self.local_participant = FakeLocalParticipant()
            self.handlers = {}
            self.remote_participants = {}

        def on(self, event: str):
            def _decorator(fn):
                self.handlers[event] = fn
                return fn

            return _decorator

        def off(self, event: str, fn) -> None:
            if self.handlers.get(event) is fn:
                self.handlers.pop(event)

        async def connect(self, url: str, token: str) -> None:
            captured["connect"] = (url, token)

        async def disconnect(self) -> None:
            captured["disconnect"] = True

    fake_room = FakeRoom()
    fake_rtc = types.SimpleNamespace(
        Room=lambda: fake_room,
        TrackPublishOptions=lambda **kwargs: types.SimpleNamespace(**kwargs),
        TrackSource=types.SimpleNamespace(SOURCE_CAMERA="camera"),
        VideoEncoding=lambda **kwargs: types.SimpleNamespace(**kwargs),
        VideoCodec=types.SimpleNamespace(VP8="VP8"),
        VideoSource=FakeVideoSource,
        LocalVideoTrack=FakeLocalVideoTrack,
        VideoFrame=FakeVideoFrame,
        VideoBufferType=types.SimpleNamespace(RGB24="RGB24"),
        AudioSource=FakeAudioSource,
        LocalAudioTrack=FakeLocalAudioTrack,
        AudioFrame=FakeAudioFrame,
    )
    fake_rtc.TrackSource.SOURCE_MICROPHONE = "microphone"
    monkeypatch.setitem(sys.modules, "livekit", types.SimpleNamespace(rtc=fake_rtc))

    async def _run() -> None:
        messages = []
        client = LiveKitRoomClient()
        await client.connect(
            "wss://livekit.example", "token", lambda data, topic, identity: messages.append((data, topic, identity))
        )

        participant = types.SimpleNamespace(identity="controller")
        packet = types.SimpleNamespace(data=b"{}", topic="tf.control", participant=participant)
        fake_room.handlers["data_received"](packet)

        controller = types.SimpleNamespace(identity="controller")
        wait_task = asyncio.create_task(client.wait_for_participant("controller", timeout_s=1.0))
        await asyncio.sleep(0)
        fake_room.remote_participants["controller"] = controller
        fake_room.handlers["participant_connected"](controller)
        await wait_task

        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        await client.publish_video_frame(frame, fps=16)
        _video_track, video_options = captured["published_track"]
        with pytest.raises(ValueError, match="dimensions changed"):
            await client.publish_video_frame(np.zeros((4, 3, 3), dtype=np.uint8), fps=16)
        pcm = np.zeros(960, dtype=np.int16).tobytes()
        await client.publish_audio_frame(pcm, sample_rate=48_000, channels=1)
        with pytest.raises(ValueError, match="audio format changed"):
            await client.publish_audio_frame(pcm, sample_rate=24_000, channels=1)
        await client.publish_status({"type": "status"})
        await client.disconnect()

        assert messages == [(b"{}", "tf.control", "controller")]
        assert captured["connect"] == ("wss://livekit.example", "token")
        assert captured["source"] == (3, 2)
        assert video_options.simulcast is False
        assert video_options.video_codec == "VP8"
        assert video_options.video_encoding.max_framerate == 16
        assert video_options.video_encoding.max_bitrate == 8_000_000
        assert captured["frame"]["buffer_type"] == "RGB24"
        assert captured["audio_source"] == (48_000, 1)
        assert captured["audio_frame"] == (pcm, 48_000, 1, 960)
        assert "captured_audio_frame" in captured
        assert captured["published_data"] == (b'{"type": "status"}', "tf.status", True)
        assert captured["unpublished_track"] == "track-sid"
        assert captured["source_closed"] is True
        assert captured["audio_source_closed"] is True
        assert captured["disconnect"] is True

    asyncio.run(_run())
