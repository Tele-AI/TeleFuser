"""LiveKit room client abstraction and SDK-backed implementation."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from .data_protocol import TF_METRICS_TOPIC, TF_STATUS_TOPIC
from .token_service import LiveKitDependencyError

DataMessageHandler = Callable[[bytes | str | dict[str, Any], str, str], None]
_VIDEO_MAX_BITRATE = 3_000_000


class RoomClient(Protocol):
    """Minimal room operations required by a TeleFuser LiveKit worker."""

    async def connect(self, url: str, token: str, on_data: DataMessageHandler) -> None: ...
    async def publish_video_track(self, name: str, width: int, height: int, *, fps: float = 16.0) -> None: ...
    async def publish_video_frame(self, frame_rgb: np.ndarray, *, fps: float = 16.0) -> None: ...
    async def publish_audio_frame(self, pcm: bytes, *, sample_rate: int, channels: int) -> None: ...
    async def publish_status(self, payload: dict[str, Any]) -> None: ...
    async def publish_metrics(self, payload: dict[str, Any]) -> None: ...
    async def disconnect(self) -> None: ...


class LiveKitRoomClient:
    """SDK-backed LiveKit room client.

    Imports LiveKit lazily so configuration and API schemas remain importable
    before the runtime worker starts.
    """

    def __init__(self) -> None:
        self._rtc: Any | None = None
        self._room: Any | None = None
        self._video_source: Any | None = None
        self._video_track: Any | None = None
        self._video_track_sid: str | None = None
        self._video_dimensions: tuple[int, int] | None = None
        self._audio_source: Any | None = None
        self._audio_track: Any | None = None
        self._audio_track_sid: str | None = None
        self._audio_format: tuple[int, int] | None = None

    async def connect(self, url: str, token: str, on_data: DataMessageHandler) -> None:
        rtc = self._load_rtc()
        self._rtc = rtc
        room = rtc.Room()
        self._room = room

        @room.on("data_received")
        def _on_data_received(packet: Any) -> None:
            participant = getattr(packet, "participant", None)
            identity = getattr(participant, "identity", "") if participant is not None else ""
            on_data(packet.data, packet.topic or "", identity)

        await room.connect(url, token)

    async def publish_video_track(self, name: str, width: int, height: int, *, fps: float = 16.0) -> None:
        room = self._require_room()
        rtc = self._require_rtc()
        if self._video_source is not None:
            return

        self._video_source = rtc.VideoSource(width, height)
        self._video_track = rtc.LocalVideoTrack.create_video_track(name, self._video_source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,
            video_encoding=rtc.VideoEncoding(
                max_framerate=fps,
                max_bitrate=_VIDEO_MAX_BITRATE,
            ),
            video_codec=rtc.VideoCodec.VP8,
        )
        publication = await room.local_participant.publish_track(self._video_track, options)
        self._video_track_sid = getattr(publication, "sid", None)
        self._video_dimensions = (width, height)

    async def publish_video_frame(self, frame_rgb: np.ndarray, *, fps: float = 16.0) -> None:
        if self._video_source is None:
            height, width = frame_rgb.shape[:2]
            await self.publish_video_track("telefuser-output", width, height, fps=fps)

        rtc = self._require_rtc()
        height, width = frame_rgb.shape[:2]
        if self._video_dimensions != (width, height):
            raise ValueError(f"LiveKit video dimensions changed from {self._video_dimensions} to {(width, height)}")

        contiguous = np.ascontiguousarray(frame_rgb)
        frame = rtc.VideoFrame(width, height, rtc.VideoBufferType.RGB24, contiguous.tobytes())
        self._video_source.capture_frame(frame)

    async def publish_audio_frame(self, pcm: bytes, *, sample_rate: int, channels: int) -> None:
        """Publish one PCM16 audio payload through a LiveKit audio source."""
        if sample_rate <= 0 or channels <= 0:
            raise ValueError("LiveKit audio sample rate and channel count must be positive")
        bytes_per_sample = channels * 2
        if not pcm or len(pcm) % bytes_per_sample:
            raise ValueError("LiveKit PCM16 byte length must align with its channel count")

        room = self._require_room()
        rtc = self._require_rtc()
        audio_format = (sample_rate, channels)
        if self._audio_source is None:
            self._audio_source = rtc.AudioSource(sample_rate, channels)
            self._audio_track = rtc.LocalAudioTrack.create_audio_track("telefuser-audio", self._audio_source)
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            publication = await room.local_participant.publish_track(self._audio_track, options)
            self._audio_track_sid = getattr(publication, "sid", None)
            self._audio_format = audio_format
        elif self._audio_format != audio_format:
            raise ValueError(f"LiveKit audio format changed from {self._audio_format} to {audio_format}")

        samples_per_channel = len(pcm) // bytes_per_sample
        frame = rtc.AudioFrame(pcm, sample_rate, channels, samples_per_channel)
        await self._audio_source.capture_frame(frame)

    async def publish_status(self, payload: dict[str, Any]) -> None:
        await self._publish_data(payload, topic=TF_STATUS_TOPIC, reliable=True)

    async def publish_metrics(self, payload: dict[str, Any]) -> None:
        await self._publish_data(payload, topic=TF_METRICS_TOPIC, reliable=False)

    async def disconnect(self) -> None:
        room = self._room
        if room is None:
            return

        try:
            if self._video_track_sid:
                try:
                    await room.local_participant.unpublish_track(self._video_track_sid)
                except Exception:
                    pass
            if self._audio_track_sid:
                try:
                    await room.local_participant.unpublish_track(self._audio_track_sid)
                except Exception:
                    pass
            if self._video_source is not None:
                await self._video_source.aclose()
            if self._audio_source is not None:
                await self._audio_source.aclose()
        finally:
            try:
                await room.disconnect()
            finally:
                self._room = None
                self._video_source = None
                self._video_track = None
                self._video_track_sid = None
                self._video_dimensions = None
                self._audio_source = None
                self._audio_track = None
                self._audio_track_sid = None
                self._audio_format = None

    async def _publish_data(self, payload: dict[str, Any], *, topic: str, reliable: bool) -> None:
        room = self._require_room()
        await room.local_participant.publish_data(json.dumps(payload).encode("utf-8"), topic=topic, reliable=reliable)

    @staticmethod
    def _load_rtc() -> Any:
        try:
            from livekit import rtc
        except ModuleNotFoundError as exc:
            raise LiveKitDependencyError(
                "LiveKit RTC SDK is required for LiveKit worker connections. "
                "Install the declared TeleFuser runtime dependencies."
            ) from exc
        return rtc

    def _require_room(self) -> Any:
        if self._room is None:
            raise RuntimeError("LiveKit room is not connected")
        return self._room

    def _require_rtc(self) -> Any:
        if self._rtc is None:
            raise RuntimeError("LiveKit RTC SDK is not loaded")
        return self._rtc
