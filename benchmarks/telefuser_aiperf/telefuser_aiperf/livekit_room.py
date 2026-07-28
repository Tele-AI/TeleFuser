"""Minimal LiveKit room client used by the TeleFuser AIPerf adapter."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
from collections.abc import Callable, Mapping
from typing import Any, Protocol
from urllib.parse import urlsplit

import orjson

LiveKitDataHandler = Callable[[bytes | str, str, str], None]
LiveKitFrameHandler = Callable[[], None]
LiveKitEventHandler = Callable[[str, Mapping[str, Any]], None]

_PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def _disable_proxy_for_loopback(url: str) -> bool:
    host = urlsplit(url).hostname
    if host is None:
        return False
    try:
        is_loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        return False
    for name in _PROXY_ENV_NAMES:
        os.environ.pop(name, None)
    return True


class LiveKitRoomClientProtocol(Protocol):
    """Minimal LiveKit subscriber surface required by the stream adapter."""

    async def connect(
        self,
        url: str,
        token: str,
        *,
        timeout_s: float,
        on_data: LiveKitDataHandler,
        on_video_frame: LiveKitFrameHandler,
        on_event: LiveKitEventHandler,
    ) -> None: ...

    async def publish_data(
        self,
        payload: dict[str, Any],
        *,
        topic: str,
        reliable: bool,
    ) -> None: ...

    async def disconnect(self) -> None: ...


class LiveKitDependencyError(RuntimeError):
    """Raised when the LiveKit SDK is unavailable."""


class LiveKitRoomClient:
    """SDK-backed LiveKit room subscriber used by the TeleFuser adapter."""

    def __init__(self) -> None:
        self._rtc = self._load_rtc()
        self._room: Any | None = None
        self._video_streams: list[Any] = []
        self._video_tasks: list[asyncio.Task[None]] = []
        self._on_video_frame: LiveKitFrameHandler | None = None
        self._on_event: LiveKitEventHandler | None = None

    async def connect(
        self,
        url: str,
        token: str,
        *,
        timeout_s: float,
        on_data: LiveKitDataHandler,
        on_video_frame: LiveKitFrameHandler,
        on_event: LiveKitEventHandler,
    ) -> None:
        self._on_video_frame = on_video_frame
        self._on_event = on_event
        room = self._rtc.Room()
        self._room = room

        @room.on("data_received")
        def _on_data_received(packet: Any) -> None:
            participant = getattr(packet, "participant", None)
            identity = getattr(participant, "identity", "") if participant is not None else ""
            on_data(packet.data, packet.topic or "", identity)

        @room.on("track_subscribed")
        def _on_track_subscribed(
            track: Any,
            publication: Any,
            participant: Any,
        ) -> None:
            if track.kind != self._rtc.TrackKind.KIND_VIDEO:
                return
            stream = self._rtc.VideoStream(track)
            self._video_streams.append(stream)
            self._video_tasks.append(asyncio.create_task(self._consume_video(stream)))
            on_event(
                "remote_track",
                {
                    "kind": "video",
                    "participant_identity": getattr(participant, "identity", ""),
                    "track_sid": getattr(publication, "sid", ""),
                },
            )

        @room.on("reconnecting")
        def _on_reconnecting() -> None:
            on_event("reconnecting", {})

        @room.on("reconnected")
        def _on_reconnected() -> None:
            on_event("reconnected", {})

        @room.on("disconnected")
        def _on_disconnected(reason: Any) -> None:
            on_event("disconnected", {"reason": str(reason)})

        options = self._rtc.RoomOptions(
            auto_subscribe=True,
            connect_timeout=timeout_s,
        )
        _disable_proxy_for_loopback(url)
        await room.connect(url, token, options)

    async def _consume_video(self, stream: Any) -> None:
        try:
            async for _ in stream:
                if self._on_video_frame is not None:
                    self._on_video_frame()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - remote media termination is data
            if self._on_event is not None:
                self._on_event(
                    "video_stream_ended",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

    async def publish_data(
        self,
        payload: dict[str, Any],
        *,
        topic: str,
        reliable: bool,
    ) -> None:
        room = self._require_room()
        await room.local_participant.publish_data(
            orjson.dumps(payload),
            topic=topic,
            reliable=reliable,
        )

    async def disconnect(self) -> None:
        room = self._room
        if room is None:
            return
        try:
            for stream in self._video_streams:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(stream.aclose(), timeout=2.0)
            for task in self._video_tasks:
                if not task.done():
                    task.cancel()
            if self._video_tasks:
                await asyncio.gather(*self._video_tasks, return_exceptions=True)
            await room.disconnect()
        finally:
            self._room = None
            self._video_streams.clear()
            self._video_tasks.clear()

    @staticmethod
    def _load_rtc() -> Any:
        try:
            from livekit import rtc
        except ModuleNotFoundError as exc:
            raise LiveKitDependencyError("The TeleFuser AIPerf adapter requires the 'livekit' package") from exc
        return rtc

    def _require_room(self) -> Any:
        if self._room is None:
            raise RuntimeError("LiveKit room is not connected")
        return self._room
