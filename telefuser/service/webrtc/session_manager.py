"""WebRTC session manager — tracks RTCPeerConnection lifecycle.

Supports two session types:

* **_Session** (server-push): output-only video/audio tracks, no DataChannel.
* **_BidirectionalSession**: DataChannel for JSON control, optional incoming
  and outgoing media tracks, ChunkRouter for fan-out.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from telefuser.utils.logging import logger

from .chunk_router import ChunkRouter
from .track import (
    AudioGeneratorTrack,
    FrameGeneratorTrack,
    IncomingAudioRelay,
    IncomingVideoRelay,
)

_SENTINEL = object()


@dataclass
class _Session:
    pc: RTCPeerConnection
    track: FrameGeneratorTrack
    audio_track: AudioGeneratorTrack | None = None


@dataclass
class _BidirectionalSession:
    pc: RTCPeerConnection
    on_close: Callable[[str], None] | None = None
    session_id: str = ""
    data_channel: object | None = None
    output_video_track: FrameGeneratorTrack | None = None
    output_audio_track: AudioGeneratorTrack | None = None
    chunk_router: ChunkRouter | None = None
    router_task: asyncio.Task | None = None
    relay_tasks: list[asyncio.Task] = field(default_factory=list)


class WebRTCSessionManager:
    """Creates and manages WebRTC peer connections for stream sessions."""

    def __init__(self, max_sessions: int = 10, configuration: RTCConfiguration | None = None) -> None:
        self._sessions: dict[str, _Session | _BidirectionalSession | object] = {}
        self._max_sessions = max_sessions
        self._configuration = configuration
        self._lock = asyncio.Lock()

    # -- Server-push sessions ------------------------------------------------

    async def create_session(
        self,
        session_id: str,
        offer_sdp: str,
        offer_type: str,
        generator: AsyncGenerator[dict, None],
        fps: int = 24,
    ) -> tuple[str, str]:
        """Process SDP offer and return (answer_sdp, answer_type)."""
        async with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(f"Max WebRTC sessions ({self._max_sessions}) reached")
            if session_id in self._sessions:
                raise RuntimeError(f"Session {session_id} already exists")
            self._sessions[session_id] = _SENTINEL

        try:
            pc = RTCPeerConnection(configuration=self._configuration or RTCConfiguration())

            has_audio_offer = "m=audio" in offer_sdp
            audio_track: AudioGeneratorTrack | None = None
            if has_audio_offer:
                audio_track = AudioGeneratorTrack()

            track = FrameGeneratorTrack(generator, fps=fps, audio_track=audio_track)

            @pc.on("connectionstatechange")
            async def _on_state_change() -> None:
                state = pc.connectionState
                if state in ("failed", "disconnected", "closed"):
                    logger.info(f"WebRTC connection {state}: session={session_id}")
                    await self.close_session(session_id, reason=f"connection_{state}")

            pc.addTrack(track)
            if audio_track is not None:
                pc.addTrack(audio_track)

            offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            async with self._lock:
                self._sessions[session_id] = _Session(pc=pc, track=track, audio_track=audio_track)

            audio_status = " + audio" if audio_track else ""
            logger.info(f"WebRTC session created: session={session_id}{audio_status}")
            return pc.localDescription.sdp, pc.localDescription.type
        except BaseException:
            async with self._lock:
                self._sessions.pop(session_id, None)
            raise

    # -- Bidirectional sessions ----------------------------------------------

    async def create_bidirectional_session(
        self,
        session_id: str,
        offer_sdp: str,
        offer_type: str,
        output_generator: AsyncGenerator[dict, None],
        on_input: Callable[[str, dict], None],
        on_close: Callable[[str], None],
        fps: int = 24,
    ) -> tuple[str, str]:
        """Create a bidirectional WebRTC session.

        The client must create a DataChannel named ``"telefuser"`` before
        generating the SDP offer.  The server reuses that single channel
        for both reading input and writing output.

        Args:
            session_id: Unique session identifier (from pipeline).
            offer_sdp: Client SDP offer string.
            offer_type: SDP type (usually ``"offer"``).
            output_generator: Async generator from ``pull_chunks()``.
            on_input: Callback ``(session_id, chunk_dict)`` for incoming data.
            on_close: Callback ``(session_id)`` when client sends ``stop``.
            fps: Target video FPS for outgoing media tracks.

        Returns:
            ``(answer_sdp, answer_type)`` tuple.
        """
        async with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(f"Max WebRTC sessions ({self._max_sessions}) reached")
            if session_id in self._sessions:
                raise RuntimeError(f"Session {session_id} already exists")
            self._sessions[session_id] = _SENTINEL

        try:
            pc = RTCPeerConnection(configuration=self._configuration or RTCConfiguration())
            session = _BidirectionalSession(pc=pc, on_close=on_close, session_id=session_id)

            has_audio_offer = "m=audio" in offer_sdp
            output_video = FrameGeneratorTrack(generator=None, fps=fps)
            output_audio: AudioGeneratorTrack | None = None
            if has_audio_offer:
                output_audio = AudioGeneratorTrack()

            session.output_video_track = output_video
            session.output_audio_track = output_audio

            pc.addTrack(output_video)
            if output_audio is not None:
                pc.addTrack(output_audio)

            # --- DataChannel (client-created, server reuses) ----------------

            @pc.on("datachannel")
            def _on_datachannel(channel) -> None:
                session.data_channel = channel
                logger.info(f"DataChannel received: session={session_id} label={channel.label}")

                @channel.on("message")
                def _on_message(message) -> None:
                    try:
                        data = json.loads(message) if isinstance(message, str) else message
                    except (json.JSONDecodeError, TypeError) as exc:
                        logger.warning(f"DataChannel message decode failed: session={session_id} {exc}")
                        return
                    if isinstance(data, dict) and data.get("type") == "stop":
                        asyncio.ensure_future(self.close_session(session_id, reason="client_stop"))
                        return
                    try:
                        on_input(session_id, data)
                    except Exception as exc:
                        logger.warning(f"DataChannel input callback failed: session={session_id} {exc}")

                router = ChunkRouter(
                    generator=output_generator,
                    video_track=output_video,
                    audio_track=output_audio,
                    data_channel_send=channel.send,
                    session_id=session_id,
                )
                session.chunk_router = router
                session.router_task = asyncio.ensure_future(router.run())

            # --- Incoming media tracks (optional) ---------------------------

            @pc.on("track")
            def _on_track(track) -> None:
                logger.info(f"Incoming track: session={session_id} kind={track.kind}")
                if track.kind == "video":
                    relay = IncomingVideoRelay(track, session_id, on_input)
                    task = asyncio.ensure_future(relay.run())
                    session.relay_tasks.append(task)
                elif track.kind == "audio":
                    relay = IncomingAudioRelay(track, session_id, on_input)
                    task = asyncio.ensure_future(relay.run())
                    session.relay_tasks.append(task)

            # --- Connection state -------------------------------------------

            @pc.on("connectionstatechange")
            async def _on_state_change() -> None:
                state = pc.connectionState
                if state in ("failed", "disconnected", "closed"):
                    logger.info(f"WebRTC connection {state}: session={session_id}")
                    await self.close_session(session_id, reason=f"connection_{state}")

            # --- SDP exchange -----------------------------------------------

            offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            async with self._lock:
                self._sessions[session_id] = session

            logger.info(f"WebRTC bidirectional session created: session={session_id}")
            return pc.localDescription.sdp, pc.localDescription.type
        except BaseException:
            async with self._lock:
                self._sessions.pop(session_id, None)
            raise

    # -- Session lifecycle ---------------------------------------------------

    async def close_session(self, session_id: str, *, reason: str = "api", notify_pipeline: bool = True) -> bool:
        async with self._lock:
            entry = self._sessions.pop(session_id, None)
        if entry is None or entry is _SENTINEL:
            return False

        logger.info(f"Closing WebRTC session: session={session_id} reason={reason}")

        if isinstance(entry, _Session):
            entry.track.stop()
            if entry.track._task is not None and not entry.track._task.done():
                try:
                    await entry.track._task
                except asyncio.CancelledError:
                    logger.info(f"WebRTC server-push track cancelled: session={session_id}")
                except Exception as exc:
                    logger.warning(f"WebRTC server-push track close failed: session={session_id} {exc}")
        elif isinstance(entry, _BidirectionalSession):
            for task in entry.relay_tasks:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"WebRTC relay task cancelled: session={session_id}")
                except Exception as exc:
                    logger.warning(f"WebRTC relay task close failed: session={session_id} {exc}")
            if entry.router_task is not None and not entry.router_task.done():
                entry.router_task.cancel()
                try:
                    await entry.router_task
                except asyncio.CancelledError:
                    logger.info(f"WebRTC chunk router cancelled: session={session_id}")
                except Exception as exc:
                    logger.warning(f"WebRTC chunk router close failed: session={session_id} {exc}")
            if entry.output_video_track is not None:
                entry.output_video_track.stop()
            if entry.output_audio_track is not None:
                entry.output_audio_track.stop()
            if notify_pipeline and entry.on_close is not None:
                try:
                    await asyncio.to_thread(entry.on_close, entry.session_id)
                except Exception as exc:
                    logger.warning(f"WebRTC pipeline close callback failed: session={session_id} {exc}")

        try:
            await asyncio.wait_for(entry.pc.close(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timed out closing WebRTC peer connection: session={session_id}")
        except Exception as exc:
            logger.warning(f"WebRTC peer connection close failed: session={session_id} {exc}")
        logger.info(f"WebRTC session closed: session={session_id} reason={reason}")
        return True

    async def close_all(self) -> None:
        async with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            await self.close_session(sid)

    def has_session(self, session_id: str) -> bool:
        entry = self._sessions.get(session_id)
        return entry is not None and entry is not _SENTINEL

    @property
    def active_session_count(self) -> int:
        return sum(1 for v in self._sessions.values() if v is not _SENTINEL)

    @property
    def server_push_session_count(self) -> int:
        return sum(1 for v in self._sessions.values() if isinstance(v, _Session))

    @property
    def bidirectional_session_count(self) -> int:
        return sum(1 for v in self._sessions.values() if isinstance(v, _BidirectionalSession))

    def session_stats(self) -> dict:
        """Single-pass session counts (avoids 3 iterations over the dict)."""
        active = server_push = bidirectional = 0
        for v in self._sessions.values():
            if v is _SENTINEL:
                continue
            active += 1
            if isinstance(v, _Session):
                server_push += 1
            elif isinstance(v, _BidirectionalSession):
                bidirectional += 1
        return {
            "webrtc_active_sessions": active,
            "webrtc_server_push_sessions": server_push,
            "webrtc_bidirectional_sessions": bidirectional,
            "webrtc_max_sessions": self._max_sessions,
        }
