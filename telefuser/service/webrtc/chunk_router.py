"""Fan-out adapter: consumes one output generator and routes to tracks + DataChannel.

The ``ChunkRouter`` reads chunks from a ``BidirectionalService.pull_chunks()``
async generator exactly once and dispatches them:

* ``frames_b64`` / ``audio_b64``  →  decoded and pushed to the outgoing
  ``FrameGeneratorTrack`` / ``AudioGeneratorTrack`` (RTP media)
* Remaining metadata fields  →  serialised to JSON and sent over the
  client-created DataChannel as ``StreamChunkMessage`` / ``StreamDoneMessage``

This avoids double-consuming the generator and prevents duplicate sends.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator, Callable

import av
import cv2
import numpy as np
from PIL import Image

from telefuser.service.api.stream_schema import StreamChunkMessage, StreamDoneMessage, serialisable_chunk
from telefuser.utils.logging import logger

_MEDIA_KEYS = frozenset({"frames_b64", "audio_b64", "audio_sample_rate", "audio_channels"})


class ChunkRouter:
    """Consumes an output generator once, routes media to tracks and metadata to DataChannel."""

    def __init__(
        self,
        generator: AsyncGenerator[dict, None],
        video_track: object | None,
        audio_track: object | None,
        data_channel_send: Callable[[str], None] | None,
        session_id: str,
        on_complete: Callable[[str], None] | None = None,
    ) -> None:
        self._generator = generator
        self._video_track = video_track
        self._audio_track = audio_track
        self._dc_send = data_channel_send
        self._session_id = session_id
        self._on_complete = on_complete
        self._chunk_count = 0

    async def run(self) -> None:
        """Main loop: consume generator, dispatch chunks."""
        cancelled = False
        try:
            async for chunk in self._generator:
                if self._route_chunk(chunk):
                    self._chunk_count += 1
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as exc:
            logger.error(f"ChunkRouter error: session={self._session_id} {exc}")
        finally:
            self._send_done()
            if self._video_track is not None:
                self._video_track.signal_done()
            if self._audio_track is not None:
                self._audio_track.signal_done()
            if not cancelled and self._on_complete is not None:
                self._on_complete(self._session_id)

    def _route_chunk(self, chunk: dict) -> bool:
        is_nested = "frames_b64" not in chunk and isinstance(chunk.get("data"), dict)
        data = chunk.get("data", {}) if is_nested else chunk

        raw_frames = data.get("frames")
        frames: list[object] | tuple[object, ...] = raw_frames if isinstance(raw_frames, (list, tuple)) else ()
        frames_b64: list[str] = data.get("frames_b64", [])
        is_video_chunk = bool(frames or frames_b64)
        if self._video_track is not None:
            if frames:
                converted_frames: list[av.VideoFrame] = []
                for source in frames:
                    if isinstance(source, av.VideoFrame):
                        frame = source
                    elif isinstance(source, Image.Image):
                        rgb = np.ascontiguousarray(source.convert("RGB"))
                        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    else:
                        logger.warning(f"ChunkRouter dropped unsupported raw video frame: session={self._session_id}")
                        continue
                    converted_frames.append(frame)
                push_frames = getattr(type(self._video_track), "push_frames", None)
                if callable(push_frames):
                    self._video_track.push_frames(converted_frames)
                else:
                    for frame in converted_frames:
                        self._video_track.push_frame(frame)
            else:
                for fb64 in frames_b64:
                    raw = base64.b64decode(fb64)
                    np_arr = np.frombuffer(raw, dtype=np.uint8)
                    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if bgr is None:
                        logger.warning(f"ChunkRouter dropped undecodable video frame: session={self._session_id}")
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                    self._video_track.push_frame(frame)

        audio_b64 = data.get("audio_b64")
        if audio_b64 and self._audio_track is not None:
            self._audio_track.feed(base64.b64decode(audio_b64))

        media_keys = _MEDIA_KEYS | {"frames"} if isinstance(raw_frames, (list, tuple)) else _MEDIA_KEYS
        metadata = {k: v for k, v in chunk.items() if k not in media_keys}
        if is_nested and isinstance(metadata.get("data"), dict):
            metadata["data"] = {k: v for k, v in metadata["data"].items() if k not in media_keys}
            if not metadata["data"]:
                del metadata["data"]
        if metadata and self._dc_send is not None:
            msg = StreamChunkMessage(
                session_id=self._session_id,
                index=chunk.get("index"),
                data=serialisable_chunk(metadata),
            )
            try:
                self._dc_send(msg.model_dump_json())
            except Exception as exc:
                logger.warning(f"ChunkRouter metadata send failed: session={self._session_id} {exc}")
        return is_video_chunk

    def _send_done(self) -> None:
        if self._dc_send is None:
            return
        done = StreamDoneMessage(
            session_id=self._session_id,
            total_chunks=self._chunk_count,
        )
        try:
            self._dc_send(done.model_dump_json())
        except Exception as exc:
            logger.warning(f"ChunkRouter done send failed: session={self._session_id} {exc}")
