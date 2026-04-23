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
    ) -> None:
        self._generator = generator
        self._video_track = video_track
        self._audio_track = audio_track
        self._dc_send = data_channel_send
        self._session_id = session_id
        self._chunk_count = 0

    async def run(self) -> None:
        """Main loop: consume generator, dispatch chunks."""
        try:
            async for chunk in self._generator:
                self._route_chunk(chunk)
                self._chunk_count += 1
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"ChunkRouter error: session={self._session_id} {exc}")
        finally:
            self._send_done()
            if self._audio_track is not None:
                self._audio_track.signal_done()

    def _route_chunk(self, chunk: dict) -> None:
        is_nested = "frames_b64" not in chunk and isinstance(chunk.get("data"), dict)
        data = chunk.get("data", {}) if is_nested else chunk

        frames_b64: list[str] = data.get("frames_b64", [])
        if frames_b64 and self._video_track is not None:
            for fb64 in frames_b64:
                raw = base64.b64decode(fb64)
                np_arr = np.frombuffer(raw, dtype=np.uint8)
                bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                self._video_track.push_frame(frame)

        audio_b64 = data.get("audio_b64")
        if audio_b64 and self._audio_track is not None:
            self._audio_track.feed(base64.b64decode(audio_b64))

        metadata = {k: v for k, v in chunk.items() if k not in _MEDIA_KEYS}
        if is_nested and isinstance(metadata.get("data"), dict):
            metadata["data"] = {k: v for k, v in metadata["data"].items() if k not in _MEDIA_KEYS}
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
            except Exception:
                pass

    def _send_done(self) -> None:
        if self._dc_send is None:
            return
        done = StreamDoneMessage(
            session_id=self._session_id,
            total_chunks=self._chunk_count,
        )
        try:
            self._dc_send(done.model_dump_json())
        except Exception:
            pass
