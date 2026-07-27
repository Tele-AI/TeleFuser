"""Stream pipeline: replay a local video file as server-push chunks.

Reads a video (with optional audio) into memory at startup, then streams
frame chunks at configurable pacing.  Audio is included as raw PCM16
base64 when the source contains an audio track.

Usage:
    telefuser stream-serve examples/stream_server/stream_video_replay.py \
        --livekit-url ws://127.0.0.1:7880 \
        --livekit-api-key devkey \
        --livekit-api-secret secret \
        -p 8088 \
        --skip-validation
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import av
import cv2
import numpy as np

VIDEO_PATH = str(Path(__file__).parent / "data" / "liveact_1.mp4")
FRAMES_PER_CHUNK = 32
OUTPUT_FPS = 24
AUDIO_SAMPLE_RATE = 48_000


class VideoReplayService:
    """Server-push service: loads a video file, streams frame chunks."""

    def __init__(
        self,
        video_path: str = VIDEO_PATH,
        frames_per_chunk: int = FRAMES_PER_CHUNK,
        fps: int = OUTPUT_FPS,
    ):
        self._video_path = video_path
        self._frames_per_chunk = frames_per_chunk
        self._fps = fps
        self._frames: list[np.ndarray] = []
        self._frames_b64: list[str] = []
        self._audio_pcm: np.ndarray | None = None
        self._audio_sample_rate = AUDIO_SAMPLE_RATE

    def start(self) -> None:
        container = av.open(self._video_path)

        for frame in container.decode(video=0):
            bgr = frame.to_ndarray(format="bgr24")
            self._frames.append(bgr)

        if container.streams.audio:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_SAMPLE_RATE)
            container.seek(0)
            audio_chunks: list[np.ndarray] = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray().flatten().astype(np.int16)
                    audio_chunks.append(arr)
            if audio_chunks:
                self._audio_pcm = np.concatenate(audio_chunks)

        container.close()

        for bgr in self._frames:
            _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self._frames_b64.append(base64.b64encode(buf.tobytes()).decode("ascii"))

        audio_info = ""
        if self._audio_pcm is not None:
            audio_info = f", {len(self._audio_pcm)} audio samples ({len(self._audio_pcm) / AUDIO_SAMPLE_RATE:.1f}s @ {AUDIO_SAMPLE_RATE}Hz)"
        print(f"[VideoReplayService] Loaded {len(self._frames)} frames{audio_info} from {self._video_path}")

    def stop(self) -> None:
        self._frames.clear()
        self._frames_b64.clear()
        self._audio_pcm = None

    async def serve(self, request: dict) -> AsyncGenerator[dict, None]:
        prompt = request.get("prompt", "")
        duration_s = request.get("duration_s", None)
        realtime = request.get("realtime", False)
        src_len = len(self._frames)

        if duration_s is not None:
            target_frames = int(float(duration_s) * self._fps)
        else:
            target_frames = src_len

        chunk_idx = 0
        emitted = 0

        while emitted < target_frames:
            remaining = target_frames - emitted
            n_frames = min(self._frames_per_chunk, remaining)

            encoded = []
            for i in range(n_frames):
                encoded.append(self._frames_b64[(emitted + i) % src_len])
            await asyncio.sleep(1.33)

            first_frame = self._frames[(emitted) % src_len]
            chunk: dict = {
                "type": "chunk",
                "index": chunk_idx,
                "frames_b64": encoded,
                "num_frames": n_frames,
                "fps": self._fps,
                "resolution": f"{first_frame.shape[1]}x{first_frame.shape[0]}",
                "prompt": prompt,
                "timestamp": time.time(),
            }

            if self._audio_pcm is not None:
                audio_len = len(self._audio_pcm)
                audio_start = emitted * self._audio_sample_rate // self._fps
                audio_end = (emitted + n_frames) * self._audio_sample_rate // self._fps
                audio_start = min(audio_start, audio_len)
                audio_end = min(audio_end, audio_len)
                audio_slice = self._audio_pcm[audio_start:audio_end]
                if len(audio_slice) > 0:
                    chunk["audio_b64"] = base64.b64encode(audio_slice.tobytes()).decode("ascii")
                    chunk["audio_sample_rate"] = self._audio_sample_rate
                    chunk["audio_channels"] = 1

            yield chunk

            emitted += n_frames
            chunk_idx += 1

            if emitted < target_frames:
                if realtime:
                    await asyncio.sleep(n_frames / self._fps)
                else:
                    await asyncio.sleep(0)


def get_service() -> VideoReplayService:
    return VideoReplayService()
