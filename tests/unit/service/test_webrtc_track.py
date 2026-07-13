from __future__ import annotations

import asyncio
import time

import av
import numpy as np
from telefuser.service.webrtc.track import FrameGeneratorTrack


def test_video_track_does_not_catch_up_after_receiver_stall() -> None:
    async def scenario() -> float:
        track = FrameGeneratorTrack(generator=None, fps=20)
        for _ in range(3):
            track.push_frame(
                av.VideoFrame.from_ndarray(
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    format="rgb24",
                )
            )

        await track.recv()
        await asyncio.sleep(0.12)
        await track.recv()
        third_started_at = time.perf_counter()
        await track.recv()
        third_delay_s = time.perf_counter() - third_started_at
        track.stop()
        return third_delay_s

    assert asyncio.run(scenario()) >= 0.04
