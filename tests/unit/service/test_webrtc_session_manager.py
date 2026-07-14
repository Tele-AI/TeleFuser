from __future__ import annotations

import asyncio
import threading

import pytest

pytest.importorskip("aiortc")

from telefuser.service.webrtc.session_manager import WebRTCSessionManager, _BidirectionalSession


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
