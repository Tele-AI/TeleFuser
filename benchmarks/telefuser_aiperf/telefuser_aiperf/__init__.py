"""TeleFuser-owned AIPerf integration."""

from __future__ import annotations

from aiperf.streaming.adapters import register_stream_adapter

from telefuser_aiperf.adapter import TeleFuserLiveKitAdapter
from telefuser_aiperf.sglang_adapter import SGLangRealtimeAdapter


def register_adapters(*, replace: bool = False) -> None:
    """Register TeleFuser adapters in the current AIPerf process."""

    register_stream_adapter(
        "telefuser_livekit",
        TeleFuserLiveKitAdapter,
        replace=replace,
    )
    register_stream_adapter(
        "sglang_realtime",
        SGLangRealtimeAdapter,
        replace=replace,
    )


__all__ = ["SGLangRealtimeAdapter", "TeleFuserLiveKitAdapter", "register_adapters"]
