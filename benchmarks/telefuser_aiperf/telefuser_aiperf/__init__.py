"""TeleFuser-owned AIPerf integration."""

from __future__ import annotations

from aiperf.streaming.adapters import register_stream_adapter

from telefuser_aiperf.adapter import TeleFuserLiveKitAdapter


def register_adapters(*, replace: bool = False) -> None:
    """Register TeleFuser adapters in the current AIPerf process."""

    register_stream_adapter(
        "telefuser_livekit",
        TeleFuserLiveKitAdapter,
        replace=replace,
    )


__all__ = ["TeleFuserLiveKitAdapter", "register_adapters"]
