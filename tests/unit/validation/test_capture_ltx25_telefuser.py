"""Tests for TeleFuser LTX-2.5 capture helpers."""

from __future__ import annotations

from tools.validation.capture_ltx25_telefuser import _release_modules


def test_capture_release_modules_releases_lazy_proxy() -> None:
    class LazyProxy:
        def __init__(self) -> None:
            self.calls = 0

        def release(self) -> None:
            self.calls += 1
