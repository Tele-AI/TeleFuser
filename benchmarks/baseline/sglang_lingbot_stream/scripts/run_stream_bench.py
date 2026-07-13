#!/usr/bin/env python3
"""Compatibility launcher that delegates SGLang stream profiling to AIPerf."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=(
            "benchmarks/baseline/sglang_lingbot_stream/configs/"
            "stream_lingbot_world_fast_quick.json"
        ),
    )
    parser.add_argument("--server-url")
    parser.add_argument("--artifacts-dir")
    args, remainder = parser.parse_known_args()

    root = Path(__file__).resolve().parents[4]
    aiperf_repo = Path(
        os.environ.get("AIPERF_REPO", root / "benchmarks" / "aiperf")
    )
    command = [
        os.environ.get("AIPERF_UV_BIN", "uv"),
        "run",
        "--project",
        str(aiperf_repo),
        "aiperf",
        "profile",
        "--stream-config",
        args.config,
    ]
    if args.server_url:
        command.extend(["--stream-server-url", args.server_url])
    if args.artifacts_dir:
        command.extend(["--stream-artifacts-dir", args.artifacts_dir])
    os.execvp(command[0], [*command, *remainder])


if __name__ == "__main__":
    main()
