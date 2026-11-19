"""Small benchmark helper for an existing TeleFuser LightX2V example."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    record = {
        "model": "LightX2V Wan2.2 distilled",
        "frames": args.frames,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "elapsed_seconds": elapsed,
        "note": "Pass this helper's output through the existing example runner; it records the standard JSON schema.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
