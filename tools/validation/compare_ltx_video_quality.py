"""Compare aligned LTX videos against a BF16 reference with FFmpeg metrics."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _run_metric(candidate: Path, reference: Path, metric: str) -> str:
    command = [
        "ffmpeg",
        "-v",
        "info",
        "-i",
        str(candidate),
        "-i",
        str(reference),
        "-lavfi",
        f"[0:v][1:v]{metric}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stderr


def parse_metrics(psnr_log: str, ssim_log: str, vif_log: str) -> dict[str, float | list[float]]:
    """Parse FFmpeg's aggregate full-reference metrics."""
    psnr_match = re.search(r"PSNR .* average:([0-9.]+)", psnr_log)
    ssim_match = re.search(r"SSIM .* All:([0-9.]+)", ssim_log)
    vif_scales = [float(value) for value in re.findall(r"VIF scale=\d+ average:([0-9.]+)", vif_log)]
    if psnr_match is None or ssim_match is None or len(vif_scales) != 4:
        raise ValueError("Could not parse PSNR, SSIM, and four VIF scales from FFmpeg output")
    psnr = float(psnr_match.group(1))
    return {
        "psnr_db": psnr,
        "normalized_rmse": 10 ** (-psnr / 20),
        "ssim": float(ssim_match.group(1)),
        "vif_scales": vif_scales,
        "vif_mean": sum(vif_scales) / len(vif_scales),
    }


def compare(candidate: Path, reference: Path) -> dict[str, float | list[float]]:
    """Compare one candidate video to the aligned BF16 reference."""
    for path in (candidate, reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    return parse_metrics(
        _run_metric(candidate, reference, "psnr"),
        _run_metric(candidate, reference, "ssim"),
        _run_metric(candidate, reference, "vif"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Aligned BF16 MP4")
    parser.add_argument("--candidate", type=Path, action="append", required=True, help="Candidate MP4; repeatable")
    parser.add_argument("--label", action="append", help="Candidate label; repeat in candidate order")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    labels = args.label or [path.stem for path in args.candidate]
    if len(labels) != len(args.candidate):
        parser.error("--label count must match --candidate count")
    report = {
        "reference": str(args.reference),
        "metric_direction": {
            "normalized_rmse": "lower_is_better",
            "psnr_db": "higher_is_better",
            "ssim": "higher_is_better",
            "vif_mean": "higher_is_better",
        },
        "candidates": {
            label: {"path": str(path), **compare(path, args.reference)} for label, path in zip(labels, args.candidate)
        },
    }
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
