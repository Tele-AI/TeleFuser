"""Validate a LingBot-Video refiner MP4 handoff against the upstream loader.

The upstream refiner reads the base result after it was written to MP4. The
native runtime uses an in-memory RGB tensor instead. This tool first verifies
that the compatibility MP4 path matches the upstream loader exactly, then can
optionally quantify the input change avoided by the native in-memory handoff.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from telefuser.pipelines.lingbot_video.refiner import load_refiner_video_file, prepare_refiner_video

DEFAULT_UPSTREAM_ROOT = Path("work_dirs/lingbot-video-master")


@contextlib.contextmanager
def _upstream_import_path(root: Path) -> Iterator[None]:
    """Temporarily expose the checked-out upstream package without modifying it."""
    root = root.resolve()
    if not (root / "lingbot_video" / "__init__.py").is_file():
        raise FileNotFoundError(f"LingBot-Video package not found under {root}")
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    """Return L0/L1 metrics for one refiner handoff tensor pair."""
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    ref = reference.float().cpu()
    got = candidate.float().cpu()
    delta = (got - ref).abs()
    exact_mismatch_count = int(torch.count_nonzero(reference.cpu() != candidate.cpu()).item())
    cosine = (
        1.0
        if exact_mismatch_count == 0
        else torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    )
    return {
        "shape_match": True,
        "dtype_match": reference.dtype == candidate.dtype,
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm().div(ref.norm().clamp_min(1e-12)).item()),
        "cosine": float(max(-1.0, min(1.0, cosine))),
        "exact_mismatch_count": exact_mismatch_count,
    }


def _pyav_decord_module() -> types.ModuleType:
    """Return the small decord surface used by the source refiner loader."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("upstream MP4 handoff validation requires the optional PyAV dependency") from exc

    class Batch:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

        def asnumpy(self) -> np.ndarray:
            return self.values

    class VideoReader:
        def __init__(self, source: str, ctx: object) -> None:
            del ctx
            container = av.open(source)
            try:
                stream = next(iter(container.streams.video), None)
                if stream is None:
                    raise ValueError(f"video has no video stream: {source}")
                self._fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
                self._frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
            finally:
                container.close()

        def __len__(self) -> int:
            return len(self._frames)

        def get_avg_fps(self) -> float:
            return self._fps

        def get_batch(self, indices: np.ndarray) -> Batch:
            return Batch(np.stack([self._frames[int(index)] for index in indices]))

    module = types.ModuleType("decord")
    module.VideoReader = VideoReader
    module.cpu = lambda _: object()
    return module


def load_upstream_refiner_video_file(
    path: str | Path,
    *,
    upstream_root: Path,
    height: int,
    width: int,
    sample_fps: int,
    vae_tc: int,
    max_frames: int | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the source MP4 loader without requiring decord to be installed."""
    decord = _pyav_decord_module()
    with _upstream_import_path(upstream_root):
        upstream_utils = importlib.import_module("lingbot_video.utils")
        with patch.dict(sys.modules, {"decord": decord}):
            return upstream_utils.load_refiner_video_tensor(
                path,
                height,
                width,
                sample_fps=sample_fps,
                vae_tc=vae_tc,
                max_frames=max_frames,
            )


def validate_refiner_handoff(
    path: str | Path,
    *,
    upstream_root: Path,
    height: int,
    width: int,
    sample_fps: int = 24,
    vae_tc: int = 4,
    max_frames: int | None = None,
    in_memory_video: torch.Tensor | None = None,
    in_memory_fps: float | None = None,
) -> dict[str, Any]:
    """Compare source and native MP4 loaders plus an optional memory handoff."""
    reference, reference_metadata = load_upstream_refiner_video_file(
        path,
        upstream_root=upstream_root,
        height=height,
        width=width,
        sample_fps=sample_fps,
        vae_tc=vae_tc,
        max_frames=max_frames,
    )
    candidate, candidate_metadata = load_refiner_video_file(
        path,
        height=height,
        width=width,
        sample_fps=sample_fps,
        vae_tc=vae_tc,
        max_frames=max_frames,
    )
    report: dict[str, Any] = {
        "input": str(path),
        "source_mp4_to_telefuser_mp4": tensor_metrics(reference, candidate),
        "metadata_match": reference_metadata == candidate_metadata,
        "reference_metadata": reference_metadata,
        "candidate_metadata": candidate_metadata,
    }
    if in_memory_video is not None:
        if in_memory_fps is None:
            raise ValueError("in_memory_fps is required when in_memory_video is provided")
        prepared, memory_metadata = prepare_refiner_video(
            in_memory_video,
            source_fps=in_memory_fps,
            height=height,
            width=width,
            sample_fps=sample_fps,
            vae_tc=vae_tc,
            max_frames=max_frames,
        )
        report["in_memory_to_mp4"] = tensor_metrics(prepared, candidate)
        report["in_memory_metadata"] = memory_metadata
        report["in_memory_metadata_match"] = memory_metadata == candidate_metadata
    return report


def exact_handoff_failures(report: dict[str, Any]) -> list[str]:
    """Return source-MP4 compatibility failures without judging native handoff quality."""
    metrics = report["source_mp4_to_telefuser_mp4"]
    failures: list[str] = []
    if not report["metadata_match"]:
        failures.append("metadata_match")
    for name in ("shape_match", "dtype_match"):
        if not metrics.get(name, False):
            failures.append(f"source_mp4_to_telefuser_mp4.{name}")
    for name in ("max_abs", "mean_abs", "relative_l2", "exact_mismatch_count"):
        if metrics.get(name) != 0:
            failures.append(f"source_mp4_to_telefuser_mp4.{name}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Base MP4 passed to the refiner.")
    parser.add_argument("--height", type=int, required=True, help="Refiner input height.")
    parser.add_argument("--width", type=int, required=True, help="Refiner input width.")
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--sample-fps", type=int, default=24)
    parser.add_argument("--vae-tc", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--in-memory-video",
        type=Path,
        help="Optional [B,3,F,H,W] torch tensor saved before MP4 encoding.",
    )
    parser.add_argument("--in-memory-fps", type=float, help="FPS for --in-memory-video.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--assert-exact",
        action="store_true",
        help="Exit nonzero unless the source MP4 compatibility path matches exactly.",
    )
    args = parser.parse_args()
    in_memory_video = None
    if args.in_memory_video is not None:
        in_memory_video = torch.load(args.in_memory_video, map_location="cpu", weights_only=True)
    report = validate_refiner_handoff(
        args.input,
        upstream_root=args.upstream_root,
        height=args.height,
        width=args.width,
        sample_fps=args.sample_fps,
        vae_tc=args.vae_tc,
        max_frames=args.max_frames,
        in_memory_video=in_memory_video,
        in_memory_fps=args.in_memory_fps,
    )
    if args.assert_exact:
        failures = exact_handoff_failures(report)
        if failures:
            raise SystemExit(f"Exact refiner MP4 handoff gate failed: {', '.join(failures)}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
