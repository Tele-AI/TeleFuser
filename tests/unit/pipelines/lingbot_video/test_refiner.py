from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from telefuser.pipelines.lingbot_video.refiner import (
    compute_refiner_sigmas,
    compute_training_aligned_indices,
    compute_training_frame_budget,
    load_refiner_first_frame,
    load_refiner_video_file,
    prepare_refiner_latent,
    prepare_refiner_video,
)
from tools.validation.capture_lingbot_video_reference import _upstream_import_path


def test_refiner_schedule_starts_at_threshold_and_descends() -> None:
    sigmas = compute_refiner_sigmas(
        sigma_max=1.0,
        sigma_min=0.0,
        num_inference_steps=8,
        shift=3.0,
        t_thresh=0.85,
        tail_steps=2,
    )

    assert sigmas is not None
    assert sigmas[0] == 0.85
    assert all(left > right for left, right in zip(sigmas, sigmas[1:]))


def test_refiner_latent_mixing() -> None:
    mixed = prepare_refiner_latent(torch.ones(1, 1, 1), torch.zeros(1, 1, 1), 0.25)

    assert torch.allclose(mixed, torch.full_like(mixed, 0.75))


def test_training_aligned_refiner_handoff_matches_upstream_sampling_contract() -> None:
    frames, vae_fps, temporal_latents = compute_training_frame_budget(10, 48.0, sample_fps=24, vae_tc=4)

    assert (frames, vae_fps, temporal_latents) == (5, 24.0, 2)
    assert torch.equal(compute_training_aligned_indices(10, frames), torch.tensor([0, 2, 4, 6, 9]))

    video = torch.linspace(0.0, 1.0, 10).reshape(1, 1, 10, 1, 1).repeat(1, 3, 1, 2, 2)
    prepared, metadata = prepare_refiner_video(video, source_fps=48.0, height=2, width=2)

    assert prepared.shape == (1, 3, 5, 2, 2)
    assert torch.equal(prepared[:, :, :, 0, 0], video[:, :, [0, 2, 4, 6, 9], 0, 0])
    assert metadata == {
        "src_fps": 48.0,
        "sample_frame": 5,
        "sample_frame_uncapped": 5,
        "max_frames": None,
        "truncated_by_max_frames": False,
        "vae_fps": 24.0,
        "t_vae": 2,
        "num_source_frames": 10,
        "align_to_training": True,
    }


def test_pyav_mp4_refiner_handoff_uses_training_aligned_sampling(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "base.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=48)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    try:
        for value in range(10):
            image = np.full((16, 16, 3), value * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    video, metadata = load_refiner_video_file(path, height=16, width=16, sample_fps=24, vae_tc=4)

    assert video.shape == (1, 3, 5, 16, 16)
    assert metadata["sample_frame"] == 5
    assert metadata["t_vae"] == 2
    assert metadata["src_fps"] == 48.0


def test_mp4_refiner_handoff_matches_upstream_with_pyav_decord_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "base.mp4"
    from diffusers.utils import export_to_video

    frames = [np.full((16, 16, 3), value * 20, dtype=np.uint8) for value in range(10)]
    export_to_video(frames, str(path), fps=48)

    class _Batch:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

        def asnumpy(self) -> np.ndarray:
            return self.values

    class _VideoReader:
        def __init__(self, source: str, ctx: object) -> None:
            del ctx
            decoded = av.open(source)
            try:
                video_stream = next(iter(decoded.streams.video))
                self.fps = float(video_stream.average_rate)
                self.frames = [frame.to_ndarray(format="rgb24") for frame in decoded.decode(video_stream)]
            finally:
                decoded.close()

        def __len__(self) -> int:
            return len(self.frames)

        def get_avg_fps(self) -> float:
            return self.fps

        def get_batch(self, indices: np.ndarray) -> _Batch:
            return _Batch(np.stack([self.frames[int(index)] for index in indices]))

    decord = types.ModuleType("decord")
    decord.VideoReader = _VideoReader
    decord.cpu = lambda _: object()
    monkeypatch.setitem(sys.modules, "decord", decord)
    upstream_root = Path("work_dirs/lingbot-video-master")
    with _upstream_import_path(upstream_root):
        upstream_utils = importlib.import_module("lingbot_video.utils")
        reference, reference_metadata = upstream_utils.load_refiner_video_tensor(path, 16, 16, sample_fps=24, vae_tc=4)
    candidate, candidate_metadata = load_refiner_video_file(path, height=16, width=16, sample_fps=24, vae_tc=4)

    assert reference_metadata == candidate_metadata
    assert torch.equal(reference, candidate)


def test_ti2v_refiner_first_frame_matches_upstream_geometry(tmp_path: Path) -> None:
    from PIL import Image

    image = np.arange(9 * 15 * 3, dtype=np.uint8).reshape(9, 15, 3)
    path = tmp_path / "first.png"
    Image.fromarray(image).save(path)
    upstream_root = Path("work_dirs/lingbot-video-master")
    with _upstream_import_path(upstream_root):
        upstream_utils = importlib.import_module("lingbot_video.utils")
        reference = upstream_utils.load_first_frame_condition_tensor(path, 8, 12, 6, 10)
    candidate = load_refiner_first_frame(path, target_height=8, target_width=12, geometry_height=6, geometry_width=10)

    assert torch.equal(reference, candidate)
