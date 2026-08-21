from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

import examples.run_examples as run_examples
from examples.run_examples import _close_pipeline


class _ClosablePipeline:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.fail:
            raise RuntimeError("close failed")


def test_close_pipeline_releases_owned_workers() -> None:
    pipeline = _ClosablePipeline()

    _close_pipeline(pipeline)

    assert pipeline.close_calls == 1


def test_close_pipeline_does_not_mask_regression_result(capsys: pytest.CaptureFixture[str]) -> None:
    pipeline = _ClosablePipeline(fail=True)

    _close_pipeline(pipeline)

    assert pipeline.close_calls == 1
    assert "Warning: failed to close pipeline: close failed" in capsys.readouterr().err


@pytest.mark.parametrize("failure_site", ["validate", "filename", "move", "emit"])
def test_run_single_closes_pipeline_for_all_post_load_failures(
    failure_site: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _ClosablePipeline()
    module = ModuleType("test_example")
    module.PPL_CONFIG = {}
    config = run_examples.Config(
        output_root=str(tmp_path / "output-root"),
        pipelines={"test": run_examples.PipelineConfig(script="test_example.py")},
    )
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    temp_path = tmp_path / "temporary.mp4"
    temp_path.write_bytes(b"video")

    def raise_lifecycle_error(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(f"injected {failure_site} failure")

    monkeypatch.setattr(run_examples, "load_config", lambda _path: config)
    monkeypatch.setattr(run_examples, "_import_example_module", lambda _path: module)
    monkeypatch.setattr(run_examples, "_patch_ppl_config", lambda _module, _overrides: None)
    monkeypatch.setattr(run_examples, "_call_get_pipeline", lambda _module, _config: pipeline)
    monkeypatch.setattr(run_examples, "_call_run", lambda _module, _pipeline, _config: object())
    monkeypatch.setattr(run_examples.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run_examples, "_validate_output", lambda _output: [])
    monkeypatch.setattr(
        run_examples,
        "_save_output",
        lambda _output, _temp_dir, _output_type, fps: (str(temp_path), 1, "1x1"),
    )
    monkeypatch.setattr(run_examples, "_generate_output_filename", lambda *_args: "result.mp4")

    if failure_site == "validate":
        monkeypatch.setattr(run_examples, "_validate_output", raise_lifecycle_error)
    elif failure_site == "filename":
        monkeypatch.setattr(run_examples, "_generate_output_filename", raise_lifecycle_error)
    elif failure_site == "move":
        monkeypatch.setattr(run_examples.shutil, "move", raise_lifecycle_error)
    else:
        monkeypatch.setattr(run_examples, "_emit_result", raise_lifecycle_error)

    with pytest.raises(RuntimeError, match=f"injected {failure_site} failure"):
        run_examples._run_single("test", None, str(output_dir))

    assert pipeline.close_calls == 1


def test_save_output_writes_tensor_video(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved_frames: list[np.ndarray] = []

    def capture_save_video(frames: list[np.ndarray], path: str, fps: float, quality: int) -> None:
        assert path == str(tmp_path / "output.mp4")
        assert fps == 24
        assert quality == 6
        saved_frames.extend(frames)

    from telefuser.utils import video as video_utils

    monkeypatch.setattr(video_utils, "save_video", capture_save_video)
    output = torch.zeros(1, 3, 2, 4, 6)

    path, frames, resolution = run_examples._save_output(output, str(tmp_path), "video", fps=24)

    assert path == str(tmp_path / "output.mp4")
    assert frames == 2
    assert resolution == "6x4"
    assert len(saved_frames) == 2
    assert saved_frames[0].shape == (4, 6, 3)
    assert saved_frames[0].dtype == np.uint8


def test_save_output_accepts_file_entrypoint_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = tmp_path / "joint-audio-video.mp4"
    artifact.write_bytes(b"artifact")

    class FakeVideoData:
        width = 1360
        height = 768

        def __init__(self, video_file: str) -> None:
            assert video_file == str(artifact)

        def __len__(self) -> int:
            return 121

    from telefuser.utils import video as video_utils

    monkeypatch.setattr(video_utils, "VideoData", FakeVideoData)

    path, frames, resolution = run_examples._save_output({"output_path": str(artifact)}, str(tmp_path), "video")

    assert path == str(artifact)
    assert frames == 121
    assert resolution == "1360x768"


def test_call_get_pipeline_forwards_matching_config_overrides() -> None:
    module = ModuleType("pipeline_config_example")

    def get_pipeline(parallelism: int, expert_backend: str, refiner_batch_cfg: bool) -> tuple[int, str, bool]:
        return parallelism, expert_backend, refiner_batch_cfg

    module.get_pipeline = get_pipeline

    assert run_examples._call_get_pipeline(
        module,
        {"gpu_count": 4, "expert_backend": "sorted", "refiner_batch_cfg": True},
    ) == (4, "sorted", True)


def test_prepare_sdpa_regression_forces_sdpa_and_disables_xformers(monkeypatch: pytest.MonkeyPatch) -> None:
    from diffusers.utils import import_utils

    monkeypatch.setattr(import_utils, "_xformers_available", True)

    overrides = run_examples._prepare_sdpa_regression({"attn_impl": "FLASH_ATTN_4", "compile": True})

    assert overrides == {"attn_impl": "TORCH_SDPA", "compile": True}
    assert not import_utils._xformers_available


def test_call_run_preserves_missing_negative_prompt_default() -> None:
    module = ModuleType("negative_prompt_example")

    def run(
        pipeline: object, negative_prompt: str | None = None, target_video_length: int | None = None
    ) -> tuple[str | None, int | None]:
        del pipeline
        return negative_prompt, target_video_length

    module.run = run

    assert run_examples._call_run(module, object(), {"target_video_length": 2}) == (None, 2)


def test_call_run_supports_standard_file_entrypoint() -> None:
    module = ModuleType("file_example")

    def run_with_file(pipeline: object, output_path: str, target_video_length: float) -> dict[str, str]:
        del pipeline
        assert target_video_length == 5
        return {"output_path": output_path}

    module.run_with_file = run_with_file

    assert run_examples._call_run(
        module,
        object(),
        {"output_path": "/tmp/result.mp4", "target_video_length": 5},
        entrypoint="run_with_file",
    ) == {"output_path": "/tmp/result.mp4"}


def test_call_run_injects_configured_i2v_input_image_path(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    module = ModuleType("i2v_example")

    def run_with_file(
        pipeline: object,
        input_image_path: str,
        prompt: str,
        output_path: str,
        num_frames: int,
    ) -> tuple[object, str, str, str, int]:
        return pipeline, input_image_path, prompt, output_path, num_frames

    module.run_with_file = run_with_file
    result = run_examples._call_run(
        module,
        "pipeline",
        {
            "input_image_path": str(image_path),
            "prompt": "test prompt",
            "output_path": "/tmp/result.mp4",
            "num_frames": 121,
        },
        entrypoint="run_with_file",
    )

    assert result == ("pipeline", str(image_path), "test prompt", "/tmp/result.mp4", 121)


def test_ltx25_two_gpu_regressions_are_registered() -> None:
    config = run_examples.load_config()

    t2v = config.pipelines["ltx25_distilled_t2v_2gpu"]
    assert t2v.script == "ltx25_distilled/ltx25_distilled_t2v_h100.py"
    assert t2v.gpu_count == 2
    assert (t2v.width, t2v.height) == (1536, 1024)
    assert t2v.use_run_with_file and t2v.require_audio
    assert t2v.ppl_config_overrides["num_frames"] == 121

    i2v = config.pipelines["ltx25_distilled_i2v_2gpu"]
    assert i2v.script == "ltx25_distilled/ltx25_distilled_i2v_h100.py"
    assert i2v.gpu_count == 2
    assert (i2v.width, i2v.height) == (896, 512)
    assert i2v.input_image_path == "examples/data/ltx25/official_guitar_man.png"
    assert i2v.use_run_with_file and i2v.require_audio
    assert i2v.ppl_config_overrides["num_frames"] == 121


def test_video_metrics_rejects_mismatched_frame_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from telefuser.utils import video as video_utils

    class FakeVideoData:
        def __init__(self, video_file: str) -> None:
            self.width = 8
            self.height = 8
            self._frame_count = 2 if video_file == "baseline.mp4" else 1

        def __len__(self) -> int:
            return self._frame_count

        def fps(self) -> float:
            return 24.0

    monkeypatch.setattr(video_utils, "VideoData", FakeVideoData)

    with pytest.raises(ValueError, match="frame count mismatch"):
        run_examples.compute_video_metrics("baseline.mp4", "current.mp4")


def test_comparison_requires_an_explicit_baseline_initialization(tmp_path: Path) -> None:
    current = tmp_path / "current.png"
    current.write_bytes(b"image")

    result = run_examples.compare_against_baseline(
        str(tmp_path), "qwen/example.py", 1, str(current), "image", 25.0, 0.85, 0.02
    )

    assert not result["passed"]
    assert not result["baseline_exists"]
    assert not (tmp_path / "baseline").exists()

    initialized = run_examples.compare_against_baseline(
        str(tmp_path), "qwen/example.py", 1, str(current), "image", 25.0, 0.85, 0.02, update_baseline=True
    )

    assert initialized["passed"]
    assert (tmp_path / "baseline" / current.name).is_file()


@pytest.mark.parametrize("metrics", [{}, {"psnr": 30.0}])
def test_video_comparison_rejects_missing_metrics(
    metrics: dict[str, float], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = tmp_path / "baseline.mp4"
    current = tmp_path / "current.mp4"
    baseline.write_bytes(b"baseline")
    current.write_bytes(b"current")
    monkeypatch.setattr(run_examples, "_get_baseline_path", lambda *_args: str(baseline))
    monkeypatch.setattr(run_examples, "compute_video_metrics", lambda *_args, **_kwargs: metrics)

    result = run_examples.compare_against_baseline(
        str(tmp_path), "wan/example.py", 1, str(current), "video", 25.0, 0.85, 0.02
    )

    assert not result["passed"]
    assert result["message"] == "Video comparison produced no PSNR/SSIM metrics"


def test_required_audio_metrics_compare_stream_contract_and_waveform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_info = {"sample_rate": 32_000, "channels": 2, "duration": 5.0}
    waveforms = {
        "baseline.mp4": np.array([0.5, -0.25, 0.1, -0.2], dtype=np.float32),
        "current.mp4": np.array([0.49, -0.24, 0.11, -0.19], dtype=np.float32),
    }
    monkeypatch.setattr(run_examples, "_probe_audio_stream", lambda _path: stream_info)
    monkeypatch.setattr(
        run_examples,
        "_decode_audio",
        lambda path, **_kwargs: waveforms[path],
    )

    metrics = run_examples._compute_audio_metrics("baseline.mp4", "current.mp4", required=True)

    assert metrics["audio_cosine"] > 0.99
    assert metrics["audio_duration_delta"] == 0.0


def test_required_audio_metrics_reject_missing_current_track(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_info = {"sample_rate": 32_000, "channels": 2, "duration": 5.0}
    monkeypatch.setattr(
        run_examples,
        "_probe_audio_stream",
        lambda path: baseline_info if path == "baseline.mp4" else None,
    )

    with pytest.raises(ValueError, match="Current video has no audio stream"):
        run_examples._compute_audio_metrics("baseline.mp4", "current.mp4", required=True)


def test_video_comparison_enforces_required_audio_cosine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.mp4"
    current = tmp_path / "current.mp4"
    baseline.write_bytes(b"baseline")
    current.write_bytes(b"current")
    monkeypatch.setattr(run_examples, "_get_baseline_path", lambda *_args: str(baseline))
    monkeypatch.setattr(
        run_examples,
        "compute_video_metrics",
        lambda *_args, **_kwargs: {"psnr": 30.0, "ssim": 0.95, "audio_cosine": 0.5},
    )

    result = run_examples.compare_against_baseline(
        str(tmp_path),
        "minimax_h3/example.py",
        4,
        str(current),
        "video",
        25.0,
        0.85,
        0.02,
        require_audio=True,
        audio_cosine_min=0.95,
    )

    assert not result["passed"]
    assert "audio cosine 0.5000 < 0.9500" in result["message"]


def test_video_comparison_rejects_unavailable_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.mp4"
    current = tmp_path / "current.mp4"
    baseline.write_bytes(b"baseline")
    current.write_bytes(b"current")
    monkeypatch.setattr(run_examples, "_get_baseline_path", lambda *_args: str(baseline))

    def raise_missing_dependency(*_args: object, **_kwargs: object) -> dict[str, float]:
        raise ModuleNotFoundError("skimage")

    monkeypatch.setattr(run_examples, "compute_video_metrics", raise_missing_dependency)

    result = run_examples.compare_against_baseline(
        str(tmp_path), "wan/example.py", 1, str(current), "video", 25.0, 0.85, 0.02
    )

    assert not result["passed"]
    assert "Comparison unavailable" in result["message"]


def test_resume_only_skips_recorded_passes_with_output(tmp_path: Path) -> None:
    pipelines = {
        "passed": run_examples.PipelineConfig(script="wan/passed.py", output_type="image"),
        "failed": run_examples.PipelineConfig(script="wan/failed.py", output_type="image"),
        "unreported": run_examples.PipelineConfig(script="wan/unreported.py", output_type="image"),
    }
    for name in pipelines:
        (tmp_path / f"wan__{name}_1gpu_1x1.png").write_bytes(b"output")
    (tmp_path / "example_report.json").write_text(
        json.dumps({"results": {"passed": {"status": "PASS"}, "failed": {"status": "FAIL"}}}),
        encoding="utf-8",
    )

    assert run_examples._get_completed_pipelines(str(tmp_path), pipelines) == {"passed"}


def test_scheduler_records_insufficient_gpu_as_skip(tmp_path: Path) -> None:
    scheduler = run_examples.PipelineScheduler(
        run_examples.GPUPool([0]),
        {"requires_two": run_examples.PipelineConfig(script="wan/example.py", gpu_count=2)},
        str(tmp_path),
    )

    assert not scheduler.has_pending()
    assert len(scheduler.results) == 1
    assert scheduler.results[0].status == "SKIP"
    assert scheduler.results[0].error_category == "INSUFFICIENT_GPUS"


def test_main_fails_when_the_requested_matrix_is_resource_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = run_examples.Config(
        output_root=str(tmp_path),
        pipelines={"requires_two": run_examples.PipelineConfig(script="wan/example.py", gpu_count=2)},
    )
    output_dir = tmp_path / "results"
    monkeypatch.setattr(run_examples, "load_config", lambda _path: config)
    monkeypatch.setattr(run_examples, "_get_date_dir", lambda _root: str(output_dir))
    monkeypatch.setattr(run_examples.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(run_examples.sys, "argv", ["run_examples.py", "--all"])

    with pytest.raises(SystemExit, match="1"):
        run_examples.main()

    report = json.loads((output_dir / "example_report.json").read_text(encoding="utf-8"))
    assert report["results"]["requires_two"]["status"] == "SKIP"
