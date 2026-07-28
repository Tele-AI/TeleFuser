from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

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
