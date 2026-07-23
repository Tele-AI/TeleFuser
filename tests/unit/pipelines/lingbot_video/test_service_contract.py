"""Tests for the split LingBot-Video CLI and service examples."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from telefuser.pipelines.lingbot_video import DEFAULT_NEGATIVE_PROMPT, DEFAULT_NEGATIVE_PROMPT_IMAGE
from telefuser.service.core.pipeline_contract import load_pipeline_contract

EXAMPLE_PATHS = {
    "dense": Path("examples/lingbot_video/lingbot_video_dense_1_3b.py"),
    "moe": Path("examples/lingbot_video/lingbot_video_moe_30b.py"),
}


def _load_example(variant: str) -> ModuleType:
    path = EXAMPLE_PATHS[variant]
    module_name = f"lingbot_video_{variant}_example_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_examples_expose_cli_and_service_entrypoints(variant: str) -> None:
    module = _load_example(variant)

    for name in ("PPL_CONFIG", "PIPELINE_CONTRACT", "get_pipeline", "run", "run_with_file"):
        assert hasattr(module, name)
    assert module.PPL_CONFIG["variant"] == variant

    prompt_payload = json.loads(module.PPL_CONFIG["prompt"])
    assert prompt_payload["duration"] == 5
    assert "comprehensive_description" in prompt_payload["caption"]
    contract, is_explicit = load_pipeline_contract(
        module,
        ppl_file=EXAMPLE_PATHS[variant].name,
        default_task="t2v",
    )
    assert is_explicit
    assert contract.pipeline_name == module.PPL_CONFIG["name"]
    assert contract.supported_tasks == ("t2i", "t2v", "i2v")
    assert contract.get_task_contract("t2i").parameters["negative_prompt"].default == DEFAULT_NEGATIVE_PROMPT_IMAGE
    assert contract.get_task_contract("t2v").parameters["negative_prompt"].default == DEFAULT_NEGATIVE_PROMPT
    assert contract.get_task_contract("i2v").required_inputs == ("first_image_path",)


def test_only_moe_contract_exposes_refiner() -> None:
    dense = _load_example("dense")
    moe = _load_example("moe")

    dense_contract, _ = load_pipeline_contract(dense, ppl_file=EXAMPLE_PATHS["dense"].name, default_task="t2v")
    moe_contract, _ = load_pipeline_contract(moe, ppl_file=EXAMPLE_PATHS["moe"].name, default_task="t2v")

    assert "refine" not in dense_contract.get_task_contract("t2v").parameters
    assert "refine" not in moe_contract.get_task_contract("t2i").parameters
    assert moe_contract.get_task_contract("t2v").parameters["refine"].default is True


@pytest.mark.parametrize(
    ("variant", "checkpoint"),
    [
        ("dense", "lingbot-video-dense-1.3b"),
        ("moe", "lingbot-video-moe-30b-a3b"),
    ],
)
def test_default_model_root_uses_tf_model_zoo_path(
    variant: str, checkpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TF_MODEL_ZOO_PATH", "/models")

    module = _load_example(variant)

    assert module.PPL_CONFIG["model_root"] == f"/models/lingbot/{checkpoint}"


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_get_pipeline_uses_fixed_variant(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example(variant)
    calls: list[tuple[object, dict[str, object]]] = []
    pipeline = SimpleNamespace()

    def capture_build(model_root: str, **kwargs: object) -> object:
        calls.append((model_root, kwargs))
        return pipeline

    monkeypatch.setattr(module, "build_pipeline", capture_build)
    result = module.get_pipeline(parallelism=4, model_root="checkpoint")

    assert result is pipeline
    assert calls[0][0] == "checkpoint"
    parallel_config = calls[0][1]["parallel_config"]
    assert parallel_config.device_ids == [0, 1, 2, 3]
    assert parallel_config.sp_ulysses_degree == 4
    assert parallel_config.enable_fsdp
    assert module.PPL_CONFIG["variant"] == variant


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_get_pipeline_supports_cfg_parallel_with_sequence_parallel(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_example(variant)
    calls: list[dict[str, object]] = []

    def capture_build(_: str, **kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(module, "build_pipeline", capture_build)
    pipeline = module.get_pipeline(parallelism=4, cfg_parallel_degree=2)

    parallel_config = calls[0]["parallel_config"]
    assert parallel_config.cfg_degree == 2
    assert parallel_config.sp_ulysses_degree == 2
    assert parallel_config.enable_fsdp
    if variant == "moe":
        assert pipeline.refiner_co_resident


def test_dense_run_with_file_supports_service_t2i(tmp_path: Path) -> None:
    module = _load_example("dense")

    class FakePipeline:
        device = "cpu"

        def generate(self, request, **_: object) -> SimpleNamespace:
            assert request.num_frames == 1
            assert request.caption == '{"scene":"test"}'
            assert (request.height, request.width) == (480, 832)
            return SimpleNamespace(output=torch.zeros(1, 3, 1, 2, 2))

    output_path = tmp_path / "dense.png"
    result = module.run_with_file(
        FakePipeline(),
        json.dumps({"caption": {"scene": "test"}, "duration": 5}),
        task="t2i",
        output_path=str(output_path),
    )

    assert result == {"output_path": str(output_path)}
    assert output_path.is_file()


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_t2i_replaces_video_output_suffix_with_png(variant: str, tmp_path: Path) -> None:
    module = _load_example(variant)

    result = module._save_output(torch.zeros(1, 3, 1, 2, 2), str(tmp_path / "result.mp4"))

    expected = tmp_path / "result.png"
    assert result == {"output_path": str(expected)}
    assert expected.is_file()


def test_dense_run_with_file_supports_service_t2v(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example("dense")
    exported: list[np.ndarray] = []

    def capture_export(frames: list[np.ndarray], output_path: str, fps: int) -> None:
        del output_path
        assert fps == 24
        exported.extend(frames)

    monkeypatch.setattr(module, "export_to_video", capture_export)

    class FakePipeline:
        device = "cpu"

        def generate(self, request, **_: object) -> SimpleNamespace:
            assert request.num_frames == 49
            return SimpleNamespace(output=torch.full((1, 3, 5, 2, 2), 0.5))

    result = module.run_with_file(
        FakePipeline(),
        json.dumps({"caption": {"scene": "test"}, "duration": 2}),
        task="t2v",
        output_path=str(tmp_path / "dense.mp4"),
    )

    assert result == {"output_path": str(tmp_path / "dense.mp4")}
    assert len(exported) == 5
    assert exported[0].dtype == np.float32
    assert float(exported[0][0, 0, 0]) == 0.5


def test_moe_run_releases_base_then_runs_refiner(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example("moe")

    class FakePipeline:
        device = "cpu"
        text_stage = object()

        def __init__(self) -> None:
            self.released = False

        def generate(self, request, **_: object) -> SimpleNamespace:
            del request
            conditions = SimpleNamespace(
                has_visual_condition=False,
                positive_prompt_embeds=torch.full((1, 2, 2), 4.0),
                negative_prompt_embeds=torch.full((1, 2, 2), 7.0),
                positive_attention_mask=torch.tensor([[True, False]]),
                negative_attention_mask=torch.tensor([[False, True]]),
            )
            return SimpleNamespace(output=torch.zeros(1, 3, 5, 2, 2), prompt_conditions=conditions)

        def release_gpu_resources(self) -> None:
            self.released = True

    class FakeRefiner:
        def __init__(self) -> None:
            self.called = False
            self.closed = False
            self.negative: torch.Tensor | None = None
            self.negative_mask: torch.Tensor | None = None

        def refine(self, lowres_video, positive, negative, positive_mask, negative_mask, **kwargs) -> torch.Tensor:
            del positive, positive_mask, kwargs
            self.called = True
            self.negative = negative
            self.negative_mask = negative_mask
            return lowres_video

        def close(self) -> None:
            self.closed = True

    pipeline = FakePipeline()
    refiner = FakeRefiner()
    monkeypatch.setattr(module, "build_refiner", lambda *args, **kwargs: refiner)
    monkeypatch.setitem(module.PPL_CONFIG, "refiner_height", 2)
    monkeypatch.setitem(module.PPL_CONFIG, "refiner_width", 2)

    frames = module.run(
        pipeline,
        json.dumps({"caption": {"scene": "test"}, "duration": 5}),
        task="t2v",
        refine=True,
    )

    assert frames.shape == (1, 3, 5, 2, 2)
    assert pipeline.released
    assert refiner.called
    assert refiner.closed
    assert torch.equal(refiner.negative, torch.zeros(1, 2, 2))
    assert torch.equal(refiner.negative_mask, torch.tensor([[True, False]]))


def test_moe_run_keeps_base_resident_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example("moe")

    class FakePipeline:
        device = "cpu"
        text_stage = object()
        refiner_co_resident = True

        def __init__(self) -> None:
            self.released = False

        def generate(self, request: object, **_: object) -> SimpleNamespace:
            del request
            conditions = SimpleNamespace(
                has_visual_condition=False,
                positive_prompt_embeds=torch.zeros(1, 2, 2),
                negative_prompt_embeds=torch.zeros(1, 2, 2),
                positive_attention_mask=torch.ones(1, 2, dtype=torch.bool),
                negative_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            )
            return SimpleNamespace(output=torch.zeros(1, 3, 5, 2, 2), prompt_conditions=conditions)

        def release_gpu_resources(self) -> None:
            self.released = True

    class FakeRefiner:
        def refine(self, lowres_video: torch.Tensor, *_: object, **__: object) -> torch.Tensor:
            return lowres_video

        def close(self) -> None:
            return None

    pipeline = FakePipeline()
    monkeypatch.setattr(module, "build_refiner", lambda *args, **kwargs: FakeRefiner())
    monkeypatch.setitem(module.PPL_CONFIG, "refiner_height", 2)
    monkeypatch.setitem(module.PPL_CONFIG, "refiner_width", 2)

    module.run(
        pipeline,
        json.dumps({"caption": {"scene": "test"}, "duration": 5}),
        task="t2v",
        refine=True,
    )

    assert not pipeline.released


def test_moe_t2i_rejects_explicit_refiner_request() -> None:
    module = _load_example("moe")

    with pytest.raises(ValueError, match="does not support t2i"):
        module.run(
            SimpleNamespace(),
            json.dumps({"caption": {"scene": "test"}, "duration": 5}),
            task="t2i",
            refine=True,
        )


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_cli_invokes_model_specific_entrypoints(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_example(variant)
    pipeline = SimpleNamespace(stop=lambda: None)
    calls: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(module, "get_pipeline", lambda *args, **kwargs: pipeline)

    def capture_run_with_file(pipe: object, *args: object, **kwargs: object) -> dict[str, str]:
        calls.append((pipe, {"args": args, **kwargs}))
        return {"output_path": "result.mp4"}

    monkeypatch.setattr(module, "run_with_file", capture_run_with_file)
    result = CliRunner().invoke(
        module.main,
        ["--output_path", "result.mp4", "--no-refine"] if variant == "moe" else ["--output_path", "result.mp4"],
    )

    assert result.exit_code == 0, result.output
    assert calls and calls[0][0] is pipeline
    assert "Output saved to result.mp4" in result.output
