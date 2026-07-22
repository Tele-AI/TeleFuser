"""Tests for the LingBot-Video service entrypoint contract."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from telefuser.pipelines.lingbot_video import DEFAULT_NEGATIVE_PROMPT, DEFAULT_NEGATIVE_PROMPT_IMAGE
from telefuser.service.core.pipeline_contract import load_pipeline_contract


def _load_service_module():
    path = Path("examples/lingbot_video/lingbot_video_service.py")
    spec = importlib.util.spec_from_file_location("lingbot_video_service_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_service_contract_declares_t2i_t2v_and_i2v() -> None:
    module = _load_service_module()

    contract, is_explicit = load_pipeline_contract(module, ppl_file="lingbot_video_service.py", default_task="t2v")

    assert is_explicit
    assert contract.supported_tasks == ("t2i", "t2v", "i2v")
    assert contract.get_task_contract("t2i").media_type == "image"
    assert contract.get_task_contract("t2i").parameters["negative_prompt"].default == DEFAULT_NEGATIVE_PROMPT_IMAGE
    assert contract.get_task_contract("t2v").parameters["refine"].type == "boolean"
    assert contract.get_task_contract("t2v").parameters["negative_prompt"].default == DEFAULT_NEGATIVE_PROMPT
    assert contract.get_task_contract("i2v").required_inputs == ("first_image_path",)


def test_service_writes_single_image_for_t2i(tmp_path: Path) -> None:
    module = _load_service_module()

    class FakePipeline:
        variant = "dense"

        def generate(self, request, **_: object) -> SimpleNamespace:
            assert request.num_frames == 1
            assert request.caption == '{"scene":"test"}'
            assert (request.height, request.width) == (480, 864)
            return SimpleNamespace(output=torch.zeros(1, 3, 1, 2, 2))

    output_path = tmp_path / "result.png"
    result = module.run_with_file(
        FakePipeline(),
        json.dumps({"caption": {"scene": "test"}, "duration": 5}),
        task="t2i",
        output_path=str(output_path),
    )

    assert result == {"output_path": str(output_path)}
    assert output_path.is_file()


def test_service_writes_video_for_t2v(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_service_module()
    exported: list[object] = []

    import diffusers.utils

    def capture_export(frames, output_path: str, fps: int) -> str:
        del output_path, fps
        exported.extend(frames)
        return "unused.mp4"

    monkeypatch.setattr(diffusers.utils, "export_to_video", capture_export)

    class FakePipeline:
        variant = "dense"

        def generate(self, request, **_: object) -> SimpleNamespace:
            assert request.num_frames == 97
            assert (request.height, request.width) == (480, 864)
            return SimpleNamespace(output=torch.full((1, 3, 5, 2, 2), 0.5))

    output_path = tmp_path / "result.mp4"
    result = module.run_with_file(
        FakePipeline(),
        json.dumps({"caption": {"scene": "test"}, "duration": 5}),
        task="t2v",
        output_path=str(output_path),
    )

    assert result == {"output_path": str(output_path)}
    assert len(exported) == 5
    assert exported[0].dtype == np.float32
    assert float(exported[0][0, 0, 0]) == 0.5


def test_service_resolution_is_aligned_to_lingbot_vae_and_dit() -> None:
    module = _load_service_module()

    for resolution in ("480p", "720p", "1080p", "2k", "4k"):
        width, height = module._lingbot_video_size("16:9", resolution)
        assert width % 16 == 0
        assert height % 16 == 0


def test_service_rejects_i2v_without_first_image_path() -> None:
    module = _load_service_module()

    with pytest.raises(ValueError, match="requires first_image_path"):
        module.run_with_file(
            object(),
            json.dumps({"caption": {"scene": "test"}, "duration": 5}),
            task="i2v",
        )


def test_service_runs_base_then_refiner_on_one_pipeline(tmp_path: Path, monkeypatch) -> None:
    module = _load_service_module()

    class FakeTextStage:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, caption: str) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            assert caption in {'{"scene":"test"}', DEFAULT_NEGATIVE_PROMPT_IMAGE}
            return torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool)

    class FakePipeline:
        variant = "moe"
        model_dir = "fake-model"
        text_stage = FakeTextStage()

        def __init__(self) -> None:
            self.released = False

        def generate(self, request, **_: object) -> SimpleNamespace:
            assert request.num_frames == 1
            conditions = SimpleNamespace(
                has_visual_condition=False,
                positive_prompt_embeds=torch.zeros(1, 1, 2),
                negative_prompt_embeds=torch.zeros(1, 1, 2),
                positive_attention_mask=torch.ones(1, 1, dtype=torch.bool),
                negative_attention_mask=torch.ones(1, 1, dtype=torch.bool),
            )
            return SimpleNamespace(output=torch.zeros(1, 3, 1, 2, 2), prompt_conditions=conditions)

        def release_gpu_resources(self) -> None:
            self.released = True

    class FakeRefiner:
        def __init__(self) -> None:
            self.called = False

        def refine(self, lowres_video, *args, **kwargs) -> torch.Tensor:
            del args, kwargs
            self.called = True
            assert lowres_video.shape == (1, 3, 1, 2, 2)
            return lowres_video

    pipeline = FakePipeline()
    refiner = FakeRefiner()
    monkeypatch.setattr(module, "build_lingbot_video_refiner_stage", lambda _: refiner)
    monkeypatch.setitem(module.PPL_CONFIG, "refiner_height", 2)
    monkeypatch.setitem(module.PPL_CONFIG, "refiner_width", 2)

    output_path = tmp_path / "refined.png"
    result = module.run_with_file(
        pipeline,
        json.dumps({"caption": {"scene": "test"}, "duration": 5}),
        task="t2i",
        refine=True,
        output_path=str(output_path),
    )

    assert result == {"output_path": str(output_path)}
    assert pipeline.released
    assert refiner.called
    assert pipeline.text_stage.calls == 0
    assert output_path.is_file()


def test_service_reencodes_text_only_conditions_for_ti2v_refiner(tmp_path: Path, monkeypatch) -> None:
    module = _load_service_module()

    class FakeTextStage:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, caption: str) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            assert caption in {'{"scene":"test"}', DEFAULT_NEGATIVE_PROMPT}
            return torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool)

    class FakePipeline:
        variant = "moe"
        model_dir = "fake-model"
        text_stage = FakeTextStage()

        def generate(self, request, **_: object) -> SimpleNamespace:
            assert request.image is not None
            conditions = SimpleNamespace(has_visual_condition=True)
            return SimpleNamespace(output=torch.zeros(1, 3, 1, 2, 2), prompt_conditions=conditions)

        def release_gpu_resources(self) -> None:
            return None

    class FakeRefiner:
        def refine(self, lowres_video, *args, **kwargs) -> torch.Tensor:
            del args, kwargs
            return lowres_video

    monkeypatch.setattr(module, "build_lingbot_video_refiner_stage", lambda _: FakeRefiner())
    monkeypatch.setitem(module.PPL_CONFIG, "refiner_height", 2)
    monkeypatch.setitem(module.PPL_CONFIG, "refiner_width", 2)
    image_path = tmp_path / "first.png"
    Image.new("RGB", (2, 2)).save(image_path)

    module.run_with_file(
        FakePipeline(),
        json.dumps({"caption": {"scene": "test"}, "duration": 5}),
        task="i2v",
        first_image_path=str(image_path),
        refine=True,
        output_path=str(tmp_path / "refined.mp4"),
    )

    assert FakePipeline.text_stage.calls == 2
