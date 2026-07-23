"""Tests for LingBot checkpoint assembly and memory lifecycle defaults."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from examples.lingbot_video import lingbot_video_moe_30b as moe_example
from telefuser.core.config import WeightOffloadType
from telefuser.pipelines.lingbot_video import LingBotVideoModelConfig


class _FakeModule:
    def __init__(self) -> None:
        self.device: str | None = None
        self.expert_execution_backend: str | None = None

    def to(self, device: str) -> _FakeModule:
        self.device = device
        return self

    def eval(self) -> _FakeModule:
        return self

    def set_attention_config(self, attention_config) -> None:
        self.attention_config = attention_config

    def set_expert_execution_backend(self, backend: str) -> None:
        self.expert_execution_backend = backend

    def promote_stability_layers_to_fp32(self) -> None:
        self.promoted_stability_layers = True


def test_moe_example_defaults_to_cpu_stage_offload(tmp_path, monkeypatch) -> None:
    loaded_transformers: list[tuple[str, str | None]] = []

    class _Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return object()

    class _TextEncoder(_FakeModule):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

    class _VAE(_FakeModule):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

    class _Scheduler:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            del args, kwargs
            return cls()

    diffusers = types.ModuleType("diffusers")
    diffusers.AutoencoderKLWan = _VAE
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = _Processor
    transformers.Qwen3VLForConditionalGeneration = _TextEncoder
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        moe_example,
        "load_lingbot_video_model_config",
        lambda *args, **kwargs: LingBotVideoModelConfig(variant="moe", num_experts=2, top_k=1),
    )
    monkeypatch.setattr(moe_example.FlowUniPCMultistepScheduler, "from_pretrained", _Scheduler.from_pretrained)

    def load_model(self, file_path, *, name=None, **kwargs) -> None:
        del kwargs
        loaded_transformers.append((file_path, name))
        self.add_module(_FakeModule(), name=name or "transformer", path=file_path)

    monkeypatch.setattr(moe_example.ModuleManager, "load_model", load_model)

    def load_from_huggingface(self, module_path, *, module_name=None, module_class=None, **kwargs) -> None:
        del kwargs
        assert module_class is not None
        self.add_module(
            module_class.from_pretrained(module_path),
            name=module_name or Path(module_path).name,
            path=str(module_path),
        )

    monkeypatch.setattr(moe_example.ModuleManager, "load_from_huggingface", load_from_huggingface)

    base = moe_example.build_pipeline(tmp_path, device="cuda")
    refiner = moe_example.build_refiner(tmp_path, device="cuda")

    assert loaded_transformers == [
        (str(tmp_path / "transformer"), "transformer"),
        (str(tmp_path / "refiner"), "transformer"),
    ]
    assert base.variant == "moe"
    assert base.denoising_stage.transformer.expert_execution_backend == "sorted"
    assert (
        base.denoising_stage.transformer.attention_config is base.denoising_stage.model_runtime_config.attention_config
    )
    assert base.denoising_stage.model_runtime_config.offload_config.offload_type is WeightOffloadType.MODEL_CPU_OFFLOAD
    assert (
        refiner.denoising_stage.model_runtime_config.offload_config.offload_type is WeightOffloadType.MODEL_CPU_OFFLOAD
    )
