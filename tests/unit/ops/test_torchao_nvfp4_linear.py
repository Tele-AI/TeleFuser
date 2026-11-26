"""Unit tests for TeleFuser NVFP4 layer selection."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch

from telefuser.ops import torchao_nvfp4_linear as nvfp4


class TinyDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        self.attn = torch.nn.Linear(4, 4)


def test_nvfp4_quality_filter_selects_only_ffn(monkeypatch) -> None:
    captured: list[str] = []

    class Config:
        def __init__(self, use_triton_kernel: bool) -> None:
            assert use_triton_kernel is False

    def quantize_(module, config, filter_fn) -> None:
        captured.extend(name for name, child in module.named_modules() if filter_fn(child, name))

    mx = ModuleType("torchao.prototype.mx_formats")
    mx.NVFP4DynamicActivationNVFP4WeightConfig = Config
    quant = ModuleType("torchao.quantization")
    quant.quantize_ = quantize_
    monkeypatch.setitem(sys.modules, "torchao.prototype.mx_formats", mx)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quant)
    monkeypatch.setattr(nvfp4.metadata, "version", lambda _name: "0.17.0")

    count = nvfp4.replace_linear_layers_with_torchao_nvfp4(TinyDiT(), include_names=("ff.",))
    assert count == 2
    assert captured == ["ff.0", "ff.1"]


def test_enable_nvfp4_triton_requires_dynamic_weights() -> None:
    parameters = [
        SimpleNamespace(data=SimpleNamespace(act_quant_kwargs=SimpleNamespace(use_triton_kernel=False)))
        for _ in range(3)
    ]
    model = SimpleNamespace(parameters=lambda: iter(parameters))
    count = nvfp4.enable_torchao_nvfp4_triton(model)
    assert count == 3
    assert all(parameter.data.act_quant_kwargs.use_triton_kernel for parameter in model.parameters())
