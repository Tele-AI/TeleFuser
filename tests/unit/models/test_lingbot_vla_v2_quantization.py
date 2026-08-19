from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.models.lingbot_vla_v2 import (
    LINGBOT_VLA_V2_DEFAULT_QUANTIZE_MODULES,
    LingBotVlaV2Model,
)
from telefuser.models.lingbot_vla_v2_quantization import linear_compute_dtype


def _empty_model() -> LingBotVlaV2Model:
    model = LingBotVlaV2Model.__new__(LingBotVlaV2Model)
    nn.Module.__init__(model)
    return model


@pytest.mark.parametrize(
    ("quant_type", "backend", "helper_path", "count_attribute"),
    [
        (
            QuantType.TORCHAO_FP8,
            QuantKernelBackend.TORCHAO,
            "telefuser.ops.torchao_fp8_linear.replace_linear_layers_with_torchao_fp8",
            "torchao_fp8_replaced_linear",
        ),
        (
            QuantType.BNB_NF4,
            QuantKernelBackend.BITSANDBYTES,
            "telefuser.ops.bnb_nf4_linear.replace_linear_layers_with_bnb_nf4",
            "bnb_nf4_replaced_linear",
        ),
    ],
)
def test_online_quantization_uses_vla_safe_linear_selection(
    monkeypatch: pytest.MonkeyPatch,
    quant_type: QuantType,
    backend: QuantKernelBackend,
    helper_path: str,
    count_attribute: str,
) -> None:
    model = _empty_model()
    calls: list[dict[str, object]] = []

    def fake_replace(_module: nn.Module, **kwargs: object) -> int:
        calls.append(kwargs)
        return 11

    monkeypatch.setattr(helper_path, fake_replace)
    model.enable_quant(QuantConfig(enabled=True, quant_type=quant_type, kernel_backend=backend))

    assert calls[0]["include_names"] == LINGBOT_VLA_V2_DEFAULT_QUANTIZE_MODULES
    exclude_names = calls[0]["exclude_names"]
    assert isinstance(exclude_names, tuple)
    assert "action_out_proj" in exclude_names
    assert "state_proj" in exclude_names
    assert getattr(model, count_attribute) == 11
    assert model.quant_type == quant_type


def test_tf_kernel_fp8_quantization_filters_action_heads_and_moe(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _empty_model()
    captured_filter: Callable[[str, nn.Module], bool] | None = None

    def fake_count(_module: nn.Module, **kwargs: object) -> int:
        nonlocal captured_filter
        captured_filter = kwargs["module_filter"]  # type: ignore[assignment]
        return 7

    def fake_enable(_module: nn.Module, **_kwargs: object) -> nn.Module:
        return _module

    monkeypatch.setattr("telefuser.ops.fp8_gemm.count_linear_layers", fake_count)
    monkeypatch.setattr("telefuser.ops.fp8_gemm.enable_fp8_gemm", fake_enable)

    model.enable_quant(QuantConfig(enabled=True, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL))

    assert captured_filter is not None
    linear = nn.Linear(2, 2)
    assert captured_filter("model.qwenvl_with_expert.qwenvl.model.language_model.layers.0.mlp.up_proj", linear)
    assert captured_filter("model.qwenvl_with_expert.qwenvl.model.visual.blocks.0.mlp.linear_fc1", linear)
    assert captured_filter("model.qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.q_proj", linear)
    assert not captured_filter("model.qwenvl_with_expert.qwen_expert.model.layers.0.mlp.shared_expert.up_proj", linear)
    assert not captured_filter("model.action_out_proj", linear)
    assert model.tf_kernel_fp8_replaced_linear == 7
    assert model.quant_type == QuantType.FP8


def test_online_quantization_rejects_unsupported_type() -> None:
    model = _empty_model()
    with pytest.raises(ValueError, match="does not support"):
        model.enable_quant(QuantConfig(enabled=True, quant_type=QuantType.INT8))


def test_linear_compute_dtype_prefers_wrapper_compute_dtype() -> None:
    wrapper = nn.Module()
    wrapper.register_buffer("weight", torch.zeros(2, 2, dtype=torch.uint8))
    wrapper.compute_dtype = torch.float16

    assert linear_compute_dtype(wrapper, torch.float32) == torch.float16


def test_linear_compute_dtype_falls_back_for_weightless_wrapper() -> None:
    assert linear_compute_dtype(nn.Identity(), torch.bfloat16) == torch.bfloat16
