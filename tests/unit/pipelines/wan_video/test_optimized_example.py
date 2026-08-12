from __future__ import annotations

import pytest
import torch

from examples.wan_video.wan21_1_3b_text_to_video_optimized_h100 import (
    make_attention_config,
    make_quant_config,
    run,
)
from telefuser.core.config import AttnImplType, QuantConfig, QuantKernelBackend, QuantType
from telefuser.models.wan_video_dit import WanModel


def test_wan_optimized_example_builds_compatible_configs() -> None:
    attention = make_attention_config("sol")
    quant = make_quant_config("torchao-fp8")

    assert attention.attn_impl is AttnImplType.SOL_ATTN
    assert attention.sparse_config is not None
    assert attention.sparse_config.sol_tau == 1.0
    assert quant.enabled
    assert quant.quant_type is QuantType.TORCHAO_FP8
    assert quant.kernel_backend is QuantKernelBackend.TORCHAO


def test_wan_optimized_example_builds_dense_attention_config() -> None:
    attention = make_attention_config("dense")

    assert attention.attn_impl is AttnImplType.TORCH_SDPA
    assert attention.sparse_config is None


def test_wan_optimized_example_rejects_unknown_attention() -> None:
    with pytest.raises(ValueError, match="attention must be"):
        make_attention_config("radial")


@pytest.mark.parametrize(
    ("name", "quant_type", "backend"),
    [
        ("none", QuantType.FP8, QuantKernelBackend.AUTO),
        ("tf-kernel-fp8", QuantType.FP8, QuantKernelBackend.TF_KERNEL),
        ("torchao-fp8", QuantType.TORCHAO_FP8, QuantKernelBackend.TORCHAO),
        ("bnb-nf4", QuantType.BNB_NF4, QuantKernelBackend.BITSANDBYTES),
    ],
)
def test_wan_optimized_example_quantization_choices(name, quant_type, backend) -> None:
    config = make_quant_config(name)
    if name == "none":
        assert not config.enabled
    else:
        assert config.enabled
    assert config.quant_type is quant_type
    assert config.kernel_backend is backend


def test_wan_optimized_example_rejects_unknown_quantization() -> None:
    with pytest.raises(ValueError, match="quantization must be"):
        make_quant_config("int8")


def test_wan_model_enables_tf_kernel_fp8_on_transformer_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    model = WanModel.__new__(WanModel)
    torch.nn.Module.__init__(model)
    model.blocks = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))])
    calls = []

    def fake_count(module, *, module_filter=None):
        calls.append(("count", module, module_filter))
        return 2

    def fake_enable(module, *, options, module_filter=None):
        calls.append(("enable", module, options, module_filter))
        return module

    monkeypatch.setattr("telefuser.ops.fp8_gemm.tf_kernel", None)
    monkeypatch.setattr("telefuser.ops.fp8_gemm.count_linear_layers", fake_count)
    monkeypatch.setattr("telefuser.ops.fp8_gemm.enable_fp8_gemm", fake_enable)

    model.enable_quant(
        QuantConfig(
            enabled=True,
            quant_type=QuantType.FP8,
            kernel_backend=QuantKernelBackend.TF_KERNEL,
        )
    )

    assert model.tf_kernel_fp8_replaced_linear == 2
    assert model.quant_type is QuantType.FP8
    assert calls[0][0] == "count"
    assert calls[0][1] is model
    assert calls[1][0] == "enable"
    assert calls[1][1] is model
    options = calls[1][2]
    assert not options.cast_output_back
    assert options.fp16_weight_storage == "discard"
    assert options.materialize_fp8_on_wrap
    module_filter = calls[1][3]
    assert module_filter("blocks.0.0", model.blocks[0][0])
    assert not module_filter("head", model.blocks[0][0])


def test_wan_optimized_run_forwards_explicit_benchmark_parameters() -> None:
    captured = {}

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return object()

    output = run(
        fake_pipeline,
        "prompt",
        seed=7,
        width=832,
        height=480,
        num_inference_steps=50,
        num_frames=81,
        cfg_scale=5.0,
        sigma_shift=5.0,
    )

    assert output is not None
    assert captured["seed"] == 7
    assert captured["width"] == 832
    assert captured["height"] == 480
    assert captured["num_inference_steps"] == 50
    assert captured["num_frames"] == 81
    assert captured["cfg_scale"] == 5.0
    assert captured["sigma_shift"] == 5.0


def test_wan_optimized_run_requires_width_and_height_together() -> None:
    with pytest.raises(ValueError, match="width and height must be provided together"):
        run(lambda **_kwargs: object(), "prompt", width=832)
