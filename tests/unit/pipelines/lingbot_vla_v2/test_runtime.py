from __future__ import annotations

import pytest

from telefuser.core.config import QuantKernelBackend, QuantType
from telefuser.pipelines.lingbot_vla_v2.runtime import (
    get_lingbot_vla_v2_pipeline,
    lingbot_vla_v2_quant_config,
)


@pytest.mark.parametrize(
    ("name", "quant_type", "backend"),
    [
        ("torchao-fp8", QuantType.TORCHAO_FP8, QuantKernelBackend.TORCHAO),
        ("torchao_fp8", QuantType.TORCHAO_FP8, QuantKernelBackend.TORCHAO),
        ("tf-kernel-fp8", QuantType.FP8, QuantKernelBackend.TF_KERNEL),
        ("bnb-nf4", QuantType.BNB_NF4, QuantKernelBackend.BITSANDBYTES),
    ],
)
def test_quantization_names_resolve_to_existing_runtime_config(
    name: str,
    quant_type: QuantType,
    backend: QuantKernelBackend,
) -> None:
    config = lingbot_vla_v2_quant_config(name)

    assert config.enabled is True
    assert config.quant_type == quant_type
    assert config.kernel_backend == backend


def test_default_runtime_quantization_keeps_bf16_path_disabled() -> None:
    assert lingbot_vla_v2_quant_config(None).enabled is False


def test_online_quantization_rejects_cpu_before_loading_models() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        get_lingbot_vla_v2_pipeline("unused", "unused", device="cpu", quantization="bnb-nf4")


def test_quantization_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="quantization must be"):
        lingbot_vla_v2_quant_config("int8")
