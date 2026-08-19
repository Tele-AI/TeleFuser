from __future__ import annotations

import gc
from importlib import metadata

import pytest
import torch
from torch import nn

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.models.lingbot_vla_v2 import LingBotVlaV2Model
from telefuser.models.lingbot_vla_v2_quantization import lingbot_vla_v2_quantization_identity

pytestmark = pytest.mark.gpu


def _package_available(distribution: str) -> bool:
    try:
        metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return False
    return True


def _quantizable_model(width: int = 64) -> LingBotVlaV2Model:
    model = LingBotVlaV2Model.__new__(LingBotVlaV2Model)
    nn.Module.__init__(model)
    model.qwenvl_with_expert = nn.Module()
    model.qwenvl_with_expert.qwenvl = nn.Module()
    model.qwenvl_with_expert.qwenvl.model = nn.Module()
    model.qwenvl_with_expert.qwenvl.model.language_model = nn.Module()
    model.qwenvl_with_expert.qwenvl.model.language_model.layers = nn.ModuleList([nn.Linear(width, width)])
    model.qwenvl_with_expert.qwenvl.model.visual = nn.Module()
    model.qwenvl_with_expert.qwenvl.model.visual.blocks = nn.ModuleList([nn.Linear(width, width)])
    model.qwenvl_with_expert.qwen_expert = nn.Module()
    model.qwenvl_with_expert.qwen_expert.model = nn.Module()
    action_layer = nn.Module()
    action_layer.self_attn = nn.Module()
    action_layer.self_attn.q_proj = nn.Linear(width, width)
    model.qwenvl_with_expert.qwen_expert.model.layers = nn.ModuleList([action_layer])
    model.action_out_proj = nn.Linear(width, width)
    return model


@pytest.mark.parametrize(
    ("distribution", "quant_type", "backend", "profile"),
    [
        ("torchao", QuantType.TORCHAO_FP8, QuantKernelBackend.TORCHAO, "torchao-fp8"),
        ("bitsandbytes", QuantType.BNB_NF4, QuantKernelBackend.BITSANDBYTES, "bnb-nf4"),
    ],
)
def test_online_quantization_repeated_forward_and_release(
    distribution: str,
    quant_type: QuantType,
    backend: QuantKernelBackend,
    profile: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not _package_available(distribution):
        pytest.skip(f"{distribution} is not installed")

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    def run_cycle() -> int:
        model = _quantizable_model().to(device=device, dtype=torch.bfloat16).eval()
        model.enable_quant(QuantConfig(enabled=True, quant_type=quant_type, kernel_backend=backend))
        selected = dict(model.named_modules())["qwenvl_with_expert.qwenvl.model.language_model.layers.0"]
        inputs = torch.randn(2, 64, device=device, dtype=torch.bfloat16)

        with torch.inference_mode():
            first = selected(inputs)
            second = selected(inputs)
        torch.cuda.synchronize(device)

        assert first.shape == (2, 64)
        assert torch.isfinite(first).all()
        assert torch.equal(first, second)
        identity = lingbot_vla_v2_quantization_identity(model)
        assert identity["profile"] == profile
        assert identity["manifest"]["selected_count"] == 3

        model.to(device="cpu")
        del first, second, selected, inputs, model
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated(device)

    first_cycle_floor = run_cycle()
    second_cycle_floor = run_cycle()

    # TorchAO may retain one process-wide dispatch cache after first use. Repeating
    # the model lifecycle must not retain another model-sized allocation.
    assert second_cycle_floor <= first_cycle_floor + 1024**2
