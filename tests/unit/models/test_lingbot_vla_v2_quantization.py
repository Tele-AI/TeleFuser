from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.models.lingbot_vla_v2 import (
    LINGBOT_VLA_V2_DEFAULT_QUANTIZE_MODULES,
    LingBotVlaV2Model,
)
from telefuser.models.lingbot_vla_v2_cuda_graph import LingBotVlaV2CudaGraphs, LingBotVlaV2DenoisingCudaGraph
from telefuser.models.lingbot_vla_v2_quantization import (
    LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZATION_MANIFEST_SHA256,
    LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZED_LINEAR_COUNT,
    build_lingbot_vla_v2_linear_manifest,
    linear_compute_dtype,
    lingbot_vla_v2_quantization_identity,
)


def _empty_model() -> LingBotVlaV2Model:
    model = LingBotVlaV2Model.__new__(LingBotVlaV2Model)
    nn.Module.__init__(model)
    return model


def _quantizable_model() -> LingBotVlaV2Model:
    model = _empty_model()
    model.qwenvl_with_expert = nn.Module()
    model.qwenvl_with_expert.qwenvl = nn.Module()
    model.qwenvl_with_expert.qwenvl.model = nn.Module()
    model.qwenvl_with_expert.qwenvl.model.language_model = nn.Module()
    model.qwenvl_with_expert.qwenvl.model.language_model.layers = nn.ModuleList([nn.Linear(4, 4)])
    model.qwenvl_with_expert.qwenvl.model.visual = nn.Module()
    model.qwenvl_with_expert.qwenvl.model.visual.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.qwenvl_with_expert.qwen_expert = nn.Module()
    model.qwenvl_with_expert.qwen_expert.model = nn.Module()
    action_layer = nn.Module()
    action_layer.self_attn = nn.Module()
    action_layer.self_attn.q_proj = nn.Linear(4, 4)
    action_layer.mlp = nn.Module()
    action_layer.mlp.shared_expert = nn.Module()
    action_layer.mlp.shared_expert.up_proj = nn.Linear(4, 4)
    model.qwenvl_with_expert.qwen_expert.model.layers = nn.ModuleList([action_layer])
    model.action_out_proj = nn.Linear(4, 4)
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
    model = _quantizable_model()
    calls: list[dict[str, object]] = []

    def fake_replace(_module: nn.Module, **kwargs: object) -> int:
        calls.append(kwargs)
        return 3

    monkeypatch.setattr(helper_path, fake_replace)
    model.enable_quant(QuantConfig(enabled=True, quant_type=quant_type, kernel_backend=backend))

    assert calls[0]["include_names"] == LINGBOT_VLA_V2_DEFAULT_QUANTIZE_MODULES
    exclude_names = calls[0]["exclude_names"]
    assert isinstance(exclude_names, tuple)
    assert "action_out_proj" in exclude_names
    assert "state_proj" in exclude_names
    assert getattr(model, count_attribute) == 3
    assert model.quant_type == quant_type
    identity = lingbot_vla_v2_quantization_identity(model)
    assert identity["profile"] in {"torchao-fp8", "bnb-nf4"}
    assert identity["kernel_backend"] == backend.name
    assert identity["manifest"]["selected_count"] == 3


def test_tf_kernel_fp8_quantization_filters_action_heads_and_moe(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _quantizable_model()
    captured_filter: Callable[[str, nn.Module], bool] | None = None

    def fake_count(_module: nn.Module, **kwargs: object) -> int:
        nonlocal captured_filter
        captured_filter = kwargs["module_filter"]  # type: ignore[assignment]
        return 3

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
    assert model.tf_kernel_fp8_replaced_linear == 3
    assert model.quant_type == QuantType.FP8


def test_cutlass_fp8_selects_fused_graph_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _empty_model()
    calls = 0

    def fake_enable() -> None:
        nonlocal calls
        calls += 1
        model.quant_type = QuantType.FP8

    monkeypatch.setattr(model, "_enable_fused_fp8_graph", fake_enable)
    model.enable_quant(QuantConfig(enabled=True, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.CUTLASS))

    assert calls == 1


def test_fp8_backend_cannot_change_after_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _quantizable_model()
    monkeypatch.setattr("telefuser.ops.fp8_gemm.count_linear_layers", lambda *_args, **_kwargs: 3)
    monkeypatch.setattr("telefuser.ops.fp8_gemm.enable_fp8_gemm", lambda module, **_kwargs: module)
    model.enable_quant(QuantConfig(enabled=True, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL))

    with pytest.raises(RuntimeError, match="cannot apply"):
        model.enable_quant(
            QuantConfig(enabled=True, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.CUTLASS)
        )


def test_online_quantization_rejects_unsupported_type() -> None:
    model = _empty_model()
    with pytest.raises(ValueError, match="does not support"):
        model.enable_quant(QuantConfig(enabled=True, quant_type=QuantType.INT8))


def test_quantization_manifest_freezes_selected_layers_and_groups() -> None:
    model = _quantizable_model()

    manifest = build_lingbot_vla_v2_linear_manifest(
        model,
        include_names=LINGBOT_VLA_V2_DEFAULT_QUANTIZE_MODULES,
        exclude_names=("action_out_proj", "shared_expert"),
    )

    assert manifest["selected_count"] == 3
    assert manifest["group_counts"] == {
        "action_expert_attention": 1,
        "qwen_language": 1,
        "qwen_visual": 1,
    }
    assert manifest["excluded_names"] == [
        "action_out_proj",
        "qwenvl_with_expert.qwen_expert.model.layers.0.mlp.shared_expert.up_proj",
    ]
    assert len(manifest["manifest_sha256"]) == 64


def test_official_base_profile_rejects_manifest_drift() -> None:
    model = _quantizable_model()
    model.config = SimpleNamespace(checkpoint_variant="base")

    with pytest.raises(
        RuntimeError,
        match=rf"expected count={LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZED_LINEAR_COUNT}",
    ):
        model.enable_quant(
            QuantConfig(
                enabled=True,
                quant_type=QuantType.TORCHAO_FP8,
                kernel_backend=QuantKernelBackend.TORCHAO,
            )
        )


def test_official_base_profile_rejects_same_count_with_different_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _quantizable_model()
    model.config = SimpleNamespace(checkpoint_variant="base")
    monkeypatch.setattr(
        "telefuser.models.lingbot_vla_v2.build_lingbot_vla_v2_linear_manifest",
        lambda *_args, **_kwargs: {
            "selected_count": LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZED_LINEAR_COUNT,
            "selected_names": [],
            "excluded_count": 0,
            "excluded_names": [],
            "group_counts": {},
            "manifest_sha256": "0" * 64,
        },
    )

    with pytest.raises(
        RuntimeError,
        match=rf"sha256={LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZATION_MANIFEST_SHA256}",
    ):
        model.enable_quant(
            QuantConfig(
                enabled=True,
                quant_type=QuantType.TORCHAO_FP8,
                kernel_backend=QuantKernelBackend.TORCHAO,
            )
        )


def test_online_quantization_is_idempotent_for_same_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _quantizable_model()
    calls = 0

    def fake_replace(_module: nn.Module, **_kwargs: object) -> int:
        nonlocal calls
        calls += 1
        return 3

    monkeypatch.setattr(
        "telefuser.ops.torchao_fp8_linear.replace_linear_layers_with_torchao_fp8",
        fake_replace,
    )
    config = QuantConfig(
        enabled=True,
        quant_type=QuantType.TORCHAO_FP8,
        kernel_backend=QuantKernelBackend.TORCHAO,
    )

    model.enable_quant(config)
    model.enable_quant(config)

    assert calls == 1


def test_online_quantization_rejects_second_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _quantizable_model()
    monkeypatch.setattr(
        "telefuser.ops.torchao_fp8_linear.replace_linear_layers_with_torchao_fp8",
        lambda _module, **_kwargs: 3,
    )
    model.enable_quant(
        QuantConfig(
            enabled=True,
            quant_type=QuantType.TORCHAO_FP8,
            kernel_backend=QuantKernelBackend.TORCHAO,
        )
    )

    with pytest.raises(RuntimeError, match="already quantized"):
        model.enable_quant(
            QuantConfig(
                enabled=True,
                quant_type=QuantType.BNB_NF4,
                kernel_backend=QuantKernelBackend.BITSANDBYTES,
            )
        )


def test_linear_compute_dtype_prefers_wrapper_compute_dtype() -> None:
    wrapper = nn.Module()
    wrapper.register_buffer("weight", torch.zeros(2, 2, dtype=torch.uint8))
    wrapper.compute_dtype = torch.float16

    assert linear_compute_dtype(wrapper, torch.float32) == torch.float16


def test_unquantized_identity_reports_bf16_without_manifest() -> None:
    identity = lingbot_vla_v2_quantization_identity(_empty_model())

    assert identity["enabled"] is False
    assert identity["profile"] == "bf16"
    assert identity["manifest"] is None


def test_linear_compute_dtype_falls_back_for_weightless_wrapper() -> None:
    assert linear_compute_dtype(nn.Identity(), torch.bfloat16) == torch.bfloat16


class _CudaGraphVelocityModel:
    config = SimpleNamespace(num_steps=10)

    def __init__(self) -> None:
        self.prefix_capture_enabled = False

    def set_prefix_cuda_graph_capture(self, enabled: bool) -> None:
        self.prefix_capture_enabled = enabled

    def build_prefix_cache(self, images, _img_masks, lang_tokens, lang_masks, _image_grid_thw):
        scale = images.mean() + lang_tokens.float().mean()
        positions = lang_tokens.clone()
        cache = {
            0: {
                "key_states": torch.ones((1, 2), device=images.device) * scale,
                "value_states": torch.zeros((1, 2), device=images.device),
            }
        }
        return lang_masks.clone(), positions, cache

    def predict_velocity(self, state, _prefix_pad_masks, past_key_values, x_t, timestep, **_kwargs):
        scale = past_key_values[0]["key_states"].mean()
        return x_t * 0.125 + state + scale + timestep.view(-1, 1, 1) * 0.01


@pytest.mark.gpu
def test_denoising_cuda_graph_replays_all_steps_with_new_inputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA Graph test requires CUDA")

    device = torch.device("cuda:0")
    model = _CudaGraphVelocityModel()
    state = torch.full((1, 2, 3), 0.2, device=device)
    noise = torch.ones((1, 2, 3), device=device)
    masks = torch.ones((1, 4), dtype=torch.bool, device=device)
    positions = torch.zeros((1, 4), dtype=torch.long, device=device)
    cache = {
        0: {"key_states": torch.full((1, 2), 0.3, device=device), "value_states": torch.zeros((1, 2), device=device)}
    }

    def eager() -> torch.Tensor:
        dt = torch.full((), -0.1, device=device)
        time = torch.ones((), device=device)
        result = noise.clone()
        for _ in range(10):
            velocity = model.predict_velocity(state, masks, cache, result, time.expand(1))
            result.add_(dt * velocity)
            time.add_(dt)
        return result

    runner = LingBotVlaV2DenoisingCudaGraph(model)
    expected = eager()
    actual = runner.run(state, masks, cache, noise, positions)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)

    state.fill_(-0.1)
    cache[0]["key_states"].fill_(0.6)
    expected_replay = eager()
    actual_replay = runner.run(state, masks, cache, noise, positions)
    torch.cuda.synchronize(device)
    assert torch.equal(actual_replay, expected_replay)


@pytest.mark.gpu
@torch.inference_mode()
def test_prefix_and_denoising_cuda_graphs_share_kv_buffers_and_replay_new_inputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA Graph test requires CUDA")

    device = torch.device("cuda:0")
    model = _CudaGraphVelocityModel()
    images = torch.full((2, 3), 0.25, device=device)
    img_masks = torch.ones((1, 2), dtype=torch.bool, device=device)
    lang_tokens = torch.tensor([[1, 2, 3, 4]], device=device)
    lang_masks = torch.ones_like(lang_tokens, dtype=torch.bool)
    grid = torch.tensor([[1, 2, 2]], device=device)
    state = torch.full((1, 2, 3), 0.2, device=device)
    noise = torch.ones((1, 2, 3), device=device)

    def eager() -> torch.Tensor:
        masks, positions, cache = model.build_prefix_cache(images, img_masks, lang_tokens, lang_masks, grid)
        del positions
        dt = torch.full((), -0.1, device=device)
        time = torch.ones((), device=device)
        result = noise.clone()
        for _ in range(10):
            velocity = model.predict_velocity(state, masks, cache, result, time.expand(1))
            result.add_(dt * velocity)
            time.add_(dt)
        return result

    runner = LingBotVlaV2CudaGraphs(model)
    expected = eager()
    actual = runner.run(images, img_masks, lang_tokens, lang_masks, state, noise, grid)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)
    assert runner.denoising.past_key_values is runner.prefix.past_key_values
    assert model.prefix_capture_enabled

    images.fill_(0.5)
    lang_tokens.add_(1)
    state.fill_(-0.1)
    expected_replay = eager()
    actual_replay = runner.run(images, img_masks, lang_tokens, lang_masks, state, noise, grid)
    torch.cuda.synchronize(device)
    assert torch.equal(actual_replay, expected_replay)

    changed_grid = torch.tensor([[1, 1, 4]], device=device)
    with pytest.raises(ValueError, match="image_grid_thw values changed"):
        runner.run(images, img_masks, lang_tokens, lang_masks, state, noise, changed_grid)

    img_masks.zero_()
    with pytest.raises(ValueError, match="img_masks values changed"):
        runner.run(images, img_masks, lang_tokens, lang_masks, state, noise, grid)
