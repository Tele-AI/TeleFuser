from unittest.mock import patch

import pytest
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, SparseAttentionConfig
from telefuser.models.wan_video_dit import SelfAttention, WanModel, precompute_freqs_cis_3d
from telefuser.ops.attention import SparseAttentionState, attention_impl
from telefuser.ops.fp8_attention import (
    dequantize_fp8_per_block,
    dequantize_fp8_per_channel,
    dequantize_fp8_per_token,
    quantize_fp8_per_block,
    quantize_fp8_qkv,
)


def test_wan_model_enables_sol_attention_state() -> None:
    model = WanModel.__new__(WanModel)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)

    sparse_config = SparseAttentionConfig(
        sparse_impl="sol",
        dense_layers=2,
        dense_timesteps=3,
        sol_tau=0.75,
        sol_threshold_type="exact",
        sol_kv_splits=2,
    )
    model.enable_sol_attention(height=64, width=64, num_frames=5, sparse_config=sparse_config)

    state = model.create_sparse_state(numeral_timestep=4, layer_idx=5)
    assert state is not None
    assert state.mask_map is None
    assert state.numeral_timestep == 4
    assert state.layer_idx == 5
    assert state.config is sparse_config
    perm = model.sol_morton_perm
    inverse = model.sol_morton_inverse
    torch.testing.assert_close(perm.index_select(0, inverse), torch.arange(perm.numel()))


def test_wan_sol_attention_uses_official_dense_guards() -> None:
    model = WanModel.__new__(WanModel)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    config = AttentionConfig.sol_attention()
    assert config.sparse_config is not None
    model.enable_sol_attention(height=480, width=832, num_frames=81, sparse_config=config.sparse_config)

    state = model.create_sparse_state(numeral_timestep=0, layer_idx=1)
    assert state is not None and state.should_use_dense()
    state.update(numeral_timestep=10, layer_idx=0)
    assert state.should_use_dense()
    state.update(numeral_timestep=10, layer_idx=1)
    assert not state.should_use_dense()
    assert model.sol_morton_perm.numel() == 21 * 30 * 52


def test_wan_sol_token_order_round_trip() -> None:
    model = WanModel.__new__(WanModel)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 2, 2)
    config = AttentionConfig.sol_attention()
    assert config.sparse_config is not None
    model.enable_sol_attention(height=64, width=64, num_frames=5, sparse_config=config.sparse_config)
    state = model.create_sparse_state()

    tokens = model.sol_morton_perm.numel()
    x = torch.arange(tokens).reshape(1, tokens, 1)
    t_mod = x.clone()
    freqs_cos = torch.arange(tokens).reshape(tokens, 1)
    freqs_sin = -freqs_cos

    ordered = model._apply_sol_token_order(x, t_mod, freqs_cos, freqs_sin, state, reorder_tokens=True)
    perm = model.sol_morton_perm
    torch.testing.assert_close(ordered[0], x.index_select(1, perm))
    torch.testing.assert_close(ordered[1], t_mod.index_select(1, perm))
    torch.testing.assert_close(ordered[2], freqs_cos.index_select(0, perm))
    torch.testing.assert_close(ordered[3], freqs_sin.index_select(0, perm))
    torch.testing.assert_close(model._restore_sol_token_order(ordered[0], state), x)


def test_wan_self_attention_dispatches_sol_through_public_ops() -> None:
    module = SelfAttention(dim=128, num_heads=1)
    sparse_config = SparseAttentionConfig(sparse_impl="sol", dense_timesteps=0)
    module.attention_config = AttentionConfig(attn_impl=AttnImplType.SOL_ATTN, sparse_config=sparse_config)
    state = SparseAttentionState(sparse_config, mask_map=None)
    captured = {}

    def fake_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs) -> torch.Tensor:
        captured.update(kwargs)
        return q

    x = torch.randn(1, 4, 128)
    with (
        patch("telefuser.models.wan_video_dit.rope_apply", side_effect=lambda tensor, *_args: tensor),
        patch("telefuser.models.wan_video_dit.attn_func", side_effect=fake_attention),
    ):
        output = module.default_forward(x, torch.empty(0), torch.empty(0), sparse_state=state)

    assert output.shape == x.shape
    assert captured["attention_config"].attn_impl is AttnImplType.SOL_ATTN
    assert captured["attention_config"].sparse_config is sparse_config
    assert captured["sparse_state"] is state


def test_wan_self_attention_casts_fp32_qkv_only_for_active_sol() -> None:
    module = SelfAttention(dim=128, num_heads=1)
    config = SparseAttentionConfig(sparse_impl="sol", dense_timesteps=1, dense_layers=0)
    state = SparseAttentionState(config, mask_map=None)
    q = torch.randn(1, 4, 1, 128)

    dense_qkv = module._prepare_sol_qkv(q, q, q, state)
    assert all(tensor.dtype is torch.float32 for tensor in dense_qkv[:3])
    assert dense_qkv[3] is None

    state.update(numeral_timestep=1)
    sol_qkv = module._prepare_sol_qkv(q, q, q, state)
    assert all(tensor.dtype is torch.bfloat16 for tensor in sol_qkv[:3])
    assert sol_qkv[3] is None


def test_wan_self_attention_casts_projection_input_once_under_autocast() -> None:
    module = SelfAttention(dim=128, num_heads=1)
    config = SparseAttentionConfig(sparse_impl="sol", dense_timesteps=1, dense_layers=0)
    state = SparseAttentionState(config, mask_map=None)
    x = torch.randn(1, 4, 128)

    with patch("telefuser.models.wan_video_dit.torch.is_autocast_enabled", return_value=True):
        dense_x = module._prepare_sol_projection_input(x, state)
        state.update(numeral_timestep=1)
        sol_x = module._prepare_sol_projection_input(x, state)

    assert dense_x is x
    assert sol_x.dtype is torch.bfloat16


def test_wan_self_attention_quantizes_qkv_for_fp8_sol() -> None:
    module = SelfAttention(dim=128, num_heads=1).to(torch.bfloat16)
    config = SparseAttentionConfig(sparse_impl="sol", dense_timesteps=0, sol_fp8=True)
    state = SparseAttentionState(config, mask_map=None)
    captured = {}

    def fake_attention(q, k, v, **kwargs):
        captured.update({"q": q, "k": k, "v": v, **kwargs})
        return q.to(torch.bfloat16)

    x = torch.randn(1, 65, 128, dtype=torch.bfloat16)
    freqs = torch.zeros(65, 64, dtype=torch.bfloat16)
    with patch("telefuser.models.wan_video_dit.attn_func", side_effect=fake_attention):
        module.default_forward(x, freqs, freqs, sparse_state=state)

    assert captured["q"].dtype is torch.float8_e4m3fn
    assert captured["q_scale"].shape == (1, 2, 1)
    restored = dequantize_fp8_per_block(captured["q"], captured["q_scale"], torch.bfloat16)
    assert torch.isfinite(restored).all()


def test_wan_self_attention_limits_fp8_sol_to_configured_layers() -> None:
    module = SelfAttention(dim=128, num_heads=1)
    config = SparseAttentionConfig(
        sparse_impl="sol",
        dense_timesteps=0,
        sol_fp8=True,
        sol_fp8_layer_start=1,
        sol_fp8_layer_end=2,
    )
    state = SparseAttentionState(config, mask_map=None)
    q = torch.randn(1, 64, 1, 128)

    bf16_qkv = module._prepare_sol_qkv(q, q, q, state)
    assert bf16_qkv[3] is None

    state.update(layer_idx=1)
    fp8_qkv = module._prepare_sol_qkv(q, q, q, state)
    assert fp8_qkv[0].dtype is torch.float8_e4m3fn
    assert fp8_qkv[3] is not None


@pytest.mark.gpu
def test_wan_self_attention_executes_sol_on_h100(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Wan Sol-Attn execution test requires H100")

    assert attention_impl.SOL_ATTN_AVAILABLE
    assert attention_impl.sol_attn is not None
    kernel_calls = 0
    sol_attn = attention_impl.sol_attn

    def tracked_sol_attn(*args, **kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        return sol_attn(*args, **kwargs)

    monkeypatch.setattr(attention_impl, "sol_attn", tracked_sol_attn)

    module = SelfAttention(dim=128, num_heads=1).eval().cuda().to(torch.bfloat16)
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)
    freqs = precompute_freqs_cis_3d(128)
    freqs_cos = torch.cat([freq.real for freq in freqs], dim=-1)[:256].cuda()
    freqs_sin = torch.cat([freq.imag for freq in freqs], dim=-1)[:256].cuda()
    sparse_config = SparseAttentionConfig(sparse_impl="sol", dense_timesteps=0, sol_tau=-1000.0)
    module.attention_config = AttentionConfig(attn_impl=AttnImplType.SOL_ATTN, sparse_config=sparse_config)
    state = SparseAttentionState(sparse_config, mask_map=None)

    output = module(x, freqs_cos, freqs_sin, sparse_state=state)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert kernel_calls == 1


@pytest.mark.gpu
def test_wan_self_attention_executes_native_fp8_sol_on_h100(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Wan FP8 Sol-Attn execution test requires H100")

    assert attention_impl.SOL_ATTN_AVAILABLE
    assert attention_impl.sol_attn is not None
    captured = {}
    sol_attn = attention_impl.sol_attn

    def tracked_sol_attn(q, k, v, **kwargs):
        captured.update({"q": q, "k": k, "v": v, **kwargs})
        return sol_attn(q, k, v, **kwargs)

    monkeypatch.setattr(attention_impl, "sol_attn", tracked_sol_attn)

    module = SelfAttention(dim=128, num_heads=1).eval().cuda().to(torch.bfloat16)
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)
    freqs = precompute_freqs_cis_3d(128)
    freqs_cos = torch.cat([freq.real for freq in freqs], dim=-1)[:256].cuda()
    freqs_sin = torch.cat([freq.imag for freq in freqs], dim=-1)[:256].cuda()
    sparse_config = SparseAttentionConfig(
        sparse_impl="sol",
        dense_timesteps=0,
        sol_tau=-1000.0,
        sol_fp8=True,
    )
    module.attention_config = AttentionConfig(attn_impl=AttnImplType.SOL_ATTN, sparse_config=sparse_config)
    state = SparseAttentionState(sparse_config, mask_map=None)

    output = module(x, freqs_cos, freqs_sin, sparse_state=state)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert captured["q"].dtype is torch.float8_e4m3fn
    assert captured["k"].dtype is torch.float8_e4m3fn
    assert captured["v"].dtype is torch.float8_e4m3fn
    assert captured["v"].stride(1) == 1
    assert captured["q_scale"].shape == (1, 256, 1)
    assert captured["k_scale"].shape == (1, 256, 1)
    assert captured["v_scale"].shape == (1, 1, 128)


@pytest.mark.gpu
def test_fused_fp8_qkv_quantization_on_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("fused FP8 QKV quantization test requires H100")

    q = torch.randn(1, 130, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = quantize_fp8_qkv(q, k, v)

    assert q_fp8.shape == q.shape
    assert k_fp8.shape == k.shape
    assert v_fp8.shape == v.shape
    assert q_scale.shape == (1, 192, 2)
    assert k_scale.shape == (1, 192, 2)
    assert v_scale.shape == (1, 2, 128)
    assert v_fp8.stride(1) == 1
    torch.testing.assert_close(
        dequantize_fp8_per_token(q_fp8, q_scale, torch.bfloat16),
        q,
        rtol=0.15,
        atol=0.05,
    )
    torch.testing.assert_close(
        dequantize_fp8_per_channel(v_fp8, v_scale, torch.bfloat16),
        v,
        rtol=0.15,
        atol=0.05,
    )


@pytest.mark.gpu
def test_native_fp8_sol_handles_partial_tail_on_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("partial-tail FP8 Sol-Attn test requires H100")

    q = torch.randn(1, 130, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = quantize_fp8_qkv(q, k, v)
    output = attention_impl.sol_attn(
        q_fp8,
        k_fp8,
        v_fp8,
        tau=-1000.0,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    reference = attention_impl.sol_attn(
        dequantize_fp8_per_token(q_fp8, q_scale, torch.bfloat16).contiguous(),
        dequantize_fp8_per_token(k_fp8, k_scale, torch.bfloat16).contiguous(),
        dequantize_fp8_per_channel(v_fp8, v_scale, torch.bfloat16).contiguous(),
        tau=-1000.0,
    )

    cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)
    assert cosine > 0.99


@pytest.mark.gpu
def test_native_fp8_sol_preserves_constant_values_at_long_sequence_on_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("long-sequence FP8 Sol-Attn test requires H100")

    q = torch.randn(1, 2048, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.ones_like(q)
    q, k, v, q_scale, k_scale, v_scale = quantize_fp8_qkv(q, k, v)
    output = attention_impl.sol_attn(
        q,
        k,
        v,
        tau=-1000.0,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    torch.testing.assert_close(output.float(), torch.ones_like(output, dtype=torch.float32), rtol=0.02, atol=0.02)


@pytest.mark.gpu
def test_native_fp8_sol_split_preserves_sparse_route_weights_on_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("split FP8 Sol-Attn test requires H100")

    torch.manual_seed(7)
    q = torch.randn(1, 4160, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    reference = attention_impl.sol_attn(q, k, v, tau=1.0, kv_splits=1)
    q, k, v, q_scale, k_scale, v_scale = quantize_fp8_qkv(q, k, v)
    output = attention_impl.sol_attn(
        q,
        k,
        v,
        tau=1.0,
        kv_splits=2,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    cosine = torch.nn.functional.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0)
    assert cosine > 0.98
