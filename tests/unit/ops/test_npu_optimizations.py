"""Unit tests for Ascend NPU performance optimizations (mocked for CPU CI)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from telefuser.models.wan_video_dit import DiTBlock, WanModel
from telefuser.ops.normalization import RMSNorm
from telefuser.ops.rotary import _apply_rotary_emb_npu


class _FakeNPUTensor:
    """Minimal wrapper exposing device.type == 'npu' around a CPU tensor."""

    def __init__(self, tensor: torch.Tensor):
        self._t = tensor
        self.device = SimpleNamespace(type="npu")
        self.dtype = tensor.dtype
        self.shape = tensor.shape

    def dim(self) -> int:
        return self._t.dim()

    def unsqueeze(self, dim: int) -> _FakeNPUTensor:
        return _FakeNPUTensor(self._t.unsqueeze(dim))

    def to(self, *args, **kwargs) -> _FakeNPUTensor:
        return self

    def repeat_interleave(self, repeats: int, dim: int) -> _FakeNPUTensor:
        return _FakeNPUTensor(self._t.repeat_interleave(repeats, dim=dim))


def test_rmsnorm_forward_npu_dispatches_to_torch_npu(monkeypatch) -> None:
    norm = RMSNorm(8, eps=1e-6)
    sentinel = torch.randn(1, 4, 8)
    fake_npu_rms_norm = MagicMock(return_value=(sentinel, None))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rms_norm=fake_npu_rms_norm))

    x_npu = _FakeNPUTensor(torch.randn(1, 4, 8))
    out = norm.forward_npu(x_npu)  # type: ignore[arg-type]

    assert out is sentinel
    fake_npu_rms_norm.assert_called_once_with(x_npu, norm.weight, norm.eps)


def test_rmsnorm_forward_npu_falls_back_on_cpu_tensor() -> None:
    norm = RMSNorm(8, eps=1e-6)
    x_cpu = torch.randn(1, 4, 8)
    expected = norm.forward_native(x_cpu)
    actual = norm.forward_npu(x_cpu)
    assert torch.allclose(actual, expected)


def test_apply_rotary_emb_npu_handles_half_and_full_dim(monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_rotary_mul(x, cos, sin, *, rotary_mode):
        calls.append((x.shape, cos.shape, sin.shape, rotary_mode))
        return x

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_rotary_mul=fake_rotary_mul))

    x = _FakeNPUTensor(torch.randn(1, 6, 2, 8))
    cos_half = _FakeNPUTensor(torch.ones(6, 1, 4))
    sin_half = _FakeNPUTensor(torch.zeros(6, 1, 4))
    _apply_rotary_emb_npu(x, cos_half, sin_half)  # type: ignore[arg-type]

    cos_full = _FakeNPUTensor(torch.ones(6, 1, 8))
    sin_full = _FakeNPUTensor(torch.zeros(6, 1, 8))
    _apply_rotary_emb_npu(x, cos_full, sin_full)  # type: ignore[arg-type]

    cos_4d = _FakeNPUTensor(torch.ones(1, 6, 1, 8))
    cos_4d.device = x.device
    sin_4d = _FakeNPUTensor(torch.zeros(1, 6, 1, 8))
    sin_4d.device = x.device
    _apply_rotary_emb_npu(x, cos_4d, sin_4d)  # type: ignore[arg-type]

    assert calls == [
        ((1, 6, 2, 8), (1, 6, 1, 8), (1, 6, 1, 8), "interleave"),
        ((1, 6, 2, 8), (1, 6, 1, 8), (1, 6, 1, 8), "interleave"),
        ((1, 6, 2, 8), (1, 6, 1, 8), (1, 6, 1, 8), "interleave"),
    ]


def test_dit_block_broadcast_t_mod_matches_expanded() -> None:
    torch.manual_seed(0)
    block = DiTBlock(has_image_input=False, dim=32, num_heads=4, ffn_dim=64, eps=1e-6).eval()
    x = torch.randn(1, 12, 32)
    ctx = torch.randn(1, 5, 32)
    t_mod_broadcast = torch.randn(1, 1, 6, 32)
    t_mod_expanded = t_mod_broadcast.expand(-1, 12, -1, -1).contiguous()
    fcos = torch.ones(12, 1, 4)
    fsin = torch.zeros(12, 1, 4)

    with torch.inference_mode():
        out_bc = block(x, ctx, t_mod_broadcast, fcos, fsin)
        out_exp = block(x, ctx, t_mod_expanded, fcos, fsin)

    assert torch.allclose(out_bc, out_exp, atol=1e-6, rtol=1e-5)


def test_wan_model_rope_freqs_cache_reuses_and_clears_on_offload() -> None:
    model = WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=1,
        has_image_input=False,
    )
    cos1, sin1 = model._get_rope_freqs(2, 3, 4, 0, 1, torch.device("cpu"), torch.float32)
    cos2, sin2 = model._get_rope_freqs(2, 3, 4, 0, 1, torch.device("cpu"), torch.float32)
    assert cos1 is cos2 and sin1 is sin2
    assert cos1.dtype == torch.float32
    assert cos1.shape == (24, 1, 6)

    model.offload_device(pin_memory=False)
    assert len(model._rope_freqs_cache) == 0


def test_wan_model_step_context_cache_parity_and_cleanup() -> None:
    from telefuser.core.config import AttentionConfig, AttnImplType

    torch.manual_seed(7)
    model = WanModel(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=4,
        text_dim=16,
        freq_dim=16,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        num_layers=2,
        has_image_input=False,
    ).eval()
    model.dtype = torch.float32

    cfg = AttentionConfig.dense_attention(AttnImplType.MINDIE_ATTN)
    model.set_attention_config(cfg)
    assert model.blocks[0].self_attn.attention_config.attn_impl == AttnImplType.MINDIE_ATTN
    assert model.blocks[0].cross_attn.attention_config.attn_impl == AttnImplType.MINDIE_ATTN
    model.set_attention_config(AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA))

    x1 = torch.randn(1, 4, 2, 4, 4)
    x2 = torch.randn(1, 4, 2, 4, 4)
    ts1 = torch.tensor([800.0])
    ts2 = torch.tensor([600.0])
    ctx = torch.randn(1, 5, 16)

    with torch.inference_mode():
        ref1 = model(x1, ts1, ctx)
        ref2 = model(x2, ts2, ctx)

        model.enable_step_context_cache()
        opt1 = model(x1, ts1, ctx)
        opt2 = model(x2, ts2, ctx)
        assert len(model._text_emb_cache) == 1
        assert len(model._cross_kv_cache) == 2

    assert torch.equal(ref1, opt1)
    assert torch.equal(ref2, opt2)

    model.clear_step_context_cache()
    assert len(model._text_emb_cache) == 0
    assert len(model._cross_kv_cache) == 0
