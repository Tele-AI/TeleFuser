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


def test_silu_and_mul_dispatches_to_npu_swiglu_and_matches_native(monkeypatch) -> None:
    from telefuser.ops.activations import silu_and_mul, silu_and_mul_reuse_input

    x_cpu = torch.randn(4, 16, dtype=torch.bfloat16)
    expected = silu_and_mul(x_cpu)

    def fake_swiglu(tensor: _FakeNPUTensor, *, dim: int = -1) -> torch.Tensor:
        assert dim == -1
        gate, val = tensor._t.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * val

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_swiglu=fake_swiglu))
    x_npu = _FakeNPUTensor(x_cpu)
    x_npu.is_cuda = False  # type: ignore[attr-defined]
    actual = silu_and_mul_reuse_input(x_npu)  # type: ignore[arg-type]
    assert torch.equal(actual, expected)


def test_indexed_scale_shift_and_gate_npu_paths_match_reference(monkeypatch) -> None:
    from telefuser.ops.normalization import indexed_gate, indexed_scale_shift

    torch.manual_seed(11)
    x = torch.randn(6, 16, dtype=torch.bfloat16)
    other = torch.randn(6, 16, dtype=torch.bfloat16)
    shift = torch.randn(3, 16, dtype=torch.bfloat16)
    scale = torch.randn(3, 16, dtype=torch.bfloat16)
    gate = torch.randn(3, 16, dtype=torch.bfloat16)
    indices = torch.tensor([0, 2, 1, 0, 2, 1], dtype=torch.long)

    ref_ss = indexed_scale_shift(x, shift, scale, indices)
    ref_gate = indexed_gate(x, gate, other, indices)

    monkeypatch.setattr("telefuser.ops.normalization._validate_indexed_modulation_inputs", lambda *a, **kw: None)

    class _NpuBackedTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="npu")

    x_n = x.clone().as_subclass(_NpuBackedTensor)
    other_n = other.clone().as_subclass(_NpuBackedTensor)
    shift_n = shift.clone().as_subclass(_NpuBackedTensor)
    scale_n = scale.clone().as_subclass(_NpuBackedTensor)
    gate_n = gate.clone().as_subclass(_NpuBackedTensor)
    idx_n = indices.clone()

    npu_ss = indexed_scale_shift(x_n, shift_n, scale_n, idx_n).as_subclass(torch.Tensor)
    npu_gate = indexed_gate(x_n, gate_n, other_n, idx_n).as_subclass(torch.Tensor)
    torch.testing.assert_close(npu_ss, ref_ss, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(npu_gate, ref_gate, atol=1e-2, rtol=1e-2)


def test_apply_qk_norm_rope_neox_npu_matches_native_partial_rotary(monkeypatch) -> None:
    from telefuser.ops.rotary import apply_qk_norm_rope_neox

    torch.manual_seed(3)
    query = torch.randn(5, 4, 16, dtype=torch.bfloat16)
    key = torch.randn(5, 4, 16, dtype=torch.bfloat16)
    q_weight = torch.randn(16, dtype=torch.bfloat16)
    k_weight = torch.randn(16, dtype=torch.bfloat16)
    angles = torch.randn(5, 6)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.bfloat16)

    ref_q, ref_k = apply_qk_norm_rope_neox(query, key, q_weight, k_weight, cache, eps=1e-5)

    def fake_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float):
        base = x.as_subclass(torch.Tensor)
        wb = w.as_subclass(torch.Tensor)
        var = base.float().pow(2).mean(-1, keepdim=True)
        out = (base * torch.rsqrt(var + eps)).to(wb.dtype) * wb
        return out.as_subclass(type(x)), None

    def fake_rotary_mul(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        xb = x.as_subclass(torch.Tensor)
        cb = cos.as_subclass(torch.Tensor)
        sb = sin.as_subclass(torch.Tensor)
        first, second = xb.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return (xb * cb + rotated * sb).to(xb.dtype).as_subclass(type(x))

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_rms_norm=fake_rms_norm, npu_rotary_mul=fake_rotary_mul),
    )

    class _NpuTensor(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="npu")

        def to(self, *args, **kwargs):  # type: ignore[override]
            kwargs.pop("device", None)
            if args and not isinstance(args[0], torch.dtype):
                args = args[1:]
            return super().to(*args, **kwargs).as_subclass(_NpuTensor)

    npu_q, npu_k = apply_qk_norm_rope_neox(
        query.clone().as_subclass(_NpuTensor),
        key.clone().as_subclass(_NpuTensor),
        q_weight.clone().as_subclass(_NpuTensor),
        k_weight.clone().as_subclass(_NpuTensor),
        cache.clone().as_subclass(_NpuTensor),
        eps=1e-5,
    )
    torch.testing.assert_close(npu_q.as_subclass(torch.Tensor), ref_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(npu_k.as_subclass(torch.Tensor), ref_k, atol=1e-2, rtol=1e-2)


def test_mindie_attn_applies_exact_power_of_two_input_prescaling(monkeypatch) -> None:
    from telefuser.ops.attention import backends

    recorded: dict[str, object] = {}

    def fake_forward(q, k, v, *, attn_mask=None, scale=None, head_first=False, **kwargs):
        recorded["q_max"] = float(q.abs().max().item())
        recorded["scale"] = scale
        recorded["head_first"] = head_first
        recorded["kwargs"] = kwargs
        return v.clone()

    monkeypatch.setattr(backends, "mindiesd_attention_forward", fake_forward)
    q = torch.full((1, 2, 8, 128), 512.0, dtype=torch.bfloat16)
    k = torch.full((1, 2, 8, 128), 512.0, dtype=torch.bfloat16)
    v = torch.full((1, 2, 8, 128), 4.0, dtype=torch.bfloat16)

    out = backends.mindie_attn(q, k, v, scale=128**-0.5, input_scale=256.0)
    assert recorded["q_max"] == 2.0
    assert recorded["scale"] == (128**-0.5) * (256.0 * 256.0)
    assert recorded["head_first"] is True
    assert recorded["kwargs"] == {}
    assert torch.equal(out, v)

    q_long = torch.full((1, 2, 4096, 128), 512.0, dtype=torch.bfloat16)
    k_long = torch.full((1, 2, 4096, 128), 512.0, dtype=torch.bfloat16)
    v_long = torch.full((1, 2, 4096, 128), 4.0, dtype=torch.bfloat16)
    backends.mindie_attn(q_long, k_long, v_long, scale=128**-0.5, input_scale=256.0)
    assert recorded["kwargs"] == {
        "opt_mode": "manual",
        "op_type": "ascend_laser_attention",
        "layout": "BNSD",
    }


def test_minimax_h3_cpu_staged_online_adaln_materialization_matches_live() -> None:
    from telefuser.models.minimax_h3_dit import MiniMaxH3DiT, MiniMaxH3DiTConfig

    torch.manual_seed(5)
    cfg = MiniMaxH3DiTConfig(
        hidden_size=32,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=1,
    )
    model = MiniMaxH3DiT(cfg).eval()
    seq = 8
    inputs = {
        "x": torch.randn(1, seq, 8),
        "audio_x": torch.randn(1, seq, 2),
        "img_position_ids": torch.zeros(1, seq, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.25, 0.75]),
        "inverse_indices": torch.tensor([0, 0, 1, 1, 0, 0, 0, 0], dtype=torch.long),
        "update_mask": torch.ones(4, dtype=torch.bool),
        "token_tags": torch.tensor([1, 1, 2, 2, 0, 0, 0, 0]),
        "prompt_embeds": torch.randn(2, 16),
        "img_pos_info": {"position_ids": torch.arange(4, 8)},
        "audio_pos_info": {"position_ids": torch.arange(2, 4)},
        "text_pos_info": {"position_ids": torch.arange(0, 2)},
        "img_pos_for_infer_output_info": {"position_ids": torch.arange(4, 8)},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, seq], dtype=torch.int32)},
    }
    expected_v, expected_a = model(**inputs)

    model.enable_online_adaln_cache(partition="FL2VA")
    assert model.materialize_online_adaln_cache_for_timesteps([0.25, 0.75, 1.0], device=torch.device("cpu")) is True
    assert model.time_embedder is None
    assert all(b.adaln_proj.linear is None for b in model.blocks)
    actual_v, actual_a = model(**inputs)
    torch.testing.assert_close(actual_v, expected_v)
    torch.testing.assert_close(actual_a, expected_a)


def test_minimax_h3_npu_parallel_topologies_fit_1_to_8_cards(monkeypatch) -> None:
    from examples.minimax_h3 import minimax_h3_fl2va_h100 as fl2va_example

    recorded: list[tuple[int, int, int, int]] = []

    def fake_loader(_root: str, **kwargs):
        recorded.append(
            (
                int(kwargs["ulysses_degree"]),
                int(kwargs["tp_degree"]),
                int(kwargs["text_encoder_tp_degree"]),
                1 if kwargs["attn_impl"] == fl2va_example.AttnImplType.MINDIE_ATTN else 0,
            )
        )
        return "ok"

    monkeypatch.setattr(fl2va_example, "load_minimax_h3_pipeline", fake_loader)
    for cards in (1, 2, 4, 8):
        fl2va_example.get_pipeline(cards, "/models/h3", device="npu:0")

    assert recorded == [
        (1, 1, 1, 1),
        (1, 2, 2, 1),
        (4, 1, 4, 1),
        (8, 1, 8, 1),
    ]


def test_raise_nofile_limit_clamps_to_hard_limit(monkeypatch) -> None:
    import resource

    from telefuser.worker.parallel_worker import _raise_nofile_limit

    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda _res: (1024, 32768))
    monkeypatch.setattr(resource, "setrlimit", lambda res, limits: calls.append((res, limits)))

    _raise_nofile_limit(65536)
    assert calls == [(resource.RLIMIT_NOFILE, (32768, 32768))]


def test_minimax_h3_adaln_cache_indices_for_skips_assert_async_on_npu(monkeypatch) -> None:
    from telefuser.models.minimax_h3_dit import MiniMaxH3AdaLNCache, MiniMaxH3DiT, MiniMaxH3DiTConfig

    cfg = MiniMaxH3DiTConfig(
        hidden_size=32,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=1,
    )
    model = MiniMaxH3DiT(cfg).eval()
    model.enable_online_adaln_cache(partition="FL2VA")
    assert model.materialize_online_adaln_cache_for_timesteps([0.25, 0.75, 1.0], device=torch.device("cpu")) is True
    assert isinstance(model._adaln_cache, MiniMaxH3AdaLNCache)

    assert_calls = 0

    def fail_assert_async(*_args, **_kwargs):
        nonlocal assert_calls
        assert_calls += 1

    class _NpuReqTensor(torch.Tensor):
        def detach(self):  # type: ignore[override]
            return super().detach().as_subclass(_NpuReqTensor)

        def to(self, *args, **kwargs):  # type: ignore[override]
            kwargs.pop("device", None)
            return super().to(*args, **kwargs)

    monkeypatch.setattr(torch, "_assert_async", fail_assert_async)
    idx = model._adaln_cache._indices_for(
        torch.tensor([0.25, 1.0]).as_subclass(_NpuReqTensor),
        SimpleNamespace(type="npu"),  # type: ignore[arg-type]
        model._adaln_cache.timesteps,
    )
    assert idx.tolist() == [0, 2]
    assert assert_calls == 0


def test_minimax_h3_video_vae_decode_moves_frames_to_cpu_and_clears_cache_on_npu(monkeypatch) -> None:
    import telefuser.pipelines.minimax_h3.vae as vae_mod
    from telefuser.core.config import ModelRuntimeConfig
    from telefuser.core.module_manager import ModuleManager

    class _FakeNpuParam(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return SimpleNamespace(type="npu")

    class _FakeVideoVAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prepared_dtype = None

        def parameters(self, recurse: bool = True):
            yield torch.zeros(()).as_subclass(_FakeNpuParam)

        def prepare_decoder_autocast_weights(self, dtype: torch.dtype) -> None:
            self.prepared_dtype = dtype

        def decode_normalized(self, latent: torch.Tensor) -> torch.Tensor:
            return latent[:, :3]

    class _DummyAutocast:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    empty_calls = 0

    def fake_empty_cache() -> None:
        nonlocal empty_calls
        empty_calls += 1

    monkeypatch.setattr(torch, "autocast", lambda **_kw: _DummyAutocast())
    monkeypatch.setattr(vae_mod.current_platform, "empty_cache", fake_empty_cache)

    video_vae = _FakeVideoVAE()
    manager = ModuleManager(device="cpu")
    manager.add_module(video_vae, "minimax_h3_video_vae")
    runtime = ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32)
    stage = vae_mod.MiniMaxH3VideoVAEStage(manager, runtime)
    stage.onload_models_flag = True

    class _LatentIgnoreNpuTo(torch.Tensor):
        def to(self, *args, **kwargs):  # type: ignore[override]
            if args and getattr(args[0], "type", None) == "npu":
                return self
            return super().to(*args, **kwargs)

    out = stage.decode_video(torch.zeros(1, 24, 1, 2, 2).as_subclass(_LatentIgnoreNpuTo))
    assert out.shape == (1, 1, 2, 2, 3)
    assert out.device.type == "cpu"
    assert out.dtype == torch.float32
    assert video_vae.prepared_dtype == torch.float16
    assert empty_calls == 1
