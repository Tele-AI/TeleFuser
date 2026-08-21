from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, SparseAttentionConfig
from telefuser.ops.attention import attention_impl, backends
from telefuser.ops.attention.attention_impl import SparseAttentionState


def test_sol_attention_config_defaults_and_validation() -> None:
    config = AttentionConfig.sol_attention()

    assert config.attn_impl is AttnImplType.SOL_ATTN
    assert config.is_sparse()
    assert config.sparse_config == SparseAttentionConfig(
        sparse_impl="sol",
        dense_timesteps=10,
        dense_layers=1,
        sol_tau=1.0,
        sol_threshold_type="diag",
        sol_kv_splits="auto",
    )

    with pytest.raises(ValueError, match="threshold type"):
        AttentionConfig.sol_attention(threshold_type="unknown")
    with pytest.raises(ValueError, match="KV splits"):
        AttentionConfig.sol_attention(kv_splits=3)


def test_sol_attention_loads_from_telefuser_kernel() -> None:
    imported_modules: list[str] = []
    kernel_module = ModuleType("telefuser.kernel.sol_attn")
    kernel_module.sol_attn = MagicMock()
    previous_available = backends.SOL_ATTN_AVAILABLE
    previous_backend = backends.sol_attn

    def import_module(name: str) -> ModuleType:
        imported_modules.append(name)
        return kernel_module

    try:
        with (
            patch("telefuser.ops.attention.backends.importlib.util.find_spec", return_value=object()),
            patch("telefuser.ops.attention.backends.importlib.import_module", side_effect=import_module),
        ):
            backends._try_import_sol_attn()

        assert imported_modules == ["telefuser.kernel.sol_attn"]
        assert backends.SOL_ATTN_AVAILABLE is True
        assert backends.sol_attn is kernel_module.sol_attn
    finally:
        backends.SOL_ATTN_AVAILABLE = previous_available
        backends.sol_attn = previous_backend


def test_sol_attention_ineligible_input_falls_back_to_sdpa() -> None:
    q = torch.randn(1, 8, 2, 16)
    kernel = MagicMock()
    config = AttentionConfig.sol_attention()
    assert config.sparse_config is not None
    state = SparseAttentionState(config.sparse_config, mask_map=None)

    with (
        patch.object(attention_impl, "SOL_ATTN_AVAILABLE", True),
        patch.object(attention_impl, "sol_attn", kernel),
    ):
        output = attention_impl.attention(q, q, q, attention_config=config, sparse_state=state)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        q.transpose(1, 2),
        q.transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)
    kernel.assert_not_called()


def test_sol_attention_dense_guard_does_not_call_kernel() -> None:
    q = torch.randn(1, 8, 2, 16)
    kernel = MagicMock()
    config = AttentionConfig.sol_attention(dense_timesteps=1)
    assert config.sparse_config is not None
    state = SparseAttentionState(config.sparse_config, mask_map=None)

    with (
        patch.object(attention_impl, "SOL_ATTN_AVAILABLE", True),
        patch.object(attention_impl, "sol_attn", kernel),
    ):
        output = attention_impl.attention(q, q, q, attention_config=config, sparse_state=state)

    assert output.shape == q.shape
    kernel.assert_not_called()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fp8_sol_uses_triton_for_unquantized_bf16_layers() -> None:
    q = torch.randn(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    kernel = MagicMock(side_effect=lambda q, _k, _v, **_kwargs: q)
    config = AttentionConfig.sol_attention(
        dense_timesteps=0,
        dense_layers=0,
        sol_fp8=True,
        sol_fp8_layer_start=10,
        sol_fp8_layer_end=20,
    )
    assert config.sparse_config is not None
    state = SparseAttentionState(config.sparse_config, mask_map=None)

    with (
        patch.object(attention_impl, "SOL_ATTN_AVAILABLE", True),
        patch.object(attention_impl, "sol_attn", kernel),
    ):
        output = attention_impl.attention(q, q, q, attention_config=config, sparse_state=state)

    assert output.shape == q.shape
    assert kernel.call_args.kwargs["force_triton"] is True


def test_fp8_dense_uses_sdpa_for_unquantized_bf16_layers() -> None:
    q = torch.randn(1, 64, 1, 128, dtype=torch.bfloat16)
    kernel = MagicMock()
    config = AttentionConfig.sol_attention(
        dense_timesteps=0,
        dense_layers=0,
        tau=-1000.0,
        sol_fp8=True,
        sol_fp8_layer_start=10,
        sol_fp8_layer_end=20,
    )
    assert config.sparse_config is not None
    state = SparseAttentionState(config.sparse_config, mask_map=None)

    with (
        patch.object(attention_impl, "SOL_ATTN_AVAILABLE", True),
        patch.object(attention_impl, "sol_attn", kernel),
    ):
        output = attention_impl.attention(q, q, q, attention_config=config, sparse_state=state)

    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        q.transpose(1, 2),
        q.transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)
    kernel.assert_not_called()


@pytest.mark.gpu
def test_sol_attention_public_ops_matches_sdpa_on_h100(monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Sol-Attn public ops test requires H100")

    assert attention_impl.SOL_ATTN_AVAILABLE
    assert attention_impl.sol_attn is not None
    kernel_calls = 0
    sol_attn = attention_impl.sol_attn

    def tracked_sol_attn(*args, **kwargs):
        nonlocal kernel_calls
        kernel_calls += 1
        return sol_attn(*args, **kwargs)

    monkeypatch.setattr(attention_impl, "sol_attn", tracked_sol_attn)

    torch.manual_seed(0)
    q = torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    config = AttentionConfig.sol_attention(dense_timesteps=0, dense_layers=0, tau=-1000.0)
    assert config.sparse_config is not None
    state = SparseAttentionState(config.sparse_config, mask_map=None)

    output = attention_impl.attention(q, k, v, attention_config=config, sparse_state=state)
    assert kernel_calls == 1
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)

    torch.testing.assert_close(output, expected, atol=0.05, rtol=0.02)
