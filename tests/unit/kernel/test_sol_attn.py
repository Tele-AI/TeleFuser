from pathlib import Path

import pytest
import torch

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from telefuser.kernel.sol_attn.interface import _backend_for_arch, _validate_inputs, sol_attn


def test_sol_attn_is_packaged_with_telefuser() -> None:
    project_root = Path(__file__).parents[3]
    with (project_root / "pyproject.toml").open("rb") as config_file:
        package_data = tomllib.load(config_file)["tool"]["setuptools"]["package-data"]

    assert package_data["telefuser.kernel.sol_attn"] == [
        "THIRD_PARTY_NOTICES.md",
        "sm100/LICENSE.flash-attention",
    ]
    assert (project_root / "telefuser" / "kernel" / "sol_attn" / "__init__.py").is_file()
    assert not (project_root / "tf-kernel" / "tf_kernel" / "_sol_attn").exists()


def test_backend_selection_prefers_cute_and_falls_back_to_triton() -> None:
    assert _backend_for_arch((9, 0), cute_available=True) == "cute_sm90"
    assert _backend_for_arch((10, 0), cute_available=True) == "cute_sm100"
    assert _backend_for_arch((12, 0), cute_available=True) == "cute_sm120"
    assert _backend_for_arch((9, 0), cute_available=False) == "triton"
    assert _backend_for_arch((8, 0), cute_available=True) == "triton"

    with pytest.raises(RuntimeError, match="compute capability >= 8.0"):
        _backend_for_arch((7, 5), cute_available=False)


def test_input_validation_rejects_cpu_tensors() -> None:
    q = torch.randn(1, 64, 2, 128, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="same CUDA device"):
        _validate_inputs(q, q, q, "diag")


@pytest.mark.gpu
def test_sol_attn_dense_limit_matches_sdpa_on_sm90() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Sol-Attn SM90 correctness test requires H100")

    torch.manual_seed(0)
    q = torch.randn(1, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    output = sol_attn(q, k, v, tau=-1000.0, thresh_type="diag", kv_splits=1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)

    torch.testing.assert_close(output, expected, atol=0.05, rtol=0.02)


@pytest.mark.gpu
@pytest.mark.parametrize("heads", [14, 56])
def test_sm90_fp8_native_after_hybrid_head_sharding(heads: int, monkeypatch) -> None:
    """Compile native FP8 Sol for hybrid and unsharded head counts, without a fallback."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Native FP8 Sol requires SM90")
    pytest.importorskip("cutlass.cute")
    from telefuser.kernel.sol_attn import interface, triton_ref

    def reject_fallback(*args, **kwargs):
        raise AssertionError("Native FP8 Sol must not fall back to Triton")

    monkeypatch.setattr(triton_ref, "sol_attn", reject_fallback)
    calls = []
    original = interface._sol_attn_cute

    def capture_native(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(args[0].dtype)
        return result

    monkeypatch.setattr(interface, "_sol_attn_cute", capture_native)
    torch.manual_seed(71)
    q, k, v = [
        torch.randn(1, 128, heads, 128, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn) for _ in range(3)
    ]
    qk_scale = torch.ones(1, 2, heads, device="cuda")
    v_scale = torch.ones(1, heads, 128, device="cuda")
    actual = interface.sol_attn(q, k, v, tau=-1000.0, q_scale=qk_scale, k_scale=qk_scale, v_scale=v_scale)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.bfloat16).transpose(1, 2),
        k.to(torch.bfloat16).transpose(1, 2),
        v.to(torch.bfloat16).transpose(1, 2),
    ).transpose(1, 2)
    torch.cuda.synchronize()
    assert calls == [torch.float8_e4m3fn]
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
