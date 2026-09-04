from types import ModuleType
from unittest.mock import MagicMock, patch

import torch

from telefuser.ops.attention import attention_impl, backends


def test_flash_attn4_dispatch_uses_cute_return_lse_argument() -> None:
    q = torch.randn(1, 3, 2, 4)
    flash_attn4 = MagicMock(return_value=(q, torch.zeros(1, 2, 3)))

    with (
        patch.object(attention_impl, "FLASH_ATTN_4_AVAILABLE", True),
        patch.object(attention_impl, "flash_attn4", flash_attn4),
    ):
        output, lse = attention_impl.attention(
            q,
            q,
            q,
            attn_impl=attention_impl.AttnImplType.FLASH_ATTN_4,
            return_lse=True,
        )

    assert output is q
    assert lse.shape == (1, 2, 3)
    flash_attn4.assert_called_once_with(
        q,
        q,
        q,
        softmax_scale=None,
        causal=False,
        return_lse=True,
    )


def test_flash_attn4_dispatch_unwraps_output_when_lse_is_disabled() -> None:
    q = torch.randn(1, 3, 2, 4)
    flash_attn4 = MagicMock(return_value=(q, None))

    with (
        patch.object(attention_impl, "FLASH_ATTN_4_AVAILABLE", True),
        patch.object(attention_impl, "flash_attn4", flash_attn4),
    ):
        output = attention_impl.attention(
            q,
            q,
            q,
            attn_impl=attention_impl.AttnImplType.FLASH_ATTN_4,
        )

    assert output is q
    flash_attn4.assert_called_once_with(
        q,
        q,
        q,
        softmax_scale=None,
        causal=False,
        return_lse=False,
    )


def test_flash_attn4_varlen_dispatch_reuses_cumulative_lengths() -> None:
    q = torch.randn(1, 5, 2, 4)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    flash_attn4_varlen = MagicMock(return_value=q[0])

    with (
        patch.object(attention_impl, "FLASH_ATTN_4_AVAILABLE", True),
        patch.object(attention_impl, "flash_attn4", MagicMock()),
        patch.object(attention_impl, "flash_attn4_varlen", flash_attn4_varlen),
    ):
        output = attention_impl.attention(
            q,
            q,
            q,
            attn_impl=attention_impl.AttnImplType.FLASH_ATTN_4,
            sequence_lengths=[2, 3],
            cu_seqlens=cu_seqlens,
            scale=0.5,
        )

    torch.testing.assert_close(output, q)
    flash_attn4_varlen.assert_called_once()
    args = flash_attn4_varlen.call_args.args
    kwargs = flash_attn4_varlen.call_args.kwargs
    for actual in args:
        torch.testing.assert_close(actual, q[0])
    torch.testing.assert_close(kwargs.pop("cu_seqlens_q"), cu_seqlens)
    torch.testing.assert_close(kwargs.pop("cu_seqlens_k"), cu_seqlens)
    assert kwargs == {
        "max_seqlen_q": 3,
        "max_seqlen_k": 3,
        "softmax_scale": 0.5,
        "causal": False,
        "return_lse": False,
    }


@torch.no_grad()
def test_flash_attn4_fixed_valid_can_return_only_live_tokens() -> None:
    q = torch.randn(1, 5, 2, 4)
    flash_attn4 = MagicMock(side_effect=lambda query, *_args, **_kwargs: (query.clone(), None))

    with (
        patch.object(attention_impl, "FLASH_ATTN_4_AVAILABLE", True),
        patch.object(attention_impl, "flash_attn4", flash_attn4),
    ):
        output = attention_impl.attention(
            q,
            q,
            q,
            attn_impl=attention_impl.AttnImplType.FLASH_ATTN_4,
            sequence_lengths=[3, 2],
            fixed_valid=True,
            pad_fixed_valid_output=False,
        )

    assert output.shape == (1, 3, 2, 4)
    torch.testing.assert_close(output, q[:, :3])
    assert flash_attn4.call_args.args[0].shape == (1, 3, 2, 4)


def test_flash_attn4_packed_falls_back_to_sdpa_without_varlen_backend() -> None:
    q = torch.randn(1, 5, 2, 4)

    with (
        patch.object(attention_impl, "FLASH_ATTN_4_AVAILABLE", True),
        patch.object(attention_impl, "flash_attn4", MagicMock()),
        patch.object(attention_impl, "flash_attn4_varlen", None),
    ):
        output = attention_impl.attention(
            q,
            q,
            q,
            attn_impl=attention_impl.AttnImplType.FLASH_ATTN_4,
            sequence_lengths=[2, 3],
        )

    expected = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                part.transpose(0, 1).unsqueeze(0),
                part.transpose(0, 1).unsqueeze(0),
                part.transpose(0, 1).unsqueeze(0),
            )
            for part in (q[0, :2], q[0, 2:])
        ],
        dim=2,
    ).transpose(1, 2)
    torch.testing.assert_close(output, expected)


def test_torch_sdpa_bsnd_layout_uses_transpose_views(monkeypatch) -> None:
    q = torch.randn(1, 7, 3, 5)
    captured: dict[str, torch.Tensor] = {}

    def fake_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        captured["query"] = query
        captured["key"] = key
        captured["value"] = value
        return torch.zeros_like(query)

    monkeypatch.setattr(attention_impl.F, "scaled_dot_product_attention", fake_sdpa)

    output = attention_impl.attention(
        q,
        q,
        q,
        attn_impl=attention_impl.AttnImplType.TORCH_SDPA,
        input_layout="BSND",
        output_layout="BSND",
    )

    assert captured["query"]._base is q
    assert captured["key"]._base is q
    assert captured["value"]._base is q
    assert captured["query"].shape == (1, 3, 7, 5)
    assert not captured["query"].is_contiguous()
    assert output.shape == q.shape


def test_sage_attention_prefers_tf_kernel() -> None:
    imported_modules: list[str] = []
    tf_kernel_module = ModuleType("tf_kernel")
    sageattention_module = ModuleType("sageattention")
    previous_available = backends.SAGE_ATTN_AVAILABLE
    previous_backend = backends.sageattention
    modules = {"tf_kernel": tf_kernel_module, "sageattention": sageattention_module}

    def import_module(name: str) -> ModuleType:
        imported_modules.append(name)
        return modules[name]

    try:
        with (
            patch("telefuser.ops.attention.backends.importlib.util.find_spec", return_value=object()),
            patch("telefuser.ops.attention.backends.importlib.import_module", side_effect=import_module),
        ):
            backends._try_import_sage_attn()

        assert imported_modules == ["tf_kernel"]
        assert backends.SAGE_ATTN_AVAILABLE is True
        assert backends.sageattention is tf_kernel_module
    finally:
        backends.SAGE_ATTN_AVAILABLE = previous_available
        backends.sageattention = previous_backend


def test_sage_attention_falls_back_to_standalone_package() -> None:
    imported_modules: list[str] = []
    sageattention_module = ModuleType("sageattention")
    previous_available = backends.SAGE_ATTN_AVAILABLE
    previous_backend = backends.sageattention

    def find_spec(name: str) -> object | None:
        return None if name == "tf_kernel" else object()

    def import_module(name: str) -> ModuleType:
        imported_modules.append(name)
        return sageattention_module

    try:
        with (
            patch("telefuser.ops.attention.backends.importlib.util.find_spec", side_effect=find_spec),
            patch("telefuser.ops.attention.backends.importlib.import_module", side_effect=import_module),
        ):
            backends._try_import_sage_attn()

        assert imported_modules == ["sageattention"]
        assert backends.SAGE_ATTN_AVAILABLE is True
        assert backends.sageattention is sageattention_module
    finally:
        backends.SAGE_ATTN_AVAILABLE = previous_available
        backends.sageattention = previous_backend


def test_sage_attention_falls_back_when_tf_kernel_cannot_load() -> None:
    imported_modules: list[str] = []
    sageattention_module = ModuleType("sageattention")
    previous_available = backends.SAGE_ATTN_AVAILABLE
    previous_backend = backends.sageattention

    def import_module(name: str) -> ModuleType:
        imported_modules.append(name)
        if name == "tf_kernel":
            raise ImportError("incompatible tf-kernel wheel")
        return sageattention_module

    try:
        with (
            patch("telefuser.ops.attention.backends.importlib.util.find_spec", return_value=object()),
            patch("telefuser.ops.attention.backends.importlib.import_module", side_effect=import_module),
        ):
            backends._try_import_sage_attn()

        assert imported_modules == ["tf_kernel", "sageattention"]
        assert backends.SAGE_ATTN_AVAILABLE is True
        assert backends.sageattention is sageattention_module
    finally:
        backends.SAGE_ATTN_AVAILABLE = previous_available
        backends.sageattention = previous_backend


def test_ring_lse_support_recognizes_sage_attention_names() -> None:
    previous_available = backends.SAGE_ATTN_AVAILABLE
    try:
        backends.SAGE_ATTN_AVAILABLE = True
        assert backends.supports_return_lse("SAGE_ATTN_2_8_8_SM90") is True
    finally:
        backends.SAGE_ATTN_AVAILABLE = previous_available


def test_ring_config_converts_fallback_name() -> None:
    from telefuser.core.config import AttentionConfig, AttnImplType
    from telefuser.ops.attention.attention_impl import _get_ring_attn_config

    config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
    with (
        patch("telefuser.ops.attention.attention_impl.supports_return_lse", return_value=False),
        patch(
            "telefuser.ops.attention.attention_impl.get_lse_fallback_impl",
            return_value="SAGE_ATTN_2_8_8_SM90",
        ),
    ):
        result = _get_ring_attn_config(config, scale=0.125, is_causal=True)

    assert result.attn_impl is AttnImplType.SAGE_ATTN_2_8_8_SM90
    assert result.scale == 0.125
    assert result.is_causal is True
