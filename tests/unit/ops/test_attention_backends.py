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
