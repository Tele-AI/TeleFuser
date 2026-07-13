from types import ModuleType
from unittest.mock import patch

from telefuser.ops.attention import backends


def test_sage_attention_prefers_tf_kernel() -> None:
    imported_modules: list[str] = []
    tf_kernel_module = ModuleType("tf_kernel.sageattn2")
    previous_available = backends.SAGE_ATTN_AVAILABLE
    previous_backend = backends.sageattention

    def import_module(name: str) -> ModuleType:
        imported_modules.append(name)
        return tf_kernel_module

    try:
        backends.SAGE_ATTN_AVAILABLE = False
        backends.sageattention = None
        with (
            patch("telefuser.ops.attention.backends.importlib.util.find_spec", return_value=object()),
            patch("telefuser.ops.attention.backends.importlib.import_module", side_effect=import_module),
        ):
            backends._try_import_sage_attn()

        assert imported_modules == ["tf_kernel.sageattn2"]
        assert backends.SAGE_ATTN_AVAILABLE is True
        assert backends.sageattention is tf_kernel_module
    finally:
        backends.SAGE_ATTN_AVAILABLE = previous_available
        backends.sageattention = previous_backend
