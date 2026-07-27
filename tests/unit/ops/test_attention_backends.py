from types import ModuleType
from unittest.mock import patch

from telefuser.ops.attention import backends


def test_sage_attention_uses_standalone_package() -> None:
    imported_modules: list[str] = []
    sageattention_module = ModuleType("sageattention")
    previous_available = backends.SAGE_ATTN_AVAILABLE
    previous_backend = backends.sageattention

    def import_module(name: str) -> ModuleType:
        imported_modules.append(name)
        return sageattention_module

    try:
        backends.SAGE_ATTN_AVAILABLE = False
        backends.sageattention = None
        with (
            patch("telefuser.ops.attention.backends.importlib.util.find_spec", return_value=object()),
            patch("telefuser.ops.attention.backends.importlib.import_module", side_effect=import_module),
        ):
            backends._try_import_sage_attn()

        assert imported_modules == ["sageattention"]
        assert backends.SAGE_ATTN_AVAILABLE is True
        assert backends.sageattention is sageattention_module
    finally:
        backends.SAGE_ATTN_AVAILABLE = previous_available
        backends.sageattention = previous_backend
