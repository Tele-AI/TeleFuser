"""Static checks for framework layering rules that can run without a GPU."""

from __future__ import annotations

import ast
from pathlib import Path


def test_models_use_public_ops_instead_of_internal_kernel_imports() -> None:
    """Model code must not couple directly to optional or internal kernels."""
    root = Path(__file__).parents[2] / "telefuser" / "models"
    violations: list[str] = []
    forbidden_prefixes = ("telefuser.kernel", "tf_kernel", "sageattention")
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                module = next((alias.name for alias in node.names if alias.name.startswith(forbidden_prefixes)), None)
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module if node.module.startswith(forbidden_prefixes) else None
            if module is not None:
                violations.append(f"{path}:{node.lineno}: {module}")
    assert not violations, "Models must import optimized operations through telefuser.ops:\n" + "\n".join(violations)
