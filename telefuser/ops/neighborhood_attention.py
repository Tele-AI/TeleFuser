"""Public dispatch for optional 3D neighborhood-attention backends."""

from __future__ import annotations

import torch

try:
    import natten

    _NATTEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    natten = None  # type: ignore[assignment]
    _NATTEN_AVAILABLE = False


def natten_available() -> bool:
    """Return whether NATTEN's required CUDA extension is usable."""

    return _NATTEN_AVAILABLE and bool(getattr(natten, "HAS_LIBNATTEN", True))


def configure_neighborhood_attention_kv_parallelism(enabled: bool) -> bool:
    """Configure NATTEN fused-attention KV parallelism when its CUDA extension is present."""
    if not natten_available():
        return False
    configure = getattr(natten, "use_kv_parallelism_in_fused_na", None)
    if configure is None:
        return False
    configure(enabled)
    return True


def neighborhood_attention_3d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    kernel_size: tuple[int, int, int],
    backend: str | None = None,
) -> torch.Tensor:
    """Run NATTEN 3D neighborhood attention with the framework's tensor contract.

    The model pre-scales query, so the NATTEN scale remains one. NATTEN requires
    matching Q/K/V dtypes; the cast mirrors the existing flash-attention paths.
    """
    if not natten_available():
        raise ImportError(
            "natten is required for 3D neighborhood attention. Install a NATTEN build that includes libnatten."
        )
    if query.dtype != value.dtype or key.dtype != value.dtype:
        query = query.to(dtype=value.dtype)
        key = key.to(dtype=value.dtype)
    return natten.na3d(query, key, value, kernel_size=kernel_size, scale=1.0, backend=backend)
