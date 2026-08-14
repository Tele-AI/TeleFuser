"""Hopper MMA atoms used by Sol-Attn."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from ._compat import sm90_utils


def make_pv_mma(
    tile_m: int = 64,
    tile_v: int = 128,
    a_dtype=cutlass.BFloat16,
    b_dtype=cutlass.BFloat16,
    source: str = "RS",
) -> cute.TiledMma:
    b_major = "K" if b_dtype is cutlass.Float8E4M3FN else "MN"
    return sm90_utils.make_tiled_mma(
        a_dtype,
        "K",
        b_major,
        tile_v,
        source=source,
        atom_layout_mnk=(tile_m // 64, 1, 1),
        b_dtype=b_dtype,
        acc_dtype=Float32,
    )


__all__ = ["make_pv_mma"]
