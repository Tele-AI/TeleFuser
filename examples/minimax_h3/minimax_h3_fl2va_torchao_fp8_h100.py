# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 FL2VA example with TorchAO FP8 online quantization."""

from __future__ import annotations

from copy import deepcopy
from functools import wraps

if __package__:
    from . import minimax_h3_fl2va_h100 as base
else:
    try:
        from examples.minimax_h3 import minimax_h3_fl2va_h100 as base
    except ModuleNotFoundError:
        import minimax_h3_fl2va_h100 as base

PPL_CONFIG = {
    **base.PPL_CONFIG,
    "name": "minimax_h3_fl2va_torchao_fp8_h100",
    "quantization": "torchao-fp8",
}
PIPELINE_MANIFEST = deepcopy(base.PIPELINE_MANIFEST)
PIPELINE_MANIFEST["pipeline_name"] = PPL_CONFIG["name"]


@wraps(base.run)
def run(*args: object, **kwargs: object) -> base.MiniMaxH3Generation:
    return base.run(*args, **kwargs)


@wraps(base.run_with_file)
def run_with_file(*args: object, **kwargs: object) -> dict[str, str]:
    return base.run_with_file(*args, **kwargs)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    **kwargs: object,
) -> base.MiniMaxH3Pipeline:
    """Load the single-GPU TorchAO FP8 FL2VA pipeline."""
    return base.get_pipeline(
        parallelism,
        model_root,
        quantization=PPL_CONFIG["quantization"],
        **kwargs,
    )


def main() -> None:
    base._main(PPL_CONFIG["quantization"])


if __name__ == "__main__":
    main()
