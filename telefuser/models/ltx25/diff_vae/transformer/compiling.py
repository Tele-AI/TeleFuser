"""``torch.compile`` setup for the isolated LTX-2.5 DiffVAE decoder."""

from __future__ import annotations

from typing import Any, Callable

import torch

from telefuser.models.ltx25.diff_vae.transformer.chunked.block import ChunkedDiffusionNABlock
from telefuser.ops.neighborhood_attention import configure_neighborhood_attention_kv_parallelism


def compile_diffusion_decoder(decoder: torch.nn.Module) -> torch.nn.Module:
    """Compile the upstream CHUNKED_COMPILE stage-5 residual methods.

    Context injection remains eager and in-place. Compiling only
    ``forward_attn_mlp`` keeps the peak allocation near the eager chunked path.
    """
    if not hasattr(torch, "compile"):
        raise RuntimeError("CHUNKED_COMPILE requires PyTorch 2.0 or newer")

    configure_neighborhood_attention_kv_parallelism(False)
    compile_kwargs: dict[str, Any] = {
        "mode": None,
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": None,
    }

    def _compile(function: Callable[..., Any]) -> Callable[..., Any]:
        with torch._dynamo.config.patch(inline_inbuilt_nn_modules=True, cache_size_limit=256):  # type: ignore[attr-defined]
            return torch.compile(function, **compile_kwargs)

    for block in decoder.diff_blocks:  # type: ignore[attr-defined]
        if isinstance(block, ChunkedDiffusionNABlock):
            block.forward_attn_mlp = _compile(block.forward_attn_mlp)  # type: ignore[method-assign]

    decoder.mark_dynamic_shapes = True  # type: ignore[attr-defined]
    return decoder
