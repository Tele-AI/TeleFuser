"""Configuration utilities for torch.compile optimization.

This module provides two types of configuration:

1. Global configs (set_global_compile_configs): Settings that affect ALL torch.compile
   calls in the process. These modify torch._dynamo.config and torch._inductor.config.
   Examples: recompile_limit, fx_graph_cache, autotune_local_cache


IMPORTANT: Global configs should be set ONCE at the start of the program, before any
torch.compile calls. Local configs are passed to each torch.compile() call individually.

Usage:

    # Set global configs (once at startup)
    from telefuser.utils.torch_compile import set_global_compile_configs
    set_global_compile_configs(recompile_limit=1024)

    # Use local configs per model
    from telefuser.core.config import CompileConfig
    config = CompileConfig(enabled=True, mode="max-autotune-no-cudagraphs")
    model = torch.compile(model, **config.get_compile_kwargs())
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    pass  # CompileConfig imported at runtime to avoid circular import


def set_global_compile_configs(
    recompile_limit: int = 8,
    accumulated_recompile_limit: int | None = None,
    fx_graph_cache: bool = True,
    fx_graph_remote_cache: bool = False,
    autotune_local_cache: bool = True,
    compute_comm_overlap: bool = True,
    capture_scalar_outputs: bool = False,
    capture_dynamic_output_shape_ops: bool = False,
) -> None:
    """Set global torch.compile configurations that affect ALL compile calls.

    These settings modify torch._dynamo.config and torch._inductor.config global state.
    Should be called ONCE at program startup, before any torch.compile() calls.

    Args:
        recompile_limit: Max recompilations per frame before fallback to eager.
            PyTorch default is 8. Higher values allow more dynamic shapes but slower.
        accumulated_recompile_limit: Total accumulated recompiles across all frames.
            PyTorch default is 256. If None, computed as recompile_limit * 8.
        fx_graph_cache: Enable inductor FX graph cache. Default True.
        fx_graph_remote_cache: Enable remote FX graph cache. Default False.
        autotune_local_cache: Enable local autotune result cache. Default True.
            IMPORTANT: Setting False will cause kernel tuning on every compile!
        compute_comm_overlap: Enable compute-communication overlap for distributed.
            Default True.
        capture_scalar_outputs: Capture scalar outputs in compiled regions.
        capture_dynamic_output_shape_ops: Capture dynamic shape operations.
    """
    # Dynamo configs
    torch._dynamo.config.recompile_limit = recompile_limit
    if accumulated_recompile_limit is None:
        accumulated_recompile_limit = recompile_limit * 8
    torch._dynamo.config.accumulated_recompile_limit = accumulated_recompile_limit

    # Inductor configs
    torch._inductor.config.fx_graph_cache = fx_graph_cache
    torch._inductor.config.fx_graph_remote_cache = fx_graph_remote_cache

    # Autotune cache - WARNING: False causes kernel tuning on every compile
    torch._inductor.config.autotune_local_cache = autotune_local_cache

    # Distributed configs
    if dist.is_initialized():
        torch._inductor.config.reorder_for_compute_comm_overlap = compute_comm_overlap
        if compute_comm_overlap:
            # L20: 64 GB/s PCIe; A100/A800 NVLink: 300 GB/s
            torch._inductor.config.intra_node_bw = 64 if "L20" in torch.cuda.get_device_name() else 300

    # Capture configs (for nested tensors, etc.)
    if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
        torch._dynamo.config.capture_scalar_outputs = capture_scalar_outputs
        torch._dynamo.config.capture_dynamic_output_shape_ops = capture_dynamic_output_shape_ops
