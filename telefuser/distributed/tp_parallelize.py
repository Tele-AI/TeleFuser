"""Tensor Parallel utilities.

Thin wrapper around PyTorch's tensor parallel API for applying parallelization
strategies to specific modules in a model.
"""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle


def _validate_tp_mesh_dim(device_mesh: DeviceMesh) -> None:
    """Validate that tensor parallelism uses a one-dimensional device mesh."""
    ndim = getattr(device_mesh, "ndim", None)
    if ndim is None:
        mesh = getattr(device_mesh, "mesh", None)
        ndim = getattr(mesh, "ndim", None)
    if ndim is not None and ndim != 1:
        raise ValueError(f"Tensor parallelism expects a 1-D DeviceMesh, got {ndim}-D.")


def parallelize_module(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None = None,
) -> nn.Module:
    """Apply tensor parallelism to a module according to the given plan.

    Args:
        module: The module to parallelize
        device_mesh: Device mesh for tensor parallelism
        parallelize_plan: Either a single ParallelStyle to apply to the whole module,
            or a dict mapping module paths to ParallelStyle instances

    Returns:
        The parallelized module

    Raises:
        ValueError: If a module path in the plan is empty
    """
    _validate_tp_mesh_dim(device_mesh)

    if parallelize_plan is None:
        return module

    # Apply single style to entire module
    if isinstance(parallelize_plan, ParallelStyle):
        return parallelize_plan._apply(module, device_mesh)

    # Apply different styles to specific submodules
    for module_path, parallelize_style in parallelize_plan.items():
        if module_path.strip() == "":
            raise ValueError("Module path must be non-empty, got empty string")
        try:
            submodule = module.get_submodule(module_path)
            parallelize_style._apply(submodule, device_mesh)
        except AttributeError:
            # Submodule not found, skip silently
            continue

    return module
