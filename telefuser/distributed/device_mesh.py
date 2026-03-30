"""DeviceMesh utility functions.

Builds PyTorch DeviceMesh based on ParallelConfig and provides utility functions
for accessing process groups and ranks across different parallelism dimensions.

Mesh layout order: DP -> CFG -> SP(ring, ulysses) -> PP -> TP
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from telefuser.core.config import ParallelConfig
from telefuser.utils.logging import logger


def create_device_mesh_from_config(parallel_config: ParallelConfig, device_type: str = "cuda") -> DeviceMesh:
    """Create PyTorch DeviceMesh from ParallelConfig.

    Mesh dimensions are built in order: DP -> CFG -> SP (ring, ulysses) -> PP -> TP
    For USP (Unified Sequence Parallelism), ring and ulysses form a 2D sub-mesh.

    Args:
        parallel_config: Parallel configuration with degrees for each dimension
        device_type: Device type ("cuda" or "cpu")

    Returns:
        PyTorch DeviceMesh instance with named dimensions
    """
    _validate_parallel_config(parallel_config)

    sp_degree = parallel_config.sp_ulysses_degree * parallel_config.sp_ring_degree

    mesh_dims = []
    dim_names = []

    # Build mesh in order: DP -> CFG -> SP (ring, ulysses) -> PP -> TP
    if parallel_config.dp_degree > 1:
        mesh_dims.append(parallel_config.dp_degree)
        dim_names.append("dp")

    if parallel_config.cfg_degree > 1:
        mesh_dims.append(parallel_config.cfg_degree)
        dim_names.append("cfg")

    # SP mesh: 2D (ring, ulysses) for USP, 1D for single strategy
    if sp_degree > 1:
        if parallel_config.sp_ring_degree > 1 and parallel_config.sp_ulysses_degree > 1:
            # 2D SP mesh for USP: (ring, ulysses)
            mesh_dims.append(parallel_config.sp_ring_degree)
            mesh_dims.append(parallel_config.sp_ulysses_degree)
            dim_names.append("ring")
            dim_names.append("ulysses")
        elif parallel_config.sp_ring_degree > 1:
            mesh_dims.append(parallel_config.sp_ring_degree)
            dim_names.append("ring")
        elif parallel_config.sp_ulysses_degree > 1:
            mesh_dims.append(parallel_config.sp_ulysses_degree)
            dim_names.append("ulysses")

    # PP mesh: Pipeline parallelism dimension
    if parallel_config.pp_degree > 1:
        mesh_dims.append(parallel_config.pp_degree)
        dim_names.append("pp")

    if parallel_config.tp_degree > 1:
        mesh_dims.append(parallel_config.tp_degree)
        dim_names.append("tp")

    # Single GPU fallback
    if not mesh_dims:
        mesh_dims = [1]
        dim_names = ["world"]

    logger.info(f"Creating DeviceMesh with dims={mesh_dims}, names={dim_names}, device_type={device_type}")

    return DeviceMesh(
        device_type=device_type,
        mesh=torch.tensor(parallel_config.device_ids).reshape(mesh_dims),
        mesh_dim_names=tuple(dim_names),
    )


def _validate_parallel_config(parallel_config: ParallelConfig) -> None:
    """Validate parallel configuration constraints.

    Raises:
        ValueError: If SP and TP are both enabled, or if world_size doesn't match expected value.
    """
    sp_degree = parallel_config.sp_ulysses_degree * parallel_config.sp_ring_degree

    # SP and TP are mutually exclusive
    if sp_degree > 1 and parallel_config.tp_degree > 1:
        raise ValueError(
            f"Not allowed to enable sequence parallel and tensor parallel together. "
            f"sp_degree={sp_degree}, tp_degree={parallel_config.tp_degree}"
        )

    # Verify world_size matches expected total
    expected_world_size = (
        parallel_config.cfg_degree
        * sp_degree
        * parallel_config.pp_degree
        * parallel_config.tp_degree
        * parallel_config.dp_degree
    )
    actual_world_size = parallel_config.world_size

    if actual_world_size != expected_world_size:
        raise ValueError(
            f"World size ({actual_world_size}) must equal "
            f"cfg({parallel_config.cfg_degree}) * sp({sp_degree}) * "
            f"pp({parallel_config.pp_degree}) * tp({parallel_config.tp_degree}) * "
            f"dp({parallel_config.dp_degree}) = {expected_world_size}"
        )


# Process group access utilities
def get_group(device_mesh: DeviceMesh | None, group_type: str) -> dist.ProcessGroup | None:
    """Get process group of specified type from DeviceMesh."""
    if device_mesh is None:
        return None
    try:
        return device_mesh.get_group(group_type)
    except (KeyError, RuntimeError):
        return None


def get_ranks(device_mesh: DeviceMesh | None, group_type: str) -> list[int]:
    """Get actual rank numbers of specified process group type."""
    group = get_group(device_mesh, group_type)
    if group is not None:
        return dist.get_process_group_ranks(group)
    return []


def get_world_size(device_mesh: DeviceMesh | None, group_type: str) -> int:
    """Get size of specified process group type (returns 1 if not present)."""
    group = get_group(device_mesh, group_type)
    return dist.get_world_size(group=group) if group is not None else 1


def get_rank(device_mesh: DeviceMesh | None, group_type: str) -> int:
    """Get current process rank in specified process group (returns 0 if not present)."""
    group = get_group(device_mesh, group_type)
    return dist.get_rank(group=group) if group is not None else 0


# Convenience accessors for specific parallelism types
def get_dp_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    """Get data parallel process group."""
    return get_group(device_mesh, "dp")


def get_dp_ranks(device_mesh: DeviceMesh | None) -> list[int]:
    """Get rank list of data parallel process group."""
    return get_ranks(device_mesh, "dp")


def get_dp_world_size(device_mesh: DeviceMesh | None) -> int:
    """Get size of data parallel process group."""
    return get_world_size(device_mesh, "dp")


def get_dp_rank(device_mesh: DeviceMesh | None) -> int:
    """Get current rank in data parallel process group."""
    return get_rank(device_mesh, "dp")


def get_cfg_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    """Get CFG parallel process group."""
    return get_group(device_mesh, "cfg")


def get_cfg_ranks(device_mesh: DeviceMesh | None) -> list[int]:
    """Get rank list of CFG parallel process group."""
    return get_ranks(device_mesh, "cfg")


def get_cfg_world_size(device_mesh: DeviceMesh | None) -> int:
    """Get size of CFG parallel process group."""
    return get_world_size(device_mesh, "cfg")


def get_cfg_rank(device_mesh: DeviceMesh | None) -> int:
    """Get current rank in CFG parallel process group."""
    return get_rank(device_mesh, "cfg")


def get_ring_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    """Get ring attention process group."""
    return get_group(device_mesh, "ring")


def get_ring_ranks(device_mesh: DeviceMesh | None) -> list[int]:
    """Get rank list of ring attention process group."""
    return get_ranks(device_mesh, "ring")


def get_ring_world_size(device_mesh: DeviceMesh | None) -> int:
    """Get size of ring attention process group."""
    return get_world_size(device_mesh, "ring")


def get_ring_rank(device_mesh: DeviceMesh | None) -> int:
    """Get current rank in ring attention process group."""
    return get_rank(device_mesh, "ring")


def get_ulysses_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    """Get ulysses attention process group."""
    return get_group(device_mesh, "ulysses")


def get_ulysses_ranks(device_mesh: DeviceMesh | None) -> list[int]:
    """Get rank list of ulysses attention process group."""
    return get_ranks(device_mesh, "ulysses")


def get_ulysses_world_size(device_mesh: DeviceMesh | None) -> int:
    """Get size of ulysses attention process group."""
    return get_world_size(device_mesh, "ulysses")


def get_ulysses_rank(device_mesh: DeviceMesh | None) -> int:
    """Get current rank in ulysses attention process group."""
    return get_rank(device_mesh, "ulysses")


def get_tp_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    """Get tensor parallel process group."""
    return get_group(device_mesh, "tp")


def get_tp_ranks(device_mesh: DeviceMesh | None) -> list[int]:
    """Get rank list of tensor parallel process group."""
    return get_ranks(device_mesh, "tp")


def get_tp_world_size(device_mesh: DeviceMesh | None) -> int:
    """Get size of tensor parallel process group."""
    return get_world_size(device_mesh, "tp")


def get_tp_rank(device_mesh: DeviceMesh | None) -> int:
    """Get current rank in tensor parallel process group."""
    return get_rank(device_mesh, "tp")


def get_mesh_dim_names(device_mesh: DeviceMesh | None) -> list[str]:
    """Get dimension names of DeviceMesh."""
    if device_mesh is None:
        return []
    return list(device_mesh.mesh_dim_names) if device_mesh.mesh_dim_names else []


def get_coordinates(device_mesh: DeviceMesh | None) -> dict[str, int]:
    """Get coordinates of current process in each mesh dimension.

    Returns:
        Dict mapping dimension names to coordinate indices
    """
    if device_mesh is None:
        return {}
    coords = device_mesh.get_coordinate()
    if coords is None:
        return {}
    dim_names = device_mesh.mesh_dim_names
    if dim_names is not None:
        return {dim_names[i]: coords[i] for i in range(len(coords))}
    return {str(i): coords[i] for i in range(len(coords))}


def get_attention_strategy(device_mesh: DeviceMesh | None) -> str:
    """Determine attention strategy from device mesh configuration.

    Returns:
        "local": No sequence parallelism
        "ulysses": Only Ulysses attention (All-to-All based)
        "ring": Only Ring attention (P2P based)
        "usp": Combined Ulysses + Ring (Unified Sequence Parallelism)
    """
    if device_mesh is None:
        return "local"
    dim_names = device_mesh.mesh_dim_names if device_mesh.mesh_dim_names else []

    has_ring = "ring" in dim_names
    has_ulysses = "ulysses" in dim_names

    if has_ring and has_ulysses:
        return "usp"
    elif has_ring:
        return "ring"
    elif has_ulysses:
        return "ulysses"
    return "local"


def get_ulysses_mesh(device_mesh: DeviceMesh | None) -> DeviceMesh | None:
    """Get the ulysses sub-mesh for All-to-All communication."""
    if device_mesh is None:
        return None
    if "ulysses" in (device_mesh.mesh_dim_names or []):
        return device_mesh["ulysses"]
    return None


def get_ring_mesh(device_mesh: DeviceMesh | None) -> DeviceMesh | None:
    """Get the ring sub-mesh for P2P communication."""
    if device_mesh is None:
        return None
    if "ring" in (device_mesh.mesh_dim_names or []):
        return device_mesh["ring"]
    return None


# PP (Pipeline Parallel) accessors
def get_pp_group(device_mesh: DeviceMesh | None) -> dist.ProcessGroup | None:
    """Get pipeline parallel process group."""
    return get_group(device_mesh, "pp")


def get_pp_ranks(device_mesh: DeviceMesh | None) -> list[int]:
    """Get rank list of pipeline parallel process group."""
    return get_ranks(device_mesh, "pp")


def get_pp_world_size(device_mesh: DeviceMesh | None) -> int:
    """Get size of pipeline parallel process group."""
    return get_world_size(device_mesh, "pp")


def get_pp_rank(device_mesh: DeviceMesh | None) -> int:
    """Get current rank in pipeline parallel process group."""
    return get_rank(device_mesh, "pp")


def is_pipeline_first_stage(device_mesh: DeviceMesh | None) -> bool:
    """Check if current rank is the first pipeline stage."""
    return get_pp_rank(device_mesh) == 0


def is_pipeline_last_stage(device_mesh: DeviceMesh | None) -> bool:
    """Check if current rank is the last pipeline stage."""
    return get_pp_rank(device_mesh) == get_pp_world_size(device_mesh) - 1
