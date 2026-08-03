"""Shared tensor collective primitives used by model parallel strategies."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist


def _resolve_world_size(group: dist.ProcessGroup | None, world_size: int | None) -> int:
    if world_size is None:
        world_size = dist.get_world_size(group=group)
    if world_size < 1:
        raise ValueError("world_size must be at least one")
    return world_size


def all_gather_stacked(
    tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    world_size: int | None = None,
) -> torch.Tensor:
    """Gather equal-shaped tensors into one rank-major contiguous buffer."""
    world_size = _resolve_world_size(group, world_size)
    if world_size == 1:
        return tensor.unsqueeze(0)

    gather_input = tensor.contiguous()
    original_shape = gather_input.shape
    if gather_input.ndim == 0:
        gather_input = gather_input.reshape(1)
    gathered = torch.empty(
        (world_size * gather_input.shape[0], *gather_input.shape[1:]),
        dtype=gather_input.dtype,
        device=gather_input.device,
    )
    dist.all_gather_into_tensor(gathered, gather_input, group=group)
    return gathered.view(world_size, *original_shape)


def all_gather_cat(
    tensor: torch.Tensor,
    *,
    dim: int,
    group: dist.ProcessGroup | None = None,
    world_size: int | None = None,
) -> torch.Tensor:
    """Gather equal-shaped shards in rank order and concatenate one dimension."""
    if tensor.ndim == 0:
        raise ValueError("all_gather_cat requires a tensor with at least one dimension")
    dim = dim if dim >= 0 else tensor.ndim + dim
    if not 0 <= dim < tensor.ndim:
        raise ValueError(f"dim={dim} is invalid for a {tensor.ndim}D tensor")
    world_size = _resolve_world_size(group, world_size)
    if world_size == 1:
        return tensor

    gather_input = tensor.movedim(dim, 0).contiguous()
    gathered = all_gather_stacked(gather_input, group=group, world_size=world_size)
    merged = gathered.flatten(0, 1).movedim(0, dim)
    return merged.contiguous()


def all_reduce_sum_(
    tensors: Iterable[torch.Tensor],
    *,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Sum tensors in place, submitting independent reductions before waiting."""
    works = [dist.all_reduce(tensor, group=group, async_op=True) for tensor in tensors]
    for work in works:
        work.wait()
