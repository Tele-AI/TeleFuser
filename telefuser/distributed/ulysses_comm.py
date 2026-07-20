"""Ulysses All-to-All communication primitives for sequence parallelism.

Ulysses requires an equal head partition: ``num_heads`` must be divisible by
the process-group world size.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc


def _get_distributed_info(process_group: dist.ProcessGroup) -> tuple[int, int]:
    """Return the process-group rank and world size."""
    return dist.get_rank(group=process_group), dist.get_world_size(group=process_group)


def _local_head_count(num_heads: int, world_size: int) -> int:
    """Return the equal Ulysses head partition or reject an invalid topology."""
    if num_heads % world_size:
        raise ValueError(
            f"Ulysses sequence parallelism requires num_heads ({num_heads}) to be divisible "
            f"by world_size ({world_size})"
        )
    return num_heads // world_size


def _wait_async_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Resolve an asynchronous functional collective tensor."""
    if isinstance(tensor, fc.AsyncCollectiveTensor):
        tensor = tensor.wait()
    return tensor


def ulysses_scatter_heads(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    *,
    async_comm: bool = True,
) -> Callable[[], torch.Tensor]:
    """Scatter global heads and gather sequence across Ulysses ranks."""
    _, world_size = _get_distributed_info(process_group)
    batch, local_seq_len, num_heads, head_dim = tensor.shape
    local_heads = _local_head_count(num_heads, world_size)

    tensor = tensor.reshape(batch, local_seq_len, world_size, local_heads, head_dim)
    tensor = tensor.permute(2, 1, 0, 3, 4).contiguous()
    comm_buffer_shape = tensor.shape

    if async_comm:
        submitted = fc.all_to_all_single(tensor.flatten(), None, None, process_group)

        def wait() -> torch.Tensor:
            result = _wait_async_tensor(submitted).reshape(comm_buffer_shape)
            return result.flatten(0, 1).permute(1, 0, 2, 3)

    else:
        submitted = tensor.flatten()
        output = torch.empty_like(submitted)
        dist.all_to_all_single(output, submitted, None, None, group=process_group, async_op=False)

        def wait() -> torch.Tensor:
            result = output.reshape(comm_buffer_shape)
            return result.flatten(0, 1).permute(1, 0, 2, 3)

    return wait


def ulysses_gather_heads(
    tensor: torch.Tensor,
    process_group: dist.ProcessGroup,
    *,
    num_heads: int,
    async_comm: bool = True,
) -> Callable[[], torch.Tensor]:
    """Gather global heads and scatter sequence across Ulysses ranks."""
    _, world_size = _get_distributed_info(process_group)
    batch, global_seq_len, local_heads, head_dim = tensor.shape
    if global_seq_len % world_size:
        raise ValueError(f"Ulysses sequence length ({global_seq_len}) must be divisible by world_size ({world_size})")
    expected_local_heads = _local_head_count(num_heads, world_size)
    if local_heads != expected_local_heads:
        raise ValueError(f"Ulysses local head count must be {expected_local_heads}, got {local_heads}")
    local_seq_len = global_seq_len // world_size

    tensor = tensor.reshape(batch, world_size, local_seq_len, local_heads, head_dim)
    tensor = tensor.permute(1, 3, 0, 2, 4).contiguous()
    comm_buffer_shape = tensor.shape

    if async_comm:
        submitted = fc.all_to_all_single(tensor.flatten(), None, None, process_group)

        def wait() -> torch.Tensor:
            result = _wait_async_tensor(submitted).reshape(comm_buffer_shape)
            return result.flatten(0, 1).permute(1, 2, 0, 3)

    else:
        submitted = tensor.flatten()
        output = torch.empty_like(submitted)
        dist.all_to_all_single(output, submitted, None, None, group=process_group, async_op=False)

        def wait() -> torch.Tensor:
            result = output.reshape(comm_buffer_shape)
            return result.flatten(0, 1).permute(1, 2, 0, 3)

    return wait
