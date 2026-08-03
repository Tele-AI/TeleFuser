"""Sequence Parallel Shard Utilities.

This module provides utilities for sharding and unsharding tensors
according to the sequence parallel strategy (Ulysses, Ring, or USP).

Key concepts:
- In Ulysses mode: Sequence is split across all SP ranks, heads are complete
- In Ring mode: Sequence is split across all SP ranks, heads are complete
- In USP mode: Sequence is split across ring_degree only, heads are complete
  (Ulysses handles head splitting via All-to-All during attention)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from .collectives import all_gather_cat
from .device_mesh import (
    get_attention_strategy,
    get_cfg_group,
    get_cfg_rank,
    get_cfg_world_size,
    get_ring_group,
    get_ring_rank,
    get_ring_world_size,
    get_ulysses_group,
    get_ulysses_rank,
    get_ulysses_world_size,
)


def get_sp_shard_degree(device_mesh: DeviceMesh) -> int:
    """Get the degree for sequence sharding.

    In USP mode, sequence is only split by ring_degree (not ring_degree * ulysses_degree).
    Ulysses handles head splitting via All-to-All during attention computation.

    Returns:
        The degree for sequence dimension sharding
    """
    strategy = get_attention_strategy(device_mesh)

    if strategy == "usp":
        # USP: sequence split by ring_degree, heads split by ulysses via All-to-All
        return get_ring_world_size(device_mesh)
    elif strategy == "ulysses":
        return get_ulysses_world_size(device_mesh)
    elif strategy == "ring":
        return get_ring_world_size(device_mesh)
    return 1  # local


def get_sp_shard_rank(device_mesh: DeviceMesh) -> int:
    """Get the rank for sequence sharding.

    In USP mode, we need the ring rank for sequence sharding.

    Returns:
        The rank for sequence dimension sharding
    """
    strategy = get_attention_strategy(device_mesh)

    if strategy == "usp":
        return get_ring_rank(device_mesh)
    elif strategy == "ulysses":
        return get_ulysses_rank(device_mesh)
    elif strategy == "ring":
        return get_ring_rank(device_mesh)
    return 0  # local


def get_sp_shard_group(device_mesh: DeviceMesh) -> dist.ProcessGroup | None:
    """Get the process group for sequence sharding communication.

    In USP mode, the shard group is the ring group (ranks with different sequence chunks).
    In Ulysses/Ring mode, it's the respective group.

    Returns:
        Process group for sequence sharding
    """
    strategy = get_attention_strategy(device_mesh)

    if strategy == "usp":
        return get_ring_group(device_mesh)
    elif strategy == "ulysses":
        return get_ulysses_group(device_mesh)
    elif strategy == "ring":
        return get_ring_group(device_mesh)
    return None  # local


def sequence_parallel_shard(
    device_mesh: DeviceMesh,
    tensors: list[torch.Tensor] | None = None,
    seq_dims: list[int] | None = None,
    seq_divisions: list[int] | None = None,
) -> None:
    """In-place sequence-parallel shard: pad -> chunk -> copy shard back into original tensor.

    Sharding strategy depends on attention strategy:
    - Ulysses: Sequence split by total SP degree
    - Ring: Sequence split by total SP degree
    - USP: Sequence split by ring_degree only (Ulysses handles head splitting)

    Args:
        device_mesh: Device mesh containing parallel configuration
        tensors: List of tensors to shard
        seq_dims: List of sequence dimensions for each tensor
        seq_divisions: Optional additional divisions for each tensor
    """
    strategy = get_attention_strategy(device_mesh)

    if strategy == "local":
        return

    shard_degree = get_sp_shard_degree(device_mesh)
    shard_rank = get_sp_shard_rank(device_mesh)

    if shard_degree == 1:
        return

    tensors = [] if tensors is None else tensors
    seq_dims = [] if seq_dims is None else seq_dims
    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have same length"

    for i, (tensor, seq_dim) in enumerate(zip(tensors, seq_dims)):
        if tensor is None:
            continue

        seq_len = tensor.size(seq_dim)
        pad_len = math.ceil(seq_len / shard_degree) * shard_degree - seq_len

        if seq_divisions is not None:
            seq_division = seq_divisions[i] * shard_degree
            pad_len = math.ceil((seq_len + pad_len) / seq_division) * seq_division - seq_len

        if pad_len > 0:
            padding = [0] * (2 * tensor.ndim)
            padding[-2 * seq_dim - 1] = pad_len
            padded_tensor = F.pad(tensor, padding)
        else:
            padded_tensor = tensor

        chunks = torch.chunk(padded_tensor, shard_degree, dim=seq_dim)
        shard = chunks[shard_rank].clone()  # Temporary shard copy
        tensor.resize_(shard.shape)
        tensor.copy_(shard)
        del shard  # Release temporary allocation


def sequence_parallel_unshard(
    device_mesh: DeviceMesh,
    tensors: list[torch.Tensor],
    seq_dims: list[int],
    seq_lens: list[int],
) -> list[torch.Tensor]:
    """Unshard tensors that were sharded by sequence parallelism.

    For USP mode, only gathers across the ring group (sequence dimension).
    Ulysses All-to-All is handled separately during attention computation.

    Args:
        device_mesh: Device mesh containing parallel configuration
        tensors: List of tensors to unshard
        seq_dims: List of sequence dimensions for each tensor
        seq_lens: List of original sequence lengths for each tensor

    Returns:
        List of unsharded tensors
    """
    assert len(tensors) == len(seq_dims), "tensors and seq_dims must have same length"
    assert len(tensors) == len(seq_lens), "tensors and seq_lens must have same length"

    attn_strategy = get_attention_strategy(device_mesh)

    if attn_strategy == "local":
        return tensors

    shard_group = get_sp_shard_group(device_mesh)
    shard_degree = get_sp_shard_degree(device_mesh)

    if shard_group is None or shard_degree == 1:
        return tensors

    unshard_tensors = []
    for tensor, seq_dim, seq_len in zip(tensors, seq_dims, seq_lens):
        unshard = all_gather_cat(tensor, dim=seq_dim, group=shard_group, world_size=shard_degree).narrow(
            dim=seq_dim, start=0, length=seq_len
        )
        unshard_tensors.append(unshard)

    return unshard_tensors


def cfg_parallel_shard(device_mesh: DeviceMesh, tensors: list[torch.Tensor]) -> None:
    """In-place shard tensors for current cfg rank.

    Does NOT keep original copies. Modifies tensors in-place along batch dimension.

    Args:
        device_mesh: Device mesh containing parallel configuration
        tensors: List of tensors to shard along batch dimension
    """
    cfg_world_size = get_cfg_world_size(device_mesh)
    if cfg_world_size == 1:
        return

    for tensor in tensors:
        if tensor is None:
            continue
        # torch.chunk returns view; need clone before resize_
        if tensor.shape[0] % cfg_world_size == 0:
            chunks = torch.chunk(tensor, cfg_world_size, dim=0)
            shard = chunks[get_cfg_rank(device_mesh)].clone()
            tensor.resize_(shard.shape)
            tensor.copy_(shard)
            del shard


def cfg_parallel_unshard(device_mesh: DeviceMesh, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Unshard tensors that were sharded by CFG parallelism.

    Args:
        device_mesh: Device mesh containing parallel configuration
        tensors: List of tensors to unshard

    Returns:
        List of unsharded tensors
    """
    cfg_world_size = get_cfg_world_size(device_mesh)
    if cfg_world_size == 1:
        return tensors

    unshard_tensors = []
    for tensor in tensors:
        unshard_tensors.append(
            all_gather_cat(tensor, dim=0, group=get_cfg_group(device_mesh), world_size=cfg_world_size)
        )

    return unshard_tensors
