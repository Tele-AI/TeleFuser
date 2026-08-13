"""Tensor-manifest helpers for chunk-boundary NCCL session migration.

The control plane transports only a small Python metadata object.  All retained
model tensors are described by that object, allocated on the target GPU, and
then copied directly with ``torch.distributed`` point-to-point NCCL operations.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def flatten_tensor_tree(value: Any, *, path: tuple[Any, ...] = ()) -> tuple[Any, list[dict[str, Any]], dict[tuple[Any, ...], torch.Tensor]]:
    """Separate a nested tree into scalar skeleton, tensor manifest, and leaves."""
    manifest: list[dict[str, Any]] = []
    leaves: dict[tuple[Any, ...], torch.Tensor] = {}

    def visit(item: Any, item_path: tuple[Any, ...]) -> Any:
        if isinstance(item, torch.Tensor):
            tensor = item.detach()
            manifest.append(
                {
                    "path": list(item_path),
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                }
            )
            leaves[item_path] = tensor
            return {"__tensor__": list(item_path)}
        if isinstance(item, dict):
            return {"__dict__": {key: visit(child, item_path + (key,)) for key, child in item.items()}}
        if isinstance(item, list):
            return {"__list__": [visit(child, item_path + (index,)) for index, child in enumerate(item)]}
        if isinstance(item, tuple):
            return {"__tuple__": [visit(child, item_path + (index,)) for index, child in enumerate(item)]}
        return {"__value__": item}

    return visit(value, path), manifest, leaves


def allocate_tensor_tree_leaves(manifest: list[dict[str, Any]], device: torch.device) -> dict[tuple[Any, ...], torch.Tensor]:
    """Allocate target GPU tensors from a source manifest."""
    dtype_table = {name.removeprefix("torch."): value for name, value in vars(torch).items() if isinstance(value, torch.dtype)}
    leaves: dict[tuple[Any, ...], torch.Tensor] = {}
    for entry in manifest:
        dtype_name = str(entry["dtype"])
        if dtype_name not in dtype_table:
            raise ValueError(f"Unsupported NCCL tensor dtype {dtype_name!r}")
        leaves[tuple(entry["path"])] = torch.empty(tuple(entry["shape"]), dtype=dtype_table[dtype_name], device=device)
    return leaves


def rebuild_tensor_tree(skeleton: Any, leaves: dict[tuple[Any, ...], torch.Tensor]) -> Any:
    """Rebuild a nested value produced by :func:`flatten_tensor_tree`."""
    if "__tensor__" in skeleton:
        return leaves[tuple(skeleton["__tensor__"])]
    if "__dict__" in skeleton:
        return {key: rebuild_tensor_tree(value, leaves) for key, value in skeleton["__dict__"].items()}
    if "__list__" in skeleton:
        return [rebuild_tensor_tree(value, leaves) for value in skeleton["__list__"]]
    if "__tuple__" in skeleton:
        return tuple(rebuild_tensor_tree(value, leaves) for value in skeleton["__tuple__"])
    if "__value__" in skeleton:
        return skeleton["__value__"]
    raise ValueError("Invalid NCCL tensor-tree skeleton")


def transfer_tensor_leaves_nccl(
    leaves: dict[tuple[Any, ...], torch.Tensor],
    *,
    peer_rank: int,
    send: bool,
) -> int:
    """Synchronously send or receive a manifest's GPU leaves over NCCL."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("NCCL process group is not initialized")
    ordered = [leaves[path].contiguous() for path in sorted(leaves, key=lambda value: tuple(map(str, value)))]
    ops = [dist.P2POp(dist.isend if send else dist.irecv, tensor, peer_rank) for tensor in ordered]
    if ops:
        requests = dist.batch_isend_irecv(ops)
        for request in requests:
            request.wait()
    return sum(tensor.numel() * tensor.element_size() for tensor in ordered)
