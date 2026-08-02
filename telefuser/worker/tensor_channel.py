"""Direct tensor transport between independently spawned worker groups."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from multiprocessing.queues import SimpleQueue
from typing import Any

import torch
import torch.multiprocessing as mp


@dataclass(frozen=True)
class WorkerTensorRef:
    """Metadata-only reference to a tensor held by a worker-to-worker channel."""

    channel_id: str
    transfer_id: int
    tensor_index: int
    shape: tuple[int, ...]
    dtype: str
    source_device: str
    nbytes: int


class WorkerTensorChannel:
    """Point-to-point tensor path that bypasses the parent process.

    The producer places tensors directly onto one queue per consumer rank. CUDA
    tensors travel as CUDA IPC handles; CPU tensors use multiprocessing shared
    memory. The parent process receives only :class:`WorkerTensorRef` objects.
    """

    def __init__(self, consumer_world_size: int, *, timeout: int = 600) -> None:
        if consumer_world_size < 1:
            raise ValueError("consumer_world_size must be at least one")
        if timeout < 1:
            raise ValueError("timeout must be at least one second")
        spawn_ctx = mp.get_context("spawn")
        self.channel_id = uuid.uuid4().hex
        self.consumer_world_size = consumer_world_size
        self.timeout = timeout
        self._queues: tuple[SimpleQueue, ...] = tuple(spawn_ctx.SimpleQueue() for _ in range(consumer_world_size))
        self._next_transfer_id = 0
        self._producer_bound = False
        self._consumer_bound = False
        self._closed = False

    def bind_producer(self) -> None:
        """Bind exactly one producer worker group."""
        if self._producer_bound:
            raise ValueError(f"Tensor channel {self.channel_id} already has a producer")
        if self._closed:
            raise RuntimeError(f"Tensor channel {self.channel_id} is closed")
        self._producer_bound = True

    def bind_consumer(self, world_size: int) -> None:
        """Bind exactly one consumer worker group with a matching rank count."""
        if world_size != self.consumer_world_size:
            raise ValueError(
                f"Tensor channel {self.channel_id} expects {self.consumer_world_size} consumer ranks, got {world_size}"
            )
        if self._consumer_bound:
            raise ValueError(f"Tensor channel {self.channel_id} already has a consumer")
        if self._closed:
            raise RuntimeError(f"Tensor channel {self.channel_id} is closed")
        self._consumer_bound = True

    def send(self, value: Any) -> Any:
        """Send every tensor leaf to all consumer ranks and return metadata refs."""
        transfer_id = self._next_transfer_id
        self._next_transfer_id += 1
        tensor_index = 0
        sent: dict[int, WorkerTensorRef] = {}

        def replace(item: Any) -> Any:
            nonlocal tensor_index
            if isinstance(item, torch.Tensor):
                existing = sent.get(id(item))
                if existing is not None:
                    return existing
                ref = WorkerTensorRef(
                    channel_id=self.channel_id,
                    transfer_id=transfer_id,
                    tensor_index=tensor_index,
                    shape=tuple(item.shape),
                    dtype=str(item.dtype),
                    source_device=str(item.device),
                    nbytes=item.numel() * item.element_size(),
                )
                tensor_index += 1
                sent[id(item)] = ref
                for queue in self._queues:
                    queue.put((ref, item))
                return ref
            if isinstance(item, dict):
                return {key: replace(child) for key, child in item.items()}
            if isinstance(item, tuple):
                return tuple(replace(child) for child in item)
            if isinstance(item, list):
                return [replace(child) for child in item]
            return item

        return replace(value)

    def receive(self, value: Any, *, rank: int, device: str | torch.device) -> Any:
        """Resolve tensor refs for one consumer rank onto its local device."""
        if not 0 <= rank < self.consumer_world_size:
            raise ValueError(f"Consumer rank {rank} is outside [0, {self.consumer_world_size})")
        queue = self._queues[rank]
        resolved: dict[WorkerTensorRef, torch.Tensor] = {}

        def receive_tensor(expected: WorkerTensorRef) -> torch.Tensor:
            while True:
                if not queue._reader.poll(self.timeout):
                    raise TimeoutError(
                        f"Tensor channel {self.channel_id} timed out receiving transfer {expected.transfer_id}"
                    )
                received_ref, tensor = queue.get()
                received_key = (received_ref.transfer_id, received_ref.tensor_index)
                expected_key = (expected.transfer_id, expected.tensor_index)
                if received_key < expected_key:
                    # The parent dropped this earlier artifact, normally after
                    # cancellation. Releasing it here keeps the FIFO usable.
                    del tensor
                    continue
                if received_ref != expected:
                    raise RuntimeError(
                        f"Tensor channel {self.channel_id} expected {expected}, received {received_ref}; "
                        "consumer calls must preserve producer order"
                    )
                return tensor

        def replace(item: Any) -> Any:
            if isinstance(item, WorkerTensorRef):
                if item.channel_id != self.channel_id:
                    return item
                cached = resolved.get(item)
                if cached is not None:
                    return cached
                tensor = receive_tensor(item)
                if (
                    tuple(tensor.shape) != item.shape
                    or str(tensor.dtype) != item.dtype
                    or str(tensor.device) != item.source_device
                    or tensor.numel() * tensor.element_size() != item.nbytes
                ):
                    raise RuntimeError(
                        f"Tensor channel {self.channel_id} received incompatible tensor metadata for {item}"
                    )
                target = torch.device(device)
                if tensor.device != target:
                    tensor = tensor.to(target, non_blocking=True)
                resolved[item] = tensor
                return tensor
            if isinstance(item, dict):
                return {key: replace(child) for key, child in item.items()}
            if isinstance(item, tuple):
                return tuple(replace(child) for child in item)
            if isinstance(item, list):
                return [replace(child) for child in item]
            return item

        return replace(value)

    def contains_ref(self, value: Any) -> bool:
        """Return whether a nested value contains a ref owned by this channel."""
        if isinstance(value, WorkerTensorRef):
            return value.channel_id == self.channel_id
        if isinstance(value, dict):
            return any(self.contains_ref(child) for child in value.values())
        if isinstance(value, tuple | list):
            return any(self.contains_ref(child) for child in value)
        return False

    def close(self) -> None:
        """Close parent-owned queue handles after both worker groups stop."""
        if self._closed:
            return
        self._closed = True
        for queue in self._queues:
            queue.close()
