"""Direct tensor transport between independently spawned worker groups."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from multiprocessing.queues import SimpleQueue
from typing import Any

import torch
import torch.multiprocessing as mp

_MAX_CUDA_IPC_POOL_PROFILES = 8


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
    shard_dim: int | None = None
    _cuda_payloads: tuple[_CudaIpcPayload, ...] | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )


@dataclass(frozen=True)
class _CudaIpcPayload:
    """Plain metadata for one view inside a producer-owned persistent CUDA pool."""

    ref: WorkerTensorRef
    profile_index: int
    slot_id: int
    handle: tuple[Any, ...]
    ready_event_handle: bytes
    byte_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class _CudaIpcSlot:
    tensor: torch.Tensor
    pending_ranks: set[int]
    completion_ranks: set[int]
    ready_event: torch.cuda.Event
    ready_event_handle: bytes
    was_used: bool = False
    transfer_key: tuple[int, int] | None = None


@dataclass
class _CudaIpcPool:
    profile_index: int
    storage: torch.Tensor
    handle: tuple[Any, ...]
    slots: list[_CudaIpcSlot]
    next_slot_id: int = 0


class WorkerTensorChannel:
    """Point-to-point tensor path that bypasses the parent process.

    CUDA tensors use persistent IPC pools whose rank-local metadata travels in
    :class:`WorkerTensorRef`. CPU tensors and dynamic-profile CUDA fallbacks use
    one multiprocessing queue per consumer rank. The parent never materializes
    tensor contents.
    """

    def __init__(
        self,
        consumer_world_size: int,
        *,
        timeout: int = 600,
        shard_dim: int | None = None,
        cuda_ipc_slots: int = 2,
    ) -> None:
        if consumer_world_size < 1:
            raise ValueError("consumer_world_size must be at least one")
        if timeout < 1:
            raise ValueError("timeout must be at least one second")
        if cuda_ipc_slots < 1:
            raise ValueError("cuda_ipc_slots must be at least one")
        spawn_ctx = mp.get_context("spawn")
        self.channel_id = uuid.uuid4().hex
        self.consumer_world_size = consumer_world_size
        self.timeout = timeout
        self.shard_dim = shard_dim
        self.cuda_ipc_slots = cuda_ipc_slots
        self._queues: tuple[SimpleQueue, ...] = tuple(spawn_ctx.SimpleQueue() for _ in range(consumer_world_size))
        self._ack_generations = spawn_ctx.Array(
            "q",
            _MAX_CUDA_IPC_POOL_PROFILES * cuda_ipc_slots * consumer_world_size,
            lock=False,
        )
        self._completion_queues: tuple[SimpleQueue, ...] = tuple(
            spawn_ctx.SimpleQueue() for _ in range(consumer_world_size)
        )
        self._cuda_pools: dict[tuple[int, tuple[int, ...], torch.dtype, str], _CudaIpcPool] = {}
        self._cuda_storage_cache: dict[tuple[bytes, int], torch.UntypedStorage] = {}
        self._cuda_event_cache: dict[tuple[bytes, int], torch.cuda.Event] = {}
        self._cuda_consumer_completion_events: dict[tuple[int, int], torch.cuda.Event] = {}
        self._cuda_completion_handles: dict[tuple[int, int, int], bytes] = {}
        self._cuda_producer_completion_events: dict[tuple[int, int, int, int], torch.cuda.Event] = {}
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
                shard_dim = self._normalize_shard_dim(item)
                ref = WorkerTensorRef(
                    channel_id=self.channel_id,
                    transfer_id=transfer_id,
                    tensor_index=tensor_index,
                    shape=tuple(item.shape),
                    dtype=str(item.dtype),
                    source_device=str(item.device),
                    nbytes=item.numel() * item.element_size(),
                    shard_dim=shard_dim,
                )
                tensor_index += 1
                if item.device.type == "cuda":
                    rank_payloads = self._stage_cuda_tensor(ref, item)
                    if rank_payloads is not None:
                        ref = dataclass_replace(ref, _cuda_payloads=rank_payloads)
                    else:
                        rank_tensors = (
                            (item,) * self.consumer_world_size
                            if shard_dim is None
                            else torch.tensor_split(item, self.consumer_world_size, dim=shard_dim)
                        )
                        for queue, rank_tensor in zip(self._queues, rank_tensors):
                            queue.put((ref, rank_tensor))
                else:
                    rank_tensors = (
                        (item,) * self.consumer_world_size
                        if shard_dim is None
                        else torch.tensor_split(item, self.consumer_world_size, dim=shard_dim)
                    )
                    for queue, rank_tensor in zip(self._queues, rank_tensors):
                        queue.put((ref, rank_tensor))
                sent[id(item)] = ref
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
        resolved: dict[WorkerTensorRef, torch.Tensor] = {}

        def replace(item: Any) -> Any:
            if isinstance(item, WorkerTensorRef):
                if item.channel_id != self.channel_id:
                    return item
                cached = resolved.get(item)
                if cached is not None:
                    return cached
                received = self._receive_payload(item, rank=rank)
                target = torch.device(device)
                if isinstance(received, _CudaIpcPayload):
                    tensor = self._copy_cuda_payload(received, rank=rank, target=target)
                else:
                    tensor = received
                expected_shape = self._rank_shape(item.shape, item.shard_dim, rank)
                expected_nbytes = item.nbytes
                if item.shard_dim is not None:
                    expected_nbytes = tensor.element_size()
                    for size in expected_shape:
                        expected_nbytes *= size
                if (
                    tuple(tensor.shape) != expected_shape
                    or str(tensor.dtype) != item.dtype
                    or tensor.numel() * tensor.element_size() != expected_nbytes
                ):
                    raise RuntimeError(
                        f"Tensor channel {self.channel_id} received incompatible tensor metadata for {item}"
                    )
                if not isinstance(received, _CudaIpcPayload) and str(tensor.device) != item.source_device:
                    raise RuntimeError(
                        f"Tensor channel {self.channel_id} received incompatible tensor metadata for {item}"
                    )
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

    def _normalize_shard_dim(self, tensor: torch.Tensor) -> int | None:
        if self.shard_dim is None:
            return None
        if tensor.ndim == 0:
            raise ValueError("Tensor channel cannot shard a scalar tensor")
        shard_dim = self.shard_dim if self.shard_dim >= 0 else tensor.ndim + self.shard_dim
        if not 0 <= shard_dim < tensor.ndim:
            raise ValueError(f"Tensor channel shard_dim={self.shard_dim} is invalid for shape {tuple(tensor.shape)}")
        if tensor.shape[shard_dim] < self.consumer_world_size:
            raise ValueError(
                f"Tensor channel cannot shard shape {tuple(tensor.shape)} along dimension {shard_dim} "
                f"across {self.consumer_world_size} consumers"
            )
        return shard_dim

    def _rank_shape(self, shape: tuple[int, ...], shard_dim: int | None, rank: int) -> tuple[int, ...]:
        if shard_dim is None:
            return shape
        base_size, remainder = divmod(shape[shard_dim], self.consumer_world_size)
        rank_shape = list(shape)
        rank_shape[shard_dim] = base_size + int(rank < remainder)
        return tuple(rank_shape)

    def _create_cuda_pool(self, ref: WorkerTensorRef, tensor: torch.Tensor) -> _CudaIpcPool:
        with torch.cuda.device(tensor.device):
            storage = torch.empty(
                (self.cuda_ipc_slots, *tensor.shape),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            slots = []
            for index in range(self.cuda_ipc_slots):
                ready_event = torch.cuda.Event(interprocess=True)
                slots.append(
                    _CudaIpcSlot(
                        tensor=storage[index],
                        pending_ranks=set(),
                        completion_ranks=set(),
                        ready_event=ready_event,
                        ready_event_handle=ready_event.ipc_handle(),
                    )
                )
        pool = _CudaIpcPool(
            profile_index=len(self._cuda_pools),
            storage=storage,
            handle=storage.untyped_storage()._share_cuda_(),
            slots=slots,
        )
        key = (ref.tensor_index, ref.shape, tensor.dtype, str(tensor.device))
        self._cuda_pools[key] = pool
        return pool

    def _ack_index(self, profile_index: int, slot_id: int, rank: int) -> int:
        return (profile_index * self.cuda_ipc_slots + slot_id) * self.consumer_world_size + rank

    def _drain_acks(self) -> None:
        for pool in self._cuda_pools.values():
            for slot_id, slot in enumerate(pool.slots):
                if slot.transfer_key is None:
                    continue
                generation = slot.transfer_key[0] + 1
                acknowledgements = {
                    rank: self._ack_generations[self._ack_index(pool.profile_index, slot_id, rank)]
                    for rank in slot.pending_ranks
                }
                slot.completion_ranks.update(
                    rank for rank, acknowledged in acknowledgements.items() if acknowledged == generation
                )
                slot.pending_ranks = {
                    rank for rank, acknowledged in acknowledgements.items() if abs(acknowledged) != generation
                }
                if not slot.pending_ranks:
                    slot.transfer_key = None

    def _acquire_cuda_slot(self, pool: _CudaIpcPool) -> tuple[int, _CudaIpcSlot]:
        deadline = None
        while True:
            self._drain_acks()
            for offset in range(len(pool.slots)):
                slot_id = (pool.next_slot_id + offset) % len(pool.slots)
                slot = pool.slots[slot_id]
                if not slot.pending_ranks:
                    pool.next_slot_id = (slot_id + 1) % len(pool.slots)
                    return slot_id, slot
            if deadline is None:
                deadline = time.monotonic() + self.timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Tensor channel {self.channel_id} timed out waiting for a reusable CUDA IPC slot")
            time.sleep(min(0.00005, remaining))

    def _stage_cuda_tensor(self, ref: WorkerTensorRef, tensor: torch.Tensor) -> tuple[_CudaIpcPayload, ...] | None:
        key = (ref.tensor_index, ref.shape, tensor.dtype, str(tensor.device))
        pool = self._cuda_pools.get(key)
        if pool is None:
            if len(self._cuda_pools) >= _MAX_CUDA_IPC_POOL_PROFILES:
                return None
            try:
                pool = self._create_cuda_pool(ref, tensor)
            except torch.OutOfMemoryError:
                return None
        slot_id, slot = self._acquire_cuda_slot(pool)
        with torch.cuda.device(tensor.device):
            producer_stream = torch.cuda.current_stream(tensor.device)
            self._wait_for_cuda_slot_reuse(
                pool,
                slot_id,
                slot,
                device=tensor.device,
                producer_stream=producer_stream,
            )
            slot.tensor.copy_(tensor, non_blocking=True)
            slot.ready_event.record(producer_stream)
        rank_views = (
            (slot.tensor,) * self.consumer_world_size
            if ref.shard_dim is None
            else torch.tensor_split(slot.tensor, self.consumer_world_size, dim=ref.shard_dim)
        )
        slot.pending_ranks = set(range(self.consumer_world_size))
        slot.completion_ranks.clear()
        slot.transfer_key = (ref.transfer_id, ref.tensor_index)
        slot.was_used = True
        return tuple(
            _CudaIpcPayload(
                ref=ref,
                profile_index=pool.profile_index,
                slot_id=slot_id,
                handle=pool.handle,
                ready_event_handle=slot.ready_event_handle,
                byte_offset=view.storage_offset() * view.element_size(),
                shape=tuple(view.shape),
                stride=tuple(view.stride()),
                dtype=view.dtype,
            )
            for view in rank_views
        )

    def _wait_for_cuda_slot_reuse(
        self,
        pool: _CudaIpcPool,
        slot_id: int,
        slot: _CudaIpcSlot,
        *,
        device: torch.device,
        producer_stream: torch.cuda.Stream,
    ) -> None:
        if not slot.was_used:
            return
        if device.index is None:
            raise RuntimeError("CUDA IPC pool device must have an explicit index")
        deadline = time.monotonic() + self.timeout
        for rank in sorted(slot.completion_ranks):
            completion_queue = self._completion_queues[rank]
            handle_key = (pool.profile_index, slot_id, rank)
            while handle_key not in self._cuda_completion_handles:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not completion_queue._reader.poll(remaining):
                    raise TimeoutError(
                        f"Tensor channel {self.channel_id} timed out waiting for CUDA completion metadata"
                    )
                profile_index, completed_slot_id, event_handle = completion_queue.get()
                self._cuda_completion_handles[(profile_index, completed_slot_id, rank)] = event_handle
            event_key = (*handle_key, device.index)
            completion_event = self._cuda_producer_completion_events.get(event_key)
            if completion_event is None:
                completion_event = torch.cuda.Event.from_ipc_handle(
                    device,
                    self._cuda_completion_handles[handle_key],
                )
                self._cuda_producer_completion_events[event_key] = completion_event
            if not completion_event.query():
                producer_stream.wait_event(completion_event)

    def _publish_cuda_completion(
        self,
        payload: _CudaIpcPayload,
        *,
        rank: int,
        device: torch.device,
        copy_stream: torch.cuda.Stream,
    ) -> None:
        event_key = (payload.profile_index, payload.slot_id)
        completion_event = self._cuda_consumer_completion_events.get(event_key)
        is_new_event = completion_event is None
        if completion_event is None:
            completion_event = torch.cuda.Event(interprocess=True)
            self._cuda_consumer_completion_events[event_key] = completion_event
        completion_event.record(copy_stream)
        if is_new_event:
            self._completion_queues[rank].put((payload.profile_index, payload.slot_id, completion_event.ipc_handle()))
        self._ack_cuda_payload(payload, rank=rank)

    def _copy_cuda_payload(
        self,
        payload: _CudaIpcPayload,
        *,
        rank: int,
        target: torch.device,
    ) -> torch.Tensor:
        source = torch.device(payload.ref.source_device)
        rebuild_device = target if target.type == "cuda" else source
        if rebuild_device.index is None:
            rebuild_device = torch.device(rebuild_device.type, torch.cuda.current_device())
        handle_key = (payload.handle[1], rebuild_device.index)
        storage = self._cuda_storage_cache.get(handle_key)
        if storage is None:
            redirected_handle = (rebuild_device.index, *payload.handle[1:])
            with torch.cuda.device(rebuild_device):
                storage = torch.UntypedStorage._new_shared_cuda(*redirected_handle)
            self._cuda_storage_cache[handle_key] = storage
        with torch.cuda.device(rebuild_device):
            copy_stream = torch.cuda.current_stream(rebuild_device)
            event_key = (payload.ready_event_handle, rebuild_device.index)
            ready_event = self._cuda_event_cache.get(event_key)
            if ready_event is None:
                ready_event = torch.cuda.Event.from_ipc_handle(rebuild_device, payload.ready_event_handle)
                self._cuda_event_cache[event_key] = ready_event
            if not ready_event.query():
                copy_stream.wait_event(ready_event)
            mapped = torch.empty(0, dtype=payload.dtype, device=rebuild_device).set_(
                storage,
                storage_offset=payload.byte_offset // torch.empty((), dtype=payload.dtype).element_size(),
                size=payload.shape,
                stride=payload.stride,
            )
            if target.type == "cuda":
                output = torch.empty(payload.shape, dtype=payload.dtype, device=target)
                output.copy_(mapped, non_blocking=True)
            else:
                output = mapped.to(target)
            self._publish_cuda_completion(payload, rank=rank, device=rebuild_device, copy_stream=copy_stream)
        return output

    def _ack_cuda_payload(self, payload: _CudaIpcPayload, *, rank: int, copied: bool = True) -> None:
        generation = payload.ref.transfer_id + 1
        self._ack_generations[self._ack_index(payload.profile_index, payload.slot_id, rank)] = (
            generation if copied else -generation
        )

    def release_local_cuda_ipc(self) -> None:
        """Release process-local CUDA IPC mappings after pending work is synchronized."""
        self._cuda_consumer_completion_events.clear()
        self._cuda_completion_handles.clear()
        self._cuda_producer_completion_events.clear()
        self._cuda_event_cache.clear()
        self._cuda_storage_cache.clear()

    def discard(self, value: Any, *, rank: int) -> int:
        """Consume and release referenced tensors without materializing a device copy."""
        if not 0 <= rank < self.consumer_world_size:
            raise ValueError(f"Consumer rank {rank} is outside [0, {self.consumer_world_size})")
        discarded: set[WorkerTensorRef] = set()

        def visit(item: Any) -> None:
            if isinstance(item, WorkerTensorRef):
                if item.channel_id != self.channel_id or item in discarded:
                    return
                received = self._receive_payload(item, rank=rank)
                if isinstance(received, _CudaIpcPayload):
                    self._ack_cuda_payload(received, rank=rank, copied=False)
                else:
                    del received
                discarded.add(item)
                return
            if isinstance(item, dict):
                for child in item.values():
                    visit(child)
            elif isinstance(item, tuple | list):
                for child in item:
                    visit(child)

        visit(value)
        return len(discarded)

    def _receive_payload(self, expected: WorkerTensorRef, *, rank: int) -> torch.Tensor | _CudaIpcPayload:
        if expected._cuda_payloads is not None:
            if len(expected._cuda_payloads) != self.consumer_world_size:
                raise RuntimeError(f"Tensor channel {self.channel_id} received incomplete CUDA IPC payload metadata")
            payload = expected._cuda_payloads[rank]
            if payload.ref != expected:
                raise RuntimeError(f"Tensor channel {self.channel_id} received mismatched CUDA IPC payload metadata")
            return payload
        queue = self._queues[rank]
        while True:
            if not queue._reader.poll(self.timeout):
                raise TimeoutError(
                    f"Tensor channel {self.channel_id} timed out receiving transfer {expected.transfer_id}"
                )
            received_ref, tensor = queue.get()
            received_key = (received_ref.transfer_id, received_ref.tensor_index)
            expected_key = (expected.transfer_id, expected.tensor_index)
            if received_key < expected_key:
                # The parent dropped this earlier artifact. Releasing it keeps
                # the FIFO usable when cancellation cleanup was delayed.
                del tensor
                continue
            if received_ref != expected:
                raise RuntimeError(
                    f"Tensor channel {self.channel_id} expected {expected}, received {received_ref}; "
                    "consumer calls must preserve producer order"
                )
            return tensor

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
        self.release_local_cuda_ipc()
        for tensor_queue in self._queues:
            tensor_queue.close()
        for completion_queue in self._completion_queues:
            completion_queue.close()
