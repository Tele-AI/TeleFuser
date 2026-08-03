from __future__ import annotations

import threading

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.distributed.vae_spatial import _gather_height
from telefuser.worker import ParallelWorker, WorkerTensorChannel, WorkerTensorRef


class _DistributedProducerStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "distributed-producer",
            ModelRuntimeConfig(
                device_type="cuda",
                device_id=0,
                parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2),
            ),
        )

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        result = tensor + dist.get_rank()
        dist.all_reduce(result)
        return result


class _GPUConsumerStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "gpu-consumer",
            ModelRuntimeConfig(
                device_type="cuda",
                device_id=2,
                parallel_config=ParallelConfig(device_ids=[2]),
            ),
        )

    def consume(self, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
        return str(tensor.device), tensor.cpu()


class _DistributedGPUConsumerStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "distributed-gpu-consumer",
            ModelRuntimeConfig(
                device_type="cuda",
                device_id=2,
                parallel_config=ParallelConfig(device_ids=[2, 3], sp_ulysses_degree=2),
            ),
        )

    def consume(self, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
        return str(tensor.device), tensor.cpu()

    def gather_height_shards(self, tensor: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor]:
        local_shape = tuple(tensor.shape)
        return local_shape, _gather_height(tensor).cpu()


class _BlockingRankDiscardChannel(WorkerTensorChannel):
    def __init__(self) -> None:
        super().__init__(consumer_world_size=2, timeout=30)
        context = mp.get_context("spawn")
        self.rank_one_started = context.Event()
        self.rank_one_release = context.Event()

    def discard(self, value: object, *, rank: int) -> int:
        if rank == 1:
            self.rank_one_started.set()
            if not self.rank_one_release.wait(timeout=30):
                raise TimeoutError("Timed out waiting to release consumer rank 1")
        return super().discard(value, rank=rank)


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 3, reason="requires three CUDA devices")
def test_distributed_worker_to_worker_tensor_path_bypasses_parent() -> None:
    channel = WorkerTensorChannel(consumer_world_size=1, timeout=30)
    producer = ParallelWorker(
        _DistributedProducerStage(),
        tensor_output_channel=channel,
        tensor_output_methods=("reduce",),
    )
    consumer = ParallelWorker(_GPUConsumerStage(), tensor_input_channels=(channel,))
    try:
        abandoned = producer.reduce(torch.arange(4, dtype=torch.float32), sync=True)
        assert isinstance(abandoned, WorkerTensorRef)
        assert consumer.discard_tensor_refs(abandoned, sync=True) == 1

        ref = producer.reduce(torch.arange(4, dtype=torch.float32), sync=True)
        assert isinstance(ref, WorkerTensorRef)
        assert ref.source_device == "cuda:0"
        device, result = consumer.consume(ref, sync=True)
        assert device == "cuda:2"
        torch.testing.assert_close(result, 2 * torch.arange(4, dtype=torch.float32) + 1)
    finally:
        consumer.close()
        producer.close()
        channel.close()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_discard_waits_for_every_consumer_rank() -> None:
    channel = _BlockingRankDiscardChannel()
    producer = ParallelWorker(
        _DistributedProducerStage(),
        tensor_output_channel=channel,
        tensor_output_methods=("reduce",),
    )
    consumer = ParallelWorker(_DistributedGPUConsumerStage(), tensor_input_channels=(channel,))
    discarded: list[int] = []

    def discard() -> None:
        discarded.append(consumer.discard_tensor_refs(abandoned, sync=True))

    try:
        abandoned = producer.reduce(torch.arange(4, dtype=torch.float32), sync=True)
        thread = threading.Thread(target=discard)
        thread.start()
        assert channel.rank_one_started.wait(timeout=10)
        thread.join(timeout=0.1)
        assert thread.is_alive()

        channel.rank_one_release.set()
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert discarded == [1]

        current = producer.reduce(torch.arange(4, dtype=torch.float32), sync=True)
        device, result = consumer.consume(current, sync=True)
        assert device == "cuda:2"
        torch.testing.assert_close(result, 2 * torch.arange(4, dtype=torch.float32) + 1)
    finally:
        channel.rank_one_release.set()
        consumer.close()
        producer.close()
        channel.close()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_sharded_channel_copies_only_each_consumer_rank_height_slice() -> None:
    channel = WorkerTensorChannel(consumer_world_size=2, timeout=30, shard_dim=-2)
    producer = ParallelWorker(
        _DistributedProducerStage(),
        tensor_output_channel=channel,
        tensor_output_methods=("reduce",),
    )
    consumer = ParallelWorker(_DistributedGPUConsumerStage(), tensor_input_channels=(channel,))
    source = torch.arange(30, dtype=torch.float32).reshape(1, 1, 1, 5, 6)
    try:
        ref = producer.reduce(source, sync=True)
        assert isinstance(ref, WorkerTensorRef)
        assert ref.shard_dim == 3
        local_shape, gathered = consumer.gather_height_shards(ref, sync=True)
    finally:
        consumer.close()
        producer.close()
        channel.close()

    assert local_shape == (1, 1, 1, 3, 6)
    torch.testing.assert_close(gathered, 2 * source + 1)
