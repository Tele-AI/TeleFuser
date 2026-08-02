from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
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
        ref = producer.reduce(torch.arange(4, dtype=torch.float32), sync=True)
        assert isinstance(ref, WorkerTensorRef)
        assert ref.source_device == "cuda:0"
        device, result = consumer.consume(ref, sync=True)
        assert device == "cuda:2"
        torch.testing.assert_close(result, 2 * torch.arange(4, dtype=torch.float32) + 1)
    finally:
        producer.close()
        consumer.close()
        channel.close()
