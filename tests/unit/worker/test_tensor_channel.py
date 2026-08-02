from __future__ import annotations

import pytest
import torch
import torch.multiprocessing as mp

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.worker.parallel_worker import ParallelWorker
from telefuser.worker.tensor_channel import WorkerTensorChannel, WorkerTensorRef


class _TensorProducerStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "tensor-producer",
            ModelRuntimeConfig(device_type="cpu", parallel_config=ParallelConfig(device_ids=[0])),
        )

    def produce(self) -> tuple[torch.Tensor, dict[str, float]]:
        return torch.arange(4, dtype=torch.float32), {"seconds": 0.25}


class _TensorConsumerStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "tensor-consumer",
            ModelRuntimeConfig(device_type="cpu", parallel_config=ParallelConfig(device_ids=[0])),
        )

    def consume(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 1


def _spawn_send(channel, metadata_queue, release_event) -> None:
    metadata_queue.put(channel.send(torch.arange(4, dtype=torch.float32)))
    release_event.wait(timeout=10)


def _spawn_receive(channel, metadata_queue, result_queue, release_event) -> None:
    ref = metadata_queue.get()
    tensor = channel.receive(ref, rank=0, device="cpu")
    result_queue.put(tensor.tolist())
    release_event.set()


def _spawn_send_cuda(channel, metadata_queue, release_event) -> None:
    torch.cuda.set_device(0)
    tensor = torch.arange(4, dtype=torch.float32, device="cuda:0")
    metadata_queue.put(channel.send(tensor))
    release_event.wait(timeout=30)


def _spawn_receive_cuda(channel, metadata_queue, result_queue, release_event) -> None:
    torch.cuda.set_device(1)
    ref = metadata_queue.get()
    tensor = channel.receive(ref, rank=0, device="cuda:1")
    torch.cuda.synchronize(1)
    result_queue.put((str(tensor.device), tensor.cpu().tolist()))
    release_event.set()


def test_tensor_channel_keeps_parent_artifact_metadata_only_and_fans_out() -> None:
    channel = WorkerTensorChannel(consumer_world_size=2, timeout=1)
    channel.bind_producer()
    channel.bind_consumer(2)
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    try:
        artifact = channel.send({"latent": source, "profile": {"seconds": 0.1}})

        assert isinstance(artifact["latent"], WorkerTensorRef)
        assert artifact["latent"].shape == (2, 3)
        assert artifact["latent"].nbytes == source.numel() * source.element_size()
        assert artifact["profile"] == {"seconds": 0.1}
        for rank in range(2):
            resolved = channel.receive(artifact, rank=rank, device="cpu")
            torch.testing.assert_close(resolved["latent"], source)
            assert resolved["profile"] == {"seconds": 0.1}
    finally:
        channel.close()


def test_tensor_channel_preserves_nested_container_types() -> None:
    channel = WorkerTensorChannel(consumer_world_size=1, timeout=1)
    try:
        artifact = channel.send((torch.ones(1), [torch.zeros(1)]))
        resolved = channel.receive(artifact, rank=0, device="cpu")
    finally:
        channel.close()

    assert isinstance(artifact, tuple)
    assert isinstance(artifact[1], list)
    assert isinstance(resolved, tuple)
    assert isinstance(resolved[1], list)


def test_tensor_channel_sends_duplicate_tensor_leaf_once() -> None:
    channel = WorkerTensorChannel(consumer_world_size=1, timeout=1)
    source = torch.ones(2)
    try:
        artifact = channel.send((source, {"same": source}))
        assert artifact[0] == artifact[1]["same"]
        resolved = channel.receive(artifact, rank=0, device="cpu")
    finally:
        channel.close()

    assert resolved[0] is resolved[1]["same"]


def test_tensor_channel_discards_cancelled_earlier_transfer() -> None:
    channel = WorkerTensorChannel(consumer_world_size=1, timeout=1)
    try:
        channel.send(torch.zeros(1))
        current = channel.send(torch.ones(1))
        resolved = channel.receive(current, rank=0, device="cpu")
    finally:
        channel.close()

    torch.testing.assert_close(resolved, torch.ones(1))


def test_tensor_channel_validates_bindings_and_rank() -> None:
    channel = WorkerTensorChannel(consumer_world_size=2, timeout=1)
    try:
        channel.bind_producer()
        with pytest.raises(ValueError, match="already has a producer"):
            channel.bind_producer()
        with pytest.raises(ValueError, match="expects 2 consumer ranks"):
            channel.bind_consumer(1)
        channel.bind_consumer(2)
        with pytest.raises(ValueError, match="outside"):
            channel.receive(None, rank=2, device="cpu")
    finally:
        channel.close()


def test_tensor_channel_transfers_between_independent_spawned_processes() -> None:
    context = mp.get_context("spawn")
    channel = WorkerTensorChannel(consumer_world_size=1, timeout=10)
    metadata_queue = context.SimpleQueue()
    result_queue = context.SimpleQueue()
    release_event = context.Event()
    producer = context.Process(target=_spawn_send, args=(channel, metadata_queue, release_event))
    consumer = context.Process(target=_spawn_receive, args=(channel, metadata_queue, result_queue, release_event))
    try:
        producer.start()
        consumer.start()
        assert result_queue.get() == [0.0, 1.0, 2.0, 3.0]
        producer.join(timeout=10)
        consumer.join(timeout=10)
        assert producer.exitcode == 0
        assert consumer.exitcode == 0
    finally:
        release_event.set()
        for process in (producer, consumer):
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
        metadata_queue.close()
        result_queue.close()
        channel.close()


def test_parallel_workers_exchange_tensor_without_parent_materialization() -> None:
    channel = WorkerTensorChannel(consumer_world_size=1, timeout=10)
    producer = ParallelWorker(
        _TensorProducerStage(),
        tensor_output_channel=channel,
        tensor_output_methods=("produce",),
    )
    consumer = ParallelWorker(_TensorConsumerStage(), tensor_input_channels=(channel,))
    try:
        ref, profile = producer.produce(sync=True)
        assert isinstance(ref, WorkerTensorRef)
        assert profile == {"seconds": 0.25}
        result = consumer.consume(ref, sync=True)
        torch.testing.assert_close(result, torch.arange(4, dtype=torch.float32) + 1)
    finally:
        producer.close()
        consumer.close()
        channel.close()


@pytest.mark.distributed
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_tensor_channel_uses_cuda_ipc_and_cross_gpu_peer_copy() -> None:
    context = mp.get_context("spawn")
    channel = WorkerTensorChannel(consumer_world_size=1, timeout=30)
    metadata_queue = context.SimpleQueue()
    result_queue = context.SimpleQueue()
    release_event = context.Event()
    producer = context.Process(target=_spawn_send_cuda, args=(channel, metadata_queue, release_event))
    consumer = context.Process(target=_spawn_receive_cuda, args=(channel, metadata_queue, result_queue, release_event))
    try:
        producer.start()
        consumer.start()
        device, values = result_queue.get()
        assert device == "cuda:1"
        assert values == [0.0, 1.0, 2.0, 3.0]
        producer.join(timeout=30)
        consumer.join(timeout=30)
        assert producer.exitcode == 0
        assert consumer.exitcode == 0
    finally:
        release_event.set()
        for process in (producer, consumer):
            if process.is_alive():
                process.terminate()
            process.join(timeout=2)
        metadata_queue.close()
        result_queue.close()
        channel.close()
