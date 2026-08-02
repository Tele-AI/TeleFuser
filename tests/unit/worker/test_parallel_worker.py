"""Tests for parallel_worker module."""

from queue import Empty
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from telefuser.worker.parallel_worker import to_device


class TestToDevice:
    """Test to_device function."""

    def test_tensor_to_cpu(self):
        """Test moving tensor to CPU."""
        tensor = torch.randn(2, 3)
        result = to_device(tensor, "cpu")

        assert result.device.type == "cpu"
        assert torch.allclose(result, tensor)
        assert result.is_shared()

    def test_tensor_already_on_cpu(self):
        """Test tensor already on CPU gets shared memory."""
        tensor = torch.randn(2, 3)  # Already on CPU
        result = to_device(tensor, "cpu")

        assert result.device.type == "cpu"
        assert result.is_shared()

    def test_dict_to_device(self):
        """Test moving dict of tensors to device."""
        data = {
            "a": torch.randn(2, 3),
            "b": torch.randn(3, 4),
        }
        result = to_device(data, "cpu")

        assert isinstance(result, dict)
        assert result["a"].device.type == "cpu"
        assert result["b"].device.type == "cpu"

    def test_list_to_device(self):
        """Test moving list of tensors to device."""
        data = [torch.randn(2, 3), torch.randn(3, 4)]
        result = to_device(data, "cpu")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].device.type == "cpu"
        assert result[1].device.type == "cpu"

    def test_tuple_to_device(self):
        """Test moving tuple of tensors to device."""
        data = (torch.randn(2, 3), torch.randn(3, 4))
        result = to_device(data, "cpu")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].device.type == "cpu"

    def test_nested_structure(self):
        """Test moving nested dict/list structure."""
        data = {
            "tensors": [torch.randn(2, 3), torch.randn(3, 4)],
            "single": torch.randn(5, 5),
        }
        result = to_device(data, "cpu")

        assert result["single"].device.type == "cpu"
        assert result["tensors"][0].device.type == "cpu"

    def test_non_tensor_data(self):
        """Test non-tensor data passes through unchanged."""
        data = {
            "string": "hello",
            "number": 42,
            "tensor": torch.randn(2, 3),
        }
        result = to_device(data, "cpu")

        assert result["string"] == "hello"
        assert result["number"] == 42
        assert result["tensor"].device.type == "cpu"


@pytest.mark.distributed
class TestParallelWorkerUnit:
    """Unit tests for ParallelWorker class (without actual multiprocessing)."""

    @patch("telefuser.worker.parallel_worker.mp")
    @patch("telefuser.worker.parallel_worker.PortAllocator")
    def test_initialization(self, mock_port_allocator, mock_mp):
        """Test ParallelWorker initialization."""
        from telefuser.worker.parallel_worker import ParallelWorker

        # Setup mocks
        mock_port_allocator.return_value.get_free_port_in_interval.return_value = 12345
        mock_mp.get_start_method.return_value = "spawn"
        mock_spawn_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_spawn_ctx
        mock_spawn_ctx.Queue.return_value = MagicMock()

        mock_stage = MagicMock()
        mock_stage.name = "TestStage"
        mock_stage.model_runtime_config.parallel_config.world_size = 2
        mock_stage.model_runtime_config.parallel_config.device_ids = [0, 1]
        mock_stage.model_runtime_config.parallel_config.queue_with_cpu = False
        mock_stage.model_runtime_config.parallel_config.timeout = 600

        worker = ParallelWorker(mock_stage)

        assert worker.world_size == 2
        assert worker.device_ids == [0, 1]
        assert worker.name == "Parallel Worker TestStage"
        mock_mp.spawn.assert_called_once()

    @patch("telefuser.worker.parallel_worker.mp")
    @patch("telefuser.worker.parallel_worker.PortAllocator")
    def test_put_data_with_queue_cpu(self, mock_port_allocator, mock_mp):
        """Test put_data with queue_with_cpu enabled."""
        from telefuser.worker.parallel_worker import ParallelWorker

        mock_port_allocator.return_value.get_free_port_in_interval.return_value = 12345
        mock_mp.get_start_method.return_value = "spawn"
        mock_spawn_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_spawn_ctx
        mock_queue = MagicMock()
        mock_spawn_ctx.SimpleQueue.return_value = mock_queue

        mock_stage = MagicMock()
        mock_stage.name = "TestStage"
        mock_stage.model_runtime_config.parallel_config.world_size = 1
        mock_stage.model_runtime_config.parallel_config.device_ids = [0]
        mock_stage.model_runtime_config.parallel_config.queue_with_cpu = True
        mock_stage.model_runtime_config.parallel_config.timeout = 600

        worker = ParallelWorker(mock_stage)
        test_data = ["test", torch.randn(2, 3)]
        worker.put_data(test_data)

        mock_queue.put.assert_called_once()

    @patch("telefuser.worker.parallel_worker.mp")
    @patch("telefuser.worker.parallel_worker.PortAllocator")
    def test_call_with_sync(self, mock_port_allocator, mock_mp):
        """Test __call__ with sync=True."""
        from telefuser.worker.parallel_worker import ParallelWorker

        mock_port_allocator.return_value.get_free_port_in_interval.return_value = 12345
        mock_mp.get_start_method.return_value = "spawn"
        mock_spawn_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_spawn_ctx
        mock_queue_out = MagicMock()
        mock_queue_out.get.return_value = torch.randn(2, 3)
        mock_spawn_ctx.SimpleQueue.return_value = mock_queue_out
        mock_queue_out._reader.poll.return_value = True

        mock_stage = MagicMock()
        mock_stage.name = "TestStage"
        mock_stage.model_runtime_config.parallel_config.world_size = 1
        mock_stage.model_runtime_config.parallel_config.device_ids = [0]
        mock_stage.model_runtime_config.parallel_config.queue_with_cpu = False
        mock_stage.model_runtime_config.parallel_config.timeout = 600

        worker = ParallelWorker(mock_stage)
        result = worker(torch.randn(2, 3), sync=True)

        assert isinstance(result, torch.Tensor)
        mock_queue_out.get.assert_called_once()

    @patch("telefuser.worker.parallel_worker.mp")
    @patch("telefuser.worker.parallel_worker.PortAllocator")
    def test_call_without_sync(self, mock_port_allocator, mock_mp):
        """Test __call__ with sync=False returns wait function."""
        from telefuser.worker.parallel_worker import ParallelWorker

        mock_port_allocator.return_value.get_free_port_in_interval.return_value = 12345
        mock_mp.get_start_method.return_value = "spawn"
        mock_spawn_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_spawn_ctx
        mock_queue_out = MagicMock()
        mock_queue_out.get.return_value = torch.randn(2, 3)
        mock_spawn_ctx.SimpleQueue.return_value = mock_queue_out
        mock_queue_out._reader.poll.return_value = True

        mock_stage = MagicMock()
        mock_stage.name = "TestStage"
        mock_stage.model_runtime_config.parallel_config.world_size = 1
        mock_stage.model_runtime_config.parallel_config.device_ids = [0]
        mock_stage.model_runtime_config.parallel_config.queue_with_cpu = False
        mock_stage.model_runtime_config.parallel_config.timeout = 600

        worker = ParallelWorker(mock_stage)
        wait_fn = worker(torch.randn(2, 3), sync=False)

        assert callable(wait_fn)
        result = wait_fn()
        assert isinstance(result, torch.Tensor)

    @patch("telefuser.worker.parallel_worker.mp")
    @patch("telefuser.worker.parallel_worker.PortAllocator")
    def test_getattr_arbitrary_method(self, mock_port_allocator, mock_mp):
        """Test __getattr__ creates wrapper for arbitrary methods."""
        from telefuser.worker.parallel_worker import ParallelWorker

        mock_port_allocator.return_value.get_free_port_in_interval.return_value = 12345
        mock_mp.get_start_method.return_value = "spawn"
        mock_spawn_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_spawn_ctx
        mock_queue_out = MagicMock()
        mock_queue_out.get.return_value = "result"
        mock_spawn_ctx.SimpleQueue.return_value = mock_queue_out
        mock_queue_out._reader.poll.return_value = True

        mock_stage = MagicMock()
        mock_stage.name = "TestStage"
        mock_stage.model_runtime_config.parallel_config.world_size = 1
        mock_stage.model_runtime_config.parallel_config.device_ids = [0]
        mock_stage.model_runtime_config.parallel_config.queue_with_cpu = False
        mock_stage.model_runtime_config.parallel_config.timeout = 600

        worker = ParallelWorker(mock_stage)
        result = worker.custom_method("arg1", key="value", sync=True)

        assert result == "result"

    @patch("telefuser.worker.parallel_worker.mp")
    @patch("telefuser.worker.parallel_worker.PortAllocator")
    def test_queue_out_exception(self, mock_port_allocator, mock_mp):
        """Test that exceptions from queue are raised."""
        from telefuser.worker.parallel_worker import ParallelWorker

        mock_port_allocator.return_value.get_free_port_in_interval.return_value = 12345
        mock_mp.get_start_method.return_value = "spawn"
        mock_spawn_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_spawn_ctx
        mock_queue_out = MagicMock()
        mock_queue_out.get.return_value = RuntimeError("Worker error")
        mock_spawn_ctx.SimpleQueue.return_value = mock_queue_out
        mock_queue_out._reader.poll.return_value = True

        mock_stage = MagicMock()
        mock_stage.name = "TestStage"
        mock_stage.model_runtime_config.parallel_config.world_size = 1
        mock_stage.model_runtime_config.parallel_config.device_ids = [0]
        mock_stage.model_runtime_config.parallel_config.queue_with_cpu = False
        mock_stage.model_runtime_config.parallel_config.timeout = 600

        worker = ParallelWorker(mock_stage)

        with pytest.raises(RuntimeError, match="Worker error"):
            worker(sync=True)

    @patch("telefuser.worker.parallel_worker.mp")
    @patch("telefuser.worker.parallel_worker.PortAllocator")
    def test_queue_timeout(self, mock_port_allocator, mock_mp):
        """Test that queue timeout raises RuntimeError."""
        from telefuser.worker.parallel_worker import ParallelWorker

        mock_port_allocator.return_value.get_free_port_in_interval.return_value = 12345
        mock_mp.get_start_method.return_value = "spawn"
        mock_spawn_ctx = MagicMock()
        mock_mp.get_context.return_value = mock_spawn_ctx
        mock_queue_out = MagicMock()
        mock_queue_out._reader.poll.return_value = False
        mock_spawn_ctx.SimpleQueue.return_value = mock_queue_out

        mock_stage = MagicMock()
        mock_stage.name = "TestStage"
        mock_stage.model_runtime_config.parallel_config.world_size = 1
        mock_stage.model_runtime_config.parallel_config.device_ids = [0]
        mock_stage.model_runtime_config.queue_with_cpu = False
        mock_stage.model_runtime_config.parallel_config.timeout = 1

        worker = ParallelWorker(mock_stage)

        with pytest.raises(RuntimeError, match="timeout"):
            worker(sync=True)


@pytest.mark.distributed
class TestWorkerLoopUnit:
    """Unit tests for _worker_loop function."""

    @patch("telefuser.worker.parallel_worker.dist")
    @patch("telefuser.worker.parallel_worker.current_platform")
    def test_worker_loop_single_process(self, mock_platform, mock_dist):
        """Test _worker_loop with world_size=1 (no distributed)."""
        import multiprocessing as mp

        from telefuser.worker.parallel_worker import _worker_loop

        mock_platform.device_type = "cpu"

        # Create mock queues
        queue_in = [MagicMock()]
        queue_out = MagicMock()

        # Simulate one task then exit
        queue_in[0].get.side_effect = [
            ("test_method", [torch.randn(2, 3)], {"key": "value"}),
            ("exit", None, None),
        ]

        mock_stage = MagicMock()
        mock_stage.device = "cpu"
        mock_stage.model_runtime_config.parallel_config.worker_intra_op_threads = 1
        mock_stage.test_method.return_value = torch.randn(2, 3)

        _worker_loop(
            rank=0,
            world_size=1,
            queue_in=queue_in,
            queue_out=queue_out,
            stage=mock_stage,
            master_port=12345,
        )

        mock_stage.test_method.assert_called_once()
        queue_out.put.assert_called()
        mock_dist.init_process_group.assert_not_called()

    @patch("telefuser.worker.parallel_worker.dist")
    @patch("telefuser.worker.parallel_worker.current_platform")
    def test_worker_loop_missing_method(self, mock_platform, mock_dist):
        """Test _worker_loop raises error for missing method."""
        from telefuser.worker.parallel_worker import _worker_loop

        mock_platform.device_type = "cpu"

        queue_in = [MagicMock()]
        queue_out = MagicMock()

        queue_in[0].get.return_value = ("nonexistent_method", [], {})

        mock_stage = MagicMock()
        mock_stage.device = "cpu"
        mock_stage.model_runtime_config.parallel_config.worker_intra_op_threads = 1
        mock_stage.name = "TestStage"
        del mock_stage.nonexistent_method  # Ensure method doesn't exist

        _worker_loop(
            rank=0,
            world_size=1,
            queue_in=queue_in,
            queue_out=queue_out,
            stage=mock_stage,
            master_port=12345,
        )

        # Should put an exception in the queue
        args, _ = queue_out.put.call_args
        assert isinstance(args[0], AttributeError)

    @patch("telefuser.worker.parallel_worker.dist")
    @patch("telefuser.worker.parallel_worker.current_platform")
    def test_worker_loop_exception_handling(self, mock_platform, mock_dist):
        """Test _worker_loop catches and propagates exceptions."""
        from telefuser.worker.parallel_worker import _worker_loop

        mock_platform.device_type = "cpu"

        queue_in = [MagicMock()]
        queue_out = MagicMock()

        queue_in[0].get.return_value = ("test_method", [], {})

        mock_stage = MagicMock()
        mock_stage.device = "cpu"
        mock_stage.model_runtime_config.parallel_config.worker_intra_op_threads = 1
        mock_stage.test_method.side_effect = RuntimeError("Method error")

        _worker_loop(
            rank=0,
            world_size=1,
            queue_in=queue_in,
            queue_out=queue_out,
            stage=mock_stage,
            master_port=12345,
        )

        args, _ = queue_out.put.call_args
        assert isinstance(args[0], RuntimeError)
