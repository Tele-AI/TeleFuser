"""Tests for Pipeline Parallel communication module."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.distributed as dist

from telefuser.distributed.pp_comm import PipelineP2PComm

# Skip if distributed not available
HAS_DISTRIBUTED = dist.is_available()

pytestmark = [
    pytest.mark.skipif(not HAS_DISTRIBUTED, reason="Distributed not available"),
    pytest.mark.distributed,
]


class TestPipelineP2PCommInit:
    """Test PipelineP2PComm initialization."""

    def test_init_with_none_group(self):
        """Test initialization with None process group (single GPU fallback)."""
        comm = PipelineP2PComm(None)

        assert comm.rank == 0
        assert comm.world_size == 1
        assert comm.is_first_stage is True
        assert comm.is_last_stage is True
        assert comm.send_dst == 0
        assert comm.recv_src == 0

    def test_init_with_process_group(self):
        """Test initialization with a mock process group."""
        mock_group = MagicMock()
        mock_group.__class__.__name__ = "ProcessGroup"

        with (
            patch.object(dist, "get_rank", return_value=1),
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(dist, "get_global_rank", side_effect=[2, 0]),
        ):
            comm = PipelineP2PComm(mock_group)

            assert comm.rank == 1
            assert comm.world_size == 4
            assert comm.is_first_stage is False
            assert comm.is_last_stage is False


class TestPipelineP2PCommMethods:
    """Test PipelineP2PComm communication methods."""

    def test_send_on_last_stage_logs_warning(self):
        """Test that send on last stage logs warning and returns None."""
        comm = PipelineP2PComm(None)  # Single GPU, is_last_stage=True

        tensor = torch.randn(1, 10, 512)
        result = comm.send(tensor)

        assert result is None

    def test_recv_on_first_stage_raises(self):
        """Test that recv on first stage raises RuntimeError."""
        comm = PipelineP2PComm(None)  # Single GPU, is_first_stage=True

        with pytest.raises(RuntimeError, match="First stage has no previous stage to receive from"):
            comm.recv(shape=(1, 10, 512))

    def test_send_recv_single_gpu(self):
        """Test send_recv on single GPU (no-op)."""
        comm = PipelineP2PComm(None)  # Single GPU

        send_tensor = torch.randn(1, 10, 512)
        result = comm.send_recv(send_tensor)

        # On single GPU, recv_buffer is None since there's no previous stage
        assert result is None

    def test_get_stage_indices_even_distribution(self):
        """Test layer distribution with even division."""
        comm = PipelineP2PComm(None)
        comm.world_size = 4
        comm.rank = 1

        # 40 layers, 4 stages -> 10 layers each
        start, end = comm.get_stage_indices(40)
        assert start == 10
        assert end == 20

    def test_get_stage_indices_with_remainder(self):
        """Test layer distribution with remainder."""
        comm = PipelineP2PComm(None)

        # Mock world_size=3, rank=0, 10 layers -> 4, 3, 3
        comm.world_size = 3
        comm.rank = 0
        start, end = comm.get_stage_indices(10)
        assert start == 0
        assert end == 4  # First stage gets extra layer

        comm.rank = 1
        start, end = comm.get_stage_indices(10)
        assert start == 4
        assert end == 7

        comm.rank = 2
        start, end = comm.get_stage_indices(10)
        assert start == 7
        assert end == 10

    def test_queue_send_on_last_stage(self):
        """Test queue_send on last stage does nothing."""
        comm = PipelineP2PComm(None)  # is_last_stage=True

        tensor = torch.randn(1, 10, 512)
        comm.queue_send(tensor)

        assert len(comm._ops) == 0

    def test_queue_recv_on_first_stage(self):
        """Test queue_recv on first stage does nothing."""
        comm = PipelineP2PComm(None)  # is_first_stage=True

        buffer = torch.randn(1, 10, 512)
        comm.queue_recv(buffer)

        assert len(comm._ops) == 0

    def test_commit_without_ops(self):
        """Test commit with no queued operations."""
        comm = PipelineP2PComm(None)

        # Should not raise any error
        comm.commit()
        comm.wait()

    def test_commit_twice_raises(self):
        """Test that calling commit twice raises RuntimeError."""
        comm = PipelineP2PComm(None)
        comm._reqs = [MagicMock()]  # Simulate pending requests

        with pytest.raises(RuntimeError, match="commit\\(\\) called twice"):
            comm.commit()


class TestPipelineP2PCommLatentMethods:
    """Test convenience methods for latent communication."""

    def test_send_latent_on_last_stage(self):
        """Test send_latent on last stage returns early."""
        comm = PipelineP2PComm(None)  # is_last_stage=True

        tensor = torch.randn(1, 10, 512)
        # Should not raise, just return
        comm.send_latent(tensor)

    def test_recv_latent_on_first_stage_raises(self):
        """Test recv_latent on first stage raises RuntimeError."""
        comm = PipelineP2PComm(None)  # is_first_stage=True

        with pytest.raises(RuntimeError, match="First stage has no previous stage"):
            comm.recv_latent(shape=(1, 10, 512))

    def test_recv_latent_without_shape_raises(self):
        """Test recv_latent without shape raises ValueError."""
        # Create a mock that simulates middle stage
        mock_group = MagicMock()

        with (
            patch.object(dist, "get_rank", return_value=1),
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(dist, "get_global_rank", side_effect=[2, 0]),
        ):
            comm = PipelineP2PComm(mock_group)

            with pytest.raises(ValueError, match="shape must be provided"):
                comm.recv_latent()

    def test_send_latent_async_on_last_stage(self):
        """Test send_latent_async on last stage returns None."""
        comm = PipelineP2PComm(None)  # is_last_stage=True

        tensor = torch.randn(1, 10, 512)
        result = comm.send_latent_async(tensor)

        assert result is None

    def test_recv_latent_async_on_first_stage_raises(self):
        """Test recv_latent_async on first stage raises RuntimeError."""
        comm = PipelineP2PComm(None)  # is_first_stage=True

        with pytest.raises(RuntimeError, match="First stage has no previous stage"):
            comm.recv_latent_async(shape=(1, 10, 512))


class TestPipelineP2PCommMultiGPU:
    """Tests that require multi-GPU setup."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_send_recv_with_mock_group(self):
        """Test send_recv initialization with mocked process group.

        Note: This test only verifies the logic without actual P2P operations.
        For real P2P tests, use the integration test below.
        """
        mock_group = MagicMock()

        # Simulate a middle stage (rank 1 of 4)
        with (
            patch.object(dist, "get_rank", return_value=1),
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(dist, "get_global_rank", side_effect=[2, 0]),
        ):
            comm = PipelineP2PComm(mock_group)

            # Verify the comm is correctly initialized for middle stage
            assert comm.is_first_stage is False
            assert comm.is_last_stage is False
            assert comm.rank == 1
            assert comm.world_size == 4


# Integration tests that require actual distributed environment
@pytest.mark.distributed
class TestPipelineP2PCommIntegration:
    """Integration tests for PipelineP2PComm with actual distributed environment."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_actual_p2p_communication(self):
        """Test actual P2P communication between processes.

        This test requires running with multiple processes:
        torchrun --nproc_per_node=2 -m pytest tests/unit/distributed/test_pp_comm.py::TestPipelineP2PCommIntegration -v
        """
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size < 2:
            pytest.skip("Need at least 2 processes for P2P test")

        # Create process group for PP
        pg = dist.new_group()

        comm = PipelineP2PComm(pg)

        if comm.is_first_stage:
            # Send tensor to next stage
            send_tensor = torch.ones(1, 10, 512, device="cuda") * comm.rank
            comm.send_latent(send_tensor)
        elif comm.is_last_stage:
            # Receive tensor from previous stage
            recv_tensor = comm.recv_latent(shape=(1, 10, 512))
            # Verify received tensor has expected value from previous stage
            expected_value = comm.rank - 1
            assert torch.allclose(recv_tensor, torch.ones_like(recv_tensor) * expected_value)
        else:
            # Middle stage: receive and send
            recv_tensor = comm.recv_latent(shape=(1, 10, 512))
            recv_tensor = recv_tensor + 1  # Modify before sending
            comm.send_latent(recv_tensor)

        dist.destroy_process_group(pg)
