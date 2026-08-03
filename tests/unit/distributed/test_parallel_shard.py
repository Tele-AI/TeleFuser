"""Tests for parallel_shard module."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Skip if distributed not available
try:
    import torch.distributed as dist

    HAS_DISTRIBUTED = dist.is_available()
except ImportError:
    HAS_DISTRIBUTED = False

pytestmark = [
    pytest.mark.skipif(not HAS_DISTRIBUTED, reason="Distributed not available"),
    pytest.mark.distributed,
]


class TestSequenceParallelShard:
    """Test sequence_parallel_shard function."""

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_rank")
    def test_no_op_when_world_size_1(
        self, mock_get_sp_shard_rank, mock_get_sp_shard_degree, mock_get_attention_strategy
    ):
        """Test no sharding when world_size is 1."""
        from telefuser.distributed.parallel_shard import sequence_parallel_shard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 1
        mock_get_sp_shard_rank.return_value = 0
        mock_mesh = MagicMock()

        tensor = torch.randn(2, 10, 64)
        original = tensor.clone()
        sequence_parallel_shard(mock_mesh, [tensor], [1])

        assert torch.allclose(tensor, original)

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_rank")
    def test_shard_even_division(self, mock_get_sp_shard_rank, mock_get_sp_shard_degree, mock_get_attention_strategy):
        """Test sharding with even division."""
        from telefuser.distributed.parallel_shard import sequence_parallel_shard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 2
        mock_get_sp_shard_rank.return_value = 0
        mock_mesh = MagicMock()

        tensor = torch.randn(2, 10, 64)
        sequence_parallel_shard(mock_mesh, [tensor], [1])

        assert tensor.shape == (2, 5, 64)  # 10 // 2 = 5

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_rank")
    def test_shard_with_padding_for_uneven(
        self, mock_get_sp_shard_rank, mock_get_sp_shard_degree, mock_get_attention_strategy
    ):
        """Test sharding with automatic padding for uneven division."""
        from telefuser.distributed.parallel_shard import sequence_parallel_shard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 4
        mock_get_sp_shard_rank.return_value = 0
        mock_mesh = MagicMock()

        tensor = torch.randn(2, 10, 64)  # 10 needs padding to 12 for 4 ranks
        sequence_parallel_shard(mock_mesh, [tensor], [1])

        assert tensor.shape == (2, 3, 64)  # 12 // 4 = 3

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_rank")
    def test_shard_with_seq_divisions(
        self, mock_get_sp_shard_rank, mock_get_sp_shard_degree, mock_get_attention_strategy
    ):
        """Test sharding with custom seq_divisions."""
        from telefuser.distributed.parallel_shard import sequence_parallel_shard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 2
        mock_get_sp_shard_rank.return_value = 0
        mock_mesh = MagicMock()

        # seq_len=10, shard_degree=2, seq_division=2 means division size = 2*2=4
        # Need to pad to 12 to be divisible by 4
        tensor = torch.randn(2, 10, 64)
        sequence_parallel_shard(mock_mesh, [tensor], [1], [2])

        assert tensor.shape[1] == 6  # 12 // 2 = 6

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_rank")
    def test_shard_handles_none_and_empty(
        self, mock_get_sp_shard_rank, mock_get_sp_shard_degree, mock_get_attention_strategy
    ):
        """Test sharding handles None tensors and empty lists."""
        from telefuser.distributed.parallel_shard import sequence_parallel_shard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 2
        mock_get_sp_shard_rank.return_value = 0
        mock_mesh = MagicMock()

        # None tensor
        sequence_parallel_shard(mock_mesh, [None], [0])

        # Empty list
        sequence_parallel_shard(mock_mesh, [], [])

        # No tensors argument
        sequence_parallel_shard(mock_mesh)


class TestSequenceParallelUnshard:
    """Test sequence_parallel_unshard function."""

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_group")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    def test_no_op_when_world_size_1(
        self, mock_get_sp_shard_degree, mock_get_sp_shard_group, mock_get_attention_strategy
    ):
        """Test no unsharding when world_size is 1."""
        from telefuser.distributed.parallel_shard import sequence_parallel_unshard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 1
        mock_get_sp_shard_group.return_value = None
        mock_mesh = MagicMock()

        tensors = [torch.randn(2, 5, 64)]
        result = sequence_parallel_unshard(mock_mesh, tensors, [1], [10])

        assert len(result) == 1
        assert torch.allclose(result[0], tensors[0])

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_group")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    @patch("telefuser.distributed.parallel_shard.all_gather_cat")
    def test_unshard_multi_gpu(
        self, mock_all_gather, mock_get_sp_shard_degree, mock_get_sp_shard_group, mock_get_attention_strategy
    ):
        """Test unsharding gathers and concatenates from all ranks."""
        from telefuser.distributed.parallel_shard import sequence_parallel_unshard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 2
        mock_get_sp_shard_group.return_value = MagicMock()
        mock_mesh = MagicMock()

        mock_all_gather.side_effect = lambda tensor, *, dim, **_: torch.cat((tensor, tensor + 1), dim=dim)

        tensors = [torch.randn(2, 5, 64)]
        result = sequence_parallel_unshard(mock_mesh, tensors, [1], [10])

        assert len(result) == 1
        assert result[0].shape[1] == 10  # Restored original seq_len

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_group")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    @patch("telefuser.distributed.parallel_shard.all_gather_cat")
    def test_unshard_multiple_tensors(
        self, mock_all_gather, mock_get_sp_shard_degree, mock_get_sp_shard_group, mock_get_attention_strategy
    ):
        """Test unsharding multiple tensors with different shapes."""
        from telefuser.distributed.parallel_shard import sequence_parallel_unshard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 2
        mock_get_sp_shard_group.return_value = MagicMock()
        mock_mesh = MagicMock()

        mock_all_gather.side_effect = lambda tensor, *, dim, **_: torch.cat((tensor, tensor), dim=dim)

        tensors = [torch.randn(2, 5, 64), torch.randn(3, 8, 32)]
        result = sequence_parallel_unshard(mock_mesh, tensors, [1, 1], [10, 16])

        assert len(result) == 2
        assert result[0].shape[1] == 10
        assert result[1].shape[1] == 16


class TestCFGParallelShard:
    """Test cfg_parallel_shard function."""

    @patch("telefuser.distributed.parallel_shard.get_cfg_world_size")
    def test_no_op_when_world_size_1(self, mock_get_cfg_world_size):
        """Test no sharding when world_size is 1."""
        from telefuser.distributed.parallel_shard import cfg_parallel_shard

        mock_get_cfg_world_size.return_value = 1
        mock_mesh = MagicMock()

        tensor = torch.randn(4, 10, 64)
        original = tensor.clone()
        cfg_parallel_shard(mock_mesh, [tensor])

        assert torch.allclose(tensor, original)

    @patch("telefuser.distributed.parallel_shard.get_cfg_world_size")
    @patch("telefuser.distributed.parallel_shard.get_cfg_rank")
    def test_shard_batch_dim(self, mock_get_cfg_rank, mock_get_cfg_world_size):
        """Test sharding on batch dimension."""
        from telefuser.distributed.parallel_shard import cfg_parallel_shard

        mock_get_cfg_world_size.return_value = 2
        mock_get_cfg_rank.return_value = 0
        mock_mesh = MagicMock()

        tensor = torch.randn(4, 10, 64)  # batch=4
        cfg_parallel_shard(mock_mesh, [tensor])

        assert tensor.shape == (2, 10, 64)  # 4 // 2 = 2

    @patch("telefuser.distributed.parallel_shard.get_cfg_world_size")
    @patch("telefuser.distributed.parallel_shard.get_cfg_rank")
    def test_shard_skips_uneven_division(self, mock_get_cfg_rank, mock_get_cfg_world_size):
        """Test sharding skips when batch not evenly divisible."""
        from telefuser.distributed.parallel_shard import cfg_parallel_shard

        mock_get_cfg_world_size.return_value = 3
        mock_get_cfg_rank.return_value = 0
        mock_mesh = MagicMock()

        tensor = torch.randn(5, 10, 64)  # 5 not divisible by 3
        original = tensor.clone()
        cfg_parallel_shard(mock_mesh, [tensor])

        assert torch.allclose(tensor, original)  # Unchanged

    @patch("telefuser.distributed.parallel_shard.get_cfg_world_size")
    def test_shard_handles_none_and_multiple(self, mock_get_cfg_world_size):
        """Test sharding handles None and multiple tensors."""
        from telefuser.distributed.parallel_shard import cfg_parallel_shard

        mock_get_cfg_world_size.return_value = 2
        mock_mesh = MagicMock()

        with patch("telefuser.distributed.parallel_shard.get_cfg_rank", return_value=0):
            # None tensor
            cfg_parallel_shard(mock_mesh, [None])

            # Multiple tensors
            tensors = [torch.randn(4, 10, 64), torch.randn(6, 8, 32)]
            cfg_parallel_shard(mock_mesh, tensors)

            assert tensors[0].shape == (2, 10, 64)
            assert tensors[1].shape == (3, 8, 32)


class TestCFGParallelUnshard:
    """Test cfg_parallel_unshard function."""

    @patch("telefuser.distributed.parallel_shard.get_cfg_world_size")
    def test_no_op_when_world_size_1(self, mock_get_cfg_world_size):
        """Test no unsharding when world_size is 1."""
        from telefuser.distributed.parallel_shard import cfg_parallel_unshard

        mock_get_cfg_world_size.return_value = 1
        mock_mesh = MagicMock()

        tensors = [torch.randn(2, 10, 64)]
        result = cfg_parallel_unshard(mock_mesh, tensors)

        assert len(result) == 1
        assert torch.allclose(result[0], tensors[0])

    @patch("telefuser.distributed.parallel_shard.get_cfg_world_size")
    @patch("telefuser.distributed.parallel_shard.get_cfg_group")
    @patch("telefuser.distributed.parallel_shard.all_gather_cat")
    def test_unshard_gather_into_tensor(self, mock_all_gather, mock_get_cfg_group, mock_get_cfg_world_size):
        """Test unsharding delegates to the shared gather primitive."""
        from telefuser.distributed.parallel_shard import cfg_parallel_unshard

        mock_get_cfg_world_size.return_value = 2
        mock_get_cfg_group.return_value = MagicMock()
        mock_mesh = MagicMock()

        mock_all_gather.side_effect = lambda tensor, *, dim, **_: torch.cat((tensor, tensor), dim=dim)

        tensors = [torch.randn(2, 10, 64), torch.randn(3, 8, 32)]
        result = cfg_parallel_unshard(mock_mesh, tensors)

        assert len(result) == 2
        assert result[0].shape == (4, 10, 64)
        assert result[1].shape == (6, 8, 32)
        assert mock_all_gather.call_count == 2


class TestShardUnshardRoundTrip:
    """Integration-style tests for shard/unshard operations."""

    @patch("telefuser.distributed.parallel_shard.get_attention_strategy")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_degree")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_rank")
    @patch("telefuser.distributed.parallel_shard.get_sp_shard_group")
    @patch("telefuser.distributed.parallel_shard.all_gather_cat")
    def test_sequence_parallel_roundtrip(
        self,
        mock_all_gather,
        mock_get_sp_shard_group,
        mock_get_sp_shard_rank,
        mock_get_sp_shard_degree,
        mock_get_attention_strategy,
    ):
        """Test sequence parallel shard and unshard restore original shape."""
        from telefuser.distributed.parallel_shard import sequence_parallel_shard, sequence_parallel_unshard

        mock_get_attention_strategy.return_value = "ulysses"
        mock_get_sp_shard_degree.return_value = 2
        mock_get_sp_shard_rank.return_value = 0
        mock_get_sp_shard_group.return_value = MagicMock()
        mock_mesh = MagicMock()

        mock_all_gather.side_effect = lambda tensor, *, dim, **_: torch.cat((tensor, tensor + 0.1), dim=dim)

        original_shape = (2, 10, 64)
        tensor = torch.randn(*original_shape)
        seq_len = original_shape[1]

        # Shard
        sequence_parallel_shard(mock_mesh, [tensor], [1])
        assert tensor.shape[1] == seq_len // 2

        # Unshard
        result = sequence_parallel_unshard(mock_mesh, [tensor], [1], [seq_len])
        assert result[0].shape == original_shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
