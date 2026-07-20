"""Tests for Ulysses All-to-All communication."""

from unittest.mock import MagicMock, patch

import pytest
import torch

try:
    import torch.distributed as dist

    HAS_DISTRIBUTED = dist.is_available()
except ImportError:
    HAS_DISTRIBUTED = False

pytestmark = [
    pytest.mark.skipif(not HAS_DISTRIBUTED, reason="Distributed not available"),
    pytest.mark.distributed,
]


class TestUlyssesScatterHeads:
    """Test head-to-sequence redistribution."""

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
    def test_scatter_heads_accepts_even_partition(self, mock_rank, mock_world_size):
        del mock_rank, mock_world_size
        from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

        tensor = torch.randn(2, 10, 32, 64)
        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single", return_value=tensor.flatten()):
            assert callable(ulysses_scatter_heads(tensor, MagicMock()))

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
    def test_scatter_heads_rejects_uneven_partition(self, mock_rank, mock_world_size):
        del mock_rank, mock_world_size
        from telefuser.distributed.ulysses_comm import ulysses_scatter_heads

        with pytest.raises(ValueError, match="divisible"):
            ulysses_scatter_heads(torch.randn(2, 10, 30, 64), MagicMock())


class TestUlyssesGatherHeads:
    """Test sequence-to-head redistribution."""

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
    def test_gather_heads_accepts_even_partition(self, mock_rank, mock_world_size):
        del mock_rank, mock_world_size
        from telefuser.distributed.ulysses_comm import ulysses_gather_heads

        tensor = torch.randn(2, 40, 8, 64)
        with patch("telefuser.distributed.ulysses_comm.fc.all_to_all_single", return_value=tensor.flatten()):
            assert callable(ulysses_gather_heads(tensor, MagicMock(), num_heads=32))

    @patch("telefuser.distributed.ulysses_comm.dist.get_world_size", return_value=4)
    @patch("telefuser.distributed.ulysses_comm.dist.get_rank", return_value=0)
    def test_gather_heads_rejects_uneven_partition(self, mock_rank, mock_world_size):
        del mock_rank, mock_world_size
        from telefuser.distributed.ulysses_comm import ulysses_gather_heads

        with pytest.raises(ValueError, match="divisible"):
            ulysses_gather_heads(torch.randn(2, 40, 8, 64), MagicMock(), num_heads=30)


def test_local_head_count_requires_even_partition() -> None:
    from telefuser.distributed.ulysses_comm import _local_head_count

    assert _local_head_count(32, 4) == 8
    with pytest.raises(ValueError, match="divisible"):
        _local_head_count(30, 4)
