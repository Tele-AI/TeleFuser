from unittest.mock import MagicMock, patch

import pytest
import torch

from telefuser.distributed.collectives import all_gather_cat, all_gather_stacked, all_reduce_sum_


def test_all_gather_stacked_uses_one_contiguous_output() -> None:
    local = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    def gather(output: torch.Tensor, tensor: torch.Tensor, *, group: object) -> None:
        ranks = output.view(2, *tensor.shape)
        ranks[0].copy_(tensor)
        ranks[1].copy_(tensor + 10)

    with patch("telefuser.distributed.collectives.dist.all_gather_into_tensor", side_effect=gather) as mocked:
        gathered = all_gather_stacked(local, group=MagicMock(), world_size=2)

    assert gathered.shape == (2, 2, 3)
    torch.testing.assert_close(gathered[0], local)
    torch.testing.assert_close(gathered[1], local + 10)
    mocked.assert_called_once()


def test_all_gather_cat_supports_arbitrary_dimensions() -> None:
    local = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    moved = local.movedim(1, 0).contiguous()
    gathered = torch.stack((moved, moved + 10))

    with patch("telefuser.distributed.collectives.all_gather_stacked", return_value=gathered):
        merged = all_gather_cat(local, dim=1, group=MagicMock(), world_size=2)

    torch.testing.assert_close(merged, torch.cat((local, local + 10), dim=1))
    assert merged.is_contiguous()


def test_all_gather_helpers_are_no_ops_for_one_rank() -> None:
    tensor = torch.randn(2, 3)

    assert all_gather_stacked(tensor, world_size=1).data_ptr() == tensor.data_ptr()
    assert all_gather_cat(tensor, dim=-1, world_size=1) is tensor


@pytest.mark.parametrize("dim", [-3, 2])
def test_all_gather_cat_rejects_invalid_dimension(dim: int) -> None:
    with pytest.raises(ValueError, match="invalid"):
        all_gather_cat(torch.zeros(2, 3), dim=dim, world_size=2)


def test_all_reduce_sum_submits_before_waiting() -> None:
    tensors = (torch.ones(1), torch.ones(1))
    works = (MagicMock(), MagicMock())

    with patch("telefuser.distributed.collectives.dist.all_reduce", side_effect=works) as mocked:
        all_reduce_sum_(tensors, group=MagicMock())

    assert mocked.call_count == 2
    assert all(call.kwargs["async_op"] is True for call in mocked.call_args_list)
    works[0].wait.assert_called_once()
    works[1].wait.assert_called_once()
