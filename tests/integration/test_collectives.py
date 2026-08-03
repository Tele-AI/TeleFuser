from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.distributed.collectives import all_gather_cat, all_gather_stacked, all_reduce_sum_
from telefuser.worker import ParallelWorker


class _CollectiveStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "collective-integration",
            ModelRuntimeConfig(
                device_type="cuda",
                torch_dtype=torch.float32,
                parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2),
            ),
        )
        self.empty_cache_after_call = False

    def run(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rank = dist.get_rank()
        local = torch.full((2, 3), float(rank), device=self.device)
        stacked = all_gather_stacked(local)
        concatenated = all_gather_cat(local, dim=1)
        value = torch.tensor([rank + 1.0], device=self.device)
        weight = torch.tensor([2.0 * rank + 1.0], device=self.device)
        all_reduce_sum_((value, weight))
        return stacked.cpu(), concatenated.cpu(), value.cpu(), weight.cpu()


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_shared_collectives_preserve_rank_order_and_reduce_all_tensors() -> None:
    worker = ParallelWorker(_CollectiveStage())
    try:
        stacked, concatenated, value, weight = worker.run(sync=True)
    finally:
        worker.close()

    assert stacked.shape == (2, 2, 3)
    torch.testing.assert_close(stacked[0], torch.zeros(2, 3))
    torch.testing.assert_close(stacked[1], torch.ones(2, 3))
    torch.testing.assert_close(concatenated, torch.tensor([[0, 0, 0, 1, 1, 1]]).expand(2, -1).float())
    torch.testing.assert_close(value, torch.tensor([3.0]))
    torch.testing.assert_close(weight, torch.tensor([4.0]))
