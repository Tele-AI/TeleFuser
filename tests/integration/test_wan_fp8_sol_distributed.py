from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.models.wan_video_dit import SelfAttention
from telefuser.ops.attention import SparseAttentionState
from telefuser.worker import ParallelWorker


class _WanFP8SolUlyssesParityStage(BaseStage):
    def __init__(self, degree: int) -> None:
        super().__init__(
            "wan-fp8-sol-ulysses-parity",
            ModelRuntimeConfig(
                device_type="cuda",
                torch_dtype=torch.bfloat16,
                parallel_config=ParallelConfig(device_ids=list(range(degree)), sp_ulysses_degree=degree),
            ),
        )
        torch.manual_seed(41)
        source = SelfAttention(dim=512, num_heads=4).eval().to(torch.bfloat16)
        config = AttentionConfig.sol_attention(
            dense_timesteps=0,
            dense_layers=0,
            tau=-1000.0,
            sol_fp8=True,
        )
        source.attention_config = config
        self.dense = deepcopy(source)
        self.parallel = deepcopy(source)
        self.device_mesh = None
        self.empty_cache_after_call = False

    def parallel_models(self) -> None:
        self.parallel = self.parallel.to(self.device)
        self.parallel.usp_flag = True
        self.device_mesh = create_device_mesh_from_config(self.model_runtime_config.parallel_config)
        if dist.get_rank() == 0:
            self.dense = self.dense.to(self.device)

    @staticmethod
    def _state(module: SelfAttention) -> SparseAttentionState:
        sparse_config = module.attention_config.sparse_config
        assert sparse_config is not None
        return SparseAttentionState(sparse_config, mask_map=None)

    def compare(self, x: torch.Tensor, freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_sequence = x.shape[1] // world_size
        start = rank * local_sequence
        stop = start + local_sequence
        parallel_local = self.parallel(
            x[:, start:stop],
            freqs[start:stop],
            freqs[start:stop],
            sparse_state=self._state(self.parallel),
            device_mesh=self.device_mesh,
        )
        gathered = [torch.empty_like(parallel_local) for _ in range(world_size)]
        dist.all_gather(gathered, parallel_local)
        parallel = torch.cat(gathered, dim=1).cpu()

        dense = torch.empty(0)
        if rank == 0:
            dense = self.dense(
                x,
                freqs,
                freqs,
                sparse_state=self._state(self.dense),
            ).cpu()
        return dense, parallel


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.parametrize("degree", [2, 4])
def test_wan_fp8_sol_ulysses_matches_single_gpu(degree: int) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} CUDA devices")
    torch.manual_seed(43)
    x = torch.randn(1, 256, 512, dtype=torch.bfloat16)
    freqs = torch.zeros(256, 64, dtype=torch.bfloat16)
    worker = ParallelWorker(_WanFP8SolUlyssesParityStage(degree))
    try:
        dense, parallel = worker.compare(x, freqs, sync=True)
    finally:
        worker.close()

    assert dense.shape == parallel.shape == x.shape
    assert torch.isfinite(parallel).all()
    torch.testing.assert_close(parallel, dense, rtol=2e-2, atol=2e-2)
