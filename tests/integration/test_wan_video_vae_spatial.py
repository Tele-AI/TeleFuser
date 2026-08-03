from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.distributed.vae_spatial import _split_height
from telefuser.models.wan_video_vae import (
    VideoVAE,
    _convert_conv3d_to_channels_last_3d,
    _count_conv3d,
    _enable_spatial_parallel_decode,
)
from telefuser.worker import ParallelWorker


class _SpatialVAEParityStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "spatial-vae-parity",
            ModelRuntimeConfig(
                device_type="cuda",
                torch_dtype=torch.float32,
                parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2),
            ),
        )
        torch.manual_seed(17)
        source = VideoVAE(dim=8, z_dim=4).eval()
        self.dense_conv2 = deepcopy(source.conv2)
        self.dense_decoder = deepcopy(source.decoder)
        self.spatial_conv2 = deepcopy(source.conv2)
        self.spatial_decoder = deepcopy(source.decoder)
        self.empty_cache_after_call = False

    def parallel_models(self) -> None:
        self.dense_conv2 = self.dense_conv2.to(self.device)
        self.dense_decoder = self.dense_decoder.to(self.device)
        self.spatial_conv2 = self.spatial_conv2.to(self.device)
        self.spatial_decoder = self.spatial_decoder.to(self.device)
        _convert_conv3d_to_channels_last_3d(self.dense_decoder)
        _convert_conv3d_to_channels_last_3d(self.spatial_decoder)
        vae = SimpleNamespace(
            model=SimpleNamespace(decoder=self.spatial_decoder),
            parallelism=1,
        )
        _enable_spatial_parallel_decode(vae)

    @staticmethod
    def _decode_chunks(
        conv2: torch.nn.Module,
        decoder: torch.nn.Module,
        chunks: list[torch.Tensor],
        *,
        pre_shard: bool = False,
    ) -> list[torch.Tensor]:
        cache: list[object] = [None] * _count_conv3d(decoder)
        outputs = []
        for chunk in chunks:
            cache_index = [0]
            encoded = conv2(chunk)
            kwargs = {}
            if pre_shard:
                kwargs["input_global_height"] = encoded.shape[-2]
                encoded = _split_height(encoded)
            outputs.append(decoder(encoded, feat_cache=cache, feat_idx=cache_index, **kwargs).cpu())
        return outputs

    def compare(self, chunks: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        dense = self._decode_chunks(self.dense_conv2, self.dense_decoder, chunks) if dist.get_rank() == 0 else []
        spatial = self._decode_chunks(self.spatial_conv2, self.spatial_decoder, chunks)
        pre_sharded = self._decode_chunks(self.spatial_conv2, self.spatial_decoder, chunks, pre_shard=True)
        return dense, spatial, pre_sharded


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_streaming_spatial_vae_matches_dense_decode_across_causal_chunks() -> None:
    torch.manual_seed(29)
    chunks = [
        torch.randn(1, 4, 1, 5, 4, dtype=torch.float32),
        torch.randn(1, 4, 1, 5, 4, dtype=torch.float32),
    ]
    worker = ParallelWorker(_SpatialVAEParityStage())
    try:
        dense_outputs, spatial_outputs, pre_sharded_outputs = worker.compare(chunks, sync=True)
    finally:
        worker.close()

    assert len(dense_outputs) == len(spatial_outputs) == len(pre_sharded_outputs) == 2
    for dense, spatial, pre_sharded in zip(dense_outputs, spatial_outputs, pre_sharded_outputs):
        torch.testing.assert_close(spatial, dense, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(pre_sharded, dense, rtol=2e-4, atol=2e-4)
