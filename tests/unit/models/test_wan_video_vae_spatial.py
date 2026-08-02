from types import SimpleNamespace
from unittest.mock import patch

import pytest

from telefuser.distributed.vae_spatial import _SpatialParallelConv2d
from telefuser.models.wan_video_vae import (
    AttentionBlock,
    CausalConv3d,
    Decoder3d,
    WanVideoVAE,
    _SpatialParallelCausalConv3d,
    _enable_spatial_parallel_decode,
)


def test_enable_spatial_parallel_decode_preserves_parameters_and_is_idempotent() -> None:
    decoder = Decoder3d(dim=8, z_dim=4)
    vae = SimpleNamespace(model=SimpleNamespace(decoder=decoder))
    parameter_ids = {name: id(parameter) for name, parameter in decoder.named_parameters()}

    with patch("telefuser.models.wan_video_vae._spatial_world_size", return_value=4):
        converted = _enable_spatial_parallel_decode(vae)
        converted_again = _enable_spatial_parallel_decode(vae)

    assert converted > 0
    assert converted_again == 0
    assert decoder._spatial_parallel is True
    assert parameter_ids == {name: id(parameter) for name, parameter in decoder.named_parameters()}
    assert all(
        isinstance(module, _SpatialParallelCausalConv3d)
        for module in decoder.modules()
        if isinstance(module, CausalConv3d)
    )
    assert any(isinstance(module, _SpatialParallelConv2d) for module in decoder.modules())
    assert all(module._spatial_parallel for module in decoder.modules() if isinstance(module, AttentionBlock))


def test_streaming_spatial_decode_rejects_native_parallelism() -> None:
    vae = SimpleNamespace(
        model=SimpleNamespace(decoder=Decoder3d(dim=8, z_dim=4)),
        parallelism=4,
    )

    with (
        patch("telefuser.models.wan_video_vae._spatial_world_size", return_value=4),
        pytest.raises(RuntimeError, match="mutually exclusive"),
    ):
        _enable_spatial_parallel_decode(vae)


def test_native_parallelism_rejects_streaming_spatial_decode() -> None:
    vae = SimpleNamespace(
        model=SimpleNamespace(decoder=SimpleNamespace(_spatial_parallel=True)),
        parallelism=1,
    )

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        WanVideoVAE.set_parallelism(vae, 4)

    assert vae.parallelism == 1
    WanVideoVAE.set_parallelism(vae, 1)
