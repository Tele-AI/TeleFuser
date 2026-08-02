from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from telefuser.pipelines.lingbot_world_fast import vae_stage
from telefuser.pipelines.lingbot_world_fast.vae_stage import (
    _VAECachePool,
    _VAEDecodeCacheState,
    _cache_tensor_bytes,
    _copy_frames_to_shared_cpu,
)


class _RecordingEncoder:
    def __init__(self) -> None:
        self.frame_counts: list[int] = []

    def cached_encode_withflag(self, video, device, is_first_clip, is_last_clip, encode_state):
        del device, encode_state
        assert is_first_clip is True
        assert is_last_clip is True
        self.frame_counts.append(video.shape[1])
        latent_frames = (video.shape[1] - 1) // 4 + 1
        values = torch.arange(latent_frames, dtype=video.dtype).view(1, latent_frames, 1, 1)
        return values.expand(16, -1, 2, 2).clone()


def test_cache_tensor_bytes_ignores_non_tensor_entries() -> None:
    cache = [
        torch.empty((2, 3), dtype=torch.float32),
        None,
        "Rep",
        torch.empty((4,), dtype=torch.bfloat16),
    ]

    assert _cache_tensor_bytes(cache) == 32


def test_vae_decode_reuses_shared_cpu_output_buffer() -> None:
    state = _VAEDecodeCacheState()
    first = _copy_frames_to_shared_cpu(state, torch.zeros(3, 4, 2, 2))
    second = _copy_frames_to_shared_cpu(state, torch.ones(3, 4, 2, 2))

    assert first.is_shared()
    assert second.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    assert torch.equal(second, torch.full_like(second, 255))


def test_vae_cache_pool_stabilizes_and_reuses_fixed_slots() -> None:
    pool = _VAECachePool(
        capacity=2,
        layout={
            0: (torch.float32, 8),
            2: (torch.bfloat16, 6),
        },
        device=torch.device("cpu"),
    )
    first_slot = pool.acquire()
    second_slot = pool.acquire()

    assert pool.try_acquire() is None
    with pytest.raises(RuntimeError, match="pool is full"):
        pool.acquire()

    cache: list[object] = [
        torch.arange(6, dtype=torch.float32).view(2, 3),
        "Rep",
        torch.arange(4, dtype=torch.bfloat16),
    ]
    original = cache[0].clone()
    pool.stabilize(cache, first_slot)

    assert torch.equal(cache[0], original)
    assert cache[0].untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    assert pool.bytes_per_session == 44

    pool.release(first_slot)
    assert pool.acquire() == first_slot
    pool.release(first_slot)
    pool.release(second_slot)


def test_vae_encode_stage_reports_capacity_without_raising() -> None:
    stage = vae_stage.LingBotWorldFastVAEEncodeStage.__new__(vae_stage.LingBotWorldFastVAEEncodeStage)
    stage._cache_registry = {}
    stage._cache_pool = _VAECachePool(capacity=1, layout={}, device=torch.device("cpu"))
    image = torch.zeros(3, 4, 4)

    assert stage.initialize_cache(1, image) is True
    assert stage.initialize_cache(2, image) is False
    assert stage.release_cache(1) is True
    assert stage.initialize_cache(2, image) is True
    assert stage.release_cache(2) is True


def test_vae_decode_stage_reports_capacity_without_raising() -> None:
    stage = vae_stage.LingBotWorldFastVAEDecodeStage.__new__(vae_stage.LingBotWorldFastVAEDecodeStage)
    stage._cache_registry = {}
    stage._cache_pool = _VAECachePool(capacity=1, layout={}, device=torch.device("cpu"))

    assert stage.initialize_cache(1) is True
    assert stage.initialize_cache(2) is False
    assert stage.release_cache(1) is True
    assert stage.initialize_cache(2) is True
    assert stage.release_cache(2) is True


def test_vae_decode_parallelization_converts_decoder_only() -> None:
    stage = vae_stage.LingBotWorldFastVAEDecodeStage.__new__(vae_stage.LingBotWorldFastVAEDecodeStage)
    decoder = object()
    stage.vae = SimpleNamespace(model=SimpleNamespace(decoder=decoder))

    with (
        patch.object(vae_stage, "_enable_spatial_parallel_decode") as enable_spatial,
        patch.object(vae_stage, "_convert_conv3d_to_channels_last_3d") as convert_channels_last,
    ):
        stage.parallel_models()

    enable_spatial.assert_called_once_with(stage.vae)
    convert_channels_last.assert_called_once_with(decoder)


def test_vae_encode_stage_encodes_bounded_prefix_once_and_repeats_tail() -> None:
    stage = vae_stage.LingBotWorldFastVAEEncodeStage.__new__(vae_stage.LingBotWorldFastVAEEncodeStage)
    stage.device = torch.device("cpu")
    stage.torch_dtype = torch.float32
    stage.vae = _RecordingEncoder()
    stage._cache_registry = {}
    stage._cache_pool = None
    assert stage.initialize_cache(1, torch.ones(3, 2, 2)) is True

    encode = vae_stage.LingBotWorldFastVAEEncodeStage.encode_condition_chunk.__wrapped__
    first = encode(stage, 1, 0, 5, 4, 2, 2)
    tail = encode(stage, 1, 4, 5, 4, 2, 2)

    assert stage.vae.frame_counts == [61]
    assert first.shape == (1, 20, 4, 2, 2)
    assert torch.equal(first[0, :4, 0], torch.ones(4, 2, 2))
    assert torch.count_nonzero(first[0, :4, 1:]) == 0
    assert torch.count_nonzero(tail[0, :4]) == 0
    assert torch.equal(tail[0, 4:], torch.full((16, 4, 2, 2), 15.0))
