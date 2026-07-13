from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from telefuser.models.wan_video_vae import WanVideoVAE, WanVideoVAEStreamingDecodeState
from telefuser.pipelines.lingbot_world_fast.denoising import LingBotWorldFastDenoisingStage


def _cache_stage() -> LingBotWorldFastDenoisingStage:
    stage = LingBotWorldFastDenoisingStage.__new__(LingBotWorldFastDenoisingStage)
    stage.device = torch.device("cpu")
    stage._cache_registry = {}
    stage._init_self_kv_cache = MagicMock(side_effect=lambda *_args: [{"owner": object()}])
    stage._init_crossattn_cache = MagicMock(side_effect=lambda *_args: [{"owner": object()}])
    return stage


def _initialize_cache(stage: LingBotWorldFastDenoisingStage, cache_handle: int) -> None:
    generator_state = torch.Generator(device="cpu").manual_seed(cache_handle).get_state().tolist()
    LingBotWorldFastDenoisingStage.initialize_cache.__wrapped__(
        stage,
        cache_handle=cache_handle,
        batch_size=1,
        kv_size=4,
        max_sequence_length=8,
        sample_shift=10.0,
        generator_state=generator_state,
    )


def test_worker_cache_registry_isolates_handles_and_releases_idempotently() -> None:
    stage = _cache_stage()

    _initialize_cache(stage, 11)
    _initialize_cache(stage, 12)

    assert stage.list_cache_handles() == (11, 12)
    assert stage.has_cache(11)
    assert stage._cache_registry[11] is not stage._cache_registry[12]
    assert stage._cache_registry[11].self_kv_cache is not stage._cache_registry[12].self_kv_cache

    with pytest.raises(ValueError, match="already registered"):
        _initialize_cache(stage, 11)

    assert stage.release_cache(11) is True
    assert stage.release_cache(11) is False
    assert stage.list_cache_handles() == (12,)


def test_worker_rejects_unknown_cache_handle() -> None:
    stage = _cache_stage()
    latent = torch.zeros(1, 1, 1, 1, 1)

    with pytest.raises(KeyError, match="Unknown cache handle 99"):
        LingBotWorldFastDenoisingStage.denoise_and_update_cache.__wrapped__(
            stage,
            cache_handle=99,
            latent_chunk=latent,
            condition_chunk=latent,
            prompt_emb=torch.zeros(1, 1, 1),
            control_chunk=None,
            current_start=0,
            max_attention_size=1,
        )


class _RecordingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cache_ids: list[int] = []

    def forward(self, x, feat_cache, feat_idx):
        self.cache_ids.append(id(feat_cache))
        feat_cache.append(float(x.flatten()[0]))
        feat_idx[0] += 1
        return x


def test_vae_streaming_decode_state_is_session_scoped() -> None:
    decoder = _RecordingDecoder()
    vae = SimpleNamespace(
        model=SimpleNamespace(conv2=lambda value: value, decoder=decoder),
        scale=[0.0, 1.0],
        z_dim=1,
        _feat_cache=[],
        _feat_idx=[0],
    )
    first = WanVideoVAEStreamingDecodeState()
    second = WanVideoVAEStreamingDecodeState()

    WanVideoVAE.cached_decode_withflag(
        vae,
        torch.ones(1, 1, 1, 1),
        device=torch.device("cpu"),
        is_first_clip=True,
        is_last_clip=False,
        decode_state=first,
    )
    WanVideoVAE.cached_decode_withflag(
        vae,
        torch.full((1, 1, 1, 1), 2.0),
        device=torch.device("cpu"),
        is_first_clip=True,
        is_last_clip=False,
        decode_state=second,
    )

    assert first.feat_cache == [1.0]
    assert second.feat_cache == [2.0]
    assert first.feat_cache is not second.feat_cache
    assert vae._feat_cache == []
    assert decoder.cache_ids == [id(first.feat_cache), id(second.feat_cache)]
