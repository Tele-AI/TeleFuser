from collections import deque
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


def _initialize_cache(
    stage: LingBotWorldFastDenoisingStage,
    cache_handle: int,
    prompt_emb: torch.Tensor | None = None,
) -> bool:
    generator_state = torch.Generator(device="cpu").manual_seed(cache_handle).get_state().tolist()
    noise_generator_state = torch.Generator(device="cpu").manual_seed(cache_handle + 100).get_state().tolist()
    return LingBotWorldFastDenoisingStage.initialize_cache.__wrapped__(
        stage,
        cache_handle=cache_handle,
        batch_size=1,
        kv_size=4,
        max_sequence_length=8,
        sample_shift=10.0,
        generator_state=generator_state,
        noise_generator_state=noise_generator_state,
        noise_shape=(1, 16, 1, 1, 1),
        prompt_emb=prompt_emb,
    )


def test_worker_cache_retains_session_prompt_embedding() -> None:
    stage = _cache_stage()
    prompt_emb = torch.randn(1, 8, 16)

    assert _initialize_cache(stage, 11, prompt_emb=prompt_emb) is True

    assert stage._cache_registry[11].prompt_emb is prompt_emb


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


def test_preallocated_cache_pool_exhausts_and_reuses_slots() -> None:
    stage = LingBotWorldFastDenoisingStage.__new__(LingBotWorldFastDenoisingStage)
    stage.device = torch.device("cpu")
    stage.torch_dtype = torch.float32
    stage.dit = SimpleNamespace(dim=8, num_heads=2, num_layers=2, device_mesh=None)
    stage._cache_registry = {}
    stage._cache_pool = None

    profile = stage.configure_cache_pool(capacity=2, batch_size=1, kv_size=4, max_sequence_length=8)
    _initialize_cache(stage, 11)
    _initialize_cache(stage, 12)

    assert profile.capacity == 2
    assert profile.allocated_bytes == 2 * profile.bytes_per_session
    assert profile.bytes_per_session == stage.estimate_session_cache_bytes(1, 4, 8)
    assert {stage._cache_registry[handle].pool_slot for handle in (11, 12)} == {0, 1}
    assert _initialize_cache(stage, 13) is False
    assert not stage.has_cache(13)

    released_slot = stage._cache_registry[11].pool_slot
    stage._cache_registry[11].self_kv_cache[0]["global_end_index"].fill_(7)
    assert stage.release_cache(11) is True
    _initialize_cache(stage, 13)

    assert stage._cache_registry[13].pool_slot == released_slot
    assert stage._cache_registry[13].self_kv_cache[0]["global_end_index"].item() == 0


def test_worker_rejects_unknown_cache_handle() -> None:
    stage = _cache_stage()
    latent = torch.zeros(1, 1, 1, 1, 1)

    with pytest.raises(KeyError, match="Unknown cache handle 99"):
        LingBotWorldFastDenoisingStage.denoise_and_update_cache.__wrapped__(
            stage,
            cache_handle=99,
            condition_chunk=latent,
            prompt_emb=torch.zeros(1, 1, 1),
            control_chunk=None,
            current_start=0,
            max_attention_size=1,
        )


def test_worker_owned_noise_rng_advances_deterministically() -> None:
    first_stage = _cache_stage()
    second_stage = _cache_stage()
    _initialize_cache(first_stage, 11)
    _initialize_cache(second_stage, 11)
    state = first_stage._cache_registry[11]
    expected_generator = torch.Generator(device="cpu")
    expected_generator.set_state(state.noise_generator.get_state())
    expected_first = torch.randn(state.noise_shape, generator=expected_generator, dtype=torch.float32)
    expected_second = torch.randn(state.noise_shape, generator=expected_generator, dtype=torch.float32)
    expected_third = torch.randn(state.noise_shape, generator=expected_generator, dtype=torch.float32)

    actual_first = first_stage._next_noise_chunk(state)
    replicated_first = second_stage._next_noise_chunk(second_stage._cache_registry[11])
    assert first_stage.advance_noise(11) is True
    assert second_stage.advance_noise(11) is True
    actual_third = first_stage._next_noise_chunk(state)
    replicated_third = second_stage._next_noise_chunk(second_stage._cache_registry[11])

    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(replicated_first, expected_first)
    assert not torch.equal(actual_third, expected_second)
    torch.testing.assert_close(actual_third, expected_third)
    torch.testing.assert_close(replicated_third, expected_third)


class _SessionCacheTestDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_embedding = nn.Sequential(nn.Linear(2, 2))


def test_session_input_caches_invalidate_on_prompt_and_control_mutation() -> None:
    stage = LingBotWorldFastDenoisingStage.__new__(LingBotWorldFastDenoisingStage)
    stage.dit = _SessionCacheTestDiT()
    state = SimpleNamespace(
        projected_context_key=None,
        prepared_control_key=None,
        prepared_control_is_sharded=False,
        session_input_cache={},
    )
    prompt = torch.zeros(1, 2)
    control = torch.zeros(1, 2)

    stage._prepare_session_inputs(state, 9, prompt, control)
    cached_context = torch.ones(1, 2)
    cached_control = (torch.ones(1, 2), ())
    state.session_input_cache.update(projected_context=cached_context, prepared_control=cached_control)
    stage._prepare_session_inputs(state, 9, prompt, control)

    assert state.session_input_cache["projected_context"] is cached_context
    assert state.session_input_cache["prepared_control"] is cached_control

    prompt.add_(1)
    stage._prepare_session_inputs(state, 9, prompt, control)

    assert "projected_context" not in state.session_input_cache
    assert state.session_input_cache["prepared_control"] is cached_control

    control.add_(1)
    stage._prepare_session_inputs(state, 9, prompt, control)

    assert "prepared_control" not in state.session_input_cache


def test_colocated_decode_consumes_worker_local_latent() -> None:
    stage = LingBotWorldFastDenoisingStage.__new__(LingBotWorldFastDenoisingStage)
    decoder = MagicMock()
    decoder.decode_chunk.return_value = torch.ones(1)
    stage._vae_decode_stage = decoder
    local_latent = torch.ones(1, 2, 3)
    next_local_latent = torch.full((1, 2, 3), 2.0)
    stage._pending_vae_decode_latents = {9: deque([local_latent, next_local_latent])}

    output = stage.decode_chunk(9, torch.empty(0, dtype=torch.uint8), True, False)
    stage.decode_chunk(9, torch.empty(0, dtype=torch.uint8), False, True)

    assert torch.equal(output, torch.ones(1))
    assert decoder.decode_chunk.call_args_list == [
        ((9, local_latent, True, False),),
        ((9, next_local_latent, False, True),),
    ]
    assert stage._pending_vae_decode_latents == {}


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
