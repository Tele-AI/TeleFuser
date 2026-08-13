from __future__ import annotations

from types import SimpleNamespace

import torch

from telefuser.models.wan22_video_vae import Wan22VideoVAEStreamingDecodeState
from telefuser.pipelines.abot_world.interactive import (
    ABotWorldInteractivePipeline,
    ABotWorldInteractiveSession,
    ABotWorldSessionLifecycle,
)


def _session(session_id: str) -> ABotWorldInteractiveSession:
    return ABotWorldInteractiveSession(
        session_id=session_id,
        prompt_emb=torch.ones(1),
        first_frame_latent=torch.ones(1),
        self_cache=[{"k": torch.ones(1)}],
        cross_cache=[{"k": torch.ones(1)}],
        scheduler=object(),
        generator=torch.Generator(device="cpu"),
        vae_decode_state=Wan22VideoVAEStreamingDecodeState(feat_cache=[torch.ones(1)]),
    )


def test_close_interactive_session_clears_only_target_state() -> None:
    pipeline = ABotWorldInteractivePipeline(device="cpu")
    pipeline.vae_stage = SimpleNamespace(vae=SimpleNamespace(_feat_cache=[torch.ones(1)], _feat_idx=[3]))
    first = _session("first")
    second = _session("second")
    pipeline._interactive_sessions = {"first": first, "second": second}

    pipeline.close_interactive_session(first)

    assert first.closed
    assert first.lifecycle == ABotWorldSessionLifecycle.CLOSED
    assert first.self_cache == []
    assert first.cross_cache == []
    assert first.vae_decode_state.feat_cache == []
    assert second.self_cache
    assert second.cross_cache
    assert second.vae_decode_state.feat_cache
    assert pipeline._interactive_sessions == {"second": second}
    # Session cleanup must not mutate the legacy model-owned cache.
    assert pipeline.vae_stage.vae._feat_idx == [3]


def test_cache_collation_and_scatter_preserve_session_isolation() -> None:
    first = _session("first")
    second = _session("second")
    first.self_cache = [
        {
            "k": torch.tensor([[[[1.0]]]]),
            "v": torch.tensor([[[[2.0]]]]),
            "global_end_index": torch.tensor([3]),
            "local_end_index": torch.tensor([3]),
        }
    ]
    second.self_cache = [
        {
            "k": torch.tensor([[[[4.0]]]]),
            "v": torch.tensor([[[[5.0]]]]),
            "global_end_index": torch.tensor([12]),
            "local_end_index": torch.tensor([3]),
        }
    ]

    collated = ABotWorldInteractivePipeline._collate_caches([first, second], "self_cache")
    assert collated[0]["k"].shape[0] == 2
    collated[0]["k"].add_(10)
    ABotWorldInteractivePipeline._scatter_caches([first, second], "self_cache", collated)

    assert first.self_cache[0]["k"].item() == 11
    assert second.self_cache[0]["k"].item() == 14
    first.self_cache[0]["k"].zero_()
    assert second.self_cache[0]["k"].item() == 14
