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


class _DecodeMetricsStage:
    def __init__(self) -> None:
        self.states: list[object] = []

    def decode_chunks(self, latents: torch.Tensor, states: list[object]) -> torch.Tensor:
        self.states = list(states)
        return latents

    @staticmethod
    def last_decode_metrics() -> dict[str, int]:
        return {
            "taew_decode_items": 2,
            "taew_decode_batch_size": 1,
            "taew_decode_invocations": 2,
            "taew_decode_mode": 2,
        }


def test_generate_next_blocks_surfaces_effective_taew_decode_metrics() -> None:
    pipeline = ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.config = SimpleNamespace(height=32, width=32)

    def denoise(noise: torch.Tensor, *_: object) -> torch.Tensor:
        return noise

    pipeline.denoise_stage = SimpleNamespace(
        dit=SimpleNamespace(use_relative_rope=True),
        _denoise_block=denoise,
    )
    decode_stage = _DecodeMetricsStage()
    pipeline.taew_decode_stage = decode_stage
    pipeline.tensor2video = lambda decoded: [object() for _ in range(decoded.shape[1])]

    def make_session(session_id: str) -> ABotWorldInteractiveSession:
        return ABotWorldInteractiveSession(
            session_id=session_id,
            prompt_emb=torch.ones(1, 1),
            first_frame_latent=torch.ones(1, 1, 1, 1, 1),
            self_cache=[],
            cross_cache=[],
            scheduler=object(),
            generator=torch.Generator(device="cpu").manual_seed(3),
            taew_decode_state=object(),
        )

    first = make_session("first")
    second = make_session("second")
    pipeline._interactive_sessions = {"first": first, "second": second}

    output = pipeline.generate_next_blocks([first, second], [{"W": True}, {"D": True}])

    assert [len(frames) for frames in output] == [3, 3]
    assert decode_stage.states == [first.taew_decode_state, second.taew_decode_state]
    assert pipeline.last_stage_metrics()["batch_size"] == 2
    assert pipeline.last_stage_metrics()["taew_decode_items"] == 2
    assert pipeline.last_stage_metrics()["taew_decode_batch_size"] == 1
    assert pipeline.last_stage_metrics()["taew_decode_invocations"] == 2
    assert pipeline.last_stage_metrics()["taew_decode_mode"] == 2
