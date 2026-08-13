from __future__ import annotations

from types import SimpleNamespace

import torch

from telefuser.models.wan22_video_vae import Wan22VideoVAEStreamingDecodeState
from telefuser.pipelines.abot_world.interactive import (
    ABotWorldInteractivePipeline,
    ABotWorldInteractiveSession,
    ABotWorldSessionLifecycle,
)


def test_session_snapshot_round_trip_preserves_causal_and_rng_state() -> None:
    source = ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32)
    source.denoise_stage = SimpleNamespace(_scheduler=lambda: object())
    generator = torch.Generator(device="cpu").manual_seed(123)
    session = ABotWorldInteractiveSession(
        session_id="migrating",
        prompt_emb=torch.tensor([1.0]),
        first_frame_latent=torch.tensor([2.0]),
        self_cache=[
            {
                "k": torch.tensor([[[[3.0]]]]),
                "v": torch.tensor([[[[4.0]]]]),
                "global_end_index": torch.tensor([12]),
                "local_end_index": torch.tensor([6]),
            }
        ],
        cross_cache=[
            {
                "k": torch.tensor([[[[5.0]]]]),
                "v": torch.tensor([[[[6.0]]]]),
                "is_init": True,
                "sequence_length": 1,
            }
        ],
        scheduler=object(),
        generator=generator,
        vae_decode_state=Wan22VideoVAEStreamingDecodeState(feat_cache=[torch.tensor([7.0])]),
        next_latent_frame=12,
        emitted_frames=45,
        ownership_epoch=4,
    )
    source._interactive_sessions[session.session_id] = session
    generator_state = generator.get_state()
    expected_next_random = torch.randn(1, generator=generator)
    generator.set_state(generator_state)

    snapshot = source.snapshot_interactive_session(session)
    source.close_interactive_session(session)
    target = ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32)
    target.denoise_stage = SimpleNamespace(_scheduler=lambda: object())
    restored = target.restore_interactive_snapshot(snapshot, owner_worker_id="gpu-1")

    assert restored.lifecycle == ABotWorldSessionLifecycle.READY
    assert restored.owner_worker_id == "gpu-1"
    assert restored.ownership_epoch == 5
    assert restored.next_latent_frame == 12
    assert restored.emitted_frames == 45
    assert restored.self_cache[0]["k"].item() == 3
    assert restored.cross_cache[0]["v"].item() == 6
    assert restored.vae_decode_state.feat_cache[0].item() == 7
    assert torch.equal(torch.randn(1, generator=restored.generator), expected_next_random)
