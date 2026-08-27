from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from telefuser.models import wan22_video_vae


class _StatefulRecordingDecoder(nn.Module):
    def forward(self, x, feat_cache, feat_idx, first_chunk: bool = False):
        del first_chunk
        feat_idx[0] += 1
        feat_cache[0] = x.detach().clone()
        return x


def test_cached_decode_uses_explicit_session_owned_state(monkeypatch) -> None:
    fake_vae = SimpleNamespace(
        model=SimpleNamespace(conv2=lambda value: value, decoder=_StatefulRecordingDecoder()),
        z_dim=1,
        _feat_cache=[],
        _feat_idx=[0],
        _get_scale_on_device=lambda _device, _dtype: [torch.zeros(1), torch.ones(1)],
    )
    monkeypatch.setattr(wan22_video_vae, "_count_conv3d", lambda _decoder: 1)
    monkeypatch.setattr(wan22_video_vae, "unpatchify", lambda video, patch_size: video)
    state = wan22_video_vae.Wan22VideoVAEStreamingDecodeState()

    wan22_video_vae.Wan22VideoVAE.cached_decode_withflag(
        fake_vae,
        torch.ones(1, 1, 1, 1, 1),
        torch.device("cpu"),
        True,
        False,
        state,
    )

    assert len(state.feat_cache) == 1
    assert state.feat_cache[0].item() == 1
    assert fake_vae._feat_cache == []


def test_cached_decode_batches_and_scatters_temporal_state(monkeypatch) -> None:
    fake_vae = SimpleNamespace(
        model=SimpleNamespace(conv2=lambda value: value, decoder=_StatefulRecordingDecoder()),
        z_dim=1,
        _get_scale_on_device=lambda _device, _dtype: [torch.zeros(1), torch.ones(1)],
    )
    monkeypatch.setattr(wan22_video_vae, "_count_conv3d", lambda _decoder: 1)
    monkeypatch.setattr(wan22_video_vae, "unpatchify", lambda video, patch_size: video)
    states = [
        wan22_video_vae.Wan22VideoVAEStreamingDecodeState(),
        wan22_video_vae.Wan22VideoVAEStreamingDecodeState(),
    ]

    output = wan22_video_vae.Wan22VideoVAE.cached_decode_batch_withflag(
        fake_vae,
        torch.tensor([1.0, 2.0]).view(2, 1, 1, 1, 1),
        torch.device("cpu"),
        True,
        False,
        states,
    )

    assert output.shape == (2, 1, 1, 1, 1)
    assert states[0].feat_cache[0].item() == 1
    assert states[1].feat_cache[0].item() == 2
    states[0].feat_cache[0].zero_()
    assert states[1].feat_cache[0].item() == 2
