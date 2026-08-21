"""LTX-2.5 video-decoding stage contracts."""

from __future__ import annotations

import torch

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.ltx25_distilled.video_decoding import LTX25VideoDecodingStage


class _ConvVideoDecoder(torch.nn.Module):
    def decode(self, latent: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        del generator
        return (latent[:, :3],)


def test_conv_vae_stage_converts_video_chunks_to_public_rgb_layout() -> None:
    manager = ModuleManager(device="cpu", torch_dtype=torch.float32)
    manager.add_module(_ConvVideoDecoder(), "ltx25_video_decoder")
    stage = LTX25VideoDecodingStage(
        manager,
        ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
        video_vae="conv",
    )

    chunks = stage.decode(torch.zeros(1, 128, 2, 8, 12), torch.Generator(device="cpu"))

    assert chunks[0].shape == (2, 8, 12, 3)
    assert torch.all(chunks[0] == 0.5)
