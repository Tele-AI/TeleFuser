from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_video.vae import (
    LingBotVideoVAEDecodeStage,
    LingBotVideoVAEEncodeStage,
    denormalize_latent,
)


class _Posterior:
    def __init__(self, sample: torch.Tensor) -> None:
        self._sample = sample

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return self._sample


class _VAE(nn.Module):
    config = SimpleNamespace(latents_mean=[1.0, 2.0], latents_std=[2.0, 4.0])

    def encode(self, values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(latent_dist=_Posterior(torch.ones(values.shape[0], 2, 1, 1, 1)))

    def decode(self, latents: torch.Tensor) -> SimpleNamespace:
        assert torch.allclose(latents, torch.ones_like(latents))
        assert latents.is_contiguous(memory_format=torch.channels_last_3d)
        return SimpleNamespace(sample=torch.zeros(latents.shape[0], 3, 1, 1, 1))


def test_vae_stages_apply_checkpoint_normalization() -> None:
    runtime = ModelRuntimeConfig(device_type="cpu")
    vae = _VAE()
    module_manager = ModuleManager(device="cpu", torch_dtype=torch.float32)
    module_manager.add_module(vae, name="vae")
    encoded = LingBotVideoVAEEncodeStage("encode", module_manager, runtime).encode(torch.ones(1, 3, 1, 2, 2))
    decoded = LingBotVideoVAEDecodeStage("decode", module_manager, runtime).decode(encoded)

    assert encoded[:, 0].item() == 0.0
    assert encoded[:, 1].item() == -0.25
    assert torch.allclose(decoded, torch.full_like(decoded, 0.5))


def test_denormalize_preserves_upstream_division_by_inverse_order() -> None:
    latent = torch.tensor([[[[[0.12345679]]]]])
    mean = torch.tensor([0.27182818])
    std = torch.tensor([0.31415927])

    actual = denormalize_latent(latent, mean, std)
    expected = latent / std.reciprocal().reshape(1, 1, 1, 1, 1) + mean.reshape(1, 1, 1, 1, 1)

    assert torch.equal(actual, expected)
