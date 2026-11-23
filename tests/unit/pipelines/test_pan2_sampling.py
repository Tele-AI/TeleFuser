"""Sampling contracts independent of model weights and CUDA kernels."""

import json
from types import SimpleNamespace

import pytest
import torch

try:
    from telefuser.core.config import AttentionConfig, ModelRuntimeConfig
    from telefuser.pipelines.pan2.data import get_pan2_default_negative_prompt
    from telefuser.pipelines.pan2.stages import PAN2DenoisingStage
except ImportError as exc:
    pytest.skip(f"Optional model dependencies unavailable: {exc}", allow_module_level=True)


class Manager:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def fetch_module(self, name: str) -> torch.nn.Module:
        return self.model


class ConstantVideoVelocity(torch.nn.Module):
    config = SimpleNamespace(out_channels=2)

    def set_attention_config(self, config: object) -> None:
        pass

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> SimpleNamespace:
        assert hidden_states.dtype == torch.bfloat16
        # Distinguish conditional/unconditional calls and preserve odd text lengths.
        value = float(kwargs["encoder_hidden_states"][0, 0, 0])
        return SimpleNamespace(sample=torch.full_like(hidden_states[:, :2], value))


def test_pan2_cfg_first_frame_and_fp32_accumulation(tmp_path) -> None:
    (tmp_path / "scheduler_config.json").write_text(json.dumps({"shift": 9.0, "num_train_timesteps": 1000}))
    model = ConstantVideoVelocity()
    stage = PAN2DenoisingStage(
        Manager(model), ModelRuntimeConfig(device_type="cpu"), str(tmp_path / "scheduler_config.json")
    )
    generator = torch.Generator().manual_seed(71)
    initial = torch.randn((1, 2, 2, 4, 4), generator=generator)
    output = stage.denoise(
        torch.full((1, 5, 4), 2.0), torch.full((1, 3, 4), 1.0), torch.ones(1, 2, 1, 4, 4), 64, 64, 5, 3, 2.5, 71
    )
    assert output.dtype == torch.float32
    torch.testing.assert_close(output[:, :, 0], initial[:, :, 0] - 2.0)
    torch.testing.assert_close(output[:, :, 1:], initial[:, :, 1:] - 3.5)


def test_pan2_negative_prompt_duration_is_request_specific() -> None:
    a = get_pan2_default_negative_prompt(121, 24)
    b = get_pan2_default_negative_prompt(241, 24)
    assert '"time_range_sec":"0 to 5"' in a
    assert '"time_range_sec":"0 to 10"' in b
    assert '"time_range_sec":"0 to 10"' not in get_pan2_default_negative_prompt(121, 24)


class CapturingVAE(torch.nn.Module):
    config = SimpleNamespace(latents_mean=(0.1234567, -0.2345678), latents_std=(0.2345678, 0.3456789))

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.pixels = pixels
        return torch.ones(1, 2, 1, 2, 2, dtype=torch.bfloat16)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.latents = latents
        return torch.full((1, 3, 1, 32, 32), -0.3333, dtype=torch.bfloat16)


def test_pan2_vae_preprocess_and_denormalization_cast_order() -> None:
    from PIL import Image
    from diffusers.image_processor import VaeImageProcessor

    from telefuser.pipelines.pan2.stages import PAN2VAEStage

    model = CapturingVAE()
    stage = PAN2VAEStage(Manager(model), ModelRuntimeConfig(device_type="cpu"))
    image = Image.new("RGB", (32, 32), (1, 128, 200))
    stage.encode_image(image, 32, 32)
    expected_pixels = VaeImageProcessor(vae_scale_factor=16).preprocess(image).unsqueeze(2).to(torch.bfloat16)
    torch.testing.assert_close(model.pixels, expected_pixels, rtol=0, atol=0)
    latent = torch.full((1, 2, 1, 2, 2), 0.713567, dtype=torch.float32)
    output = stage.decode_video(latent)
    mean = torch.tensor(model.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(model.config.latents_std).view(1, -1, 1, 1, 1)
    torch.testing.assert_close(model.latents, (latent * std + mean).to(torch.bfloat16), rtol=0, atol=0)
    expected = (
        VaeImageProcessor()
        .postprocess(torch.full((1, 3, 32, 32), -0.3333, dtype=torch.bfloat16), output_type="pt")
        .float()
        .unsqueeze(2)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
