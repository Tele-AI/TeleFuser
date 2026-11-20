"""Sampling contracts independent of expensive model weights and CUDA kernels."""

import json
from types import SimpleNamespace

import pytest
import torch

try:
    from telefuser.core.config import AttentionConfig, ModelRuntimeConfig
    from telefuser.pipelines.minimax_h3_image.data import build_image_packed_sequence, get_image_canvas_size
    from telefuser.pipelines.minimax_h3_image.stages import MiniMaxH3ImageDenoisingStage
    from telefuser.pipelines.pan2.data import get_pan2_default_negative_prompt
    from telefuser.pipelines.pan2.stages import PAN2DenoisingStage
except ImportError as exc:
    pytest.skip(f"Optional model dependencies unavailable: {exc}", allow_module_level=True)


class Manager:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def fetch_module(self, name: str) -> torch.nn.Module:
        return self.model


class ConstantImageVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(latents_dim=2, patch_size=(1, 2, 2))
        self.final_layer = SimpleNamespace(video_out=SimpleNamespace(weight=torch.empty(6, 8, 1)))
        self.inputs = []

    def set_attention_config(self, config: object) -> None:
        pass

    def forward(self, **kwargs: object) -> torch.Tensor:
        self.inputs.append({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})
        count = len(kwargs["img_pos_info"]["position_ids"])
        return torch.full((count, 8), 2.0)


def test_h3_velocity_direction_fixed_heads_and_reference_freeze() -> None:
    model = ConstantImageVelocity()
    stage = MiniMaxH3ImageDenoisingStage(
        Manager(model), ModelRuntimeConfig(device_type="cpu"), [1.0, 0.96, 0.88, 0.75, 0.53, 0.23, 0.0]
    )
    text = {"hidden_states": torch.randn(3, 4), "token_tags": torch.ones(3, dtype=torch.long)}
    reference = torch.randn(1, 2, 4, 4)
    generator = torch.Generator().manual_seed(31)
    initial = torch.randn((1, 2, 1, 4, 4), generator=generator)
    output = stage.denoise(text, [reference], 64, 64, torch.Generator().manual_seed(31).get_state())
    torch.testing.assert_close(output, initial[:, :, 0] + 2.0)
    assert [x["pdd_step_index"] for x in model.inputs] == list(range(6))
    ref_ids = model.inputs[0]["img_pos_info"]["position_ids"][:4]
    frozen = model.inputs[0]["x"][0, ref_ids]
    for call in model.inputs:
        torch.testing.assert_close(call["x"][0, ref_ids], frozen)
        assert torch.all(call["unique_timesteps"][call["inverse_indices"]][ref_ids] == torch.tensor(0.999))
    assert model._static_prompt is None


def test_image_layout_preserves_vision_tags_and_separate_reference_grids() -> None:
    pos, tags, images, text, nref = build_image_packed_sequence(torch.tensor([1, 0, 1]), [(4, 6)], 8, 4, (1, 2, 2), 0)
    assert nref == 6 and len(images) == 14 and len(text) == 3
    assert tags[:3].tolist() == [1, 0, 1]
    assert pos.dtype == torch.float64
    assert torch.all(pos[images[:6], 0] == 3) and torch.all(pos[images[6:], 0] == 4)
    assert get_image_canvas_size(512, 512, 2048**2, 32) == (2048, 2048)


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


def test_h3_optimized_attention_uses_block_aligned_separate_padding() -> None:
    model = ConstantImageVelocity()
    config = ModelRuntimeConfig(device_type="cpu", attention_config=AttentionConfig.sol_attention())
    stage = MiniMaxH3ImageDenoisingStage(Manager(model), config, [1.0, 0.96, 0.88, 0.75, 0.53, 0.23, 0.0])
    text = {"hidden_states": torch.randn(3, 4), "token_tags": torch.ones(3, dtype=torch.long)}
    stage.denoise(text, [], 64, 64, torch.Generator().manual_seed(31).get_state())
    call = model.inputs[0]
    assert call["x"].shape[1] == 64
    assert call["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, 7, 64]
    assert call["img_pos_info"]["position_ids"].tolist() == [3, 4, 5, 6]


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


def test_h3_image_output_quantizes_by_truncation() -> None:
    from telefuser.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline

    pipeline = MiniMaxH3ImagePipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.text_stage = SimpleNamespace(encode_image_prompt=lambda *args: {})
    pipeline.vae_stage = SimpleNamespace(decode_image=lambda *args: torch.full((1, 3, 32, 32), 0.5))
    pipeline.denoising_stage = SimpleNamespace(denoise=lambda *args: torch.zeros(1, 2, 2, 2))
    image = pipeline("A red fox.", height=32, width=32, min_generation_area=1024)
    assert image.getpixel((0, 0)) == (127, 127, 127)
