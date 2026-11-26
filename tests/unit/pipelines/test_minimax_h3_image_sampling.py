"""Sampling contracts independent of model weights and CUDA kernels."""

import json
from types import SimpleNamespace

import pytest
import torch

try:
    from telefuser.core.config import AttentionConfig, ModelRuntimeConfig
    from telefuser.pipelines.minimax_h3_image.data import build_image_packed_sequence, get_image_canvas_size
    from telefuser.pipelines.minimax_h3_image.stages import MiniMaxH3ImageDenoisingStage
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
    """Verify h3 velocity direction fixed heads and reference freeze."""
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
    """Verify image layout preserves vision tags and separate reference grids."""
    pos, tags, images, text, nref = build_image_packed_sequence(torch.tensor([1, 0, 1]), [(4, 6)], 8, 4, (1, 2, 2), 0)
    assert nref == 6 and len(images) == 14 and len(text) == 3
    assert tags[:3].tolist() == [1, 0, 1]
    assert pos.dtype == torch.float64
    assert torch.all(pos[images[:6], 0] == 3) and torch.all(pos[images[6:], 0] == 4)
    assert get_image_canvas_size(512, 512, 2048**2, 32) == (2048, 2048)


def test_h3_optimized_attention_uses_block_aligned_separate_padding() -> None:
    """Verify h3 optimized attention uses block aligned separate padding."""
    model = ConstantImageVelocity()
    config = ModelRuntimeConfig(device_type="cpu", attention_config=AttentionConfig.sol_attention())
    stage = MiniMaxH3ImageDenoisingStage(Manager(model), config, [1.0, 0.96, 0.88, 0.75, 0.53, 0.23, 0.0])
    text = {"hidden_states": torch.randn(3, 4), "token_tags": torch.ones(3, dtype=torch.long)}
    stage.denoise(text, [], 64, 64, torch.Generator().manual_seed(31).get_state())
    call = model.inputs[0]
    assert call["x"].shape[1] == 64
    assert call["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, 7, 64]
    assert call["img_pos_info"]["position_ids"].tolist() == [3, 4, 5, 6]


def test_h3_image_output_quantizes_by_truncation() -> None:
    """Verify h3 image output quantizes by truncation."""
    from telefuser.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline

    pipeline = MiniMaxH3ImagePipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.text_stage = SimpleNamespace(encode_image_prompt=lambda *args: {})
    pipeline.vae_stage = SimpleNamespace(decode_image=lambda *args: torch.full((1, 3, 32, 32), 0.5))
    pipeline.denoising_stage = SimpleNamespace(denoise=lambda *args: torch.zeros(1, 2, 2, 2))
    image = pipeline("A red fox.", height=32, width=32, min_generation_area=1024)
    assert image.getpixel((0, 0)) == (127, 127, 127)


def test_decoder_graph_rejects_weight_offload() -> None:
    """Reject a graph configuration whose weight addresses change between requests."""
    from telefuser.core.config import OffloadConfig, WeightOffloadType
    from telefuser.pipelines.minimax_h3_image import MiniMaxH3ImagePipelineConfig

    with pytest.raises(ValueError, match="resident VAE weights"):
        MiniMaxH3ImagePipelineConfig(
            processor_path="processor",
            pdd_config_path="pdd.json",
            scheduler_config_path="scheduler.json",
            vae_cuda_graph=True,
            vae_config=ModelRuntimeConfig(
                offload_config=OffloadConfig(offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD)
            ),
        )
