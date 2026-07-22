from __future__ import annotations

import torch
import torch.nn as nn

from telefuser.core.config import ModelRuntimeConfig
from telefuser.pipelines.lingbot_video.data import DEFAULT_NEGATIVE_PROMPT, LingBotVideoRequest
from telefuser.pipelines.lingbot_video.denoising import LingBotVideoDenoisingStage, transformer_timestep
from telefuser.pipelines.lingbot_video.pipeline import LingBotVideoPipeline, LingBotVideoPipelineConfig


class _TextStage:
    def prepare_ti2v_vlm_image(self, pixel_values: torch.Tensor) -> object:
        return object()

    def encode(self, caption: str, images: object | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        del caption, images
        return torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.long)


class _DenoisingStage:
    device = torch.device("cpu")

    def predict_noise_with_cfg(self, latents: torch.Tensor, *_: object) -> torch.Tensor:
        return torch.ones_like(latents)


class _Scheduler:
    timesteps = torch.tensor([2, 1])

    def set_timesteps(self, *_: object, **__: object) -> None:
        return None

    def step(self, prediction: torch.Tensor, timestep: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        del timestep
        return latent - prediction


def test_pipeline_runs_t2v_sampling_with_injected_runtime() -> None:
    pipeline = LingBotVideoPipeline(device="cpu")
    pipeline.init(None, LingBotVideoPipelineConfig(num_inference_steps=2))
    pipeline.set_runtime(text_stage=_TextStage(), denoising_stage=_DenoisingStage(), scheduler=_Scheduler())

    generation = pipeline.generate(LingBotVideoRequest(caption="{}", height=16, width=16, num_frames=5), decode=False)

    assert generation.output.shape == (1, 16, 2, 2, 2)
    assert not generation.prompt_conditions.has_visual_condition
    assert torch.equal(generation.prompt_conditions.positive_prompt_embeds, torch.zeros(1, 1, 4))
    assert pipeline(LingBotVideoRequest(caption="{}", height=16, width=16, num_frames=5), decode=False).shape == (
        1,
        16,
        2,
        2,
        2,
    )


class _DtypeProbeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedder = nn.Linear(1, 1, bias=False)

    def forward(
        self, latents: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        del timestep, mask
        return (latents + embeds.mean()).to(torch.bfloat16)


def test_denoising_stage_returns_fp32_scheduler_prediction() -> None:
    stage = LingBotVideoDenoisingStage(
        "denoising", _DtypeProbeTransformer(), ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32)
    )
    latents = torch.zeros(1, 1, 1, 1, 1)
    positive = torch.full((1, 1, 1), 3.0)
    negative = torch.full((1, 1, 1), 1.0)

    prediction = stage.predict_noise_with_cfg(latents, torch.tensor([1]), positive, negative, guidance_scale=2.0)

    assert prediction.dtype is torch.float32
    assert torch.equal(prediction, torch.full_like(latents, 5.0))


class _RecordingTextStage(_TextStage):
    def __init__(self) -> None:
        self.images: list[object | None] = []

    def encode(self, caption: str, images: object | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.images.append(images)
        return super().encode(caption, images)


def test_pipeline_uses_source_negative_caption_when_not_overridden() -> None:
    class _CapturingTextStage(_TextStage):
        def __init__(self) -> None:
            self.captions: list[str] = []

        def encode(self, caption: str, images: object | None = None) -> tuple[torch.Tensor, torch.Tensor]:
            self.captions.append(caption)
            return super().encode(caption, images)

    text_stage = _CapturingTextStage()
    pipeline = LingBotVideoPipeline(device="cpu")
    pipeline.init(None, LingBotVideoPipelineConfig(num_inference_steps=2))
    pipeline.set_runtime(text_stage=text_stage, denoising_stage=_DenoisingStage(), scheduler=_Scheduler())

    pipeline.generate(LingBotVideoRequest(caption="{}", height=16, width=16, num_frames=5), decode=False)

    assert text_stage.captions == ["{}", DEFAULT_NEGATIVE_PROMPT]


class _VAEEncodeStage:
    def __init__(self) -> None:
        self.pixel_values: torch.Tensor | None = None

    def encode(self, pixel_values: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        del generator
        self.pixel_values = pixel_values
        return torch.zeros(1, 16, 1, 2, 2)


def test_pipeline_ti2v_reuses_one_visual_condition_for_both_cfg_branches() -> None:
    text_stage = _RecordingTextStage()
    vae_encode_stage = _VAEEncodeStage()
    pipeline = LingBotVideoPipeline(device="cpu")
    pipeline.init(None, LingBotVideoPipelineConfig(num_inference_steps=2))
    pipeline.set_runtime(
        text_stage=text_stage,
        denoising_stage=_DenoisingStage(),
        scheduler=_Scheduler(),
        vae_encode_stage=vae_encode_stage,
    )

    generation = pipeline.generate(
        LingBotVideoRequest(caption="{}", height=16, width=16, num_frames=5, image=torch.full((1, 3, 8, 8), 128.0)),
        decode=False,
    )

    assert generation.prompt_conditions.has_visual_condition
    assert len(text_stage.images) == 2
    assert all(images is not None for images in text_stage.images)
    assert vae_encode_stage.pixel_values is not None
    assert vae_encode_stage.pixel_values.shape == (1, 3, 1, 16, 16)
    assert torch.equal(generation.output[:, :, :1], torch.zeros_like(generation.output[:, :, :1]))


def test_transformer_timestep_preserves_upstream_bfloat16_rounding() -> None:
    timestep = torch.tensor([999], dtype=torch.int64)

    actual = transformer_timestep(timestep, torch.bfloat16)

    expected = ((timestep.float() / 1000).bfloat16() * 1000).float()
    assert torch.equal(actual, expected)


class _OffloadStage:
    def __init__(self) -> None:
        self.calls = 0
        self.onload_models_flag = True

    def offload_models(self) -> None:
        self.calls += 1


def test_pipeline_releases_stages_before_a_separate_refiner_is_loaded() -> None:
    text_stage = _OffloadStage()
    denoising_stage = _OffloadStage()
    vae_encode_stage = _OffloadStage()
    vae_decode_stage = _OffloadStage()
    pipeline = LingBotVideoPipeline(device="cpu")
    pipeline.set_runtime(
        text_stage=text_stage,
        denoising_stage=denoising_stage,
        scheduler=_Scheduler(),
        vae_encode_stage=vae_encode_stage,
        vae_decode_stage=vae_decode_stage,
    )

    pipeline.release_gpu_resources()

    for stage in (text_stage, denoising_stage, vae_encode_stage, vae_decode_stage):
        assert stage.calls == 1
        assert not stage.onload_models_flag
