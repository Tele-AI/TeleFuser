"""Monolithic LTX-2.5 distilled reference-path contracts."""

from __future__ import annotations

import torch
from PIL import Image

from telefuser.models.ltx25.embeddings import LTX25EmbeddingsProcessorOutput
from telefuser.models.ltx25.sampler import ancestral_noise_generator
from telefuser.pipelines.ltx25_distilled.reference import (
    LTX25DistilledReference,
    LTX25ReferenceComponents,
    LTX25ReferenceImageCondition,
    LTX25ReferenceRequest,
    _LazyCallable,
    _LazyTextEncoder,
    _release_modules,
)


def test_lazy_callable_proxies_instrumentation_attributes() -> None:
    class Model:
        velocity_model = object()

    lazy = _LazyCallable(lambda: Model())

    assert lazy.velocity_model is not None


def test_release_modules_unloads_retained_lazy_callable() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_calls = 0

        def to(self, *args: object, **kwargs: object) -> "Model":
            self.to_calls += 1
            return self

    model = Model()
    lazy = _LazyCallable(lambda: model)
    assert lazy.resolve() is model

    _release_modules(lazy)

    assert model.to_calls == 1
    assert lazy._model is None


def test_lazy_text_encoder_can_remain_resident_without_cpu_offload() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_calls = 0

        def encode(self, prompts: list[str]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
            assert prompts == ["test"]
            return (), torch.empty((1, 0), dtype=torch.long), torch.empty((1, 0), dtype=torch.long)

        def to(self, *args: object, **kwargs: object) -> "Model":
            self.to_calls += 1
            return self

    model = Model()
    loads = 0

    def load() -> Model:
        nonlocal loads
        loads += 1
        return model

    encoder = _LazyTextEncoder(load, release_after_call=False)

    encoder.encode(["test"])
    encoder.encode(["test"])

    assert loads == 1
    assert model.to_calls == 0


class _TextEncoder:
    def encode(self, prompts: list[str]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        assert prompts == ["A test prompt"]
        return (torch.ones((1, 2, 4)),), torch.tensor([[1, 2]]), torch.tensor([[1, 1]])


class _EmbeddingsProcessor:
    def __call__(
        self, hidden_states: tuple[torch.Tensor, ...], attention_mask: torch.Tensor
    ) -> LTX25EmbeddingsProcessorOutput:
        assert len(hidden_states) == 1
        return LTX25EmbeddingsProcessorOutput(
            video_encoding=torch.ones((1, 2, 4)),
            audio_encoding=torch.ones((1, 2, 4)),
            attention_mask=attention_mask,
        )


class _Transformer:
    def __call__(self, video: object, audio: object, perturbations: object) -> tuple[torch.Tensor, torch.Tensor]:
        del perturbations
        return torch.zeros_like(video.latent), torch.zeros_like(audio.latent)  # type: ignore[union-attr]


class _RecordingTransformer(_Transformer):
    def __init__(self) -> None:
        self.video_latents: list[torch.Tensor] = []

    def __call__(self, video: object, audio: object, perturbations: object) -> tuple[torch.Tensor, torch.Tensor]:
        self.video_latents.append(video.latent.detach().clone())  # type: ignore[union-attr]
        return super().__call__(video, audio, perturbations)


class _Upsampler:
    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)


class _IdentityStatistics:
    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def un_normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class _VideoEncoder:
    def __init__(self) -> None:
        self.pixel_shapes: list[torch.Size] = []

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        self.pixel_shapes.append(pixels.shape)
        _, _, _, height, width = pixels.shape
        return torch.ones((1, 128, 1, height // 32, width // 32), dtype=pixels.dtype)


def test_reference_preserves_two_stage_rng_and_latent_contracts() -> None:
    reference = LTX25DistilledReference(
        LTX25ReferenceComponents(
            text_encoder=_TextEncoder(),
            embeddings_processor=_EmbeddingsProcessor(),
            transformer=_Transformer(),
            spatial_upsampler=_Upsampler(),
            latent_statistics=_IdentityStatistics(),
        ),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        capture_prompt_intermediates=True,
    )
    request = LTX25ReferenceRequest(
        prompt="A test prompt",
        seed=42,
        height=256,
        width=384,
        num_frames=9,
        frame_rate=24.0,
    )

    result = reference.generate(request)

    assert result.stage1_video.latent.shape == (1, 128, 2, 4, 6)
    assert result.stage1_audio.latent.shape == (1, 8, 9, 16)
    assert result.stage2_video.latent.shape == (1, 128, 2, 8, 12)
    assert result.stage2_audio.latent.shape == (1, 8, 9, 16)
    assert result.trace.video_context.shape == (1, 2, 4)
    assert torch.equal(result.trace.artifacts["gemma_token_ids"], torch.tensor([[1, 2]]))
    assert torch.equal(result.trace.artifacts["gemma_attention_mask"], torch.tensor([[1, 1]]))
    assert "gemma_hidden_state_0" in result.trace.artifacts
    for step_index in range(3):
        assert f"stage2_step{step_index}_updated_video_latent" in result.trace.artifacts
        assert f"stage2_step{step_index}_updated_audio_latent" in result.trace.artifacts

    initial_generator = torch.Generator().manual_seed(request.seed)
    expected_stage1_video = torch.randn((1, 48, 128), generator=initial_generator, dtype=torch.bfloat16)
    expected_stage1_audio = torch.randn((1, 9, 128), generator=initial_generator, dtype=torch.bfloat16)
    torch.testing.assert_close(result.trace.artifacts["stage1_initial_video_noise"], expected_stage1_video)
    torch.testing.assert_close(result.trace.artifacts["stage1_initial_audio_noise"], expected_stage1_audio)

    expected_stage2_video_noise = torch.randn((1, 192, 128), generator=initial_generator, dtype=torch.bfloat16)
    expected_stage2_audio_noise = torch.randn((1, 9, 128), generator=initial_generator, dtype=torch.bfloat16)
    expected_stage2_video = expected_stage2_video_noise * torch.tensor(0.909375, dtype=torch.bfloat16)
    expected_stage2_audio = expected_stage2_audio_noise * torch.tensor(0.909375, dtype=torch.bfloat16)
    torch.testing.assert_close(result.trace.artifacts["stage2_initial_video_noise"], expected_stage2_video)
    torch.testing.assert_close(result.trace.artifacts["stage2_initial_audio_noise"], expected_stage2_audio)
    assert torch.equal(result.decoder_generator_state, initial_generator.get_state())

    ancestral = ancestral_noise_generator(request.seed, "cpu")
    expected_ancestral_video = torch.randn((1, 48, 128), generator=ancestral, dtype=torch.bfloat16)
    expected_ancestral_audio = torch.randn((1, 9, 128), generator=ancestral, dtype=torch.bfloat16)
    torch.testing.assert_close(result.trace.artifacts["stage1_step0_ancestral_noise_video"], expected_ancestral_video)
    torch.testing.assert_close(result.trace.artifacts["stage1_step0_ancestral_noise_audio"], expected_ancestral_audio)


def test_reference_applies_i2v_conditions_before_each_stage_noising() -> None:
    encoder = _VideoEncoder()
    transformer = _RecordingTransformer()
    reference = LTX25DistilledReference(
        LTX25ReferenceComponents(
            text_encoder=_TextEncoder(),
            embeddings_processor=_EmbeddingsProcessor(),
            transformer=transformer,
            spatial_upsampler=_Upsampler(),
            latent_statistics=_IdentityStatistics(),
            video_encoder=encoder,
        ),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    request = LTX25ReferenceRequest(
        prompt="A test prompt",
        seed=42,
        height=256,
        width=384,
        num_frames=9,
        frame_rate=24.0,
        images=(LTX25ReferenceImageCondition(Image.new("RGB", (4, 4), "white"), crf=0),),
    )

    result = reference.generate(request)

    assert encoder.pixel_shapes == [torch.Size((1, 3, 1, 128, 192)), torch.Size((1, 3, 1, 256, 384))]
    torch.testing.assert_close(
        transformer.video_latents[1][:, :24], torch.ones_like(transformer.video_latents[1][:, :24])
    )
    torch.testing.assert_close(
        result.stage1_video.latent[:, :, 0], torch.ones_like(result.stage1_video.latent[:, :, 0])
    )
    torch.testing.assert_close(
        result.stage2_video.latent[:, :, 0], torch.ones_like(result.stage2_video.latent[:, :, 0])
    )
