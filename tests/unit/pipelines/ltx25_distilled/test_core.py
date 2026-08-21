"""Unit tests for isolated LTX-2.5 distilled sampling helpers."""

from __future__ import annotations

from dataclasses import replace

import torch

from telefuser.models.ltx25.sampler import LTX25EulerAncestralStep
from telefuser.pipelines.ltx25_distilled.core import (
    LTX25SimpleDenoiser,
    euler_ancestral_denoising_loop,
    euler_denoising_loop,
    modality_from_latent_state,
)
from telefuser.pipelines.ltx25_distilled.latent import LatentState


class _IdentityDenoiserModel:
    def __call__(
        self, video: object, audio: object, perturbations: object
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        del perturbations
        return (
            video.latent if video is not None else None,  # type: ignore[union-attr]
            audio.latent if audio is not None else None,  # type: ignore[union-attr]
        )


class _ConstantDenoiserModel:
    def __call__(
        self, video: object, audio: object, perturbations: object
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        del perturbations
        return (
            torch.ones_like(video.latent) if video is not None else None,  # type: ignore[union-attr]
            torch.ones_like(audio.latent) if audio is not None else None,  # type: ignore[union-attr]
        )


def _state(latent: torch.Tensor, *, denoise_mask: torch.Tensor | None = None) -> LatentState:
    mask = denoise_mask if denoise_mask is not None else torch.ones_like(latent[..., :1])
    return LatentState(
        latent=latent,
        denoise_mask=mask,
        positions=torch.zeros(latent.shape[0], 1, latent.shape[1], 2),
        clean_latent=torch.full_like(latent, 9.0),
    )


def test_modality_uses_masked_per_token_timesteps() -> None:
    state = _state(torch.zeros(1, 2, 3), denoise_mask=torch.tensor([[[1.0], [0.0]]]))

    modality = modality_from_latent_state(state, torch.zeros(1, 4, 5), torch.tensor([0.75]))

    torch.testing.assert_close(modality.timesteps, torch.tensor([[[0.75], [0.0]]]))
    assert modality.positions is state.positions


def test_euler_loop_preserves_conditioning_tokens() -> None:
    state = _state(torch.zeros(1, 2, 1), denoise_mask=torch.tensor([[[1.0], [0.0]]]))
    denoiser = LTX25SimpleDenoiser(torch.zeros(1, 1, 1), None)

    result, audio = euler_denoising_loop(
        torch.tensor([0.5, 0.0]),
        state,
        None,
        _ConstantDenoiserModel(),  # type: ignore[arg-type]
        denoiser,
        model_dtype=torch.float32,
    )

    assert audio is None
    assert result is not None
    torch.testing.assert_close(result.latent, torch.tensor([[[1.0], [9.0]]]))


def test_euler_loop_matches_upstream_scalar_sigma_velocity_rounding() -> None:
    torch.manual_seed(23)
    state = _state(torch.randn(1, 256, 1, dtype=torch.bfloat16))
    sigmas = torch.tensor([0.725, 0.421875])
    denoiser = LTX25SimpleDenoiser(torch.zeros(1, 1, 1), None)

    result, _ = euler_denoising_loop(
        sigmas,
        state,
        None,
        _ConstantDenoiserModel(),  # type: ignore[arg-type]
        denoiser,
        model_dtype=torch.bfloat16,
    )
    expected_velocity = ((state.latent.float() - torch.ones_like(state.latent).float()) / sigmas[0].item()).to(
        torch.bfloat16
    )
    expected = (state.latent.float() + expected_velocity.float() * (sigmas[1] - sigmas[0])).to(torch.bfloat16)

    assert result is not None
    torch.testing.assert_close(result.latent, expected)


def test_noising_preserves_clean_conditioning_tokens() -> None:
    from telefuser.pipelines.ltx25_distilled.pipeline import _noised_state

    state = _state(torch.tensor([[[2.0], [0.0]]]), denoise_mask=torch.tensor([[[0.0], [1.0]]]))
    noised = _noised_state(state, 1.0, torch.Generator().manual_seed(17))

    torch.testing.assert_close(noised.latent[:, :1], state.clean_latent[:, :1])
    assert not torch.equal(noised.latent[:, 1:], state.clean_latent[:, 1:])


def test_ancestral_loop_draws_video_noise_before_audio_noise() -> None:
    video_state = _state(torch.zeros(1, 2, 1))
    audio_state = _state(torch.zeros(1, 3, 1))
    sigmas = torch.tensor([0.8, 0.4, 0.0])
    seed = 123
    denoiser = LTX25SimpleDenoiser(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))

    video_result, audio_result = euler_ancestral_denoising_loop(
        sigmas,
        video_state,
        audio_state,
        _IdentityDenoiserModel(),  # type: ignore[arg-type]
        denoiser,
        noise_seed=seed,
        model_dtype=torch.float32,
    )

    generator = torch.Generator().manual_seed(seed)
    stepper = LTX25EulerAncestralStep()
    expected_video = stepper.step(
        video_state.latent.float(),
        video_state.latent,
        sigmas,
        0,
        torch.randn(video_state.latent.shape, generator=generator),
    )
    expected_audio = stepper.step(
        audio_state.latent.float(),
        audio_state.latent,
        sigmas,
        0,
        torch.randn(audio_state.latent.shape, generator=generator),
    )

    assert video_result is not None
    assert audio_result is not None
    torch.testing.assert_close(video_result.latent, expected_video)
    torch.testing.assert_close(audio_result.latent, expected_audio)
