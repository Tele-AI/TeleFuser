"""Faithful LTX-2.5 distilled denoising helpers.

This module intentionally owns only the single-GPU sampling mechanics.  Stage
assembly, prompting, and decoding remain separate so the sampler can be tested
without loading the 22B transformer.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from telefuser.models.ltx25.sampler import LTX25EulerAncestralStep
from telefuser.models.ltx25.transformer import LTX25AVTransformer, Modality

from .latent import LatentState


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Apply a batch of scalar sigmas to the corresponding token masks."""
    if sigma.ndim != 1:
        raise ValueError(f"sigma must have shape (batch,), got {tuple(sigma.shape)}")
    return denoise_mask * sigma.view(-1, *([1] * (denoise_mask.ndim - 1)))


def post_process_latent(denoised: torch.Tensor, state: LatentState) -> torch.Tensor:
    """Keep conditioning tokens fixed after a denoiser prediction."""
    return (denoised * state.denoise_mask + state.clean_latent.float() * (1 - state.denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(state: LatentState, context: torch.Tensor, sigma: torch.Tensor) -> Modality:
    """Translate an isolated latent state into the transformer's public input."""
    return Modality(
        latent=state.latent,
        sigma=sigma,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
        keyframes_mask=state.keyframes_mask,
    )


class LTX25SimpleDenoiser:
    """One conditioned transformer call without CFG or perturbations."""

    def __init__(self, video_context: torch.Tensor | None, audio_context: torch.Tensor | None) -> None:
        self.video_context = video_context
        self.audio_context = audio_context

    def __call__(
        self,
        transformer: LTX25AVTransformer,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video_state is None and audio_state is None:
            raise ValueError("At least one latent modality must be provided")
        if video_state is not None and self.video_context is None:
            raise ValueError("video_context is required when video_state is present")
        if audio_state is not None and self.audio_context is None:
            raise ValueError("audio_context is required when audio_state is present")

        sigma = sigmas[step_index]
        video = (
            modality_from_latent_state(video_state, self.video_context, sigma.expand(video_state.latent.shape[0]))
            if video_state is not None
            else None
        )
        audio = (
            modality_from_latent_state(audio_state, self.audio_context, sigma.expand(audio_state.latent.shape[0]))
            if audio_state is not None
            else None
        )
        return transformer(video=video, audio=audio, perturbations=None)


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    transformer: LTX25AVTransformer,
    denoiser: LTX25SimpleDenoiser,
    *,
    model_dtype: torch.dtype,
) -> tuple[LatentState | None, LatentState | None]:
    """Run the deterministic rectified-flow Euler loop used by distilled stage 2."""
    for step_index in range(sigmas.numel() - 1):
        denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_index)
        video_state = _euler_step(video_state, denoised_video, sigmas, step_index, model_dtype)
        audio_state = _euler_step(audio_state, denoised_audio, sigmas, step_index, model_dtype)
    return video_state, audio_state


def euler_ancestral_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    transformer: LTX25AVTransformer,
    denoiser: LTX25SimpleDenoiser,
    *,
    noise_seed: int,
    stepper: LTX25EulerAncestralStep | None = None,
    model_dtype: torch.dtype,
) -> tuple[LatentState | None, LatentState | None]:
    """Run stage-1 ancestral Euler sampling with upstream noise ordering."""
    if video_state is None and audio_state is None:
        raise ValueError("At least one latent modality must be provided")
    stepper = stepper or LTX25EulerAncestralStep()
    present_state = video_state if video_state is not None else audio_state
    generator = torch.Generator(device=present_state.latent.device).manual_seed(noise_seed)

    for step_index in range(sigmas.numel() - 1):
        denoised_video, denoised_audio = denoiser(transformer, video_state, audio_state, sigmas, step_index)
        terminal_step = bool(sigmas[step_index + 1] == 0)
        video_state = _ancestral_step(
            video_state, denoised_video, sigmas, step_index, terminal_step, stepper, generator, model_dtype
        )
        audio_state = _ancestral_step(
            audio_state, denoised_audio, sigmas, step_index, terminal_step, stepper, generator, model_dtype
        )
    return video_state, audio_state


def _euler_step(
    state: LatentState | None,
    denoised: torch.Tensor | None,
    sigmas: torch.Tensor,
    step_index: int,
    model_dtype: torch.dtype,
) -> LatentState | None:
    if state is None or denoised is None:
        return state
    denoised = post_process_latent(denoised, state)
    sigma = sigmas[step_index]
    sigma_next = sigmas[step_index + 1]
    if bool(sigma_next == 0):
        return replace(state, latent=denoised.to(model_dtype))
    # Preserve EulerDiffusionStep's BF16 velocity rounding point.  The ratio
    # form is algebraically equivalent but diverges from the upstream trajectory.
    velocity = ((state.latent.float() - denoised.float()) / sigma.to(torch.float32).item()).to(state.latent.dtype)
    latent = (state.latent.float() + velocity.float() * (sigma_next - sigma)).to(model_dtype)
    return replace(state, latent=latent)


def _ancestral_step(
    state: LatentState | None,
    denoised: torch.Tensor | None,
    sigmas: torch.Tensor,
    step_index: int,
    terminal_step: bool,
    stepper: LTX25EulerAncestralStep,
    generator: torch.Generator,
    model_dtype: torch.dtype,
) -> LatentState | None:
    if state is None or denoised is None:
        return state
    denoised = post_process_latent(denoised, state)
    if terminal_step:
        return replace(state, latent=denoised.to(model_dtype))
    noise = torch.randn(state.latent.shape, generator=generator, dtype=state.latent.dtype, device=state.latent.device)
    latent = stepper.step(
        sample=state.latent.float(),
        denoised_sample=denoised,
        sigmas=sigmas,
        step_index=step_index,
        noise=noise,
    )
    return replace(state, latent=post_process_latent(latent, state).to(model_dtype))


__all__ = [
    "LTX25SimpleDenoiser",
    "euler_ancestral_denoising_loop",
    "euler_denoising_loop",
    "modality_from_latent_state",
    "post_process_latent",
    "timesteps_from_mask",
]
