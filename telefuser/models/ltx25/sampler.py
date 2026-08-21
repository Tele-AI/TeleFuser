"""Exact LTX-2.5 distilled sampler constants and ancestral Euler step."""

from __future__ import annotations

import torch

from .checkpoint import parse_model_version

LTX25_STAGE1_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
LTX25_STAGE2_DISTILLED_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
ANCESTRAL_NOISE_SEED_OFFSET = 10_000


def distilled_sigmas(stage: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Create the upstream distilled sigma schedule as float32."""
    if stage == 1:
        values = LTX25_STAGE1_DISTILLED_SIGMAS
    elif stage == 2:
        values = LTX25_STAGE2_DISTILLED_SIGMAS
    else:
        raise ValueError(f"stage must be 1 or 2, got {stage}")
    return torch.tensor(values, dtype=torch.float32, device=device)


def uses_ancestral_stage1_sampler(model_version: str | None) -> bool:
    """Return whether a checkpoint generation uses LTX-2.5 ancestral stage 1."""
    return parse_model_version(model_version) >= (2, 5)


class LTX25EulerAncestralStep:
    """Upstream LTX-2.5 rectified-flow ancestral Euler update."""

    def __init__(self, eta: float = 1.0, s_noise: float = 1.0) -> None:
        self.eta = eta
        self.s_noise = s_noise

    def step(
        self,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one LTX rectified-flow ancestral Euler step."""
        sigma = sigmas[step_index].to(torch.float32)
        sigma_next = sigmas[step_index + 1].to(torch.float32)
        if bool(sigma_next == 0):
            return denoised_sample.to(sample.dtype)
        if self.eta > 0 and noise is None:
            raise ValueError("LTX25EulerAncestralStep requires noise when eta > 0")

        sigma_down = sigma_next * (1.0 + (sigma_next / sigma - 1.0) * self.eta)
        ratio = sigma_down / sigma
        result = ratio * sample.float() + (1.0 - ratio) * denoised_sample.float()
        if self.eta > 0:
            alpha_next = 1.0 - sigma_next
            alpha_down = 1.0 - sigma_down
            renoise = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0).sqrt()
            result = (alpha_next / alpha_down) * result + noise.float() * self.s_noise * renoise
        return result.to(sample.dtype)


def ancestral_noise_generator(seed: int, device: torch.device | str) -> torch.Generator:
    """Create the independent upstream generator used for stage-1 ancestral noise."""
    return torch.Generator(device=device).manual_seed(seed + ANCESTRAL_NOISE_SEED_OFFSET)
