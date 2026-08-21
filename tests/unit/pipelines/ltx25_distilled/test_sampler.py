"""LTX-2.5 distilled sampler contract tests."""

from __future__ import annotations

import pytest
import torch

from telefuser.models.ltx25.sampler import (
    LTX25_STAGE1_DISTILLED_SIGMAS,
    LTX25_STAGE2_DISTILLED_SIGMAS,
    LTX25EulerAncestralStep,
    ancestral_noise_generator,
    distilled_sigmas,
    uses_ancestral_stage1_sampler,
)


def test_distilled_schedules_are_frozen_upstream_values() -> None:
    assert tuple(distilled_sigmas(1).tolist()) == pytest.approx(LTX25_STAGE1_DISTILLED_SIGMAS)
    assert tuple(distilled_sigmas(2).tolist()) == pytest.approx(LTX25_STAGE2_DISTILLED_SIGMAS)


def test_ancestral_selection_starts_at_ltx_25() -> None:
    assert not uses_ancestral_stage1_sampler("2.4")
    assert uses_ancestral_stage1_sampler("2.5")
    assert uses_ancestral_stage1_sampler("2.5-rc1")


def test_ancestral_step_matches_rectified_flow_equation() -> None:
    sample = torch.tensor([2.0, -1.0])
    denoised = torch.tensor([1.0, 3.0])
    noise = torch.tensor([0.5, -0.25])
    sigmas = torch.tensor([1.0, 0.5])
    actual = LTX25EulerAncestralStep().step(sample, denoised, sigmas, 0, noise)

    expected = 0.25 * sample + 0.75 * denoised
    expected = (0.5 / 0.75) * expected + noise * torch.sqrt(torch.tensor(0.5**2 - 0.25**2 * 0.5**2 / 0.75**2))
    torch.testing.assert_close(actual, expected)


def test_terminal_step_returns_denoised_without_noise() -> None:
    result = LTX25EulerAncestralStep().step(torch.ones(2), torch.tensor([3.0, 4.0]), torch.tensor([0.5, 0.0]), 0)
    torch.testing.assert_close(result, torch.tensor([3.0, 4.0]))


def test_ancestral_generator_is_offset_from_initial_seed() -> None:
    actual = torch.randn(4, generator=ancestral_noise_generator(7, "cpu"))
    expected = torch.randn(4, generator=torch.Generator().manual_seed(10_007))
    torch.testing.assert_close(actual, expected)
