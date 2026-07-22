from __future__ import annotations

import torch

from telefuser.pipelines.lingbot_video.refiner import LingBotVideoRefinerStage


class _EncodeStage:
    def encode(self, values: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        del generator
        return values[:, :1]


class _DecodeStage:
    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _DenoiseStage:
    def predict_noise_with_cfg(self, latents: torch.Tensor, *_: object) -> torch.Tensor:
        return torch.ones_like(latents)


class _Scheduler:
    sigma_max = 1.0
    sigma_min = 0.0
    timesteps = torch.tensor([1])

    def set_timesteps(self, *_: object, **__: object) -> None:
        return None

    def step(self, prediction: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        del timestep
        return sample - prediction


def test_refiner_reinjects_clean_first_frame_after_scheduler_step() -> None:
    stage = LingBotVideoRefinerStage(
        denoising_stage=_DenoiseStage(),
        vae_encode_stage=_EncodeStage(),
        vae_decode_stage=_DecodeStage(),
        scheduler=_Scheduler(),
    )
    video = torch.zeros(1, 3, 2, 1, 1)
    first_frame = torch.ones(1, 3, 1, 1)

    result = stage.refine(
        video,
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        None,
        None,
        num_inference_steps=1,
        guidance_scale=1.0,
        shift=1.0,
        t_thresh=0.5,
        tail_steps=0,
        clean_first_frame=first_frame,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(result[:, :, :1], torch.ones_like(result[:, :, :1]))


def test_refiner_accepts_captured_noise() -> None:
    stage = LingBotVideoRefinerStage(
        denoising_stage=_DenoiseStage(),
        vae_encode_stage=_EncodeStage(),
        vae_decode_stage=_DecodeStage(),
        scheduler=_Scheduler(),
    )
    video = torch.zeros(1, 3, 2, 1, 1)
    noise = torch.zeros(1, 1, 2, 1, 1)

    result = stage.refine(
        video,
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        None,
        None,
        num_inference_steps=1,
        guidance_scale=1.0,
        shift=1.0,
        t_thresh=0.5,
        tail_steps=0,
        noise=noise,
    )

    assert torch.equal(result, -torch.ones_like(result))
