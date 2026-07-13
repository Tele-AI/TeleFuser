from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from telefuser.pipelines.lingbot_world_fast.denoising import LingBotWorldFastDenoisingStage, LingBotWorldFastTimesteps
from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline
from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig


def _build_runtime_pipeline() -> LingBotWorldFastPipeline:
    pipeline = LingBotWorldFastPipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.text_device = torch.device("cpu")
    pipeline.vae_device = torch.device("cpu")
    pipeline.config = SimpleNamespace(
        control_type="cam",
        max_area=16 * 16,
        orig_height=16,
        orig_width=16,
        local_attn_size=-1,
        sink_size=0,
    )
    pipeline.dit = SimpleNamespace(
        patch_size=(1, 2, 2),
        dim=8,
        num_heads=2,
        num_layers=1,
    )
    pipeline.denoise_stage = MagicMock()
    pipeline._next_cache_handle = 0
    pipeline.timesteps = LingBotWorldFastTimesteps()
    pipeline.encode_prompt = MagicMock(return_value=torch.zeros(1, 4, 8))
    pipeline._prepare_image_tensor = MagicMock(return_value=torch.zeros(3, 16, 16))
    pipeline._encode_condition_video = MagicMock(
        side_effect=lambda _image, frame_num: torch.zeros(17, (frame_num - 1) // 4 + 1, 2, 2)
    )
    return pipeline


def _create_runtime(frame_num: int, seed: int = 42):
    pipeline = _build_runtime_pipeline()
    runtime = pipeline._create_initialized_session(
        LingBotWorldFastSessionConfig(
            prompt="baseline",
            image=Image.new("RGB", (16, 16)),
            frame_num=frame_num,
            chunk_size=3,
            seed=seed,
        )
    )
    return pipeline, runtime


def test_aligned_81_frame_runtime_has_seven_complete_latent_chunks() -> None:
    pipeline, runtime = _create_runtime(frame_num=81)

    assert runtime.latent_f == 21
    assert len(runtime.noise_chunks) == 7
    assert len(runtime.condition_chunks) == 7
    assert all(chunk.shape[2] == 3 for chunk in runtime.noise_chunks)
    assert all(chunk.shape[2] == 3 for chunk in runtime.condition_chunks)
    assert runtime.cache_handle == 0
    assert not hasattr(runtime, "self_kv_cache")
    pipeline.denoise_stage.initialize_cache.assert_called_once()
    pipeline._encode_condition_video.assert_called_once()


def test_generation_sessions_receive_isolated_cache_and_decoder_handles() -> None:
    pipeline = _build_runtime_pipeline()
    config = LingBotWorldFastSessionConfig(
        prompt="baseline",
        image=Image.new("RGB", (16, 16)),
        frame_num=9,
        chunk_size=3,
    )

    first = pipeline._create_initialized_session(config)
    second = pipeline._create_initialized_session(config)

    assert first.cache_handle == 0
    assert second.cache_handle == 1
    assert first.decoder_state is not second.decoder_state
    assert pipeline.denoise_stage.initialize_cache.call_count == 2


def test_cache_initialization_failure_triggers_global_cleanup() -> None:
    pipeline = _build_runtime_pipeline()
    pipeline.denoise_stage.initialize_cache.side_effect = RuntimeError("rank initialization failed")

    with pytest.raises(RuntimeError, match="rank initialization failed"):
        pipeline._create_initialized_session(
            LingBotWorldFastSessionConfig(
                prompt="baseline",
                image=Image.new("RGB", (16, 16)),
                frame_num=9,
            )
        )

    pipeline.denoise_stage.release_cache.assert_called_once_with(0)


def test_runtime_noise_is_reproducible_and_seed_dependent() -> None:
    _, first = _create_runtime(frame_num=21, seed=7)
    _, repeated = _create_runtime(frame_num=21, seed=7)
    _, different = _create_runtime(frame_num=21, seed=8)

    first_noise = torch.cat(first.noise_chunks, dim=2)
    repeated_noise = torch.cat(repeated.noise_chunks, dim=2)
    different_noise = torch.cat(different.noise_chunks, dim=2)

    torch.testing.assert_close(first_noise, repeated_noise)
    assert not torch.equal(first_noise, different_noise)


def test_denoising_generator_state_advances_between_chunks() -> None:
    class Scheduler:
        sigmas = torch.tensor([1.0, 0.0])
        timesteps = torch.tensor([10.0, 0.0])

        @staticmethod
        def add_noise(x0: torch.Tensor, noise: torch.Tensor, _timestep: torch.Tensor) -> torch.Tensor:
            return x0 + noise

    stage = LingBotWorldFastDenoisingStage.__new__(LingBotWorldFastDenoisingStage)
    stage.torch_dtype = torch.float32
    stage.dit = MagicMock(side_effect=lambda **kwargs: torch.zeros_like(kwargs["x"]))
    timesteps = torch.tensor([10.0, 0.0])
    latent = torch.zeros(1, 1, 1, 1, 1)
    generator = torch.Generator(device="cpu").manual_seed(123)

    def denoise(active_generator: torch.Generator) -> torch.Tensor:
        return stage.denoise_chunk(
            latent_chunk=latent,
            condition_chunk=latent,
            prompt_emb=torch.zeros(1, 1, 1),
            timesteps=timesteps,
            scheduler=Scheduler(),
            control_chunk=None,
            self_kv_cache=[],
            crossattn_cache=[],
            current_start=0,
            max_attention_size=1,
            generator=active_generator,
        )

    first = denoise(generator)
    second = denoise(generator)
    repeated = denoise(torch.Generator(device="cpu").manual_seed(123))

    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, second)


def test_final_chunk_marks_runtime_inactive_and_releases_denoise_state() -> None:
    pipeline = LingBotWorldFastPipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.vae_device = torch.device("cpu")
    pipeline.denoise_stage = object()
    pipeline.decode_video_cached = MagicMock(return_value=torch.zeros(1))
    expected_frames = [Image.new("RGB", (8, 8)) for _ in range(9)]
    pipeline.tensor2video = MagicMock(return_value=expected_frames)
    cached_latent = torch.zeros(1, 1, 3, 2, 2)
    runtime = SimpleNamespace(
        current_chunk_index=0,
        noise_chunks=[torch.zeros_like(cached_latent)],
        condition_chunks=[torch.zeros_like(cached_latent)],
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        chunk_size=3,
        frame_tokens=1,
        world_kv_cached_latents={0: cached_latent},
        world_kv_binding=None,
        active=True,
        emitted_frames=0,
    )

    frames = pipeline.generate_next_chunk(runtime, control=torch.zeros(1))

    assert frames == expected_frames
    assert runtime.current_chunk_index == 1
    assert runtime.emitted_frames == 9
    assert runtime.active is False


@pytest.mark.xfail(strict=True, reason="Non-aligned frame counts are silently truncated instead of rejected")
def test_runtime_rejects_non_aligned_frame_count() -> None:
    with pytest.raises(ValueError, match="frame_num"):
        _create_runtime(frame_num=13)


@pytest.mark.xfail(strict=True, reason="Frame counts smaller than one complete chunk are not validated")
def test_runtime_rejects_frame_count_smaller_than_first_chunk() -> None:
    with pytest.raises(ValueError, match="frame_num"):
        _create_runtime(frame_num=5)
