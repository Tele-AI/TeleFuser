from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from telefuser.models.lingbot_world_fast_dit import _set_cache_index
from telefuser.pipelines.lingbot_world_fast.denoising import LingBotWorldFastDenoisingStage
from telefuser.pipelines.lingbot_world_fast.pipeline import (
    LingBotWorldFastPipeline,
    LingBotWorldFastPipelineConfig,
)
from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig
from telefuser.pipelines.lingbot_world_v2 import LingBotWorldV2Pipeline, LingBotWorldV2PipelineConfig


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
        vae_encode_config=SimpleNamespace(torch_dtype=torch.float32),
    )
    pipeline.dit = SimpleNamespace(
        patch_size=(1, 2, 2),
        dim=8,
        num_heads=2,
        num_layers=1,
    )
    pipeline.denoise_stage = MagicMock()
    pipeline.vae_encode_worker = MagicMock()
    pipeline.vae_decode_worker = MagicMock()
    pipeline._next_cache_handle = 0
    pipeline.encode_prompt = MagicMock(return_value=torch.zeros(1, 4, 8))
    pipeline._prepare_image_tensor = MagicMock(return_value=torch.zeros(3, 16, 16))
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


def test_v1_and_v2_defaults_match_the_shared_source_contract() -> None:
    image = Image.new("RGB", (16, 16))

    assert LingBotWorldFastPipelineConfig().vae_encode_config.torch_dtype == torch.float32
    assert LingBotWorldFastPipelineConfig().vae_decode_config.torch_dtype == torch.float32
    assert LingBotWorldV2PipelineConfig().vae_encode_config.torch_dtype == torch.float32
    assert LingBotWorldV2PipelineConfig().vae_decode_config.torch_dtype == torch.float32
    assert LingBotWorldFastSessionConfig(prompt="v1", image=image).frame_policy == "truncate"


def test_denoising_cache_cursors_use_mutable_scalar_tensors() -> None:
    stage = object.__new__(LingBotWorldFastDenoisingStage)
    stage.dit = SimpleNamespace(dim=8, num_heads=2, num_layers=1, device_mesh=None)
    stage.device = torch.device("cpu")
    stage.torch_dtype = torch.float32

    cache = stage._init_self_kv_cache(batch_size=1, kv_size=4)[0]

    assert cache["global_end_index"].shape == ()
    assert cache["local_end_index"].shape == ()
    assert cache["global_end_index"].dtype == torch.int64
    assert cache["local_end_index"].dtype == torch.int64

    copied_cache = cache.copy()
    _set_cache_index(copied_cache, "global_end_index", 7)
    _set_cache_index(copied_cache, "local_end_index", 11)

    assert cache["global_end_index"].item() == 7
    assert cache["local_end_index"].item() == 11


def test_v1_and_v2_share_source_image_geometry_and_preprocessing() -> None:
    pipeline = LingBotWorldFastPipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.vae_device = torch.device("cpu")
    pipeline.config = SimpleNamespace(vae_encode_config=SimpleNamespace(torch_dtype=torch.float32))
    image = Image.fromarray(np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3), mode="RGB")

    actual = pipeline._prepare_image_tensor(image, height=6, width=8)
    source_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    expected = torch.nn.functional.interpolate(
        source_tensor.sub(0.5).div(0.5).unsqueeze(0),
        size=(6, 8),
        mode="bicubic",
    ).squeeze(0)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert LingBotWorldFastPipeline._best_output_size(1024, 768, 480 * 832) == (720, 544)
    assert LingBotWorldV2Pipeline._best_output_size is LingBotWorldFastPipeline._best_output_size
    assert LingBotWorldV2Pipeline._prepare_image_tensor is LingBotWorldFastPipeline._prepare_image_tensor


@pytest.mark.parametrize(("width", "height"), [(32, 16), (32, 32), (16, 32)])
def test_default_intrinsics_follow_reference_image_geometry(width: int, height: int) -> None:
    pipeline = _build_runtime_pipeline()
    pipeline.config.max_area = width * height

    context = pipeline.control_context(
        LingBotWorldFastSessionConfig(
            prompt="geometry",
            image=Image.new("RGB", (width, height)),
            frame_num=9,
        )
    )

    assert (context.orig_width, context.orig_height) == (width, height)
    torch.testing.assert_close(
        context.intrinsics,
        torch.tensor([float(width), float(width), width * 0.5, height * 0.5]),
    )


def test_explicit_intrinsics_use_their_calibration_size() -> None:
    pipeline = _build_runtime_pipeline()
    intrinsics = [1200.0, 1190.0, 960.0, 540.0]

    context = pipeline.control_context(
        LingBotWorldFastSessionConfig(
            prompt="calibrated",
            image=Image.new("RGB", (1024, 1024)),
            frame_num=9,
            intrinsics=intrinsics,
            intrinsics_width=1920,
            intrinsics_height=1080,
        )
    )

    assert (context.orig_width, context.orig_height) == (1920, 1080)
    torch.testing.assert_close(context.intrinsics, torch.tensor(intrinsics))


def test_intrinsics_calibration_size_must_be_a_positive_pair() -> None:
    pipeline = _build_runtime_pipeline()
    config = LingBotWorldFastSessionConfig(
        prompt="invalid calibration",
        image=Image.new("RGB", (16, 16)),
        frame_num=9,
        intrinsics=[16.0, 16.0, 8.0, 8.0],
        intrinsics_width=16,
    )

    with pytest.raises(ValueError, match="provided together"):
        pipeline.control_context(config)


def test_aligned_81_frame_runtime_has_seven_complete_latent_chunks() -> None:
    pipeline, runtime = _create_runtime(frame_num=81)

    assert runtime.latent_f == 21
    assert runtime.chunk_count == 7
    assert not hasattr(runtime, "noise_generator")
    assert runtime.condition_image is not None
    assert not hasattr(runtime, "noise_chunks")
    assert not hasattr(runtime, "condition_chunks")
    assert runtime.cache_handle == 0
    assert not hasattr(runtime, "self_kv_cache")
    pipeline.denoise_stage.initialize_cache.assert_called_once()


def test_generation_sessions_receive_isolated_worker_cache_handles() -> None:
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
    assert pipeline.denoise_stage.initialize_cache.call_count == 2
    assert pipeline.vae_encode_worker.initialize_cache.call_count == 2
    assert pipeline.vae_decode_worker.initialize_cache.call_count == 2


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


def test_runtime_passes_reproducible_noise_rng_state_to_denoise_actor() -> None:
    first_pipeline, first = _create_runtime(frame_num=21, seed=7)
    repeated_pipeline, repeated = _create_runtime(frame_num=21, seed=7)
    different_pipeline, different = _create_runtime(frame_num=21, seed=8)

    first_state = first_pipeline.denoise_stage.initialize_cache.call_args.kwargs["noise_generator_state"]
    repeated_state = repeated_pipeline.denoise_stage.initialize_cache.call_args.kwargs["noise_generator_state"]
    different_state = different_pipeline.denoise_stage.initialize_cache.call_args.kwargs["noise_generator_state"]

    assert first_state == repeated_state
    assert first_state != different_state
    assert first.cache_handle == repeated.cache_handle == different.cache_handle == 0


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


def test_runtime_truncates_non_aligned_latent_frame_count() -> None:
    _, runtime = _create_runtime(frame_num=13)

    assert runtime.latent_f == 3
    assert runtime.config.frame_num == 9
    assert runtime.chunk_count == 1


def test_strict_frame_policy_rejects_non_aligned_latent_frame_count() -> None:
    pipeline = _build_runtime_pipeline()
    with pytest.raises(ValueError, match="frame_num"):
        pipeline._create_initialized_session(
            LingBotWorldFastSessionConfig(
                prompt="baseline",
                image=Image.new("RGB", (16, 16)),
                frame_num=13,
                chunk_size=3,
                frame_policy="strict",
            )
        )


def test_runtime_rejects_frame_count_smaller_than_first_chunk() -> None:
    with pytest.raises(ValueError, match="frame_num"):
        _create_runtime(frame_num=5)


def test_runtime_rejects_non_positive_chunk_size() -> None:
    pipeline = _build_runtime_pipeline()
    with pytest.raises(ValueError, match="chunk_size"):
        pipeline.control_context(
            LingBotWorldFastSessionConfig(
                prompt="baseline",
                image=Image.new("RGB", (16, 16)),
                chunk_size=0,
                frame_num=9,
            )
        )


def test_control_mode_must_match_the_initialized_pipeline() -> None:
    pipeline = _build_runtime_pipeline()
    with pytest.raises(ValueError, match="does not match"):
        pipeline.control_context(
            LingBotWorldFastSessionConfig(
                prompt="baseline",
                image=Image.new("RGB", (16, 16)),
                control_mode="act",
                frame_num=9,
            )
        )
