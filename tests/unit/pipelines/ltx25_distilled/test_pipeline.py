"""ModuleManager and stage-composition contracts for LTX-2.5."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from examples.ltx25_distilled import ltx25_distilled_i2v_h100 as i2v_example
from telefuser.core.config import AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.ltx25_distilled.pipeline import (
    LTX25DistilledPipeline,
    build_ltx25_distilled_config,
)


class _TextEncoder(torch.nn.Module):
    def encode(self, prompts: list[str]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        batch = len(prompts)
        hidden = torch.ones(batch, 2, 4)
        return (hidden,), torch.ones(batch, 2, dtype=torch.long), torch.ones(batch, 2, dtype=torch.long)


class _Embeddings(torch.nn.Module):
    def forward(self, hidden: tuple[torch.Tensor, ...], mask: torch.Tensor) -> SimpleNamespace:
        del mask
        return SimpleNamespace(video_encoding=hidden[0], audio_encoding=hidden[0])


class _DurationHead(torch.nn.Module):
    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        del video, audio
        return torch.tensor([1.0])


class _Transformer(torch.nn.Module):
    def set_attention_config(self, attention_config: object) -> None:
        self.attention_config = attention_config

    def forward(
        self, video: object, audio: object, perturbations: object
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        del perturbations
        return (
            torch.zeros_like(video.latent) if video is not None else None,  # type: ignore[union-attr]
            torch.zeros_like(audio.latent) if audio is not None else None,  # type: ignore[union-attr]
        )


class _Upsampler(torch.nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)


class _Statistics(torch.nn.Module):
    def un_normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def normalize(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


class _VideoDecoder(torch.nn.Module):
    def decode_video(self, latent: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
        del generator
        return (latent,)


class _Identity(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


def _module_manager() -> ModuleManager:
    manager = ModuleManager(device="cpu", torch_dtype=torch.bfloat16)
    modules = {
        "ltx25_gemma4": _TextEncoder(),
        "ltx25_embeddings_processor": _Embeddings(),
        "ltx25_duration_head": _DurationHead(),
        "ltx25_video_encoder": _Identity(),
        "ltx25_transformer": _Transformer(),
        "ltx25_spatial_upsampler": _Upsampler(),
        "ltx25_video_latent_statistics": _Statistics(),
        "ltx25_video_decoder": _VideoDecoder(),
        "ltx25_audio_decoder": _Identity(),
        "ltx25_vocoder": _Identity(),
    }
    for name, module in modules.items():
        manager.add_module(module, name, path="unused")
    return manager


def test_pipeline_composes_six_manager_backed_stages() -> None:
    pipeline = LTX25DistilledPipeline(device="cpu", torch_dtype=torch.bfloat16)
    pipeline.init(
        _module_manager(),
        build_ltx25_distilled_config("cpu", torch.bfloat16, "diff", "none"),
    )

    assert [stage.name for stage in pipeline._get_stages()] == [
        "ltx25_text_encoding",
        "ltx25_video_conditioning",
        "ltx25_denoising",
        "ltx25_latent_upsampling",
        "ltx25_video_decoding",
        "ltx25_audio_decoding",
    ]
    assert len(pipeline._model_info) == 10


def test_pipeline_runs_two_stage_contract_with_manager_owned_modules() -> None:
    pipeline = LTX25DistilledPipeline(device="cpu", torch_dtype=torch.bfloat16)
    pipeline.init(
        _module_manager(),
        build_ltx25_distilled_config("cpu", torch.bfloat16, "diff", "none"),
    )

    result = pipeline(
        "A test prompt",
        seed=7,
        height=256,
        width=384,
        num_frames=9,
        frame_rate=24.0,
    )

    assert result.video_latent.shape == (1, 128, 2, 8, 12)
    assert result.audio_latent.shape == (1, 8, 9, 16)
    assert result.video_chunks == (result.video_latent,)
    assert result.audio.shape == result.audio_latent.squeeze(0).shape
    assert result.audio.dtype == torch.float32


def test_pipeline_resolves_auto_duration_through_text_stage() -> None:
    pipeline = LTX25DistilledPipeline(device="cpu", torch_dtype=torch.bfloat16)
    pipeline.init(
        _module_manager(),
        build_ltx25_distilled_config("cpu", torch.bfloat16, "diff", "none"),
    )

    result = pipeline("A test prompt", seed=7, height=256, width=384)

    assert result.num_frames == 25
    assert result.video_latent.shape == (1, 128, 4, 8, 12)


def test_build_config_supports_ulysses_and_attention_selection() -> None:
    config = build_ltx25_distilled_config(
        "cuda",
        torch.bfloat16,
        "diff",
        "cpu",
        parallelism=4,
        attn_impl=AttnImplType.TORCH_SDPA,
    )

    denoising = config.denoising_config
    assert denoising.attention_config.attn_impl == AttnImplType.TORCH_SDPA
    assert denoising.parallel_config.device_ids == [0, 1, 2, 3]
    assert denoising.parallel_config.sp_ulysses_degree == 4
    assert denoising.parallel_config.enable_fsdp
    assert denoising.offload_config.offload_type == WeightOffloadType.NO_CPU_OFFLOAD
    assert config.text_encoding_config.offload_config.offload_type == WeightOffloadType.MODEL_CPU_OFFLOAD


def test_build_config_rejects_sparse_attention_and_invalid_sp_degree() -> None:
    for parallelism in (0, 3, 33):
        try:
            build_ltx25_distilled_config("cuda", torch.bfloat16, "diff", "none", parallelism=parallelism)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parallelism={parallelism} should be rejected")

    try:
        build_ltx25_distilled_config(
            "cuda",
            torch.bfloat16,
            "diff",
            "none",
            attn_impl=AttnImplType.RADIAL_ATTN,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("sparse attention should be rejected")


def test_i2v_run_with_file_accepts_service_image_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    opened_paths = []
    image = SimpleNamespace(convert=lambda mode: mode)

    def stop_run(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("stop")

    monkeypatch.setattr(i2v_example.Image, "open", lambda path: opened_paths.append(path) or image)
    monkeypatch.setattr(i2v_example, "run", stop_run)

    with pytest.raises(RuntimeError, match="stop"):
        i2v_example.run_with_file(
            object(),
            prompt="test",
            output_path="result.mp4",
            first_image_path="service-input.png",
        )

    assert opened_paths == ["service-input.png"]
