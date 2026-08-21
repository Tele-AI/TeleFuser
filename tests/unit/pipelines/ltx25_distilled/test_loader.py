"""LTX-2.5 model-pack loading contracts."""

from __future__ import annotations

from pathlib import Path

import torch

from telefuser.core.config import WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx25 import LTX25ModelPaths
from telefuser.pipelines.ltx25_distilled import loader
from telefuser.pipelines.ltx25_distilled.pipeline import build_ltx25_distilled_config


class _Module(torch.nn.Module):
    pass


def test_model_pack_loader_registers_every_component_with_module_manager(monkeypatch) -> None:
    paths = LTX25ModelPaths(
        *(Path(name) for name in ("transformer", "text", "diff", "conv", "audio", "up", "duration"))
    )
    monkeypatch.setattr(LTX25ModelPaths, "from_model_root", staticmethod(lambda model_root: paths))

    single_checkpoint_classes = (
        loader.LTX25Gemma4TextEncoder,
        loader.LTX25DurationHead,
        loader.LTX25VideoEncoder,
        loader.LTX25AVTransformer,
        loader.LTX25SpatialUpsampler,
        loader.DiffusionVideoDecoder,
    )
    for model_class in single_checkpoint_classes:
        monkeypatch.setattr(model_class, "from_checkpoint", staticmethod(lambda *args, **kwargs: _Module()))
    monkeypatch.setattr(
        loader.LTX25EmbeddingsProcessor,
        "from_checkpoints",
        staticmethod(lambda *args, **kwargs: _Module()),
    )
    monkeypatch.setattr(loader, "load_video_latent_statistics", lambda path: _Module())
    monkeypatch.setattr(
        loader,
        "load_ltx25_audio_decoder_and_vocoder",
        lambda *args, **kwargs: (_Module(), _Module()),
    )

    manager = ModuleManager(device="cpu", torch_dtype=torch.bfloat16)
    actual_paths = loader.load_ltx25_distilled_modules(manager, "unused", video_vae="diff")

    assert actual_paths is paths
    assert manager.module_names == [
        "ltx25_gemma4",
        "ltx25_embeddings_processor",
        "ltx25_duration_head",
        "ltx25_video_encoder",
        "ltx25_transformer",
        "ltx25_spatial_upsampler",
        "ltx25_video_latent_statistics",
        "ltx25_video_decoder",
        "ltx25_audio_decoder",
        "ltx25_vocoder",
    ]
    assert all(isinstance(module, _Module) for module in manager.modules)


def test_cpu_offload_config_uses_async_denoiser_and_model_offload_for_other_stages() -> None:
    config = build_ltx25_distilled_config("cuda", torch.bfloat16, "diff", "cpu")

    assert config.denoising_config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD
    assert config.text_encoding_config.offload_config.offload_type == WeightOffloadType.MODEL_CPU_OFFLOAD
    assert config.video_decoding_config.offload_config.offload_type == WeightOffloadType.MODEL_CPU_OFFLOAD
