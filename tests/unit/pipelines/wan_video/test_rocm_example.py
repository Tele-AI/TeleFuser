from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

from telefuser.core.config import AttnImplType, WeightOffloadType


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example_path() -> Path:
    return Path(__file__).resolve().parents[4] / "examples/wan_video/wan21_1_3b_text_to_video_rocm.py"


def test_rocm_example_defaults_to_sdpa_and_eager() -> None:
    module = _load_module(_example_path())

    assert module.PPL_CONFIG["attn_impl"] == AttnImplType.TORCH_SDPA
    assert module.PPL_CONFIG["enable_vfi"] is False


def test_rocm_example_uses_overlapping_two_tile_vae_geometry() -> None:
    module = _load_module(_example_path())

    tile_size = module.PPL_CONFIG["vae_tile_size"]
    tile_stride = module.PPL_CONFIG["vae_tile_stride"]

    assert tile_size == (60, 62)
    assert tile_stride == (30, 54)
    # Zero overlap (tile_size == tile_stride) breaks the tiled-decode border mask.
    assert all(s > d for s, d in zip(tile_size, tile_stride))


def test_rocm_example_get_pipeline_loads_official_layout() -> None:
    module = _load_module(_example_path())
    pipeline = MagicMock()

    with (
        patch.object(module, "ModuleManager") as manager_cls,
        patch.object(module, "Wan21VideoPipeline", return_value=pipeline),
    ):
        result = module.get_pipeline(1, "/models/Wan2.1-T2V-1.3B")

    assert result is pipeline
    manager = manager_cls.return_value
    loaded = [call.args[0] for call in manager.load_models.call_args_list]
    assert loaded == [
        ["/models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"],
        [["/models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"]],
        ["/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"],
    ]
    for call in manager.load_models.call_args_list:
        assert call.kwargs["low_cpu_mem_usage"] is True
    pipeline.init.assert_called_once()

    pipe_config = pipeline.init.call_args.args[1]
    assert pipe_config.dit_config.attention_config.attn_impl == AttnImplType.TORCH_SDPA
    assert pipe_config.dit_config.compile_config.enabled is False
    assert pipe_config.enable_clip_stage is False
    assert pipe_config.enable_vfi is False
    assert pipe_config.text_encoding_config.offload_config.offload_type == WeightOffloadType.MODEL_CPU_OFFLOAD
    assert pipe_config.text_encoding_config.offload_config.pin_cpu_memory is False


def test_rocm_example_multi_gpu_enables_denoising_parallel() -> None:
    module = _load_module(_example_path())
    pipeline = MagicMock()

    with (
        patch.object(module, "ModuleManager"),
        patch.object(module, "Wan21VideoPipeline", return_value=pipeline),
    ):
        module.get_pipeline(2, "/models/Wan2.1-T2V-1.3B")

    pipe_config = pipeline.init.call_args.args[1]
    assert pipe_config.enable_denoising_parallel is True
    assert pipe_config.dit_config.parallel_config.device_ids == [0, 1]
    assert pipe_config.dit_config.parallel_config.cfg_degree == 2
    assert pipe_config.dit_config.parallel_config.sp_ulysses_degree == 1
