from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from telefuser.core.config import AttentionConfig, CompileConfig, ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.lingbot_world_fast.denoising import LingBotWorldFastDenoisingStage


def test_denoising_stage_parallel_models_enables_ulysses_and_fsdp() -> None:
    dit = MagicMock()
    dit.get_fsdp_module_names.return_value = ["blocks"]
    parallel_config = ParallelConfig(
        device_ids=[0, 1, 2, 3],
        sp_ulysses_degree=4,
        enable_fsdp=True,
    )
    runtime_config = ModelRuntimeConfig(
        device_type="cuda",
        device_id=0,
        torch_dtype=torch.bfloat16,
        attention_config=AttentionConfig(),
        parallel_config=parallel_config,
    )
    module_manager = MagicMock()
    module_manager.fetch_module.return_value = dit
    stage = LingBotWorldFastDenoisingStage("denoise", module_manager, runtime_config)
    device_mesh = MagicMock()
    fsdp_model = MagicMock()

    with (
        patch(
            "telefuser.pipelines.lingbot_world_fast.denoising.create_device_mesh_from_config",
            return_value=device_mesh,
        ) as create_mesh,
        patch(
            "telefuser.pipelines.lingbot_world_fast.denoising.shard_model",
            return_value=fsdp_model,
        ) as shard,
    ):
        stage.parallel_models()

    create_mesh.assert_called_once_with(parallel_config)
    dit.enable_usp.assert_called_once_with(device_mesh)
    shard.assert_called_once_with(
        module=dit,
        device_id=stage.device,
        wrap_module_names=["blocks"],
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    assert stage.dit is fsdp_model
    assert stage.onload_models_flag is True


def test_denoising_stage_parallel_models_without_fsdp_keeps_model() -> None:
    dit = MagicMock()
    parallel_config = ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2)
    runtime_config = ModelRuntimeConfig(
        device_type="cuda",
        device_id=0,
        parallel_config=parallel_config,
    )
    module_manager = MagicMock()
    module_manager.fetch_module.return_value = dit
    stage = LingBotWorldFastDenoisingStage("denoise", module_manager, runtime_config)
    device_mesh = MagicMock()

    with (
        patch(
            "telefuser.pipelines.lingbot_world_fast.denoising.create_device_mesh_from_config",
            return_value=device_mesh,
        ),
        patch("telefuser.pipelines.lingbot_world_fast.denoising.shard_model") as shard,
    ):
        stage.parallel_models()

    dit.enable_usp.assert_called_once_with(device_mesh)
    shard.assert_not_called()
    assert stage.dit is dit


def test_denoising_stage_rejects_uneven_ulysses_head_partition() -> None:
    stage = LingBotWorldFastDenoisingStage.__new__(LingBotWorldFastDenoisingStage)
    stage.device = torch.device("cpu")
    stage.torch_dtype = torch.float32
    stage.dit = SimpleNamespace(dim=80, num_heads=40, num_layers=1, device_mesh=MagicMock())

    with (
        patch(
            "telefuser.pipelines.lingbot_world_fast.denoising.get_ulysses_world_size",
            return_value=3,
        ),
        pytest.raises(ValueError, match="divisible"),
    ):
        stage._init_self_kv_cache(batch_size=1, kv_size=1)


def test_denoising_stage_parallel_models_compiles_after_fsdp() -> None:
    dit = MagicMock()
    dit.get_fsdp_module_names.return_value = ["blocks"]
    parallel_config = ParallelConfig(
        device_ids=[0, 1, 2, 3],
        sp_ulysses_degree=4,
        enable_fsdp=True,
    )
    compile_config = CompileConfig(enabled=True, backend="eager")
    runtime_config = ModelRuntimeConfig(
        device_type="cuda",
        device_id=0,
        attention_config=AttentionConfig(),
        parallel_config=parallel_config,
        compile_config=compile_config,
    )
    module_manager = MagicMock()
    module_manager.fetch_module.return_value = dit
    stage = LingBotWorldFastDenoisingStage("denoise", module_manager, runtime_config)
    fsdp_model = MagicMock()
    compiled_model = MagicMock()

    with (
        patch("telefuser.pipelines.lingbot_world_fast.denoising.create_device_mesh_from_config"),
        patch("telefuser.pipelines.lingbot_world_fast.denoising.shard_model", return_value=fsdp_model),
        patch("telefuser.pipelines.lingbot_world_fast.denoising.torch.compile", return_value=compiled_model) as compile,
    ):
        stage.parallel_models()

    compile.assert_called_once_with(fsdp_model, **compile_config.get_compile_kwargs())
    assert stage.dit is compiled_model
