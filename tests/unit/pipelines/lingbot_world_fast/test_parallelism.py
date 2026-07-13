from unittest.mock import MagicMock, patch

import torch

from telefuser.core.config import AttentionConfig, ModelRuntimeConfig, ParallelConfig
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
    stage = LingBotWorldFastDenoisingStage("denoise", dit, runtime_config)
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
    stage = LingBotWorldFastDenoisingStage("denoise", dit, runtime_config)
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
