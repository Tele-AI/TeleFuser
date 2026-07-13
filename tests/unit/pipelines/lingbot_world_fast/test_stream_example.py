from unittest.mock import MagicMock, patch

import torch

from examples.lingbot import stream_lingbot_world_fast as stream_example
from telefuser.core.config import AttnImplType


def test_stream_get_pipeline_maps_ppl_config_to_internal_workers() -> None:
    pipeline = MagicMock()

    with (
        patch.object(stream_example, "LingBotWorldFastPipeline", return_value=pipeline) as pipeline_cls,
        patch.object(stream_example, "ModuleManager") as module_manager_cls,
    ):
        result = stream_example.get_pipeline(
            parallelism=4,
            model_root="/models/Wan2.2-I2V-A14B",
            fast_model_root="/models/lingbot-world-fast",
        )

    assert result is pipeline
    pipeline_cls.assert_called_once_with(device="cuda", torch_dtype=torch.bfloat16)
    module_manager_cls.assert_called_once_with(device="cpu")

    config = pipeline.init.call_args.args[1]
    assert config.checkpoint_dir == "/models/Wan2.2-I2V-A14B"
    assert config.fast_checkpoint_subdir == "/models/lingbot-world-fast"
    assert config.control_type == "cam"
    assert config.max_area == 480 * 832
    assert config.attention_config.attn_impl == AttnImplType.SAGE_ATTN_2_8_8_SM90
    assert config.parallel_config.device_ids == [0, 1, 2, 3]
    assert config.parallel_config.sp_ulysses_degree == 4
    assert config.parallel_config.enable_fsdp is False


def test_stream_get_service_uses_ppl_pipeline_and_fps() -> None:
    pipeline = MagicMock()

    with patch.object(stream_example, "get_pipeline", return_value=pipeline) as get_pipeline:
        service = stream_example.get_service()

    get_pipeline.assert_called_once_with()
    assert service.pipeline is pipeline
    assert service.default_fps == stream_example.PPL_CONFIG["target_fps"]
