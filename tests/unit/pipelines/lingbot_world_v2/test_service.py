from unittest.mock import MagicMock, patch

import torch
from PIL import Image
from click.testing import CliRunner

from examples.lingbot import lingbot_world_v2_image_to_video_h100 as offline_example
from examples.lingbot import stream_lingbot_world_v2 as stream_example
from examples.lingbot import stream_lingbot_world_v2_7gpu_split_async_vae as split_async_stream_example
from telefuser.core.config import AttnImplType
from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig


def test_v2_stream_get_pipeline_maps_ppl_config_to_internal_workers() -> None:
    pipeline = MagicMock()

    with patch.object(stream_example, "LingBotWorldV2Pipeline", return_value=pipeline) as pipeline_cls:
        result = stream_example.get_pipeline(
            parallelism=4,
            model_root="/models/Wan2.2-I2V-A14B",
            v2_model_root="/models/lingbot-world-v2-14b-causal-fast/transformers",
        )

    assert result is pipeline
    pipeline_cls.assert_called_once_with(device="cuda", torch_dtype=torch.bfloat16)
    config = pipeline.init.call_args.args[0]
    assert config.checkpoint_dir == "/models/Wan2.2-I2V-A14B"
    assert config.fast_checkpoint_path == "/models/lingbot-world-v2-14b-causal-fast/transformers"
    assert config.local_attn_size == 18
    assert config.sink_size == 6
    assert config.timestep_indices == (0, 250, 500, 750)
    assert config.attention_config.attn_impl == AttnImplType.TORCH_SDPA
    assert config.parallel_config.device_ids == [0, 1, 2, 3]


def test_v2_split_async_7gpu_stream_config_separates_condition_vae_and_dit() -> None:
    pipeline = MagicMock()

    with patch.object(split_async_stream_example, "LingBotWorldV2Pipeline", return_value=pipeline) as pipeline_cls:
        result = split_async_stream_example.get_pipeline(
            model_root="/models/Wan2.2-I2V-A14B",
            v2_model_root="/models/lingbot-world-v2-14b-causal-fast/transformers",
        )

    assert result is pipeline
    pipeline_cls.assert_called_once_with(device="cuda", torch_dtype=torch.bfloat16)
    config = pipeline.init.call_args.args[0]
    assert config.vae_config.device_id == 0
    assert config.async_vae_config.device_id == 1
    assert config.text_encoding_config.device_id == 0
    assert config.parallel_config.device_ids == [2, 3, 4, 5, 6]
    assert config.parallel_config.sp_ulysses_degree == 5
    assert config.enable_async_vae is True
    assert config.enable_condition_prefetch is True
    assert config.vae_queue_size == 1
    assert config.local_attn_size == 18
    assert config.sink_size == 6
    assert config.timestep_indices == (0, 250, 500, 750)
    assert config.attention_config.attn_impl == AttnImplType.TORCH_SDPA


def test_v2_stream_service_constructs_v2_session_from_ppl_config() -> None:
    pipeline = MagicMock()

    with patch.object(stream_example, "get_pipeline", return_value=pipeline) as get_pipeline:
        service = stream_example.get_service(gpu_num=4)

    get_pipeline.assert_called_once_with(parallelism=4)
    session_id = service.create_session({"image": Image.new("RGB", (8, 8))})
    session_config = pipeline.control_context.call_args.args[0]

    assert isinstance(session_config, LingBotWorldFastSessionConfig)
    assert session_config.frame_num == 1917
    assert session_config.chunk_size == 4
    assert session_config.frame_policy == "truncate"
    assert session_config.sample_shift == 10.0
    assert service.default_fps == 16
    assert service.max_generation_seconds == 120.0
    assert session_id in service._sessions


def test_v2_offline_cli_forwards_only_supported_run_arguments(tmp_path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (8, 8)).save(image_path)
    output_path = tmp_path / "output.mp4"
    pipeline = MagicMock()

    with (
        patch.object(offline_example, "get_pipeline", return_value=pipeline),
        patch.object(offline_example, "run", return_value=[]) as run,
        patch.object(offline_example, "save_video"),
    ):
        result = CliRunner().invoke(
            offline_example.main,
            [
                "--image_path",
                str(image_path),
                "--action_path",
                str(tmp_path),
                "--output",
                str(output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "frame_num" not in run.call_args.kwargs
    assert "max_sequence_length" not in run.call_args.kwargs
    pipeline.close.assert_called_once_with()
