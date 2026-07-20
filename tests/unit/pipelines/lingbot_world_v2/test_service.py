from unittest.mock import MagicMock, patch

import torch
from PIL import Image
from click.testing import CliRunner

from examples.lingbot import lingbot_world_v2_image_to_video_h100 as offline_example
from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig


def test_v2_unified_example_get_pipeline_maps_ppl_config_to_internal_workers() -> None:
    pipeline = MagicMock()
    module_manager = MagicMock()

    with (
        patch.object(offline_example, "ModuleManager", return_value=module_manager),
        patch.object(offline_example, "LingBotWorldV2Pipeline", return_value=pipeline) as pipeline_cls,
    ):
        result = offline_example.get_pipeline(
            parallelism=4,
            model_root="/models/Wan2.2-I2V-A14B",
            v2_model_root="/models/lingbot-world-v2-14b-causal-fast/transformers",
        )

    assert result is pipeline
    pipeline_cls.assert_called_once_with(device="cuda", torch_dtype=torch.bfloat16)
    assert pipeline.init.call_args.args[0] is module_manager
    config = pipeline.init.call_args.args[1]
    assert config.local_attn_size == 18
    assert config.sink_size == 6
    assert config.timestep_indices == (0, 250, 500, 750)
    assert config.attention_config.attn_impl == offline_example.PPL_CONFIG["attn_impl"]
    assert config.parallel_config.device_ids == [0, 1, 2, 3]
    assert config.vae_encode_config.device_id == 0
    assert config.vae_encode_config.parallel_config.device_ids == [0]
    assert config.vae_decode_config.device_id == 0
    assert config.vae_decode_config.parallel_config.device_ids == [0]
    load_calls = module_manager.load_model.call_args_list
    assert [call.args[0] for call in load_calls[:2]] == [
        "/models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth",
        "/models/Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
    ]
    assert load_calls[2].args[0] == [
        f"/models/lingbot-world-v2-14b-causal-fast/transformers/model-{index:05d}-of-00008.safetensors"
        for index in range(1, 9)
    ]


def test_v2_offline_multi_gpu_defaults_to_the_shared_vae_worker() -> None:
    pipeline = MagicMock()
    module_manager = MagicMock()

    with (
        patch.object(offline_example, "ModuleManager", return_value=module_manager),
        patch.object(offline_example, "LingBotWorldV2Pipeline", return_value=pipeline),
    ):
        offline_example.get_pipeline(
            parallelism=4,
            model_root="/models/Wan2.2-I2V-A14B",
            v2_model_root="/models/lingbot-world-v2-14b-causal-fast/transformers",
        )

    config = pipeline.init.call_args.args[1]
    assert config.parallel_config.device_ids == [0, 1, 2, 3]
    assert config.parallel_config.sp_ulysses_degree == 4
    assert config.vae_encode_config.device_id == 0
    assert config.vae_encode_config.parallel_config.device_ids == [0]
    assert config.vae_decode_config.device_id == 0
    assert config.vae_decode_config.parallel_config.device_ids == [0]


def test_v2_unified_example_uses_ppl_configured_vae_stage_devices_independently() -> None:
    pipeline = MagicMock()
    module_manager = MagicMock()

    with (
        patch.dict(offline_example.PPL_CONFIG, {"vae_encode_device_id": 5, "vae_decode_device_id": 6}),
        patch.object(offline_example, "ModuleManager", return_value=module_manager),
        patch.object(offline_example, "LingBotWorldV2Pipeline", return_value=pipeline),
    ):
        offline_example.get_pipeline(
            parallelism=4,
            model_root="/models/Wan2.2-I2V-A14B",
            v2_model_root="/models/lingbot-world-v2-14b-causal-fast/transformers",
        )

    config = pipeline.init.call_args.args[1]
    assert config.parallel_config.device_ids == [0, 1, 2, 3]
    assert config.vae_encode_config.device_id == 5
    assert config.vae_encode_config.parallel_config.device_ids == [5]
    assert config.vae_decode_config.device_id == 6
    assert config.vae_decode_config.parallel_config.device_ids == [6]


def test_v2_unified_example_service_constructs_v2_session_from_ppl_config() -> None:
    pipeline = MagicMock()

    with patch.object(offline_example, "get_pipeline", return_value=pipeline) as get_pipeline:
        service = offline_example.get_service(gpu_num=4)

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
