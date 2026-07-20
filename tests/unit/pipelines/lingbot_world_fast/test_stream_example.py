import inspect
from unittest.mock import MagicMock, patch

import torch

from examples.lingbot import lingbot_world_fast_image_to_video_h100 as offline_example
from examples.stream_server import webrtc_bidirectional_demo as webrtc_demo
from telefuser.core.config import AttnImplType


def test_webrtc_demo_uses_stream_service_defaults() -> None:
    source = inspect.getsource(webrtc_demo)

    assert webrtc_demo.DEFAULT_IMAGE_PATH == offline_example.DEFAULT_IMAGE_PATH
    assert webrtc_demo.DEFAULT_PROMPT == offline_example.DEFAULT_PROMPT
    assert source.count("HTML_TEMPLATE =") == 1
    assert "DEFAULT_OPTIONS" not in source
    assert "--intrinsics-path" not in source
    assert "--image-path" not in source
    assert 'type="file"' in webrtc_demo.HTML_TEMPLATE
    assert 'id="image-preview" src="/default-image"' in webrtc_demo.HTML_TEMPLATE
    assert '$("image-preview").src = imagePreviewObjectUrl' in webrtc_demo.HTML_TEMPLATE
    assert "requestBody.image = image" in webrtc_demo.HTML_TEMPLATE
    assert "requestBody.image_path = DEFAULT_IMAGE_PATH" in webrtc_demo.HTML_TEMPLATE
    assert 'type: "control_state"' in webrtc_demo.HTML_TEMPLATE
    assert 'window.addEventListener("blur", () => releaseAllControls(true))' in webrtc_demo.HTML_TEMPLATE
    assert 'document.addEventListener("visibilitychange"' in webrtc_demo.HTML_TEMPLATE
    assert 'id="reset-pose"' in webrtc_demo.HTML_TEMPLATE


def test_unified_example_get_pipeline_maps_ppl_config_to_internal_workers() -> None:
    pipeline = MagicMock()

    with patch.object(offline_example, "LingBotWorldFastPipeline", return_value=pipeline) as pipeline_cls:
        result = offline_example.get_pipeline(
            parallelism=4,
            model_root="/models/Wan2.2-I2V-A14B",
            fast_model_root="/models/lingbot-world-fast",
        )

    assert result is pipeline
    pipeline_cls.assert_called_once_with(device="cuda", torch_dtype=torch.bfloat16)

    assert len(pipeline.init.call_args.args) == 1
    config = pipeline.init.call_args.args[0]
    assert config.checkpoint_dir == "/models/Wan2.2-I2V-A14B"
    assert config.fast_checkpoint_path == "/models/lingbot-world-fast"
    assert config.control_type == "cam"
    assert config.max_area == 480 * 832
    assert config.local_attn_size == -1
    assert config.attention_config.attn_impl == AttnImplType.SAGE_ATTN_2_8_8_SM90
    assert config.parallel_config.device_ids == [0, 1, 2, 3]
    assert config.parallel_config.sp_ulysses_degree == 4
    assert config.parallel_config.enable_fsdp is offline_example.PPL_CONFIG["enable_fsdp"]


def test_unified_example_get_service_uses_passed_gpu_num_and_ppl_fps() -> None:
    pipeline = MagicMock()

    with patch.object(offline_example, "get_pipeline", return_value=pipeline) as get_pipeline:
        service = offline_example.get_service(gpu_num=4)

    get_pipeline.assert_called_once_with(parallelism=4)
    assert service.pipeline is pipeline
    assert service.default_fps == offline_example.PPL_CONFIG["target_fps"]
    assert service.default_session_config["control_translation_scale"] == 3.0


def test_unified_example_get_service_retains_example_default_gpu_num() -> None:
    pipeline = MagicMock()

    with patch.object(offline_example, "get_pipeline", return_value=pipeline) as get_pipeline:
        service = offline_example.get_service()

    get_pipeline.assert_called_once_with(parallelism=offline_example.PPL_CONFIG["parallelism"])
    assert service.pipeline is pipeline


def test_get_pipeline_uses_module_model_zoo_path() -> None:
    pipeline = MagicMock()
    with patch.object(offline_example, "LingBotWorldFastPipeline", return_value=pipeline):
        offline_example.get_pipeline()

    assert len(pipeline.init.call_args.args) == 1
    config = pipeline.init.call_args.args[0]
    assert config.checkpoint_dir == str(offline_example.TF_MODEL_ZOO_PATH / "Wan2.2-I2V-A14B")
    assert config.fast_checkpoint_path == str(offline_example.TF_MODEL_ZOO_PATH / "lingbot/lingbot-world-fast")
