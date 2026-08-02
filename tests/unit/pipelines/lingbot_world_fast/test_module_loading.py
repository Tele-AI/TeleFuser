from unittest.mock import MagicMock, patch

import torch

from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline, LingBotWorldFastPipelineConfig


def test_pipeline_consumes_preloaded_lingbot_components_from_module_manager(tmp_path) -> None:
    text_encoder = MagicMock()
    dit = MagicMock(control_type="cam")
    text_encoder.to.return_value = text_encoder
    text_encoder.eval.return_value = text_encoder
    dit.to.return_value = dit
    dit.eval.return_value = dit
    dit.requires_grad_.return_value = dit
    module_manager = MagicMock()

    def fetch_module(name: str, require_model_path: bool = False):
        if name == "wan_video_text_encoder":
            return (
                (text_encoder, str(tmp_path / "models_t5_umt5-xxl-enc-bf16.pth"))
                if require_model_path
                else text_encoder
            )
        if name == "lingbot_world_fast_dit":
            return dit
        return None

    module_manager.fetch_module.side_effect = fetch_module

    with (
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.ParallelWorker"),
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.LingBotWorldFastVAEEncodeStage"),
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.LingBotWorldFastVAEDecodeStage"),
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.LingBotWorldFastDenoisingStage"),
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.HuggingfaceTokenizer"),
    ):
        pipeline = LingBotWorldFastPipeline(device="cpu", torch_dtype=torch.float32)
        pipeline.init(
            module_manager,
            LingBotWorldFastPipelineConfig(dit_config=ModelRuntimeConfig(torch_dtype=torch.float32)),
        )

    assert module_manager.load_model.call_count == 0
    assert module_manager.fetch_module.call_args_list[0].args == ("wan_video_text_encoder",)
    assert module_manager.fetch_module.call_args_list[0].kwargs == {"require_model_path": True}
    assert module_manager.fetch_module.call_args_list[1].args == ("lingbot_world_fast_dit",)
    dit.set_causal_attention_window.assert_called_once_with(-1, 0)


def test_non_colocated_dit_and_vae_workers_use_direct_tensor_channel(tmp_path) -> None:
    text_encoder = MagicMock()
    dit = MagicMock(control_type="cam")
    text_encoder.to.return_value = text_encoder
    text_encoder.eval.return_value = text_encoder
    dit.to.return_value = dit
    dit.eval.return_value = dit
    dit.requires_grad_.return_value = dit
    module_manager = MagicMock()
    module_manager.fetch_module.side_effect = lambda name, require_model_path=False: (
        (text_encoder, str(tmp_path / "models_t5_umt5-xxl-enc-bf16.pth"))
        if name == "wan_video_text_encoder" and require_model_path
        else dit
        if name == "lingbot_world_fast_dit"
        else None
    )
    condition_channel = MagicMock()
    latent_channel = MagicMock()

    with (
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.ParallelWorker") as worker_cls,
        patch(
            "telefuser.pipelines.lingbot_world_fast.pipeline.WorkerTensorChannel",
            side_effect=(condition_channel, latent_channel),
        ) as channel_cls,
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.LingBotWorldFastVAEEncodeStage"),
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.LingBotWorldFastVAEDecodeStage"),
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.LingBotWorldFastDenoisingStage"),
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.HuggingfaceTokenizer"),
    ):
        pipeline = LingBotWorldFastPipeline(device="cpu", torch_dtype=torch.float32)
        pipeline.init(
            module_manager,
            LingBotWorldFastPipelineConfig(
                dit_config=ModelRuntimeConfig(
                    torch_dtype=torch.float32,
                    parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2),
                ),
                vae_decode_config=ModelRuntimeConfig(
                    torch_dtype=torch.float32,
                    parallel_config=ParallelConfig(device_ids=[2]),
                ),
            ),
        )

    assert channel_cls.call_args_list[0].args == (2,)
    assert channel_cls.call_args_list[0].kwargs == {"timeout": 600}
    assert channel_cls.call_args_list[1].args == (1,)
    assert channel_cls.call_args_list[1].kwargs == {"timeout": 600}
    assert pipeline.uses_direct_condition_handoff is True
    assert pipeline.uses_direct_vae_handoff is True
    assert worker_cls.call_args_list[0].kwargs == {
        "tensor_output_channel": condition_channel,
        "tensor_output_methods": ("encode_condition_chunk",),
    }
    assert worker_cls.call_args_list[1].kwargs == {
        "tensor_output_channel": latent_channel,
        "tensor_output_methods": ("denoise_and_update_cache",),
        "tensor_input_channels": (condition_channel,),
    }
    assert worker_cls.call_args_list[2].kwargs == {"tensor_input_channels": (latent_channel,)}
