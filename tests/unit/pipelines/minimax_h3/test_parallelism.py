from unittest.mock import MagicMock, patch

import pytest
import torch

from telefuser.core.config import (
    ModelRuntimeConfig,
    OffloadConfig,
    ParallelConfig,
    QuantConfig,
    QuantType,
    WeightOffloadType,
)
from telefuser.pipelines.minimax_h3.denoising import (
    MiniMaxH3DenoisingStage,
    _build_local_embedding_layout,
)
from telefuser.pipelines.minimax_h3.text_encoding import MiniMaxH3TextEncodingStage
from telefuser.pipelines.minimax_h3.vae import MiniMaxH3VideoVAEStage


def _stage(parallel_config: ParallelConfig) -> tuple[MiniMaxH3DenoisingStage, MagicMock]:
    transformer = MagicMock()
    transformer.parameters.return_value = [torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))]
    transformer.get_fsdp_module_names.return_value = ["blocks"]
    manager = MagicMock()
    manager.fetch_module.return_value = transformer
    runtime = ModelRuntimeConfig(
        device_type="cuda",
        device_id=0,
        torch_dtype=torch.bfloat16,
        parallel_config=parallel_config,
        offload_config=OffloadConfig(offload_type=WeightOffloadType.NO_CPU_OFFLOAD),
    )
    return MiniMaxH3DenoisingStage(manager, runtime), transformer


def test_local_embedding_layout_selects_only_rank_owned_rows() -> None:
    layout = _build_local_embedding_layout(
        seq_len=12,
        text_pos=torch.tensor([0, 1, 2]),
        img_pos=torch.tensor([3, 5, 6, 9, 11]),
        audio_pos=torch.tensor([4, 7, 8, 10]),
        world_size=2,
        rank=1,
        device=torch.device("cpu"),
    )

    assert layout["row_start"] == 6
    assert layout["row_stop"] == 12
    assert torch.equal(layout["text_source_ids"], torch.empty(0, dtype=torch.long))
    assert torch.equal(layout["img_global_ids"], torch.tensor([6, 9, 11]))
    assert torch.equal(layout["img_row_ids"], torch.tensor([0, 3, 5]))
    assert torch.equal(layout["audio_global_ids"], torch.tensor([7, 8, 10]))
    assert torch.equal(layout["audio_row_ids"], torch.tensor([1, 2, 4]))


def test_parallel_models_enables_ulysses_and_preserves_fp32_fsdp_parameters() -> None:
    stage, transformer = _stage(ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2, enable_fsdp=True))
    device_mesh = MagicMock()
    fsdp_model = MagicMock()
    with (
        patch(
            "telefuser.pipelines.minimax_h3.denoising.create_device_mesh_from_config",
            return_value=device_mesh,
        ),
        patch(
            "telefuser.pipelines.minimax_h3.denoising.shard_model_fsdp2_inference",
            return_value=fsdp_model,
        ) as shard,
    ):
        stage.parallel_models()

    transformer.enable_usp.assert_called_once_with(device_mesh)
    shard.assert_called_once()
    call = shard.call_args.kwargs
    assert call["wrap_module_names"] == ["blocks"]
    assert call["device_mesh"] is device_mesh
    assert len(call["ignored_states"]) == 1
    assert stage.transformer is fsdp_model


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("cfg_degree", {"cfg_degree": 2}),
        ("sp_ring_degree", {"sp_ring_degree": 2}),
        ("pp_degree", {"pp_degree": 2}),
    ],
)
def test_parallel_models_rejects_unsupported_degrees(field: str, kwargs: dict[str, int]) -> None:
    stage, _ = _stage(ParallelConfig(device_ids=[0, 1], **kwargs))
    with pytest.raises(NotImplementedError, match=field):
        stage.parallel_models()


def test_parallel_models_enables_tp_before_ulysses_and_keeps_shards_resident() -> None:
    stage, transformer = _stage(ParallelConfig(device_ids=[0, 1, 2, 3], sp_ulysses_degree=2, tp_degree=2))
    device_mesh = MagicMock()
    calls = MagicMock()
    calls.attach_mock(transformer.enable_tp, "enable_tp")
    calls.attach_mock(transformer.enable_usp, "enable_usp")

    with patch(
        "telefuser.pipelines.minimax_h3.denoising.create_device_mesh_from_config",
        return_value=device_mesh,
    ):
        stage.parallel_models()

    assert [call[0] for call in calls.method_calls] == ["enable_tp", "enable_usp"]
    transformer.enable_tp.assert_called_once_with(device_mesh)
    transformer.enable_usp.assert_called_once_with(device_mesh)
    transformer.to.assert_called_once_with(stage.device)
    assert stage.onload_models_flag is True


def test_parallel_models_rejects_tp_with_fsdp() -> None:
    stage, _ = _stage(ParallelConfig(device_ids=[0, 1], tp_degree=2, enable_fsdp=True))
    with (
        patch("telefuser.pipelines.minimax_h3.denoising.create_device_mesh_from_config"),
        pytest.raises(ValueError, match="cannot be combined with FSDP"),
    ):
        stage.parallel_models()


def test_online_quantization_is_applied_once_after_stage_onload() -> None:
    stage, transformer = _stage(ParallelConfig())
    stage.model_runtime_config.quant_config = QuantConfig(enabled=True, quant_type=QuantType.TORCHAO_FP8)
    transformer.quant_type = None

    def enable_quant(config: QuantConfig) -> None:
        transformer.quant_type = config.quant_type

    transformer.enable_quant.side_effect = enable_quant
    with patch("telefuser.pipelines.minimax_h3.denoising.current_platform.empty_cache") as empty_cache:
        stage._ensure_online_quantized()
        stage._ensure_online_quantized()

    transformer.enable_quant.assert_called_once_with(stage.model_runtime_config.quant_config)
    empty_cache.assert_called_once_with()


def test_text_encoder_direct_handoff_keeps_token_tags_on_cpu() -> None:
    manager = MagicMock()
    manager.fetch_module.return_value = MagicMock()
    stage = MiniMaxH3TextEncodingStage(
        manager,
        ModelRuntimeConfig(device_type="cpu"),
        processor=MagicMock(),
    )
    stage.onload_models_flag = True
    condition = MagicMock()
    hidden_states = torch.zeros(2, 4)
    cpu_tags = torch.tensor([1, 1])
    condition.hidden_states = hidden_states
    condition.token_tags.cpu.return_value = cpu_tags

    with patch.object(stage, "_encode_impl", return_value=condition):
        result = stage.encode_for_denoising(
            task="t2va",
            prompt="move",
            images=[],
            videos=[],
            condition_labels=[],
        )

    assert result["hidden_states"] is hidden_states
    assert result["token_tags"] is cpu_tags
    condition.token_tags.cpu.assert_called_once_with()


def test_text_encoder_parallel_models_enables_fsdp_residency() -> None:
    encoder = MagicMock()
    encoder.get_fsdp_module_names.return_value = ["fsdp_language_layers"]
    manager = MagicMock()
    manager.fetch_module.return_value = encoder
    runtime = ModelRuntimeConfig(
        device_type="cuda",
        parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2, enable_fsdp=True),
        offload_config=OffloadConfig(offload_type=WeightOffloadType.NO_CPU_OFFLOAD),
    )
    stage = MiniMaxH3TextEncodingStage(manager, runtime, processor=MagicMock())
    device_mesh = MagicMock()
    fsdp_encoder = MagicMock()

    with (
        patch(
            "telefuser.pipelines.minimax_h3.text_encoding.create_device_mesh_from_config",
            return_value=device_mesh,
        ),
        patch(
            "telefuser.pipelines.minimax_h3.text_encoding.shard_model_fsdp2_inference",
            return_value=fsdp_encoder,
        ) as shard,
    ):
        stage.parallel_models()

    shard.assert_called_once_with(
        module=encoder,
        device_mesh=device_mesh,
        wrap_module_names=["fsdp_language_layers"],
    )
    assert stage.text_encoder is fsdp_encoder
    assert stage.onload_models_flag is True


def test_text_encoder_parallel_models_enables_tp_residency() -> None:
    encoder = MagicMock()
    manager = MagicMock()
    manager.fetch_module.return_value = encoder
    runtime = ModelRuntimeConfig(
        device_type="cuda",
        parallel_config=ParallelConfig(device_ids=[0, 1, 2, 3], tp_degree=4),
        offload_config=OffloadConfig(offload_type=WeightOffloadType.NO_CPU_OFFLOAD),
    )
    stage = MiniMaxH3TextEncodingStage(manager, runtime, processor=MagicMock())
    device_mesh = MagicMock()

    with patch(
        "telefuser.pipelines.minimax_h3.text_encoding.create_device_mesh_from_config",
        return_value=device_mesh,
    ):
        stage.parallel_models()

    encoder.enable_tp.assert_called_once_with(device_mesh)
    encoder.to.assert_called_once_with(stage.device)
    assert stage.onload_models_flag is True


def test_text_encoder_parallel_models_requires_fsdp() -> None:
    manager = MagicMock()
    manager.fetch_module.return_value = MagicMock()
    stage = MiniMaxH3TextEncodingStage(
        manager,
        ModelRuntimeConfig(
            device_type="cuda",
            parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2),
            offload_config=OffloadConfig(offload_type=WeightOffloadType.NO_CPU_OFFLOAD),
        ),
        processor=MagicMock(),
    )

    with pytest.raises(NotImplementedError, match="requires TP or FSDP"):
        stage.parallel_models()


def test_video_vae_parallel_models_enables_tile_residency() -> None:
    video_vae = MagicMock()
    manager = MagicMock()
    manager.fetch_module.return_value = video_vae
    runtime = ModelRuntimeConfig(
        device_type="cuda",
        parallel_config=ParallelConfig(device_ids=[0, 1, 2, 3], tp_degree=4),
        offload_config=OffloadConfig(offload_type=WeightOffloadType.NO_CPU_OFFLOAD),
    )
    stage = MiniMaxH3VideoVAEStage(manager, runtime)
    device_mesh = MagicMock()
    group = MagicMock()

    with (
        patch(
            "telefuser.pipelines.minimax_h3.vae.create_device_mesh_from_config",
            return_value=device_mesh,
        ),
        patch("telefuser.pipelines.minimax_h3.vae.get_tp_group", return_value=group),
    ):
        stage.parallel_models()

    video_vae.enable_parallel_tiling.assert_called_once_with(group)
    video_vae.to.assert_called_once_with(stage.device)
    assert stage.onload_models_flag is True
