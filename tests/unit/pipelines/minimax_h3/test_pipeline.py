from pathlib import Path

import pytest
import torch

from telefuser.core.config import (
    AttnImplType,
    FeatureCacheConfig,
    ModelRuntimeConfig,
    ParallelConfig,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.minimax_h3.data import minimax_h3_validate_canonical_request
from telefuser.pipelines.minimax_h3.denoising import MiniMaxH3DenoisingStage
from telefuser.pipelines.minimax_h3.material_io import MiniMaxH3MaterialFacts
from telefuser.pipelines.minimax_h3.pipeline import MiniMaxH3Pipeline, MiniMaxH3PipelineConfig
from telefuser.pipelines.minimax_h3.resolved_plan import (
    MiniMaxH3MaterialPlanItem,
    minimax_h3_resolve_plan,
)
from telefuser.pipelines.minimax_h3.text_encoding import MiniMaxH3TextCondition
from telefuser.pipelines.minimax_h3.vae import MiniMaxH3PreparedCondition
from tools.validation.minimax_h3_trajectory_stage import MiniMaxH3TrajectoryDenoisingStage


class _ZeroVelocityDiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, **kwargs):
        video_rows = kwargs["img_pos_for_infer_output_info"]["position_ids"].numel()
        audio_rows = kwargs["audio_pos_info"]["position_ids"].numel()
        device = self.anchor.device
        return (
            torch.zeros(video_rows, 96, device=device),
            torch.zeros(audio_rows, 32, device=device),
        )


def test_deferred_fl2va_geometry_uses_first_keyframe_ratio() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="fl2va",
        prompt="move",
        conditions=[{"type": "image", "role": "keyframe", "uri": "frame.png", "frame_index": 0}],
        target={"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 4.0},
        seed=0,
    )
    _, plan = MiniMaxH3Pipeline._resolve_deferred_plan(
        canonical,
        {0: MiniMaxH3MaterialFacts(width=1920, height=1080)},
    )
    assert plan.shape["geometry"] == "resolved_v2"
    assert plan.shape["width"] > plan.shape["height"]
    assert plan.shape["width"] % 32 == 0
    assert plan.shape["height"] % 32 == 0


def test_ref2va_deferred_duration_uses_audio_stream_duration() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="move",
        conditions=[
            {
                "type": "video",
                "role": "reference",
                "uri": "reference.mp4",
                "start_time_seconds": 1.0,
            }
        ],
        target={"short_edge": 768, "aspect_ratio": "auto"},
        seed=0,
    )
    _, plan = MiniMaxH3Pipeline._resolve_deferred_plan(
        canonical,
        {
            0: MiniMaxH3MaterialFacts(
                width=1920,
                height=1080,
                duration_seconds=10.0,
                video_duration_seconds=10.0,
                audio_duration_seconds=6.0,
                has_audio=True,
            )
        },
    )
    assert plan.shape["frame_count"] == 124


def test_ref2va_deferred_duration_rejects_silent_video() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="move",
        conditions=[{"type": "video", "role": "reference", "uri": "silent.mp4"}],
        target={"short_edge": 768, "aspect_ratio": "auto"},
        seed=0,
    )
    with pytest.raises(ValueError, match="exactly one probed condition with an audio stream"):
        MiniMaxH3Pipeline._resolve_deferred_plan(
            canonical,
            {
                0: MiniMaxH3MaterialFacts(
                    width=1920,
                    height=1080,
                    duration_seconds=6.0,
                    video_duration_seconds=6.0,
                    has_audio=False,
                )
            },
        )


def test_ref2va_start_time_must_precede_soundtrack_end() -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="ref2va",
        prompt="move",
        conditions=[
            {
                "type": "video",
                "role": "reference",
                "uri": "reference.mp4",
                "start_time_seconds": 7.0,
            }
        ],
        target={"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 4.0},
        seed=0,
    )
    with pytest.raises(ValueError, match="soundtrack duration"):
        MiniMaxH3Pipeline._resolve_deferred_plan(
            canonical,
            {
                0: MiniMaxH3MaterialFacts(
                    width=1920,
                    height=1080,
                    duration_seconds=10.0,
                    video_duration_seconds=10.0,
                    audio_duration_seconds=6.0,
                    has_audio=True,
                )
            },
        )


def test_ref2va_labels_preserve_mixed_material_order() -> None:
    materials = [
        MiniMaxH3MaterialPlanItem(0, "reference", "image", "a", "image.reference_preserve"),
        MiniMaxH3MaterialPlanItem(1, "reference", "audio", "b", "audio"),
        MiniMaxH3MaterialPlanItem(2, "reference", "video", "c", "video.reference_preserve"),
    ]
    prepared = [
        MiniMaxH3PreparedCondition(materials[0], "image"),
        MiniMaxH3PreparedCondition(materials[1], "audio", has_audio=True),
        MiniMaxH3PreparedCondition(materials[2], "video_audio", has_audio=True),
    ]
    assert MiniMaxH3Pipeline._condition_labels(prepared) == [
        ("image", 1),
        ("audio", 1),
        ("audio", 2),
        ("video", 1),
    ]


def test_t2va_denoising_stage_runs_complete_packed_contract_on_cpu() -> None:
    manager = ModuleManager(device="cpu")
    manager.add_module(_ZeroVelocityDiT(), "minimax_h3_transformer")
    stage = MiniMaxH3DenoisingStage(
        manager,
        ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
    )
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="move",
        conditions=[],
        target={"short_edge": 768, "aspect_ratio": "1:1", "duration_seconds": 4.0},
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    text = MiniMaxH3TextCondition(
        hidden_states=torch.zeros(3, 5120, dtype=torch.bfloat16),
        token_tags=torch.ones(3, dtype=torch.long),
    )
    transported = stage.denoise_for_video_vae(
        plan=plan,
        text={"hidden_states": text.hidden_states, "token_tags": text.token_tags},
        conditions=[],
        num_inference_steps=2,
    )
    result = transported["remainder"]
    assert transported["video_latent"].shape == (1, 24, plan.shape["video_latent_t"], 48, 48)
    assert result.audio_latent.shape == (2, 32, plan.shape["audio_latent_t"])
    assert result.packed["seq_len"] % 64 == 0
    assert result.runtime_metrics["peak_allocated_bytes"] == 0
    assert result.runtime_metrics["peak_reserved_bytes"] == 0
    assert result.runtime_metrics["feature_cache_computed_steps"] == 1
    assert result.runtime_metrics["feature_cache_skipped_steps"] == 0


def test_denoising_rejects_corrupt_transported_token_tags() -> None:
    manager = ModuleManager(device="cpu")
    manager.add_module(_ZeroVelocityDiT(), "minimax_h3_transformer")
    stage = MiniMaxH3DenoisingStage(
        manager,
        ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
    )
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="move",
        conditions=[],
        target={"short_edge": 768, "aspect_ratio": "1:1", "duration_seconds": 4.0},
        seed=0,
    )
    text = MiniMaxH3TextCondition(
        hidden_states=torch.zeros(3, 5120, dtype=torch.bfloat16),
        token_tags=torch.tensor([1, 2**62, 1], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="token_tags must contain only"):
        stage.denoise(
            plan=minimax_h3_resolve_plan(canonical),
            text=text,
            conditions=[],
            num_inference_steps=2,
        )


def test_trajectory_stage_captures_selected_boundaries_without_changing_result(tmp_path: Path) -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="move",
        conditions=[],
        target={"short_edge": 768, "aspect_ratio": "1:1", "duration_seconds": 4.0},
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    text = MiniMaxH3TextCondition(
        hidden_states=torch.zeros(3, 5120, dtype=torch.bfloat16),
        token_tags=torch.ones(3, dtype=torch.long),
    )

    def build_stage(stage_class, **kwargs):
        manager = ModuleManager(device="cpu")
        manager.add_module(_ZeroVelocityDiT(), "minimax_h3_transformer")
        return stage_class(
            manager,
            ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
            **kwargs,
        )

    expected = build_stage(MiniMaxH3DenoisingStage).denoise(
        plan=plan,
        text=text,
        conditions=[],
        num_inference_steps=4,
    )
    trajectory_path = tmp_path / "trajectory.pt"
    actual = build_stage(
        MiniMaxH3TrajectoryDenoisingStage,
        trajectory_path=trajectory_path,
    ).denoise(
        plan=plan,
        text=text,
        conditions=[],
        num_inference_steps=4,
    )

    assert torch.equal(actual.video_latent, expected.video_latent)
    assert torch.equal(actual.audio_latent, expected.audio_latent)
    artifact = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    assert artifact["selected_steps"] == [0, 1, 2]
    assert artifact["observed_transformer_steps"] == 3
    assert artifact["observed_scheduler_steps"] == 3
    assert set(artifact["steps"]) == {"0", "1", "2"}
    assert torch.count_nonzero(artifact["steps"]["0"]["scheduler_input"]["input_visual_latent"]) > 0
    assert torch.count_nonzero(artifact["steps"]["0"]["scheduler_input"]["input_audio_latent"]) > 0
    assert torch.equal(
        artifact["steps"]["0"]["scheduler_input"]["noise_pred_visual"],
        torch.zeros_like(artifact["steps"]["0"]["scheduler_input"]["noise_pred_visual"]),
    )


def test_trajectory_stage_can_capture_only_first_update_from_full_schedule(tmp_path: Path) -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="move",
        conditions=[],
        target={"short_edge": 768, "aspect_ratio": "1:1", "duration_seconds": 4.0},
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    text = MiniMaxH3TextCondition(
        hidden_states=torch.zeros(3, 5120, dtype=torch.bfloat16),
        token_tags=torch.ones(3, dtype=torch.long),
    )

    def capture(path: Path, max_updates: int | None = None) -> dict:
        manager = ModuleManager(device="cpu")
        manager.add_module(_ZeroVelocityDiT(), "minimax_h3_transformer")
        stage = MiniMaxH3TrajectoryDenoisingStage(
            manager,
            ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
            trajectory_path=path,
            max_updates=max_updates,
        )
        stage.denoise(plan=plan, text=text, conditions=[], num_inference_steps=4)
        return torch.load(path, map_location="cpu", weights_only=True)

    full = capture(tmp_path / "full.pt")
    first = capture(tmp_path / "first.pt", max_updates=1)

    assert first["configured_num_updates"] == 3
    assert first["num_updates"] == 1
    assert first["trajectory_truncated"] is True
    assert first["selected_steps"] == [0]
    assert first["observed_transformer_steps"] == 1
    assert first["observed_scheduler_steps"] == 1
    assert torch.equal(
        first["steps"]["0"]["scheduler_input"]["noise_pred_visual"],
        full["steps"]["0"]["scheduler_input"]["noise_pred_visual"],
    )


def test_pipeline_wraps_multi_gpu_stages_and_closes_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    from telefuser.pipelines.minimax_h3 import pipeline as pipeline_module

    class _Stage:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _Worker:
        def __init__(self, stage: object, **kwargs: object) -> None:
            self.stage = stage
            self.kwargs = kwargs
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _Channel:
        instances = []

        def __init__(self, consumer_world_size: int, **kwargs: object) -> None:
            self.consumer_world_size = consumer_world_size
            self.kwargs = kwargs
            self.closed = False
            self.instances.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(pipeline_module.AutoProcessor, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(pipeline_module, "MiniMaxH3TextEncodingStage", _Stage)
    monkeypatch.setattr(pipeline_module, "MiniMaxH3VideoVAEStage", _Stage)
    monkeypatch.setattr(pipeline_module, "MiniMaxH3AudioVAEStage", _Stage)
    monkeypatch.setattr(pipeline_module, "MiniMaxH3DenoisingStage", _Stage)
    monkeypatch.setattr(pipeline_module, "ParallelWorker", _Worker)
    monkeypatch.setattr(pipeline_module, "WorkerTensorChannel", _Channel)
    monkeypatch.setattr(pipeline_module.dist, "is_initialized", lambda: False)

    pipeline = MiniMaxH3Pipeline(device="cpu")
    pipeline.init(
        ModuleManager(device="cpu"),
        MiniMaxH3PipelineConfig(
            processor_path="processor",
            text_encoder_config=ModelRuntimeConfig(
                device_type="cpu",
                parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2),
            ),
            dit_config=ModelRuntimeConfig(
                device_type="cpu",
                parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2),
            ),
            video_vae_config=ModelRuntimeConfig(
                device_type="cpu",
                parallel_config=ParallelConfig(device_ids=[0, 1], tp_degree=2),
            ),
        ),
    )
    assert isinstance(pipeline.text_stage, _Worker)
    assert isinstance(pipeline.video_vae_stage, _Worker)
    assert isinstance(pipeline.denoising_stage, _Worker)
    assert len(_Channel.instances) == 3
    assert pipeline._uses_direct_text_handoff
    assert pipeline._uses_direct_visual_handoff
    assert pipeline._uses_direct_video_latent_handoff
    assert pipeline.text_stage.kwargs["tensor_output_methods"] == ("encode_for_denoising",)
    assert pipeline.video_vae_stage.kwargs["tensor_output_methods"] == ("encode_visual_for_denoising",)
    assert pipeline.denoising_stage.kwargs["tensor_output_methods"] == ("denoise_for_video_vae",)
    pipeline.stop()
    assert pipeline.text_stage.closed
    assert pipeline.video_vae_stage.closed
    assert pipeline.denoising_stage.closed
    assert all(channel.closed for channel in _Channel.instances)


def test_condition_transport_payload_drops_source_media() -> None:
    material = MiniMaxH3MaterialPlanItem(0, "reference", "image", "a", "image.reference_preserve")
    visual_rows = torch.zeros(2, 96)
    condition = MiniMaxH3PreparedCondition(
        material,
        "image",
        image=object(),
        visual_rows=visual_rows,
        latent_t=1,
        latent_h=2,
        latent_w=3,
    )

    payload = MiniMaxH3Pipeline._condition_transport_payload(condition)

    assert "image" not in payload
    assert "video_frames" not in payload
    assert payload["visual_rows"] is visual_rows
    assert MiniMaxH3PreparedCondition(**payload).latent_w == 3


def test_pipeline_resolves_deferred_worker_result() -> None:
    marker = object()
    assert MiniMaxH3Pipeline._resolve_stage_result(lambda: marker) is marker
    assert MiniMaxH3Pipeline._resolve_stage_result(marker) is marker


def test_example_loader_allows_release_length_parallel_denoising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import examples.minimax_h3.common as common

    component_root = tmp_path / "Ref2VA"
    for name in ("transformer", "text_encoder"):
        directory = component_root / name
        directory.mkdir(parents=True)
        (directory / "model-00001-of-00001.safetensors").touch()

    captured = {}

    class _Manager:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def load_model(self, *args, **kwargs) -> None:
            pass

    class _Pipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def init(self, manager, config) -> None:
            captured["config"] = config

    monkeypatch.setattr(common, "ModuleManager", _Manager)
    monkeypatch.setattr(common, "MiniMaxH3Pipeline", _Pipeline)

    feature_cache_config = FeatureCacheConfig(enabled=True, model_type="MiniMax-H3-Base")
    common.load_minimax_h3_pipeline(
        tmp_path,
        partition="Ref2VA",
        device="cpu",
        ulysses_degree=2,
        tp_degree=2,
        text_encoder_tp_degree=4,
        attn_impl="TORCH_SDPA",
        feature_cache_config=feature_cache_config,
    )

    config = captured["config"]
    assert config.dit_config.parallel_config.timeout == 1800
    assert config.dit_config.parallel_config.sp_ulysses_degree == 2
    assert config.dit_config.parallel_config.tp_degree == 2
    assert config.dit_config.parallel_config.enable_fsdp is False
    assert config.dit_config.attention_config.attn_impl is AttnImplType.TORCH_SDPA
    assert config.dit_config.feature_cache_config is feature_cache_config
    assert config.dit_config.offload_config.offload_type is WeightOffloadType.NO_CPU_OFFLOAD
    assert config.text_encoder_config.parallel_config.sp_ulysses_degree == 1
    assert config.text_encoder_config.parallel_config.tp_degree == 4
    assert config.text_encoder_config.parallel_config.enable_fsdp is False
    assert config.text_encoder_config.offload_config.offload_type is WeightOffloadType.NO_CPU_OFFLOAD
    assert config.video_vae_config.offload_config.offload_type is WeightOffloadType.NO_CPU_OFFLOAD
    assert config.video_vae_config.parallel_config.tp_degree == 4
    assert config.audio_vae_config.offload_config.offload_type is WeightOffloadType.NO_CPU_OFFLOAD

    common.load_minimax_h3_pipeline(
        tmp_path,
        partition="Ref2VA",
        device="cuda:0",
        quantization="torchao-fp8",
    )
    quantized_config = captured["config"]
    assert quantized_config.dit_config.quant_config.enabled is True
    assert quantized_config.dit_config.offload_config.offload_type is WeightOffloadType.NO_CPU_OFFLOAD
    assert quantized_config.text_encoder_config.offload_config.offload_type is WeightOffloadType.MODEL_CPU_OFFLOAD
    assert quantized_config.video_vae_config.offload_config.offload_type is WeightOffloadType.MODEL_CPU_OFFLOAD
    assert quantized_config.audio_vae_config.offload_config.offload_type is WeightOffloadType.MODEL_CPU_OFFLOAD


def test_example_writer_preserves_complete_generated_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import examples.minimax_h3.common as common

    commands = []
    monkeypatch.setattr(common, "save_wav", lambda *args, **kwargs: None)
    monkeypatch.setattr(common, "save_video", lambda *args, **kwargs: None)
    monkeypatch.setattr(common.subprocess, "run", lambda command, **kwargs: commands.append(command))
    generation = SimpleNamespace(
        video=torch.zeros(1, 2, 2, 2, 3),
        audio=torch.zeros(1, 2, 16),
        video_fps=24,
        audio_sample_rate=32_000,
    )
    output = tmp_path / "result.mp4"

    common.save_generation(generation, output)

    assert len(commands) == 1
    command = commands[0]
    assert "-shortest" not in command
    assert command[-1] == str(output)
    assert command[command.index("-map") + 1] == "0:v:0"
