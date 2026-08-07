import json
from pathlib import Path

import pytest

from examples.minimax_h3 import minimax_h3_fl2va_bnb_nf4_h100 as bnb_nf4_example
from examples.minimax_h3 import minimax_h3_fl2va_h100 as fl2va_example
from examples.minimax_h3 import minimax_h3_fl2va_tf_kernel_fp8_h100 as tf_kernel_fp8_example
from examples.minimax_h3 import minimax_h3_fl2va_torchao_fp8_h100 as torchao_fp8_example
from examples.minimax_h3 import minimax_h3_ref2va_h100 as ref2va_example
from examples.minimax_h3.common import (
    MINIMAX_H3_DEFAULT_FL2VA_IMAGE,
    MINIMAX_H3_DEFAULT_REF2VA_AUDIO,
    MINIMAX_H3_DEFAULT_REF2VA_VIDEO,
    load_minimax_h3_pipeline,
    load_minimax_h3_request,
    minimax_h3_adaln_cache_timesteps,
    minimax_h3_quant_config,
    partition_for_minimax_h3_request,
)
from examples.minimax_h3.minimax_h3_cache_calibrate import _apply_cache_profile
from examples.minimax_h3.minimax_h3_fl2va_h100 import build_fl2va_conditions
from examples.minimax_h3.minimax_h3_ref2va_h100 import (
    build_ref2va_conditions,
    default_ref2va_conditions,
    parse_ref2va_ordered_materials,
)
from telefuser.core.config import AttnImplType, FeatureCacheConfig, QuantKernelBackend, QuantType
from telefuser.pipelines.minimax_h3.task_profiles import MINIMAX_H3_FINITE_ASPECT_RATIOS
from telefuser.service.core.pipeline_contract import PipelineContract


def test_fl2va_example_builds_every_public_keyframe_signature() -> None:
    assert build_fl2va_conditions(mode="t2va", image=None, last_image=None) == []
    assert [item["frame_index"] for item in build_fl2va_conditions(mode="first-frame", image="a", last_image=None)] == [
        0
    ]
    assert [item["frame_index"] for item in build_fl2va_conditions(mode="last-frame", image=None, last_image="b")] == [
        -1
    ]
    assert [item["frame_index"] for item in build_fl2va_conditions(mode="first-last", image="a", last_image="b")] == [
        0,
        -1,
    ]


def test_fl2va_example_keeps_legacy_mode_inference_and_accepts_last_only() -> None:
    assert build_fl2va_conditions(mode=None, image=None, last_image=None) == []
    assert build_fl2va_conditions(mode=None, image=None, last_image="last.png")[0]["frame_index"] == -1
    with pytest.raises(ValueError, match="does not accept"):
        build_fl2va_conditions(mode="t2va", image="first.png", last_image=None)


def test_default_materials_are_source_controlled_example_inputs() -> None:
    assert MINIMAX_H3_DEFAULT_FL2VA_IMAGE.is_file()
    assert MINIMAX_H3_DEFAULT_REF2VA_VIDEO.is_file()
    assert MINIMAX_H3_DEFAULT_REF2VA_AUDIO.is_file()
    assert [item["type"] for item in default_ref2va_conditions()] == ["video", "audio"]


def test_ref2va_ordered_cli_materials_preserve_mixed_reference_order() -> None:
    materials = ["video=https://example.com/motion.mp4", "image=subject.png", "audio=voice.mp3"]
    conditions = build_ref2va_conditions(images=[], videos=[], audios=[], materials=materials)

    assert [condition["type"] for condition in conditions] == ["video", "image", "audio"]
    assert [condition["uri"] for condition in conditions] == [
        "https://example.com/motion.mp4",
        "subject.png",
        "voice.mp3",
    ]
    with pytest.raises(ValueError, match="cannot be combined"):
        build_ref2va_conditions(images=["subject.png"], videos=[], audios=[], materials=materials)
    with pytest.raises(ValueError, match="TYPE=URI"):
        parse_ref2va_ordered_materials(["video"])
    with pytest.raises(ValueError, match="unsupported type"):
        parse_ref2va_ordered_materials(["text=not-supported"])


def test_examples_expose_standard_pipeline_service_entrypoints() -> None:
    for example in (fl2va_example, ref2va_example):
        assert example.PPL_CONFIG["model_root"]
        assert callable(example.get_pipeline)
        assert callable(example.run)
        assert callable(example.run_with_file)

    for example in (fl2va_example, ref2va_example):
        assert example.PPL_CONFIG["online_adaln_cache"] is True
        assert example.PIPELINE_MANIFEST["entrypoints"] == {
            "get_pipeline": "get_pipeline",
            "run_with_file": "run_with_file",
        }

    fl2va_contract = PipelineContract.from_mapping(fl2va_example.PIPELINE_MANIFEST, fallback_name="fl2va")
    ref2va_contract = PipelineContract.from_mapping(ref2va_example.PIPELINE_MANIFEST, fallback_name="ref2va")
    assert fl2va_contract.supported_tasks == ("t2v", "i2v", "fl2v")
    assert ref2va_contract.supported_tasks == ("s2v",)

    assert fl2va_example.PIPELINE_MANIFEST["supported_tasks"] == ["t2v", "i2v", "fl2v"]
    assert ref2va_example.PIPELINE_MANIFEST["supported_tasks"] == ["s2v"]
    for example, task in ((fl2va_example, "t2v"), (ref2va_example, "s2v")):
        aspect_ratio = example.PIPELINE_MANIFEST["task_contracts"][task]["parameters"]["aspect_ratio"]
        assert aspect_ratio["enum"] == list(MINIMAX_H3_FINITE_ASPECT_RATIOS)
    conditions = ref2va_example.PIPELINE_MANIFEST["task_contracts"]["s2v"]["parameters"]["conditions"]
    assert conditions["type"] == "array"
    assert conditions["required"] is True


def test_standard_get_pipeline_forwards_parallel_runtime_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    sentinel = object()

    def fake_loader(model_root: str, **kwargs: object) -> object:
        calls.append((model_root, kwargs))
        return sentinel

    monkeypatch.setattr(fl2va_example, "load_minimax_h3_pipeline", fake_loader)
    result = fl2va_example.get_pipeline(
        4,
        "/models/h3",
        device="cuda:1",
        num_inference_steps=20,
        enable_fsdp=True,
        enable_feature_cache=True,
    )

    assert result is sentinel
    assert calls == [
        (
            "/models/h3",
            {
                "partition": "FL2VA",
                "device": "cuda:1",
                "num_inference_steps": 20,
                "ulysses_degree": 2,
                "tp_degree": 2,
                "text_encoder_tp_degree": 4,
                "enable_fsdp": True,
                "online_adaln_cache": True,
                "attn_impl": AttnImplType.FLASH_ATTN_4,
                "feature_cache_config": FeatureCacheConfig(
                    enabled=True,
                    model_type="MiniMax-H3-Base",
                    n_derivatives=1,
                    taylor_threshold=2,
                ),
                "quantization": None,
            },
        )
    ]


def test_cache_calibration_applies_validated_h3_profile(tmp_path: Path) -> None:
    output_path = tmp_path / "MiniMax-H3-Base.json"
    output_path.write_text(json.dumps({"K": 4, "retention_ratio": 0.2, "thresh": 0.12}), encoding="utf-8")

    _apply_cache_profile(
        output_path,
        max_consecutive_skips=2,
        retention_ratio=0.2,
        schedule_threshold=0.03,
    )

    params = json.loads(output_path.read_text(encoding="utf-8"))
    assert params == {"K": 2, "retention_ratio": 0.2, "thresh": 0.03}


@pytest.mark.parametrize(
    ("name", "quant_type", "backend"),
    [
        ("torchao-fp8", QuantType.TORCHAO_FP8, QuantKernelBackend.TORCHAO),
        ("torchao_fp8", QuantType.TORCHAO_FP8, QuantKernelBackend.TORCHAO),
        ("tf-kernel-fp8", QuantType.FP8, QuantKernelBackend.TF_KERNEL),
        ("bnb-nf4", QuantType.BNB_NF4, QuantKernelBackend.BITSANDBYTES),
    ],
)
def test_quantization_names_resolve_to_runtime_config(
    name: str,
    quant_type: QuantType,
    backend: QuantKernelBackend,
) -> None:
    config = minimax_h3_quant_config(name)
    assert config.enabled is True
    assert config.quant_type == quant_type
    assert config.kernel_backend == backend


def test_quantization_rejects_unsupported_parallel_and_cpu_profiles(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single-GPU"):
        load_minimax_h3_pipeline(
            tmp_path,
            partition="FL2VA",
            ulysses_degree=2,
            quantization="torchao-fp8",
        )

    (tmp_path / "FL2VA").mkdir()
    with pytest.raises(ValueError, match="CUDA"):
        load_minimax_h3_pipeline(
            tmp_path,
            partition="FL2VA",
            device="cpu",
            quantization="bnb-nf4",
        )


@pytest.mark.parametrize(
    ("example", "quantization"),
    [
        (torchao_fp8_example, "torchao-fp8"),
        (tf_kernel_fp8_example, "tf-kernel-fp8"),
        (bnb_nf4_example, "bnb-nf4"),
    ],
)
def test_dedicated_quantized_examples_forward_fixed_backend(
    monkeypatch: pytest.MonkeyPatch,
    example: object,
    quantization: str,
) -> None:
    calls = []
    sentinel = object()

    def fake_get_pipeline(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(example.base, "get_pipeline", fake_get_pipeline)
    assert example.get_pipeline(1, "/models/h3", device="cuda:1") is sentinel
    assert calls == [((1, "/models/h3"), {"device": "cuda:1", "quantization": quantization})]
    assert example.PIPELINE_MANIFEST["pipeline_name"] == example.PPL_CONFIG["name"]


def test_fl2va_run_maps_standard_service_tasks_to_model_conditions() -> None:
    calls = []
    marker = object()

    class Pipeline:
        def __call__(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return marker

    pipeline = Pipeline()
    assert fl2va_example.run(pipeline, task="t2v") is marker
    assert calls[-1]["task"] == "t2va"
    assert calls[-1]["conditions"] == []
    with pytest.raises(ValueError, match="requires first_image_path"):
        fl2va_example.run(pipeline, task="i2v")
    with pytest.raises(ValueError, match="requires first_image_path and last_image_path"):
        fl2va_example.run(pipeline, task="fl2v", first_image_path="first.png")

    fl2va_example.run(pipeline, task="i2v", first_image_path="first.png")
    assert [item["frame_index"] for item in calls[-1]["conditions"]] == [0]

    fl2va_example.run(pipeline, task="fl2v", first_image_path="first.png", last_image_path="last.png")
    assert [item["frame_index"] for item in calls[-1]["conditions"]] == [0, -1]


def test_ref2va_run_preserves_ordered_service_conditions() -> None:
    calls = []
    marker = object()
    conditions = [
        {"type": "audio", "role": "reference", "uri": "voice.mp3"},
        {"type": "image", "role": "reference", "uri": "subject.png"},
    ]

    class Pipeline:
        def __call__(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return marker

    assert ref2va_example.run(Pipeline(), conditions=conditions) is marker
    assert calls[0]["conditions"] is conditions
    assert calls[0]["task"] == "ref2va"


def test_ref2va_run_request_preserves_ordered_json_conditions(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "ref2va",
                "prompt": "preserve order",
                "conditions": [
                    {"type": "audio", "role": "reference", "uri": "voice.mp3"},
                    {"type": "image", "role": "reference", "uri": "subject.png"},
                ],
                "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
                "num_inference_steps": 50,
            }
        ),
        encoding="utf-8",
    )
    calls = []
    marker = object()

    class Pipeline:
        def __call__(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return marker

    assert ref2va_example.run_request(Pipeline(), request_path, num_inference_steps=20) is marker
    assert [condition["type"] for condition in calls[0]["conditions"]] == ["audio", "image"]
    assert calls[0]["conditions"][0]["uri"] == str(tmp_path / "voice.mp3")
    assert calls[0]["num_inference_steps"] == 20


def test_run_with_file_returns_service_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved = []
    marker = object()

    class Pipeline:
        def __call__(self, **kwargs: object) -> object:
            return marker

    monkeypatch.setattr(fl2va_example, "save_generation", lambda result, path: saved.append((result, path)))
    output_path = tmp_path / "result.mp4"
    response = fl2va_example.run_with_file(Pipeline(), output_path=str(output_path))

    assert response == {"output_path": str(output_path)}
    assert saved == [(marker, str(output_path))]


def test_request_loader_resolves_relative_materials_without_reordering(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "ref2va",
                "prompt": "preserve order",
                "conditions": [
                    {"type": "audio", "role": "reference", "uri": "voice.mp3"},
                    {"type": "video", "role": "reference", "uri": "https://example.com/reference.mp4"},
                ],
                "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
            }
        ),
        encoding="utf-8",
    )

    request = load_minimax_h3_request(request_path)

    assert request["conditions"][0]["type"] == "audio"
    assert request["conditions"][0]["uri"] == str(tmp_path / "voice.mp3")
    assert request["conditions"][1]["type"] == "video"
    assert request["conditions"][1]["uri"] == "https://example.com/reference.mp4"
    assert partition_for_minimax_h3_request(request) == "REF2VA"


def test_request_loader_rejects_unknown_top_level_fields(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"task": "t2va", "prompt": "move", "target": {}, "unexpected": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        load_minimax_h3_request(request_path)


def test_adaln_cache_schedule_covers_both_modalities_and_condition_timesteps() -> None:
    timesteps = minimax_h3_adaln_cache_timesteps(
        {"task": "t2va", "flow_shift": 12.0, "audio_flow_shift": 3.0, "num_inference_steps": 4}
    )

    assert timesteps == sorted(timesteps)
    assert 0.0 in timesteps
    assert 0.999 in timesteps
    assert 1.0 in timesteps
    assert len(timesteps) > 4


def test_adaln_cache_loader_rejects_fsdp_execution() -> None:
    from examples.minimax_h3.common import load_minimax_h3_pipeline

    with pytest.raises(ValueError, match="FSDP"):
        load_minimax_h3_pipeline(
            "/missing",
            partition="FL2VA",
            ulysses_degree=2,
            enable_fsdp=True,
            adaln_cache_path="/cache",
        )


def test_online_adaln_cache_loader_rejects_fsdp_execution() -> None:
    from examples.minimax_h3.common import load_minimax_h3_pipeline

    with pytest.raises(ValueError, match="FSDP"):
        load_minimax_h3_pipeline(
            "/missing",
            partition="FL2VA",
            ulysses_degree=2,
            enable_fsdp=True,
            online_adaln_cache=True,
        )
