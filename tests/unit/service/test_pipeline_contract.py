from __future__ import annotations

from pathlib import Path
from typing import Any

from telefuser.service.core import pipeline_service as pipeline_service_module
from telefuser.service.core.config import SecurityLevel
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template
from telefuser.service.core.pipeline_contract import load_pipeline_contract
from telefuser.service.core.pipeline_service import PipelineService


def test_load_pipeline_contract_from_fake_pipeline() -> None:
    from tests.server.pipeline import fake_t2v_pipeline

    contract, declared = load_pipeline_contract(
        fake_t2v_pipeline,
        ppl_file=str(Path(fake_t2v_pipeline.__file__)),
        default_task="t2v",
    )

    assert declared is True
    assert contract.pipeline_name == "fake_t2v_pipeline"
    assert contract.supported_tasks == ("t2v", "i2v", "t2i", "i2i")
    assert contract.supported_media_types == ("video", "image")
    assert contract.entrypoints.get_pipeline == "get_pipeline"
    assert contract.entrypoints.run_with_file == "run_with_file"
    assert contract.get_task_contract("t2v") is not None
    assert contract.get_task_contract("t2v").required_inputs == tuple()


def test_load_pipeline_contract_preserves_parameter_metadata() -> None:
    manifest = build_pipeline_manifest(
        pipeline_name="parameterized_pipeline",
        supported_tasks=["i2v"],
        task_contracts={
            "i2v": build_task_contract_template(
                "i2v",
                parameter_overrides={
                    "resolution": {
                        "default": "720p",
                        "enum": ["480p", "720p"],
                        "description": "Runtime resolution override.",
                    }
                },
                excluded_parameters=("target_video_length",),
            )
        },
    )

    class DummyModule:
        PIPELINE_MANIFEST = manifest

    contract, declared = load_pipeline_contract(DummyModule, ppl_file="/tmp/dummy.py", default_task="i2v")

    assert declared is True
    task_contract = contract.get_task_contract("i2v")
    assert task_contract is not None
    assert task_contract.parameters["resolution"].default == "720p"
    assert task_contract.parameters["resolution"].enum == ("480p", "720p")
    assert "target_video_length" not in task_contract.to_metadata()["parameters"]


def test_pipeline_service_supports_manifest_defined_entrypoints(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "manifest_pipeline.py"
    pipeline_file.write_text(
        """
PIPELINE_MANIFEST = {
    \"contract_version\": \"v1\",
    \"pipeline_name\": \"manifest_pipeline\",
    \"supported_tasks\": [\"t2i\"],
    \"supported_media_types\": [\"image\"],
    \"execution_mode\": \"serial_single_pipeline\",
    \"effective_max_concurrent_tasks\": 1,
    \"entrypoints\": {
        \"get_pipeline\": \"build_pipeline\",
        \"run_with_file\": \"serve_run\",
    },
    \"task_contracts\": {
        \"t2i\": {
            \"media_type\": \"image\",
            \"required_inputs\": [],
            \"optional_inputs\": [],
        }
    },
}

class DummyPipeline:
    pass

def build_pipeline(parallelism=1):
    return DummyPipeline()

def serve_run(pipeline, output_path, **kwargs):
    return {\"output_path\": output_path}
""".strip()
    )

    service = PipelineService(security_level=SecurityLevel.NONE)
    try:
        assert service.start_pipeline(str(pipeline_file), parallelism=1, task="t2i", skip_validation=True) is True

        metadata = service.server_metadata()
        assert metadata["pipeline_name"] == "manifest_pipeline"
        assert metadata["declared_pipeline_contract"] is True
        assert metadata["supported_tasks"] == ["t2i"]
        assert metadata["entrypoints"] == {"get_pipeline": "build_pipeline", "run_with_file": "serve_run"}
        assert metadata["task_contracts"]["t2i"]["media_type"] == "image"
    finally:
        service.close()


def test_build_pipeline_manifest_uses_standard_task_templates() -> None:
    manifest = build_pipeline_manifest(
        pipeline_name="wan22_distill_example",
        supported_tasks=["i2v", "fl2v"],
        task_contracts={
            "i2v": build_task_contract_template("i2v", excluded_parameters=("target_video_length",)),
        },
    )

    assert manifest["supported_tasks"] == ["i2v", "fl2v"]
    assert manifest["supported_media_types"] == ["video"]
    assert manifest["task_contracts"]["i2v"]["required_inputs"] == ["first_image_path"]
    assert "target_video_length" not in manifest["task_contracts"]["i2v"]["parameters"]
    assert manifest["task_contracts"]["fl2v"]["required_inputs"] == ["first_image_path", "last_image_path"]


def test_pipeline_service_logs_startup_contract_summary(tmp_path: Path, monkeypatch) -> None:
    pipeline_file = tmp_path / "logging_manifest_pipeline.py"
    pipeline_file.write_text(
        """
PIPELINE_MANIFEST = {
    \"contract_version\": \"v1\",
    \"pipeline_name\": \"logging_manifest_pipeline\",
    \"supported_tasks\": [\"t2i\"],
    \"supported_media_types\": [\"image\"],
    \"execution_mode\": \"serial_single_pipeline\",
    \"effective_max_concurrent_tasks\": 1,
    \"entrypoints\": {
        \"get_pipeline\": \"build_pipeline\",
        \"run_with_file\": \"serve_run\",
    },
    \"task_contracts\": {
        \"t2i\": {
            \"media_type\": \"image\",
            \"required_inputs\": [],
            \"optional_inputs\": [],
            \"parameters\": {
                \"prompt\": {\"type\": \"string\", \"required\": True, \"description\": \"Prompt\"},
                \"resolution\": {\"type\": \"string\", \"default\": \"1024x1024\", \"enum\": [\"1024x1024\", \"1024x768\"]},
                \"internal_secret\": {\"type\": \"string\", \"default\": \"hidden\", \"exposed\": False},
            },
        }
    },
}

class DummyPipeline:
    pass

def build_pipeline(parallelism=1):
    return DummyPipeline()

def serve_run(pipeline, output_path, **kwargs):
    return {\"output_path\": output_path}
""".strip()
    )

    info_messages: list[str] = []

    def fake_info(message: Any) -> None:
        info_messages.append(str(message))

    monkeypatch.setattr(pipeline_service_module.logger, "info", fake_info)

    service = PipelineService(security_level=SecurityLevel.NONE)
    try:
        assert service.start_pipeline(str(pipeline_file), parallelism=1, task="t2i", skip_validation=True) is True
    finally:
        service.close()

    pipeline_summary = next(
        message for message in info_messages if message.startswith("Pipeline contract startup summary:")
    )
    task_summary = next(message for message in info_messages if message.startswith("Pipeline task startup summary:"))

    assert '"pipeline_name": "logging_manifest_pipeline"' in pipeline_summary
    assert '"supported_tasks": ["t2i"]' in pipeline_summary
    assert '"metadata_endpoint": "/v1/service/metadata"' in pipeline_summary

    assert '"task": "t2i"' in task_summary
    assert '"prompt": {"default": null, "enum": [], "required": true, "type": "string"}' in task_summary
    assert (
        '"resolution": {"default": "1024x1024", "enum": ["1024x768", "1024x1024"], "required": false, "type": "string"}'
        not in task_summary
    )
    assert (
        '"resolution": {"default": "1024x1024", "enum": ["1024x1024", "1024x768"], "required": false, "type": "string"}'
        in task_summary
    )
    assert "internal_secret" not in task_summary
