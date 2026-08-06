"""Contract parity tests for service-capable example entrypoints."""

from __future__ import annotations

import ast
import asyncio
import functools
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from examples.run_examples import load_config
from telefuser.service.api.api_server import ApiServer
from telefuser.service.api.schema import TaskRequest
from telefuser.service.core.file_service import FileService
from telefuser.service.core.pipeline_contract import PipelineContract, TaskContract, load_pipeline_contract
from telefuser.service.core.pipeline_runner import PipelineRunner, _select_kwargs
from telefuser.service.core.task_manager import TaskManager
from telefuser.service.core.task_service import MediaGenerationService
from telefuser.service_types import PipelineRunStatus, TaskStatus

SERVICE_EXAMPLES = {
    "wan21_i2v_service": (Path("examples/wan_video/wan21_14b_image_to_video_480p_service.py"), "i2v", True),
    "minimax_h3_fl2va": (Path("examples/minimax_h3/minimax_h3_fl2va_h100.py"), "t2v", True),
    "minimax_h3_fl2va_torchao_fp8": (Path("examples/minimax_h3/minimax_h3_fl2va_torchao_fp8_h100.py"), "t2v", True),
    "minimax_h3_fl2va_bnb_nf4": (Path("examples/minimax_h3/minimax_h3_fl2va_bnb_nf4_h100.py"), "t2v", True),
    "minimax_h3_ref2va": (Path("examples/minimax_h3/minimax_h3_ref2va_h100.py"), "s2v", True),
    "wan22_i2v_distill": (Path("examples/wan_video/wan22_14b_image_to_video_distill_h100.py"), "i2v", True),
    "lingbot_video_dense": (Path("examples/lingbot_video/lingbot_video_dense_1_3b.py"), "t2i", True),
    "lingbot_video_moe": (Path("examples/lingbot_video/lingbot_video_moe_30b.py"), "t2i", True),
    "wan21_t2v": (Path("examples/wan_video/wan21_1_3b_text_to_video_h100.py"), "t2v", False),
    "wan21_i2v": (Path("examples/wan_video/wan21_14b_image_to_video_h100.py"), "i2v", False),
    "wan21_i2v_lora": (Path("examples/wan_video/wan21_14b_image_to_video_lora_h100.py"), "i2v", False),
    "wan22_i2v_lora": (Path("examples/wan_video/wan22_14b_image_to_video_lora_h100.py"), "i2v", False),
    "hunyuan_i2v": (Path("examples/hunyuan_video/hunyuan_video_i2v.py"), "i2v", False),
    "longcat_t2v": (Path("examples/longcat_video/longcat_text_to_video.py"), "t2v", False),
    "longcat_i2v": (Path("examples/longcat_video/longcat_image_to_video.py"), "i2v", False),
    "longcat_continue": (Path("examples/longcat_video/longcat_video_continue.py"), "vc", False),
    "longcat_unify": (Path("examples/longcat_video/longcat_video_unify.py"), "i2v", False),
    "longcat_t2v_refine": (Path("examples/longcat_video/longcat_text_to_video_refine.py"), "t2v", False),
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"


def _has_run_with_file(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_with_file" for node in tree.body
    )


def _configured_service_example_paths() -> set[Path]:
    return {
        Path(pipeline.script)
        for pipeline in load_config().pipelines.values()
        if _has_run_with_file(EXAMPLES_ROOT / pipeline.script)
    }


def test_configured_service_examples_are_covered_by_parity_suite() -> None:
    configured_examples = _configured_service_example_paths()
    covered_examples = {path.relative_to("examples") for path, _, _ in SERVICE_EXAMPLES.values()}

    assert configured_examples
    assert configured_examples.issubset(covered_examples)


def _load_example(name: str) -> ModuleType:
    path, _, _ = SERVICE_EXAMPLES[name]
    module_name = f"service_parity_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parameter_value(parameter_type: str) -> object:
    values = {
        "boolean": True,
        "integer": 7,
        "number": 1.5,
        "array": [{"type": "image", "role": "reference", "uri": "input.png"}],
        "object": {},
    }
    return values.get(parameter_type, "test-value")


def _task_data(task_contract: TaskContract, output_path: Path) -> dict[str, Any]:
    input_image_path = output_path.parent / "input.png"
    Image.new("RGB", (1, 1)).save(input_image_path)
    task_data: dict[str, Any] = {
        "task_id": "parity-task",
        "task": task_contract.task,
        "prompt": '{"caption":{"scene":"test"},"duration":5}',
        "negative_prompt": "test negative prompt",
        "seed": 7,
        "output_path": str(output_path),
        "first_image_path": str(input_image_path),
        "last_image_path": "last.png",
        "ref_video_path": "input.mp4",
        "audio_path": "input.wav",
    }
    for name, parameter in task_contract.parameters.items():
        if parameter.default is not None:
            task_data.setdefault(name, parameter.default)
        elif parameter.required:
            task_data.setdefault(name, _parameter_value(parameter.type))

    for name in task_contract.required_inputs:
        task_data.setdefault(name, f"{name}.bin")

    return task_data


def _assert_required_inputs_are_represented(task_contract: TaskContract, kwargs: dict[str, Any]) -> None:
    aliases = {"first_image_path": "image"}
    for input_name in task_contract.required_inputs:
        assert input_name in kwargs or aliases.get(input_name) in kwargs


@pytest.mark.parametrize("example_name", SERVICE_EXAMPLES)
def test_service_runner_matches_direct_example_entrypoint(example_name: str, tmp_path: Path) -> None:
    module = _load_example(example_name)
    example_path, task, expects_explicit_contract = SERVICE_EXAMPLES[example_name]
    contract, declared = load_pipeline_contract(module, ppl_file=example_path.name, default_task=task)
    task_contract = contract.get_task_contract(task)

    assert declared is expects_explicit_contract
    assert task_contract is not None
    entrypoint = getattr(module, contract.entrypoints.run_with_file)
    task_data = _task_data(task_contract, tmp_path / "result.bin")
    expected_kwargs = _select_kwargs(entrypoint, task_data=task_data, module=module)
    calls: list[dict[str, Any]] = []

    @functools.wraps(entrypoint)
    def capture_entrypoint(pipeline: object, **kwargs: Any) -> dict[str, str]:
        assert pipeline is sentinel
        calls.append(kwargs)
        return {"output_path": kwargs["output_path"]}

    sentinel = object()
    direct_result = capture_entrypoint(sentinel, **expected_kwargs)
    runner = PipelineRunner(pipeline=sentinel, run_with_file=capture_entrypoint, module=module)
    service_result = asyncio.run(runner.run(task_data=task_data))

    assert service_result.status == PipelineRunStatus.SUCCESS
    assert service_result.output_path == direct_result["output_path"]
    assert calls == [expected_kwargs, expected_kwargs]
    _assert_required_inputs_are_represented(task_contract, expected_kwargs)
    assert {
        name for name, parameter in task_contract.parameters.items() if parameter.required and parameter.exposed
    }.issubset(expected_kwargs)


@pytest.mark.parametrize("example_name", SERVICE_EXAMPLES)
def test_task_request_reaches_example_entrypoint_with_contract_defaults(example_name: str, tmp_path: Path) -> None:
    module = _load_example(example_name)
    example_path, task, expects_explicit_contract = SERVICE_EXAMPLES[example_name]
    contract, declared = load_pipeline_contract(module, ppl_file=example_path.name, default_task=task)
    task_contract = contract.get_task_contract(task)
    assert declared is expects_explicit_contract
    assert task_contract is not None

    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    file_service = FileService(tmp_path)
    server.file_service = file_service
    server.validate_task_supported = lambda requested_task: None
    contract_metadata = contract.to_metadata()["task_contracts"][task]
    server.get_task_contract = lambda requested_task: contract_metadata if requested_task == task else None

    request_data: dict[str, Any] = {
        "task": task,
        "prompt": '{"caption":{"scene":"test"},"duration":5}',
        "aspect_ratio": "16:9",
        "output_format": "png",
    }
    for input_name in task_contract.required_inputs:
        if input_name == "first_image_path":
            input_path = tmp_path / "input.png"
            Image.new("RGB", (1, 1)).save(input_path)
            request_data[input_name] = str(input_path)
        else:
            request_data[input_name] = f"{input_name}.bin"
    for name, parameter in task_contract.parameters.items():
        if parameter.required and parameter.default is None:
            request_data.setdefault(name, _parameter_value(parameter.type))
    request = TaskRequest(**request_data)
    explicit_fields = set(request.model_fields_set)
    asyncio.run(
        server.task_app_service.submit(
            request,
            explicit_fields=explicit_fields,
            ensure_processing=False,
        )
    )

    captured_task_data: list[dict[str, Any]] = []

    class CapturingInferenceService:
        async def run_task_with_stop_event(
            self,
            task_data: dict[str, Any],
            stop_event: threading.Event,
            output_root: str | None = None,
        ) -> dict[str, Any]:
            assert not stop_event.is_set()
            assert output_root == str(file_service.output_dir)
            captured_task_data.append(task_data)
            return {
                "status": PipelineRunStatus.SUCCESS.value,
                "output_path": task_data["output_path"],
                "peak_memory_mb": 0.0,
                "inference_time_s": 0.0,
            }

    response = asyncio.run(
        MediaGenerationService(file_service, CapturingInferenceService()).generate_media_with_stop_event(
            request, threading.Event()
        )
    )

    assert response is not None
    assert response.task_status == TaskStatus.COMPLETED
    assert len(captured_task_data) == 1

    entrypoint = getattr(module, contract.entrypoints.run_with_file)
    expected_kwargs = _select_kwargs(entrypoint, task_data=captured_task_data[0], module=module)
    calls: list[dict[str, Any]] = []
    sentinel = object()

    @functools.wraps(entrypoint)
    def capture_entrypoint(pipeline: object, **kwargs: Any) -> dict[str, str]:
        assert pipeline is sentinel
        calls.append(kwargs)
        return {"output_path": kwargs["output_path"]}

    direct_result = capture_entrypoint(sentinel, **expected_kwargs)
    runner = PipelineRunner(pipeline=sentinel, run_with_file=capture_entrypoint, module=module)
    service_result = asyncio.run(runner.run(task_data=captured_task_data[0]))

    assert service_result.status == PipelineRunStatus.SUCCESS
    assert service_result.output_path == direct_result["output_path"]
    assert calls == [expected_kwargs, expected_kwargs]
    _assert_required_inputs_are_represented(task_contract, expected_kwargs)
    for name, parameter in task_contract.parameters.items():
        if parameter.exposed and parameter.default is not None and name not in explicit_fields:
            assert captured_task_data[0][name] == parameter.default


@pytest.mark.parametrize("example_name", SERVICE_EXAMPLES)
def test_http_task_route_applies_example_contract_defaults(example_name: str, tmp_path: Path) -> None:
    module = _load_example(example_name)
    example_path, task, expects_explicit_contract = SERVICE_EXAMPLES[example_name]
    contract, declared = load_pipeline_contract(module, ppl_file=example_path.name, default_task=task)
    task_contract = contract.get_task_contract(task)
    assert declared is expects_explicit_contract
    assert task_contract is not None

    class ContractMetadataService:
        def supported_tasks(self) -> tuple[str, ...]:
            return contract.supported_tasks

        def get_task_contract(self, requested_task: str) -> TaskContract | None:
            return contract.get_task_contract(requested_task)

    task_manager = TaskManager()
    server = ApiServer(task_manager=task_manager, enable_openai_api=False)
    server.file_service = FileService(tmp_path)
    server.inference_service = ContractMetadataService()
    request_data: dict[str, Any] = {
        "task": task,
        "prompt": '{"caption":{"scene":"test"},"duration":5}',
        "aspect_ratio": "16:9",
        "output_format": "png",
    }
    for input_name in task_contract.required_inputs:
        request_data[input_name] = f"{input_name}.bin"

    for name, parameter in task_contract.parameters.items():
        if parameter.required and parameter.default is None:
            request_data.setdefault(name, _parameter_value(parameter.type))
    try:
        with TestClient(server.app) as client:
            response = client.post("/v1/tasks/create", json=request_data)
    finally:
        asyncio.run(server.cleanup())

    assert response.status_code == 200, response.text
    queued_request = task_manager.get_task(response.json()["task_id"]).message
    explicit_fields = set(request_data)
    assert queued_request.task == task
    assert queued_request.prompt == request_data["prompt"]
    assert set(task_contract.required_inputs).issubset(queued_request.model_fields_set)
    for name, parameter in task_contract.parameters.items():
        if (
            parameter.exposed
            and parameter.default is not None
            and name not in explicit_fields
            and name != "output_path"
        ):
            assert getattr(queued_request, name) == parameter.default
