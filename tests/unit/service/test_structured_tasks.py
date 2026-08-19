from __future__ import annotations

import asyncio
import base64
import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from examples.lingbot_vla_v2 import lingbot_vla_v2_native_service
from telefuser.client import TFClient, TaskFailedError
from telefuser.pipelines.lingbot_vla_v2.pipeline import LingBotVlaV2CanonicalActionChunk
from telefuser.service.api.api_server import ApiServer
from telefuser.service.api.schema import StructuredTaskRequest
from telefuser.service.core.task_manager import TaskManager
from telefuser.service.core.task_service import StructuredInferenceService


def _encoded_image() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _payload() -> dict:
    image = _encoded_image()
    return {
        "task": "vla_action",
        "instruction": "pick up the red block",
        "state": [0.0] * 14,
        "camera_high": image,
        "camera_left_wrist": image,
        "camera_right_wrist": image,
        "seed": 7,
    }


class _StructuredPipelineService:
    is_running = True

    def supported_tasks(self) -> tuple[str, ...]:
        return ("vla_action",)

    def get_task_contract(self, task: str) -> dict:
        assert task == "vla_action"
        return lingbot_vla_v2_native_service.PIPELINE_CONTRACT["task_contracts"][task]

    async def run_task_with_stop_event(self, task_data, stop_event, **kwargs) -> dict:
        assert task_data["instruction"] == "pick up the red block"
        assert "output_path" not in task_data
        return {
            "status": "success",
            "raw": {
                "canonical_normalized_actions": [[0.0] * 55 for _ in range(2)],
                "horizon": 2,
                "action_dim": 55,
                "checkpoint_variant": "base",
                "policy_verified": False,
                "verification_status": "unverified_official_6b_base",
            },
            "peak_memory_mb": 128.0,
            "inference_time_s": 0.25,
        }


def test_structured_route_uses_scheduler_and_exposes_result_metrics(tmp_path: Path) -> None:
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.initialize_services(tmp_path, _StructuredPipelineService())

    with TestClient(server.get_app()) as client:
        created = client.post("/v1/tasks/structured", json=_payload())

        assert created.status_code == 200
        created_body = created.json()
        assert created_body["task_status"] == "pending"
        assert "output_path" not in created_body

        deadline = time.monotonic() + 2.0
        while True:
            status = client.get(f"/v1/tasks/{created_body['task_id']}/status")
            assert status.status_code == 200
            body = status.json()
            if body["status"] == "completed":
                break
            assert time.monotonic() < deadline
            time.sleep(0.01)

    assert body["output_path"] is None
    assert body["media_type"] == "structured"
    assert body["peak_memory_mb"] == 128.0
    assert body["inference_time_s"] == 0.25
    assert body["result"]["horizon"] == 2
    assert len(body["result"]["canonical_normalized_actions"][0]) == 55
    assert "camera_high" not in body


def test_structured_route_rejects_media_contract(tmp_path: Path) -> None:
    inference_service = _StructuredPipelineService()
    inference_service.supported_tasks = lambda: ("t2i",)
    inference_service.get_task_contract = lambda task: {"media_type": "image", "parameters": {}}
    server = ApiServer(task_manager=TaskManager(), enable_openai_api=False)
    server.initialize_services(tmp_path, inference_service)

    with TestClient(server.get_app()) as client:
        response = client.post("/v1/tasks/structured", json={"task": "t2i"})

    assert response.status_code == 400
    assert "does not declare a structured result contract" in response.json()["detail"]


def test_structured_service_rejects_non_json_pipeline_result() -> None:
    class InvalidService:
        async def run_task_with_stop_event(self, task_data, stop_event):
            return {"status": "success", "raw": {"value": torch.zeros(1)}}

    service = StructuredInferenceService(InvalidService())
    request = StructuredTaskRequest(task="vla_action")

    with pytest.raises(RuntimeError, match="finite JSON-serializable"):
        asyncio.run(service.execute_with_stop_event(request, threading.Event()))


def test_native_vla_entrypoint_returns_action_contract() -> None:
    class Pipeline:
        def __call__(self, observation, seed=None):
            assert observation.task == "pick up the red block"
            assert seed == 7
            return LingBotVlaV2CanonicalActionChunk(
                canonical_normalized_actions=torch.zeros(2, 55),
                horizon=2,
                action_dim=55,
            )

    result = lingbot_vla_v2_native_service.run_structured(Pipeline(), **_payload())

    assert result["horizon"] == 2
    assert result["action_dim"] == 55
    assert len(result["canonical_normalized_actions"][0]) == 55


def test_native_vla_pipeline_keeps_bf16_default_and_forwards_quantization(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    sentinel = object()

    def fake_get_pipeline(*_args, **kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(lingbot_vla_v2_native_service, "get_lingbot_vla_v2_pipeline", fake_get_pipeline)
    assert lingbot_vla_v2_native_service.get_pipeline() is sentinel
    assert calls[-1]["quantization"] is None

    monkeypatch.setitem(lingbot_vla_v2_native_service.PPL_CONFIG, "quantization", "torchao-fp8")
    assert lingbot_vla_v2_native_service.get_pipeline() is sentinel
    assert calls[-1]["quantization"] == "torchao-fp8"


def test_unified_client_encodes_vla_inputs_and_returns_result(tmp_path: Path) -> None:
    image_path = tmp_path / "camera.png"
    Image.new("RGB", (8, 8)).save(image_path)
    client = TFClient("http://127.0.0.1:8000")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"task_id": "task-1", "task_status": "pending"}
    client._session.post = Mock(return_value=response)
    client.wait_for_completion = Mock(return_value={"status": "completed", "result": {"horizon": 2}})

    result = client.predict_vla_actions(
        instruction="pick up the red block",
        state=[0.0] * 14,
        camera_high_path=str(image_path),
        camera_left_wrist_path=str(image_path),
        camera_right_wrist_path=str(image_path),
        seed=7,
    )

    assert result == {"horizon": 2}
    request = client._session.post.call_args
    assert request.args[0].endswith("/v1/tasks/structured")
    assert request.kwargs["json"]["task"] == "vla_action"
    assert request.kwargs["json"]["camera_high"] == base64.b64encode(image_path.read_bytes()).decode("ascii")


def test_unified_client_rejects_missing_structured_result() -> None:
    client = TFClient()
    client.create_vla_action_task = Mock(return_value={"task_id": "task-1"})
    client.wait_for_completion = Mock(return_value={"status": "completed", "result": None})

    with pytest.raises(TaskFailedError, match="without a structured result"):
        client.predict_vla_actions(
            instruction="pick",
            state=[0.0] * 14,
            camera_high_path="unused",
            camera_left_wrist_path="unused",
            camera_right_wrist_path="unused",
        )


def test_pipeline_pool_preserves_structured_result() -> None:
    from telefuser.service.core.pipeline_pool import PipelinePool

    result = {"status": "success", "raw": {"horizon": 2}}
    handle = SimpleNamespace(
        _dead=False,
        run_task=AsyncMock(return_value=result),
        shutdown=Mock(),
    )
    pool = PipelinePool(
        num_replicas=1,
        replica_device_ids=[["0"]],
        security_level_name="NONE",
    )
    pool._handles = [handle]
    pool._instance_status = ["idle"]
    pool._available.put_nowait(0)

    received = asyncio.run(
        pool.run_task_with_stop_event(
            {"task": "vla_action"},
            threading.Event(),
        )
    )

    assert received == result
    assert pool._instance_status == ["idle"]


def test_pipeline_runner_closes_close_only_pipeline() -> None:
    from telefuser.service.core.pipeline_runner import PipelineRunner

    class Pipeline:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    pipeline = Pipeline()

    def run_structured(pipeline, **kwargs):
        return {"value": 1}

    async def scenario() -> None:
        runner = PipelineRunner(pipeline=pipeline, run_with_file=run_structured)
        result = await runner.run(task_data={"task": "vla_action"})
        assert result.raw == {"value": 1}
        await runner.shutdown()

    asyncio.run(scenario())
    assert pipeline.closed is True
