from __future__ import annotations

import pytest
from fastapi import HTTPException

from telefuser.service.api.schema import TaskRequest
from telefuser.service.api.task_contract_runtime import apply_task_contract_defaults, validate_required_task_parameters


def test_apply_task_contract_defaults_overrides_implicit_schema_defaults() -> None:
    message = TaskRequest(task="t2v")
    contract = {
        "parameters": {
            "resolution": {"default": "480p"},
            "seed": {"default": 7},
            "num_inference_steps": {"default": 8, "exposed": False},
            "output_format": {"default": "jpg"},
        }
    }

    apply_task_contract_defaults(message, task_contract=contract, explicit_fields={"task"})

    assert message.resolution == "480p"
    assert message.seed == 7
    assert message.output_path == f"{message.task_id}.mp4"
    assert not hasattr(message, "num_inference_steps")


def test_validate_required_task_parameters_rejects_missing_prompt() -> None:
    message = TaskRequest(task="t2v")
    contract = {
        "parameters": {
            "prompt": {
                "required": True,
                "default": "",
                "exposed": True,
            }
        }
    }

    with pytest.raises(HTTPException) as exc:
        validate_required_task_parameters(message, task_contract=contract)

    assert exc.value.status_code == 400
    assert exc.value.detail["missing_parameters"] == ["prompt"]
