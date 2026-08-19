"""Native TeleFuser service contract for LingBot-VLA v2 action inference."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from telefuser.pipelines.lingbot_vla_v2.pipeline import LingBotVlaV2Pipeline
from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline
from telefuser.pipelines.lingbot_vla_v2.service import (
    LingBotVlaV2ActionRequest,
    predict_lingbot_vla_v2_action,
)
from telefuser.utils.logging import logger

TF_MODEL_ZOO_PATH = Path(os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")).expanduser()

PPL_CONFIG = {
    "model_root": str(TF_MODEL_ZOO_PATH / "lingbot" / "lingbot-vla-v2-6b"),
    "qwen3vl_root": str(TF_MODEL_ZOO_PATH / "Qwen3-VL-4B-Instruct"),
    "device": "cuda:0",
    "quantization": None,
    "max_image_bytes": 10 * 1024 * 1024,
}

PIPELINE_CONTRACT = {
    "contract_version": "v1",
    "pipeline_name": "lingbot_vla_v2_6b_base",
    "supported_tasks": ["vla_action"],
    "supported_media_types": ["structured"],
    "execution_mode": "serial_single_pipeline",
    "effective_max_concurrent_tasks": 1,
    "entrypoints": {
        "get_pipeline": "get_pipeline",
        "run_with_file": "run_structured",
    },
    "task_contracts": {
        "vla_action": {
            "media_type": "structured",
            "required_inputs": ["camera_high", "camera_left_wrist", "camera_right_wrist"],
            "optional_inputs": [],
            "parameters": {
                "instruction": {
                    "type": "string",
                    "required": True,
                    "description": "Robot instruction.",
                },
                "state": {
                    "type": "array",
                    "required": True,
                    "description": "Raw 14-dimensional RobotWin state.",
                },
                "camera_high": {
                    "type": "string",
                    "required": True,
                    "description": "Base64-encoded high camera image.",
                },
                "camera_left_wrist": {
                    "type": "string",
                    "required": True,
                    "description": "Base64-encoded left wrist camera image.",
                },
                "camera_right_wrist": {
                    "type": "string",
                    "required": True,
                    "description": "Base64-encoded right wrist camera image.",
                },
                "seed": {
                    "type": "integer",
                    "required": False,
                    "default": None,
                    "description": "Optional deterministic inference seed.",
                },
            },
        }
    },
}


def get_pipeline(parallelism: int = 1) -> LingBotVlaV2Pipeline:
    """Load one policy replica for the native TeleFuser service."""
    if parallelism != 1:
        raise ValueError("LingBot-VLA v2 supports parallelism=1 per replica; use --num-replicas for a pipeline pool")
    logger.info(f"Loading LingBot-VLA v2 service profile quantization={PPL_CONFIG['quantization'] or 'bf16'}")
    return get_lingbot_vla_v2_pipeline(
        PPL_CONFIG["model_root"],
        PPL_CONFIG["qwen3vl_root"],
        device=PPL_CONFIG["device"],
        warmup=True,
        quantization=PPL_CONFIG["quantization"],
    )


def run_structured(
    pipeline: LingBotVlaV2Pipeline,
    instruction: str,
    state: list[float],
    camera_high: str,
    camera_left_wrist: str,
    camera_right_wrist: str,
    seed: int | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return one JSON-serializable canonical normalized action chunk."""
    request = LingBotVlaV2ActionRequest(
        task=instruction,
        state=state,
        camera_high=camera_high,
        camera_left_wrist=camera_left_wrist,
        camera_right_wrist=camera_right_wrist,
        seed=seed,
    )
    response = predict_lingbot_vla_v2_action(
        pipeline,
        request,
        max_image_bytes=int(PPL_CONFIG["max_image_bytes"]),
    )
    return response.model_dump(mode="json")
