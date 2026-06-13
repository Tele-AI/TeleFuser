from __future__ import annotations

import os
from pathlib import Path

import torch

from telefuser.core.config import ModelRuntimeConfig
from telefuser.pipelines.lingbot_world_fast.pipeline import (
    LingBotWorldFastPipeline,
    LingBotWorldFastPipelineConfig,
)
from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService


class _LocalModuleManager:
    def __init__(self) -> None:
        self._modules: list[tuple[str, object, str]] = []

    def fetch_module(
        self,
        model_name: str,
        file_path: str | None = None,
        require_model_path: bool = False,
        index: int | None = None,
    ):
        matches = [(module, path) for name, module, path in self._modules if name == model_name]
        if not matches:
            return None
        module, path = matches[0]
        return (module, path) if require_model_path else module

    def add_module(self, module, name: str, path: str = "manual") -> None:
        self._modules.append((name, module, path))

    def get_model_info(self):
        return [{"name": name, "path": path} for name, _, path in self._modules]


def get_service() -> LingBotWorldFastService:
    checkpoint_dir = os.environ.get("LINGBOT_WORLD_CHECKPOINT_DIR", "")
    if not checkpoint_dir:
        raise RuntimeError("Set LINGBOT_WORLD_CHECKPOINT_DIR to the LingBot-World base checkpoint directory")

    # Allow separating "base" weights (VAE + text encoder + tokenizer) from the DiT fast weights directory.
    # If set to an absolute path, Path join keeps the absolute path (so it works as a standalone folder).
    fast_subdir = os.environ.get("LINGBOT_WORLD_FAST_CHECKPOINT_SUBDIR", "lingbot_world_fast")

    mm = _LocalModuleManager()
    vae_device_type = os.environ.get("LINGBOT_WORLD_VAE_DEVICE", "cpu")
    vae_device_id = int(os.environ.get("LINGBOT_WORLD_VAE_DEVICE_ID", "0"))
    text_device_type = os.environ.get("LINGBOT_WORLD_TEXT_DEVICE", "cpu")
    text_device_id = int(os.environ.get("LINGBOT_WORLD_TEXT_DEVICE_ID", "0"))
    dit_device = os.environ.get("LINGBOT_WORLD_DIT_DEVICE", "cuda")
    max_area = int(os.environ.get("LINGBOT_WORLD_MAX_AREA", str(480 * 832)))

    pipeline = LingBotWorldFastPipeline(device=dit_device, torch_dtype=torch.bfloat16)
    pipeline.init(
        mm,
        LingBotWorldFastPipelineConfig(
            checkpoint_dir=checkpoint_dir,
            fast_checkpoint_subdir=fast_subdir,
            vae_config=ModelRuntimeConfig(
                device_type=vae_device_type,
                device_id=vae_device_id,
                torch_dtype=torch.bfloat16,
            ),
            text_encoding_config=ModelRuntimeConfig(
                device_type=text_device_type,
                device_id=text_device_id,
                torch_dtype=torch.bfloat16,
            ),
            dit_torch_dtype=torch.bfloat16,
            control_type=os.environ.get("LINGBOT_WORLD_CONTROL_TYPE", "cam"),
            max_area=max_area,
        ),
    )
    return LingBotWorldFastService(pipeline)
