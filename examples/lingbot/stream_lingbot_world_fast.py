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

TF_MODEL_ZOO_PATH = Path(os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")).expanduser().resolve()
PPL_CONFIG = dict(
    name="lingbot_world_fast_stream",
    # LingBot-World-Fast reuses the Wan2.2 base weights (VAE + T5 text encoder + ``google/umt5-xxl``
    # tokenizer), which ship in the shared Wan2.2-I2V-A14B directory. The DiT fast weights live in their
    # own ``lingbot-world-fast`` directory, passed as an absolute path so the pipeline keeps it standalone
    # rather than nesting it under ``checkpoint_dir``.
    checkpoint_dir=str(TF_MODEL_ZOO_PATH / "Wan2.2-I2V-A14B"),
    fast_checkpoint_subdir=str(TF_MODEL_ZOO_PATH / "lingbot-world-fast"),
    control_type="cam",
    vae_device="cuda",
    vae_device_id=0,
    text_device="cuda",
    text_device_id=0,
    dit_device="cuda",
    max_area=480 * 832,
    torch_dtype=torch.bfloat16,
)


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
    dtype = PPL_CONFIG["torch_dtype"]
    mm = _LocalModuleManager()

    pipeline = LingBotWorldFastPipeline(device=PPL_CONFIG["dit_device"], torch_dtype=dtype)
    pipeline.init(
        mm,
        LingBotWorldFastPipelineConfig(
            checkpoint_dir=PPL_CONFIG["checkpoint_dir"],
            fast_checkpoint_subdir=PPL_CONFIG["fast_checkpoint_subdir"],
            vae_config=ModelRuntimeConfig(
                device_type=PPL_CONFIG["vae_device"],
                device_id=PPL_CONFIG["vae_device_id"],
                torch_dtype=dtype,
            ),
            text_encoding_config=ModelRuntimeConfig(
                device_type=PPL_CONFIG["text_device"],
                device_id=PPL_CONFIG["text_device_id"],
                torch_dtype=dtype,
            ),
            dit_torch_dtype=dtype,
            control_type=PPL_CONFIG["control_type"],
            max_area=PPL_CONFIG["max_area"],
        ),
    )
    return LingBotWorldFastService(pipeline)
