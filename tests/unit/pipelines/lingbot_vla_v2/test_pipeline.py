from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_vla_v2 import (
    LingBotVlaV2Observation,
    LingBotVlaV2Pipeline,
    LingBotVlaV2PipelineConfig,
)
from telefuser.pipelines.lingbot_vla_v2.robot_profile import ROBOTWIN_CAMERA_KEYS


class _ImageProcessor:
    def __call__(self, images: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        assert len(images) == 3
        assert all(image.shape == (3, 8, 8) for image in images)
        return {
            "pixel_values": torch.zeros(12, 6),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs) -> str:
        return messages[0]["content"]

    def __call__(self, prompts, **kwargs) -> dict[str, torch.Tensor]:
        length = kwargs["max_length"]
        return {
            "input_ids": torch.zeros(1, length, dtype=torch.long),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            max_state_dim=55,
            max_action_dim=55,
            n_action_steps=4,
            tokenizer_max_length=6,
            checkpoint_variant="base",
            policy_verified=False,
            verification_status="unverified_official_6b_base",
        )
        self.sample_count = 0

    def sample_actions(self, **inputs) -> torch.Tensor:
        self.sample_count += 1
        assert inputs["state"].shape == (1, 55)
        assert inputs["images"].shape == (1, 3, 4, 6)
        return torch.zeros(1, self.config.n_action_steps, self.config.max_action_dim, device=self.anchor.device)


def test_pipeline_returns_normalized_canonical_action_chunk() -> None:
    policy = _Policy()
    processor = SimpleNamespace(image_processor=_ImageProcessor(), tokenizer=_Tokenizer())
    manager = ModuleManager(torch_dtype=torch.float32, device="cpu")
    manager.add_module(policy, "lingbot_vla_v2")
    manager.add_module(processor, "lingbot_vla_v2_processor")
    pipeline = LingBotVlaV2Pipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.init(
        manager,
        LingBotVlaV2PipelineConfig(
            policy_config=ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
            image_size=8,
        ),
    )
    observation = LingBotVlaV2Observation(
        task="pick up the block",
        state=[0.0] * 14,
        images={key: np.zeros((8, 8, 3), dtype=np.uint8) for key in ROBOTWIN_CAMERA_KEYS},
    )

    try:
        chunk = pipeline(observation, seed=7)
    finally:
        pipeline.close()

    assert chunk.horizon == 4
    assert chunk.action_dim == 55
    assert chunk.canonical_normalized_actions.shape == (4, 55)
    assert chunk.checkpoint_variant == "base"
    assert chunk.policy_verified is False
    assert chunk.verification_status == "unverified_official_6b_base"


def test_pipeline_prepares_resident_policy_and_disables_per_call_cache_eviction() -> None:
    policy = _Policy()
    processor = SimpleNamespace(image_processor=_ImageProcessor(), tokenizer=_Tokenizer())
    manager = ModuleManager(torch_dtype=torch.float32, device="cpu")
    manager.add_module(policy, "lingbot_vla_v2")
    manager.add_module(processor, "lingbot_vla_v2_processor")
    pipeline = LingBotVlaV2Pipeline(device="cpu", torch_dtype=torch.float32)
    pipeline.init(
        manager,
        LingBotVlaV2PipelineConfig(
            policy_config=ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
            image_size=8,
        ),
    )

    assert pipeline.clear_memory_after_call is False
    assert pipeline.policy_stage.onload_models_flag is False

    pipeline.prepare_for_inference()
    assert pipeline.policy_stage.onload_models_flag is True

    pipeline.warmup()
    assert policy.sample_count == 1

    pipeline.close()
    assert pipeline.policy_stage.onload_models_flag is False
