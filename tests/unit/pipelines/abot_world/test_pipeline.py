from __future__ import annotations

import pytest
import torch

from examples.abot_world._loader import _attention_backend, _env_flag
from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.abot_world import ABotWorldPipeline
from telefuser.pipelines.abot_world.pipeline import ABotWorldPipelineConfig


def test_cuda_graph_configuration_is_opt_in() -> None:
    assert ABotWorldPipelineConfig().cuda_graph_enabled is False


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
    ],
)
def test_cuda_graph_environment_flag_is_explicit(
    monkeypatch: pytest.MonkeyPatch, raw_value: str, expected: bool
) -> None:
    monkeypatch.setenv("TELEFUSER_ABOT_CUDA_GRAPH_ENABLED", raw_value)

    assert _env_flag("TELEFUSER_ABOT_CUDA_GRAPH_ENABLED") is expected


def test_cuda_graph_environment_flag_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEFUSER_ABOT_CUDA_GRAPH_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="TELEFUSER_ABOT_CUDA_GRAPH_ENABLED must be a boolean"):
        _env_flag("TELEFUSER_ABOT_CUDA_GRAPH_ENABLED")


def test_attention_environment_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEFUSER_ABOT_ATTENTION", "unknown")

    with pytest.raises(ValueError, match="Unsupported TELEFUSER_ABOT_ATTENTION"):
        _attention_backend(0)


def test_action_context_uses_official_wasd_ijkl_channel_layout() -> None:
    action = ABotWorldPipeline.build_action_context(
        {"W": True, "D": True, "L": True},
        latent_frames=4,
        height=32,
        width=64,
        device="cpu",
        dtype=torch.float32,
    )
    assert action.shape == (1, 32, 4, 32, 64)
    # Every key is expanded into four contiguous channels, in W,A,S,D,I,J,K,L order.
    assert torch.all(action[:, 0:4] == 1)
    assert torch.all(action[:, 12:16] == 1)
    assert torch.all(action[:, 28:32] == 1)
    assert torch.all(action[:, 4:12] == 0)
    assert torch.all(action[:, 16:28] == 0)


def test_action_context_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown ABot action keys"):
        ABotWorldPipeline.build_action_context(
            {"SPACE": True}, latent_frames=1, height=32, width=32, device="cpu", dtype=torch.float32
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            ABotWorldPipelineConfig(
                dit_config=ModelRuntimeConfig(
                    device_type="cpu",
                    parallel_config=ParallelConfig(device_ids=[0, 1], dp_degree=2),
                )
            ),
            "exactly one GPU",
        ),
        (ABotWorldPipelineConfig(latent_frames=2), "1 mod 3"),
        (ABotWorldPipelineConfig(local_attn_size=6, sink_size=6), "smaller than local_attn_size"),
        (ABotWorldPipelineConfig(height=481), "divisible by 32"),
    ],
)
def test_pipeline_rejects_release_incompatible_configuration(config: ABotWorldPipelineConfig, message: str) -> None:
    pipeline = ABotWorldPipeline(device="cpu")

    with pytest.raises(ValueError, match=message):
        pipeline.init(ModuleManager(device="cpu"), config)
