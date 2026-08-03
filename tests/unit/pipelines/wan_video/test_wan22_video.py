from __future__ import annotations

from unittest.mock import Mock

import torch

from telefuser.core.config import ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.wan_video import wan22_video


class _FakeStage:
    def __init__(self, _name: str, _module_manager: object, runtime_config: ModelRuntimeConfig, *_args: object) -> None:
        self.model_runtime_config = runtime_config


class _FakeWorker:
    instances: list["_FakeWorker"] = []

    def __init__(self, stage: _FakeStage, **kwargs: object) -> None:
        self.stage = stage
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


class _FakeChannel:
    instances: list["_FakeChannel"] = []

    def __init__(self, consumer_world_size: int, *, timeout: int) -> None:
        self.consumer_world_size = consumer_world_size
        self.timeout = timeout
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


def _parallel_runtime_config() -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        parallel_config=ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2, timeout=123),
    )


def test_parallel_vae_and_denoise_use_bidirectional_tensor_channels(monkeypatch) -> None:
    _FakeWorker.instances = []
    _FakeChannel.instances = []
    monkeypatch.setattr(wan22_video, "VAEStage", _FakeStage)
    monkeypatch.setattr(wan22_video, "MoeDitDenoisingStage", _FakeStage)
    monkeypatch.setattr(wan22_video, "TextEncodingStage", _FakeStage)
    monkeypatch.setattr(wan22_video, "ParallelWorker", _FakeWorker)
    monkeypatch.setattr(wan22_video, "WorkerTensorChannel", _FakeChannel)
    module_manager = Mock()
    module_manager.get_model_info.return_value = {}
    config = wan22_video.Wan22VideoPipelineConfig(
        vae_config=_parallel_runtime_config(),
        dit_high_config=_parallel_runtime_config(),
        dit_low_config=_parallel_runtime_config(),
        enable_vae_parallel=True,
        enable_denoising_parallel=True,
    )
    pipeline = wan22_video.Wan22VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)

    pipeline.init(module_manager, config)

    assert pipeline._uses_direct_tensor_handoff
    assert len(_FakeChannel.instances) == 2
    vae_to_denoise, denoise_to_vae = _FakeChannel.instances
    vae_worker, denoise_worker = _FakeWorker.instances
    assert vae_worker.kwargs == {
        "tensor_output_channel": vae_to_denoise,
        "tensor_output_methods": ("process",),
        "tensor_input_channels": (denoise_to_vae,),
    }
    assert denoise_worker.kwargs == {
        "tensor_output_channel": denoise_to_vae,
        "tensor_output_methods": ("process",),
        "tensor_input_channels": (vae_to_denoise,),
    }

    pipeline.close()

    assert all(worker.closed for worker in _FakeWorker.instances)
    assert all(channel.closed for channel in _FakeChannel.instances)
