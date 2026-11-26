"""Contracts for the public LTX-Video 2B TeleFuser pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from telefuser.core.config import QuantType
from telefuser.pipelines.ltx_video_2b.pipeline import LTXVideo2BPipeline, _nvfp4_runtime


def test_nvfp4_profiles() -> None:
    quality = _nvfp4_runtime("quality").quant_config
    full = _nvfp4_runtime("full").quant_config
    assert quality.enabled and quality.quant_type == QuantType.NVFP4
    assert quality.quantize_modules == (".ff.",)
    assert full.quantize_modules is None
    with pytest.raises(ValueError, match="nvfp4_profile"):
        _nvfp4_runtime("unknown")


def test_pipeline_validates_geometry_and_forwards_request() -> None:
    stage = SimpleNamespace(nvfp4_linear_count=56)
    stage.process = lambda prompt, **kwargs: np.zeros((kwargs["num_frames"], kwargs["height"], kwargs["width"], 3))
    pipeline = LTXVideo2BPipeline(device="cpu", torch_dtype=torch.bfloat16)
    pipeline.generation_stage = stage
    frames = pipeline("prompt", height=64, width=96, num_frames=9, num_inference_steps=1)
    assert frames.shape == (9, 64, 96, 3)
    assert pipeline.nvfp4_linear_count == 56
    with pytest.raises(ValueError, match="multiples of 32"):
        pipeline("prompt", height=65, width=96, num_frames=9)
    with pytest.raises(ValueError, match="8.k.1"):
        pipeline("prompt", height=64, width=96, num_frames=8)


def test_benchmark_uses_public_telefuser_pipeline() -> None:
    source = open("tools/validation/benchmark_ltx_nvfp4.py", encoding="utf-8").read()
    assert "from telefuser.pipelines.ltx_video_2b import LTXVideo2BPipeline" in source
    assert "from diffusers import LTXPipeline" not in source
    assert "quantize_(" not in source
