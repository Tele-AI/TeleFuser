# SPDX-License-Identifier: Apache-2.0
"""Load released PAN2 components without adding a framework registry or loader."""

from pathlib import Path

import torch

from telefuser.core.config import (
    AttentionConfig,
    ModelRuntimeConfig,
    OffloadConfig,
    ParallelConfig,
    QuantConfig,
    QuantKernelBackend,
    QuantType,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.models.pan2_conditioning import PAN2VAE, PAN2TextEncoder
from telefuser.models.pan2_dit import PAN2DiT
from telefuser.pipelines.pan2 import PAN2Pipeline, PAN2PipelineConfig


def build_pipeline(
    model_root: str,
    *,
    device_ids: list[int],
    sp: int = 1,
    tp: int = 1,
    fp8: bool = False,
    sol: bool = False,
    sol_fp8: bool = False,
) -> PAN2Pipeline:
    root = Path(model_root)
    torch.cuda.set_device(device_ids[0])
    parallel = ParallelConfig(device_ids=device_ids, sp_ulysses_degree=sp, tp_degree=tp)
    parallel.validate()
    if sol_fp8 and not sol:
        raise ValueError("--sol_fp8 requires --sol")
    offload = OffloadConfig(offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD, pin_cpu_memory=False)
    attention = (
        AttentionConfig.sol_attention(dense_timesteps=5, dense_layers=1, sol_fp8=sol_fp8) if sol else AttentionConfig()
    )
    quant = QuantConfig(enabled=fp8, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL)
    runtime = ModelRuntimeConfig(
        device_id=device_ids[0],
        offload_config=offload,
        parallel_config=parallel,
        attention_config=attention,
        quant_config=quant,
    )
    manager = ModuleManager(device="cpu", torch_dtype=torch.bfloat16)
    for component, name, model_class in (
        ("transformer", "pan2_transformer", PAN2DiT),
        ("text_encoder", "pan2_text_encoder", PAN2TextEncoder),
        ("vae", "pan2_vae", PAN2VAE),
    ):
        manager.load_model(
            sorted(str(p) for p in (root / component).glob("*.safetensors")),
            name=name,
            model_class=model_class,
            model_resource="diffusers",
            converter_kwargs={"config_path": root / component / "config.json"},
            low_cpu_mem_usage=True,
        )
    pipeline = PAN2Pipeline(device=f"cuda:{device_ids[0]}", torch_dtype=torch.bfloat16)
    pipeline.init(
        manager,
        PAN2PipelineConfig(
            tokenizer_path=str(root / "tokenizer"),
            scheduler_config_path=str(root / "scheduler/scheduler_config.json"),
            dit_config=runtime,
            text_encoder_config=ModelRuntimeConfig(device_id=device_ids[0], offload_config=offload),
            vae_config=ModelRuntimeConfig(device_id=device_ids[0], offload_config=offload),
        ),
    )
    return pipeline
