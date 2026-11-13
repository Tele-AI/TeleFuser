# SPDX-License-Identifier: Apache-2.0
"""Explicit checkpoint loading through TeleFuser's existing ModuleManager API."""

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
from telefuser.models.minimax_h3_image_dit import MiniMaxH3ImageDiT
from telefuser.models.minimax_h3_image_encoder import MiniMaxH3ImageEncoder
from telefuser.models.minimax_h3_image_vae import MiniMaxH3ImageVAE
from telefuser.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline, MiniMaxH3ImagePipelineConfig


def build_pipeline(
    model_root: str,
    text_encoder_path: str,
    *,
    device_ids: list[int],
    sp: int = 1,
    tp: int = 1,
    fp8: bool = False,
    sol: bool = False,
    sol_fp8: bool = False,
) -> MiniMaxH3ImagePipeline:
    root, encoder = Path(model_root), Path(text_encoder_path)
    torch.cuda.set_device(device_ids[0])
    parallel = ParallelConfig(device_ids=device_ids, sp_ulysses_degree=sp, tp_degree=tp)
    parallel.validate()
    if sol_fp8 and not sol:
        raise ValueError("--sol_fp8 requires --sol")
    offload = OffloadConfig(offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD, pin_cpu_memory=False)
    attention = (
        AttentionConfig.sol_attention(dense_timesteps=1, dense_layers=1, sol_fp8=sol_fp8) if sol else AttentionConfig()
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
    manager.load_model(
        sorted(str(p) for p in (root / "transformer").glob("model-*.safetensors"))
        + [str(root / "pdd_fused_heads.safetensors")],
        name="minimax_h3_image_transformer",
        model_class=MiniMaxH3ImageDiT,
        converter_kwargs={"config_path": root / "transformer/config.json"},
        low_cpu_mem_usage=True,
    )
    manager.load_model(
        sorted(str(p) for p in encoder.glob("model-*.safetensors")),
        name="minimax_h3_text_encoder",
        model_class=MiniMaxH3ImageEncoder,
        converter_kwargs={"config_path": encoder},
        low_cpu_mem_usage=True,
    )
    manager.load_model(
        str(root / "vae_diffusers/diffusion_pytorch_model.safetensors"),
        name="minimax_h3_image_vae",
        model_class=MiniMaxH3ImageVAE,
        model_resource="diffusers",
        converter_kwargs={"config_path": root / "vae_diffusers/config.json"},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipeline = MiniMaxH3ImagePipeline(device=f"cuda:{device_ids[0]}", torch_dtype=torch.bfloat16)
    pipeline.init(
        manager,
        MiniMaxH3ImagePipelineConfig(
            processor_path=str(root / "processor"),
            pdd_config_path=str(root / "pdd_config.json"),
            scheduler_config_path=str(root / "scheduler/scheduler_config.json"),
            text_encoder_config=ModelRuntimeConfig(device_id=device_ids[0], offload_config=offload),
            dit_config=runtime,
            vae_config=ModelRuntimeConfig(device_id=device_ids[0], offload_config=offload, torch_dtype=torch.bfloat16),
        ),
    )
    return pipeline
