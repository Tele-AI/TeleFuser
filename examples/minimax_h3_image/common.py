# SPDX-License-Identifier: Apache-2.0
"""Explicit checkpoint loading through TeleFuser's existing ModuleManager API."""

from pathlib import Path

import torch

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
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
    dit_resident: bool = False,
    resident: bool = False,
    text_encoder_tp: int | None = None,
    vae_parallel: bool | None = None,
    vae_cuda_graph: bool | None = None,
) -> MiniMaxH3ImagePipeline:
    root, encoder = Path(model_root), Path(text_encoder_path)
    torch.cuda.set_device(device_ids[0])
    parallel = ParallelConfig(device_ids=device_ids, sp_ulysses_degree=sp, tp_degree=tp)
    parallel.validate()
    offload = OffloadConfig(offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD, pin_cpu_memory=False)
    no_offload = OffloadConfig(offload_type=WeightOffloadType.NO_CPU_OFFLOAD)
    text_encoder_tp = (len(device_ids) if resident else 1) if text_encoder_tp is None else text_encoder_tp
    if text_encoder_tp < 1 or text_encoder_tp > len(device_ids):
        raise ValueError("text_encoder_tp must be between one and the number of devices")
    vae_parallel = resident and len(device_ids) > 1 if vae_parallel is None else vae_parallel
    vae_cuda_graph = resident if vae_cuda_graph is None else vae_cuda_graph
    dit_offload = no_offload if dit_resident or resident else offload
    encoder_offload = no_offload if resident or text_encoder_tp > 1 else offload
    vae_offload = no_offload if resident or vae_parallel or vae_cuda_graph else offload
    vae_dtype = torch.float32 if resident else torch.bfloat16
    attention = (
        AttentionConfig.sol_attention(dense_timesteps=1, dense_layers=1, sol_fp8=sol_fp8) if sol else AttentionConfig()
    )
    quant = QuantConfig(enabled=fp8, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL)
    runtime = ModelRuntimeConfig(
        device_id=device_ids[0],
        offload_config=dit_offload,
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
        torch_dtype=vae_dtype,
        low_cpu_mem_usage=True,
    )
    if resident:
        # Preserve the FP32 encoder/norms/RoPE while using BF16 decoder GEMMs.
        manager.fetch_module("minimax_h3_image_vae").to(dtype=torch.bfloat16)
    pipeline = MiniMaxH3ImagePipeline(device=f"cuda:{device_ids[0]}", torch_dtype=torch.bfloat16)
    pipeline.init(
        manager,
        MiniMaxH3ImagePipelineConfig(
            processor_path=str(root / "processor"),
            pdd_config_path=str(root / "pdd_config.json"),
            scheduler_config_path=str(root / "scheduler/scheduler_config.json"),
            text_encoder_config=ModelRuntimeConfig(
                device_id=device_ids[0],
                offload_config=encoder_offload,
                attention_config=AttentionConfig.dense_attention(
                    AttnImplType.TORCH_CUDNN if resident else AttnImplType.TORCH_SDPA
                ),
                parallel_config=ParallelConfig(
                    device_ids=device_ids[:text_encoder_tp],
                    tp_degree=text_encoder_tp,
                ),
            ),
            dit_config=runtime,
            vae_config=ModelRuntimeConfig(
                device_id=device_ids[0],
                offload_config=vae_offload,
                torch_dtype=torch.bfloat16,
                parallel_config=ParallelConfig(
                    device_ids=device_ids if vae_parallel else device_ids[:1],
                    tp_degree=len(device_ids) if vae_parallel else 1,
                ),
            ),
            vae_cuda_graph=vae_cuda_graph,
        ),
    )
    return pipeline
