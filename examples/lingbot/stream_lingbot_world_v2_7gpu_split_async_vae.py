"""LingBot-World v2 7-GPU streaming service with split condition/VAE/DiT placement."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService
from telefuser.pipelines.lingbot_world_v2.pipeline import LingBotWorldV2Pipeline, LingBotWorldV2PipelineConfig

RESOLUTION_AREAS = {"480p": 480 * 832, "720p": 720 * 1280}

PPL_CONFIG = dict(
    parallelism=5,
    total_gpus=7,
    condition_device_id=0,
    async_vae_device_id=1,
    dit_device_ids=[2, 3, 4, 5, 6],
    control_mode="cam",
    resolution="480p",
    target_fps=16,
    max_duration_seconds=120.0,
    chunk_size=4,
    frame_policy="truncate",
    sample_shift=10.0,
    max_attention_size=None,
    attn_impl=AttnImplType.TORCH_SDPA,
    enable_fsdp=False,
    local_attn_size=18,
    sink_size=6,
    timestep_indices=(0, 250, 500, 750),
    vae_torch_dtype=torch.float32,
    torch_dtype=torch.bfloat16,
    enable_async_vae=True,
    vae_queue_size=1,
    enable_condition_prefetch=True,
)


def get_pipeline(
    model_root: str | None = None,
    v2_model_root: str | None = None,
) -> LingBotWorldV2Pipeline:
    if model_root is None or v2_model_root is None:
        model_zoo_path = Path(os.environ["TF_MODEL_ZOO_PATH"]).expanduser()
        default_model_root = str(model_zoo_path / "Wan2.2-I2V-A14B")
        default_v2_model_root = str(model_zoo_path / "lingbot" / "lingbot-world-v2-14b-causal-fast" / "transformers")
    else:
        default_model_root, default_v2_model_root = model_root, v2_model_root

    dtype = PPL_CONFIG["torch_dtype"]
    vae_dtype = PPL_CONFIG["vae_torch_dtype"]
    pipeline = LingBotWorldV2Pipeline(device="cuda", torch_dtype=dtype)
    pipeline.init(
        LingBotWorldV2PipelineConfig(
            checkpoint_dir=model_root or default_model_root,
            fast_checkpoint_path=v2_model_root or default_v2_model_root,
            vae_config=ModelRuntimeConfig(
                device_type="cuda",
                device_id=PPL_CONFIG["condition_device_id"],
                torch_dtype=vae_dtype,
            ),
            async_vae_config=ModelRuntimeConfig(
                device_type="cuda",
                device_id=PPL_CONFIG["async_vae_device_id"],
                torch_dtype=vae_dtype,
            ),
            text_encoding_config=ModelRuntimeConfig(
                device_type="cuda",
                device_id=PPL_CONFIG["condition_device_id"],
                torch_dtype=dtype,
            ),
            dit_torch_dtype=dtype,
            control_type=PPL_CONFIG["control_mode"],
            max_area=RESOLUTION_AREAS[PPL_CONFIG["resolution"]],
            local_attn_size=PPL_CONFIG["local_attn_size"],
            sink_size=PPL_CONFIG["sink_size"],
            timestep_indices=PPL_CONFIG["timestep_indices"],
            attention_config=AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"]),
            parallel_config=ParallelConfig(
                device_ids=list(PPL_CONFIG["dit_device_ids"]),
                sp_ulysses_degree=PPL_CONFIG["parallelism"],
                enable_fsdp=PPL_CONFIG["enable_fsdp"],
            ),
            enable_async_vae=PPL_CONFIG["enable_async_vae"],
            vae_queue_size=PPL_CONFIG["vae_queue_size"],
            enable_condition_prefetch=PPL_CONFIG["enable_condition_prefetch"],
        )
    )
    return pipeline


def get_service(gpu_num: int = PPL_CONFIG["total_gpus"]) -> LingBotWorldFastService:
    if gpu_num < PPL_CONFIG["total_gpus"]:
        raise ValueError(
            f"Split async VAE config needs at least {PPL_CONFIG['total_gpus']} visible GPUs, got {gpu_num}"
        )
    pipeline = get_pipeline()
    return LingBotWorldFastService(
        pipeline,
        default_fps=PPL_CONFIG["target_fps"],
        max_generation_seconds=PPL_CONFIG["max_duration_seconds"],
        default_session_config={
            "control_mode": PPL_CONFIG["control_mode"],
            "max_duration_seconds": PPL_CONFIG["max_duration_seconds"],
            "chunk_size": PPL_CONFIG["chunk_size"],
            "frame_policy": PPL_CONFIG["frame_policy"],
            "sample_shift": PPL_CONFIG["sample_shift"],
            "max_attention_size": PPL_CONFIG["max_attention_size"],
        },
    )
