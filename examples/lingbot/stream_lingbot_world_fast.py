"""LingBot-World-Fast bidirectional streaming service example.

Run the four-GPU stream service:
    telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py \
        -p 8088 --skip-validation
"""

from __future__ import annotations

import torch

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_world_fast.pipeline import (
    LingBotWorldFastPipeline,
    LingBotWorldFastPipelineConfig,
)
from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService

TF_MODEL_ZOO_PATH = "/hhb-data/aigc/model_zoo"
RESOLUTION_AREAS = {"480p": 480 * 832, "720p": 720 * 1280}

PPL_CONFIG = dict(
    name="lingbot_world_fast_stream",
    model_root=TF_MODEL_ZOO_PATH + "/Wan2.2-I2V-A14B",
    fast_model_root=TF_MODEL_ZOO_PATH + "/lingbot-world-fast",
    parallelism=4,
    control_mode="cam",
    resolution="480p",
    target_fps=16,
    attn_impl=AttnImplType.SAGE_ATTN_2_8_8_SM90,
    enable_fsdp=False,
    local_attn_size=-1,
    sink_size=0,
    torch_dtype=torch.bfloat16,
)


def get_pipeline(
    parallelism: int = PPL_CONFIG["parallelism"],
    model_root: str = PPL_CONFIG["model_root"],
    fast_model_root: str = PPL_CONFIG["fast_model_root"],
) -> LingBotWorldFastPipeline:
    """Load LingBot-World-Fast with internal multi-GPU workers."""
    if parallelism < 1:
        raise ValueError(f"parallelism must be positive, got {parallelism}")

    dtype = PPL_CONFIG["torch_dtype"]
    pipeline = LingBotWorldFastPipeline(device="cuda", torch_dtype=dtype)
    pipeline.init(
        ModuleManager(device="cpu"),
        LingBotWorldFastPipelineConfig(
            checkpoint_dir=model_root,
            fast_checkpoint_subdir=fast_model_root,
            vae_config=ModelRuntimeConfig(device_type="cuda", device_id=0, torch_dtype=dtype),
            text_encoding_config=ModelRuntimeConfig(device_type="cuda", device_id=0, torch_dtype=dtype),
            dit_torch_dtype=dtype,
            control_type=PPL_CONFIG["control_mode"],
            max_area=RESOLUTION_AREAS[PPL_CONFIG["resolution"]],
            local_attn_size=PPL_CONFIG["local_attn_size"],
            sink_size=PPL_CONFIG["sink_size"],
            attention_config=AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"]),
            parallel_config=ParallelConfig(
                device_ids=list(range(parallelism)) if parallelism > 1 else None,
                sp_ulysses_degree=parallelism,
                enable_fsdp=PPL_CONFIG["enable_fsdp"],
            ),
        ),
    )
    return pipeline


def get_service() -> LingBotWorldFastService:
    """Build the service loaded by the TeleFuser stream server."""
    pipeline = get_pipeline()
    return LingBotWorldFastService(pipeline, default_fps=PPL_CONFIG["target_fps"])
