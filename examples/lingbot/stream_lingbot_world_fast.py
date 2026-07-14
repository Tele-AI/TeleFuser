"""LingBot-World-Fast bidirectional streaming service example.

Run the four-GPU stream service:
    TF_MODEL_ZOO_PATH=/path/to/model_zoo \
    telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py \
        --gpu-num 4 -p 8088 --skip-validation

Select physical GPUs with CUDA_VISIBLE_DEVICES. For example, use physical GPUs
2 and 3 with --gpu-num 2 and CUDA_VISIBLE_DEVICES=2,3.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline, LingBotWorldFastPipelineConfig
from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService

RESOLUTION_AREAS = {"480p": 480 * 832, "720p": 720 * 1280}

PPL_CONFIG = dict(
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
    model_root: str | None = None,
    fast_model_root: str | None = None,
) -> LingBotWorldFastPipeline:
    """Load LingBot-World-Fast with internal multi-GPU workers."""
    if model_root is None or fast_model_root is None:
        model_zoo_path = Path(os.environ["TF_MODEL_ZOO_PATH"]).expanduser()
        default_model_root = str(model_zoo_path / "Wan2.2-I2V-A14B")
        default_fast_model_root = str(model_zoo_path / "lingbot" / "lingbot-world-fast")
    else:
        default_model_root, default_fast_model_root = model_root, fast_model_root
    if parallelism < 1:
        raise ValueError(f"parallelism must be positive, got {parallelism}")
    dtype = PPL_CONFIG["torch_dtype"]
    pipeline = LingBotWorldFastPipeline(device="cuda", torch_dtype=dtype)
    pipeline.init(
        LingBotWorldFastPipelineConfig(
            checkpoint_dir=model_root or default_model_root,
            fast_checkpoint_path=fast_model_root or default_fast_model_root,
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


def get_service(gpu_num: int = PPL_CONFIG["parallelism"]) -> LingBotWorldFastService:
    """Build the service loaded by the TeleFuser stream server."""
    pipeline = get_pipeline(parallelism=gpu_num)
    return LingBotWorldFastService(pipeline, default_fps=PPL_CONFIG["target_fps"])
