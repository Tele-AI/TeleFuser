"""Internal ABot-World checkpoint loader for the interactive example."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    ModelRuntimeConfig,
    OffloadConfig,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.models.abot_world_dit import ABotWorldDiT
from telefuser.models.taew2_2 import TAEHV
from telefuser.models.wan22_video_vae import Wan22VideoVAE
from telefuser.models.wan_video_text_encoder import WanTextEncoder
from telefuser.ops.attention.backends import (
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_4_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    sageattention,
)
from telefuser.pipelines.abot_world import ABotWorldPipeline, ABotWorldPipelineConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_ROOT = (
    Path(os.environ.get("TF_MODEL_ZOO_PATH", _PROJECT_ROOT.parent / "model_zoo")) / "ABot-World-0-5B-LF"
)
DEFAULT_PROMPT = "A smooth first-person exploration through a vivid natural landscape."


def _env_flag(name: str, default: bool = False) -> bool:
    """Read an explicit boolean environment override for an experimental path."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _attention_backend(device_id: int) -> AttnImplType:
    """Choose the explicitly requested ABot attention backend when safe."""
    requested = os.environ.get("TELEFUSER_ABOT_ATTENTION", "auto").strip().lower()
    if requested in {"sage_sm90", "sageattn_sm90", "sage_attn_sm90"}:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(device_id) != (9, 0):
            raise RuntimeError("TELEFUSER_ABOT_ATTENTION=sage_sm90 requires an SM90 H100-class CUDA device")
        if (
            not SAGE_ATTN_AVAILABLE
            or sageattention is None
            or not hasattr(sageattention, "sageattn_qk_int8_pv_fp8_cuda_sm90")
        ):
            raise RuntimeError("sage_sm90 requires the tf_kernel SM90 SageAttention extension")
        return AttnImplType.SAGE_ATTN_2_8_8_SM90
    if requested not in {"", "auto"}:
        raise ValueError(f"Unsupported TELEFUSER_ABOT_ATTENTION value {requested!r}; use 'auto' or 'sage_sm90'.")
    if FLASH_ATTN_4_AVAILABLE:
        return AttnImplType.FLASH_ATTN_4
    if FLASH_ATTN_3_AVAILABLE:
        return AttnImplType.FLASH_ATTN_3
    return AttnImplType.TORCH_SDPA


def get_pipeline(
    model_root: str | Path = _DEFAULT_MODEL_ROOT,
    *,
    height: int = 480,
    width: int = 832,
    latent_frames: int = 31,
    device_id: int = 0,
    pipeline_class: type[ABotWorldPipeline] = ABotWorldPipeline,
) -> ABotWorldPipeline:
    """Load the downloaded ABot checkpoint with VAE/T5 model CPU offload."""
    root = Path(model_root).expanduser()
    required = (
        "diffusion_pytorch_model.safetensors",
        "Wan2.2_VAE.pth",
        "taew2_2.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"ABot model root {root} is missing: {', '.join(missing)}")

    model_manager = ModuleManager(device="cpu")
    model_manager.load_model(
        str(root / "Wan2.2_VAE.pth"),
        name="wan_video_vae",
        model_class=Wan22VideoVAE,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_manager.add_module(
        TAEHV(str(root / "taew2_2.pth")).eval().requires_grad_(False),
        name="abot_world_taew_decoder",
        path=str(root / "taew2_2.pth"),
    )
    model_manager.load_model(
        str(root / "models_t5_umt5-xxl-enc-bf16.pth"),
        name="wan_video_text_encoder",
        model_class=WanTextEncoder,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model_manager.load_model(
        str(root / "diffusion_pytorch_model.safetensors"),
        name="abot_world_dit",
        model_class=ABotWorldDiT,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    cpu_offload = OffloadConfig(offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD)
    device = f"cuda:{device_id}"
    pipeline = pipeline_class(device=device, torch_dtype=torch.bfloat16)
    pipeline.init(
        model_manager,
        ABotWorldPipelineConfig(
            vae_config=ModelRuntimeConfig(
                device_type="cuda", device_id=device_id, torch_dtype=torch.float32, offload_config=cpu_offload
            ),
            text_encoding_config=ModelRuntimeConfig(
                device_type="cuda", device_id=device_id, torch_dtype=torch.bfloat16, offload_config=cpu_offload
            ),
            dit_config=ModelRuntimeConfig(
                device_type="cuda",
                device_id=device_id,
                torch_dtype=torch.bfloat16,
                attention_config=AttentionConfig.dense_attention(_attention_backend(device_id)),
            ),
            height=height,
            width=width,
            latent_frames=latent_frames,
            local_attn_size=18,
            sink_size=6,
            cuda_graph_enabled=_env_flag("TELEFUSER_ABOT_CUDA_GRAPH_ENABLED"),
        ),
    )
    return pipeline
