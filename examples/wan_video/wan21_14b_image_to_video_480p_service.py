from __future__ import annotations

import os

import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template
from telefuser.utils.video import save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="wan21_14B_i2v_480p_service",
    model_root=TF_MODEL_ZOO_PATH + "/Wan2.1-I2V-14B-480P",
    negative_prompt="Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="480p",
    cfg_scale=5.0,
    cfg_skip_ratio=0.0,
    seed=42,
    tiled=False,
    sample_solver="unipc",
    attn_impl=AttnImplType.TORCH_SDPA,
    sigma_shift=5.0,
    target_fps=16,
)

PIPELINE_MANIFEST = build_pipeline_manifest(
    pipeline_name=PPL_CONFIG["name"],
    supported_tasks=["i2v"],
    task_contracts={
        "i2v": build_task_contract_template(
            "i2v",
            parameter_overrides={
                "prompt": {
                    "description": "Positive guidance text prompt for Wan2.1 I2V 480P inference.",
                    "required": True,
                },
                "negative_prompt": {
                    "default": "",
                    "description": "Optional negative prompt appended before the built-in Wan2.1 negative prompt.",
                },
                "seed": {
                    "default": PPL_CONFIG["seed"],
                    "description": "Random seed used for the denoising trajectory.",
                },
                "resolution": {
                    "default": PPL_CONFIG["resolution"],
                    "description": "Output resolution for this service example.",
                    "enum": ["480p"],
                },
                "target_video_length": {
                    "default": 5,
                    "description": "Accepted by the OpenAI-compatible API but ignored by this fixed-workload benchmark service.",
                    "exposed": False,
                },
            },
            excluded_parameters=("aspect_ratio",),
        )
    },
)


def get_dit_path_list(model_root: str) -> list[str]:
    """Generate DiT model shard paths based on model_root."""
    return [
        f"{model_root}/diffusion_pytorch_model-00001-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00002-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00003-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00004-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00005-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00006-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00007-of-00007.safetensors",
    ]


def get_pipeline(parallelism: int = 1, model_root: str = PPL_CONFIG["model_root"]):
    """Build Wan2.1 14B 480P I2V pipeline for TeleFuser service."""
    dit_path_list = get_dit_path_list(model_root)
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,
    )
    module_manager.load_models([f"{model_root}/Wan2.1_VAE.pth"], torch_dtype=torch.bfloat16)
    module_manager.load_models([dit_path_list], torch_dtype=torch.bfloat16)
    module_manager.load_models([f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth"], torch_dtype=torch.bfloat16)

    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.clip_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_clip_stage = True

    if parallelism > 1:
        cfg_scale = PPL_CONFIG["cfg_scale"]
        if cfg_scale > 1:
            pipe_config.dit_config.parallel_config.cfg_degree = 2
            pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism // 2
        else:
            pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism

        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.enable_denoising_parallel = True

    pipe.init(module_manager, pipe_config)
    return pipe


def run(
    pipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    seed: int = PPL_CONFIG["seed"],
    resolution: str = PPL_CONFIG["resolution"],
):
    """Convert a static image to a video sequence using the fixed 480P Wan2.1 benchmark workload."""
    if resolution != "480p":
        raise ValueError(f"Unsupported benchmark resolution: {resolution}")

    # Keep TeleFuser aligned with the official Diffusers Wan2.1-I2V-14B-480P
    # benchmark shape instead of the repo-wide 480p area heuristic.
    width, height = 832, 480
    video = pipeline(
        prompt=prompt,
        input_image=image,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
    )
    return video


def run_with_file(
    pipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    output_path: str,
    resolution: str = PPL_CONFIG["resolution"],
    **kwargs,
) -> dict:
    """Run the fixed benchmark workload and save the result to file."""
    video = run(
        pipeline,
        image,
        prompt,
        negative_prompt,
        seed,
        resolution,
    )
    save_video(
        video,
        output_path,
        fps=PPL_CONFIG["target_fps"],
        quality=6,
    )
    return {"output_path": str(output_path)}
