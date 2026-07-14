"""Wan2.1 14B Text-to-Video (T2V) example.


This example demonstrates text-to-video generation using Wan2.1 14B model
without requiring an input image.

Usage:
    python wan21_14b_text_to_video_h100.py --prompt "A cat playing piano"
"""

import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="wan21_14B_t2v_h100",
    model_root=TF_MODEL_ZOO_PATH + "/Wan2.1-T2V-14B",
    negative_prompt="Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="720p",
    cfg_scale=5.0,
    seed=42,
    tiled=False,
    sample_solver="unipc",
    attn_impl=AttnImplType.TORCH_SDPA,
    sigma_shift=5.0,
    target_fps=16,
)


def get_dit_path_list(model_root):
    """Generate DiT model paths based on model_root."""
    return [
        f"{model_root}/diffusion_pytorch_model-00001-of-00006.safetensors",
        f"{model_root}/diffusion_pytorch_model-00002-of-00006.safetensors",
        f"{model_root}/diffusion_pytorch_model-00003-of-00006.safetensors",
        f"{model_root}/diffusion_pytorch_model-00004-of-00006.safetensors",
        f"{model_root}/diffusion_pytorch_model-00005-of-00006.safetensors",
        f"{model_root}/diffusion_pytorch_model-00006-of-00006.safetensors",
    ]


def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """
    Args:
        parallelism (int): Number of parallel GPUs for inference: 2, 4 or 8
        model_root (str): Root directory of the model
    """
    dit_path_list = get_dit_path_list(model_root)
    # Load models
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [dit_path_list],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_clip_stage = False  # T2V model, no CLIP stage
    if parallelism > 1:
        # Configure parallel based on cfg_scale
        # For cfg_scale > 1: cfg_degree=2, sp_ulysses_degree=parallelism//2
        # For cfg_scale == 1: cfg_degree=1, sp_ulysses_degree=parallelism
        cfg_scale = PPL_CONFIG["cfg_scale"]

        if cfg_scale > 1:
            pipe_config.dit_config.parallel_config.cfg_degree = 2
            pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism // 2
        else:
            pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism

        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.enable_denoising_parallel = True
        pipe_config.dit_config.parallel_config.timeout = 3600
    pipe.init(module_manager, pipe_config)
    return pipe


def run(
    pipeline,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    aspect_ratio="16:9",
):
    """
    Generate video from text prompt.

    Args:
        pipeline: Preloaded video generation pipeline object
        prompt (str): Positive guidance text prompt
        negative_prompt (str, optional): Negative guidance prompt
        seed (int, optional): Random seed
        resolution (str): Resolution such as 720p, 480p
        aspect_ratio (str): Aspect ratio such as 16:9, 9:16, 1:1

    Returns:
        List[PIL.Image]: Generated video sequence
    """
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=16,
        width_division_factor=16,
    )
    video = pipeline(
        prompt=prompt,
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


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--prompt",
    default="A stylish woman walking down a Tokyo street filled with warm golden sunlight and cherry blossoms floating in the wind. The camera follows her from behind as she strolls leisurely, creating a cinematic atmosphere.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--resolution", default=PPL_CONFIG["resolution"], help="480p or 720p")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio: 16:9, 9:16, 1:1, etc.")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Root directory of the model")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
def main(
    gpu_num,
    prompt,
    negative_prompt,
    resolution,
    aspect_ratio,
    model_root,
    seed,
):
    """Text to video conversion using Wan2.1 14B model"""
    pipe = get_pipeline(gpu_num, model_root)

    # Run inference
    start = time.time()
    video = run(pipe, prompt, negative_prompt, seed=seed, resolution=resolution, aspect_ratio=aspect_ratio)
    elapsed_time = time.time() - start

    print(f"Video generation time: {elapsed_time:.2f} seconds")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
