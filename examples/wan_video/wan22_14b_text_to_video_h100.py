"""Wan2.2 14B Text-to-Video (T2V) example.

This example demonstrates text-to-video generation using Wan2.2 14B model
without requiring an input image.

Usage:
    python wan22_14b_text_to_video_h100.py --prompt "A cat playing piano"
"""

import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, FeatureCacheConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan22_video import (
    Wan22VideoPipeline,
    Wan22VideoPipelineConfig,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

PPL_CONFIG = dict(
    name="wan22_A14B_t2v_h100",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-14B",
    negative_prompt="Overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="720p",
    cfg_scale_high=5.0,
    cfg_scale_low=5.0,
    seed=42,
    tiled=False,
    sigma_shift=5.0,
    boundary=0.9,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    dit_high_path_list=[
        "dit_high_noise_model_bfloat16_5d6fd.safetensors",
    ],
    dit_low_path_list=[
        "dit_low_noise_model_bfloat16_c55d6.safetensors",
    ],
    enable_feature_cache_dit_high=True,
    enable_feature_cache_dit_low=True,
    model_type="Wan2_2-T2V-14B",
    target_fps=16,
)


def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """
    Args:
        parallelism (int): Number of parallel GPUs for inference: 2, 4 or 8
        model_root (str): Root directory of the model files
    """
    ppl_config = PPL_CONFIG
    model_root = ppl_config["model_root"]
    # Load models
    module_manager = ModuleManager(device="cpu")
    # vae
    module_manager.load_model(
        f"{model_root}/Wan2.1_VAE.pth",
        torch_dtype=torch.bfloat16,
    )
    # dit high
    module_manager.load_model(
        [os.path.join(model_root, filename) for filename in ppl_config["dit_high_path_list"]],
        torch_dtype=torch.bfloat16,
    )
    # dit low
    module_manager.load_model(
        [os.path.join(model_root, filename) for filename in ppl_config["dit_low_path_list"]],
        torch_dtype=torch.bfloat16,
    )

    # t5
    module_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )
    pipe = Wan22VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan22VideoPipelineConfig()
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_high_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_high_config.attention_config = AttentionConfig.dense_attention(ppl_config["attn_impl"])
    pipe_config.dit_low_config.attention_config = AttentionConfig.dense_attention(ppl_config["attn_impl"])
    pipe_config.sample_solver = ppl_config["sample_solver"]
    pipe_config.dit_low_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    # Configure feature cache
    if ppl_config.get("enable_feature_cache_dit_high", False):
        pipe_config.dit_high_config.feature_cache_config = FeatureCacheConfig(
            enabled=True, model_type=ppl_config["model_type"]
        )
    if ppl_config.get("enable_feature_cache_dit_low", False):
        pipe_config.dit_low_config.feature_cache_config = FeatureCacheConfig(
            enabled=True, model_type=ppl_config["model_type"]
        )
    if parallelism > 1:
        # Configure parallel based on cfg_scale
        # For cfg_scale > 1: cfg_degree=2, sp_ulysses_degree=parallelism//2
        # For cfg_scale == 1: cfg_degree=1, sp_ulysses_degree=parallelism
        cfg_scale_high = PPL_CONFIG["cfg_scale_high"]
        cfg_scale_low = PPL_CONFIG["cfg_scale_low"]

        if cfg_scale_high > 1:
            pipe_config.dit_high_config.parallel_config.cfg_degree = 2
            pipe_config.dit_high_config.parallel_config.sp_ulysses_degree = parallelism // 2
        else:
            pipe_config.dit_high_config.parallel_config.sp_ulysses_degree = parallelism

        if cfg_scale_low > 1:
            pipe_config.dit_low_config.parallel_config.cfg_degree = 2
            pipe_config.dit_low_config.parallel_config.sp_ulysses_degree = parallelism // 2
        else:
            pipe_config.dit_low_config.parallel_config.sp_ulysses_degree = parallelism

        pipe_config.dit_high_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_low_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.enable_denoising_parallel = True
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
        cfg_scale_high=PPL_CONFIG["cfg_scale_high"],
        cfg_scale_low=PPL_CONFIG["cfg_scale_low"],
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
@click.option(
    "--model_root",
    default="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-T2V-14B",
    help="Root directory of the model files",
)
def main(
    gpu_num,
    prompt,
    negative_prompt,
    resolution,
    aspect_ratio,
    model_root,
):
    """Text to video conversion using Wan2.2 14B model"""
    pipe = get_pipeline(gpu_num, model_root)

    # Run inference
    start = time.time()
    video = run(pipe, prompt, negative_prompt, resolution=resolution, aspect_ratio=aspect_ratio)
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
