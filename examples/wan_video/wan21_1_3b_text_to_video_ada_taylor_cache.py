"""Wan2.1 1.3B Text-to-Video with AdaTaylorCache V2 Feature Cache.

This example demonstrates how to use AdaTaylorCache V2 feature caching to accelerate
video generation with the Wan2.1 1.3B Text-to-Video model.

AdaTaylorCache V2 combines:
- AdaTaylorCache's adaptive skip logic based on magnitude ratios and error accumulation
- Hybrid approximation strategy: Taylor series for small elapsed steps, residual reuse for large
- Window-based derivative calculation for better accuracy with adaptive skip intervals

This provides better quality-speed trade-offs compared to simple residual caching.

Key improvements in V2:
- Hybrid strategy: When elapsed steps <= taylor_threshold, use Taylor expansion
  When elapsed steps > taylor_threshold, fall back to residual reuse (more stable)
- Window-based derivatives: Uses actual step window instead of fixed dt=1

Usage:
    python wan21_1_3b_text_to_video_ada_taylor_cache.py --gpu_num 1 \
        --n_derivatives 1 --taylor_threshold 2
"""

import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, FeatureCacheConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

PPL_CONFIG = dict(
    name="wan21_1.3B_t2v_ada_taylor_cache",
    negative_prompt="Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="480p",
    cfg_scale=6.0,
    tiled=True,
    target_fps=16,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    model_type="Wan2.1-T2V-1.3B",
    sigma_shift=8.0,
    # AdaTaylorCache V2 parameters
    n_derivatives=1,
    taylor_threshold=2,  # Hybrid strategy threshold
)


def get_pipeline(
    parallelism=1,
    model_root="/dev/shm/Wan2.1-T2V-1.3B/",
    enable_feature_cache=True,
    n_derivatives=1,
    taylor_threshold=2,
):
    """
    Initialize the video generation pipeline with AdaTaylorCache V2 feature caching.

    Args:
        parallelism: Number of parallel GPUs for inference: 2, 4 or 8
        model_root: Root directory of the model files
        enable_feature_cache: Whether to enable AdaTaylorCache feature caching
        n_derivatives: Order of Taylor series expansion (1 or 2 recommended)
        taylor_threshold: Threshold for switching to residual reuse (default: 2)
            - elapsed <= threshold: Use Taylor series expansion (higher accuracy)
            - elapsed > threshold: Use residual reuse (more stable fallback)
    """
    # Load models
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,  # VAE is loaded with bfloat16
    )
    module_manager.load_models(
        [[f"{model_root}/diffusion_pytorch_model.safetensors"]],
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
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]

    # Configure feature cache for faster inference
    if enable_feature_cache:
        pipe_config.dit_config.feature_cache_config = FeatureCacheConfig(
            enabled=True,
            model_type=PPL_CONFIG["model_type"],
            n_derivatives=n_derivatives,
            taylor_threshold=taylor_threshold,
        )

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
    pipe.init(module_manager, pipe_config)

    logger.info(
        f"AdaTaylorCache V2 pipeline initialized: enable={enable_feature_cache}, "
        f"n_derivatives={n_derivatives}, taylor_threshold={taylor_threshold}"
    )
    return pipe


def run(
    pipeline,
    prompt,
    negative_prompt="",
    seed=42,
    resolution=PPL_CONFIG["resolution"],
    aspect_ratio="16:9",
):
    """
    Generate video from text prompt using AdaTaylorCache V2 feature caching.

    Args:
        pipeline: Preloaded video generation pipeline object
        prompt: Positive guidance text prompt
        negative_prompt: Negative guidance prompt
        seed: Random seed
        resolution: Resolution such as 720p, 480p
        aspect_ratio: Aspect ratio such as 16:9

    Returns:
        Generated video frames
    """
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=2,
        width_division_factor=2,
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


def run_with_file(
    pipeline,
    prompt,
    negative_prompt,
    seed,
    resolution,
    output_path,
    aspect_ratio: str = "16:9",
    **kwargs,
):
    """Run pipeline and save to file."""
    video = run(
        pipeline,
        prompt,
        negative_prompt,
        seed,
        resolution,
        aspect_ratio,
    )
    logger.info(f"Saving video to {output_path}")
    save_video(
        video,
        output_path,
        fps=PPL_CONFIG["target_fps"],
        quality=6,
    )


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--seed", default=42, help="Random seed")
@click.option("--resolution", default=PPL_CONFIG["resolution"], help="Resolution (480p, 720p)")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio")
@click.option("--model_root", default="/dev/shm/Wan2.1-T2V-1.3B/", help="Root directory of the model files")
@click.option("--enable_feature_cache", is_flag=True, default=True, help="Enable AdaTaylorCache V2 feature caching")
@click.option("--n_derivatives", default=1, help="Taylor series order (1 or 2 recommended)")
@click.option(
    "--taylor_threshold",
    default=2,
    help="Hybrid strategy threshold (default: 2). "
    "elapsed<=threshold: Taylor expansion, elapsed>threshold: residual reuse",
)
def main(
    gpu_num,
    prompt,
    negative_prompt,
    seed,
    resolution,
    aspect_ratio,
    model_root,
    enable_feature_cache,
    n_derivatives,
    taylor_threshold,
):
    """Text to video conversion using Wan2.1 1.3B model with AdaTaylorCache V2 feature caching.

    AdaTaylorCache V2 combines adaptive skip logic with a hybrid approximation
    strategy for efficient and accurate feature caching.

    Key features:
    - Adaptive skip: Uses error accumulation to dynamically decide when to skip
    - Hybrid approximation:
      * Small elapsed (<=threshold): Taylor series for higher-order accuracy
      * Large elapsed (>threshold): Residual reuse for stability
    - Window-based derivatives: Uses actual skip window for better accuracy
    - Better quality-speed trade-off compared to V1

    Key parameters:
    - n_derivatives: Taylor series order (1-2 recommended)
    - taylor_threshold: Switch point between Taylor and residual reuse (default: 2)
    """
    # Initialize pipeline with AdaTaylorCache V2
    pipe = get_pipeline(
        gpu_num,
        model_root,
        enable_feature_cache=enable_feature_cache,
        n_derivatives=n_derivatives,
        taylor_threshold=taylor_threshold,
    )

    # Run inference
    start = time.time()
    video = run(
        pipe,
        prompt,
        negative_prompt,
        seed,
        resolution,
        aspect_ratio,
    )
    elapsed_time = time.time() - start

    print(f"\n{'=' * 60}")
    print("Video generation completed!")
    print(f"  - Time: {elapsed_time:.2f} seconds")
    print(f"  - AdaTaylorCache V2: {'enabled' if enable_feature_cache else 'disabled'}")
    if enable_feature_cache:
        print(f"  - n_derivatives: {n_derivatives}")
        print(f"  - taylor_threshold: {taylor_threshold}")
        print(f"    * elapsed <= {taylor_threshold}: Taylor series expansion")
        print(f"    * elapsed > {taylor_threshold}: Residual reuse (fallback)")
        print("  - Adaptive skip: Uses error-based skip logic")
        print("  - Window-based derivatives: Uses actual skip window for accuracy")
    print(f"{'=' * 60}\n")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=16, quality=6)
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
