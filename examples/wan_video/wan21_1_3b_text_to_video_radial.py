"""
Wan2.1 T2V 1.3B with Radial Attention - Efficient Video Generation.

This example demonstrates how to use radial attention for memory-efficient
video generation with Wan2.1 models. Radial attention uses sparse attention
patterns where nearby frames have denser attention than distant frames.

Requirements:
    - flashinfer: For radial attention backend
    - Or sageattention/spas_sage_attn: For alternative backend

Usage:
    # Standard generation (dense attention with FLASH_ATTN_2)
    python wan21_1_3b_text_to_video_radial.py
    # With radial attention
    python wan21_1_3b_text_to_video_radial.py --enable_radial
    # With custom radial attention parameters
    python wan21_1_3b_text_to_video_radial.py \\
        --enable_radial \\
        --dense_timesteps 20 \\
        --decay_factor 0.8
"""

import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

PPL_CONFIG = dict(
    name="wan21_1.3B_t2v_radial",
    negative_prompt="Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="480p",
    cfg_scale=6.0,
    tiled=True,
    target_fps=16,
    sample_solver="euler",
    sigma_shift=8.0,
)


def get_pipeline(
    parallelism=1,
    model_root="/dev/shm/Wan2.1-T2V-1.3B/",
    enable_radial=False,
    dense_layers=0,
    dense_timesteps=40,
    decay_factor=1.0,
    use_sage_attention=False,
):
    """
    Args:
        parallelism (int): Number of parallel GPUs for inference: 1, 2, 4 or 8
        model_root (str): Root directory of the model files
        enable_radial (bool): Enable radial sparse attention
        dense_layers (int): Number of layers to use dense attention
        dense_timesteps (int): Number of timesteps to use dense attention
        decay_factor (float): Decay factor for radial attention window
        use_sage_attention (bool): Use sage attention backend
    """
    # Load models
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [[f"{model_root}/diffusion_pytorch_model.safetensors"]],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth"],
        torch_dtype=torch.bfloat16,
    )

    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()

    # Configure attention
    if enable_radial:
        attention_config = AttentionConfig.radial_attention(
            dense_timesteps=dense_timesteps,
            dense_layers=dense_layers,
            decay_factor=decay_factor,
            use_sage_attention=use_sage_attention,
        )
        logger.info(
            f"Using radial attention: dense_timesteps={dense_timesteps}, "
            f"dense_layers={dense_layers}, decay_factor={decay_factor}"
        )
    else:
        attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
        logger.info("Using dense FLASH_ATTN_2 attention")

    pipe_config.dit_config.attention_config = attention_config
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_clip_stage = False  # T2V model

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
    Generate video from text prompt.

    Args:
        pipeline: Preloaded video generation pipeline object
        prompt: Positive guidance text prompt
        negative_prompt: Negative guidance prompt
        seed: Random seed
        resolution: Resolution such as 720p, 480p
        aspect_ratio: Aspect ratio such as 16:9

    Returns:
        Generated video
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
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard.",
    help="Text prompt for video generation",
)
@click.option("--negative_prompt", default="", help="Negative prompt")
@click.option("--seed", default=42, help="Random seed")
@click.option("--resolution", default=PPL_CONFIG["resolution"], help="Resolution (480p, 720p)")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio (16:9, 4:3, 1:1)")
@click.option("--model_root", default="/dev/shm/Wan2.1-T2V-1.3B/", help="Root directory of the model files")
@click.option("--enable_radial", is_flag=True, help="Enable radial sparse attention")
@click.option("--dense_layers", default=0, help="Number of layers to use dense attention")
@click.option("--dense_timesteps", default=40, help="Number of timesteps to use dense attention")
@click.option("--decay_factor", default=1.0, help="Decay factor for attention window")
@click.option("--use_sage_attention", is_flag=True, help="Use sage attention backend")
def main(
    gpu_num,
    prompt,
    negative_prompt,
    seed,
    resolution,
    aspect_ratio,
    model_root,
    enable_radial,
    dense_layers,
    dense_timesteps,
    decay_factor,
    use_sage_attention,
):
    """
    Text to video generation using Wan2.1 1.3B with optional radial attention.

    Examples:
        # Standard generation (dense FLASH_ATTN_2)
        python wan21_1_3b_text_to_video_radial.py

        # With radial attention
        python wan21_1_3b_text_to_video_radial.py --enable_radial

        # With custom radial attention parameters
        python wan21_1_3b_text_to_video_radial.py \\
            --enable_radial \\
            --dense_timesteps 20 \\
            --decay_factor 0.8
    """
    click.echo(f"GPUs: {gpu_num}")
    click.echo(f"Model root: {model_root}")
    click.echo(f"Attention: {'Radial (sparse)' if enable_radial else 'FLASH_ATTN_2 (dense)'}")
    if enable_radial:
        click.echo(f"  - Dense layers: {dense_layers}")
        click.echo(f"  - Dense timesteps: {dense_timesteps}")
        click.echo(f"  - Decay factor: {decay_factor}")
        click.echo(f"  - Use sage attention: {use_sage_attention}")

    # Load pipeline
    click.echo("Loading pipeline...")
    pipe = get_pipeline(
        gpu_num,
        model_root,
        enable_radial=enable_radial,
        dense_layers=dense_layers,
        dense_timesteps=dense_timesteps,
        decay_factor=decay_factor,
        use_sage_attention=use_sage_attention,
    )

    # Run inference
    click.echo("Generating video...")
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

    click.echo(f"Video generation time: {elapsed_time:.2f} seconds")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    suffix = "_radial" if enable_radial else "_dense"
    filename = get_example_name(__file__).replace(".py", f"{suffix}_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=16, quality=6)
    click.echo(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
