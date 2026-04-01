"""
Cache Calibrator Example for Wan2.2 14B Image-to-Video

This script runs the pipeline once to collect calibration data
and generates a parameter JSON file for AdaTaylorCache.

Usage:
    python wan22_14b_image_to_video_cache_calibrate.py \
        --model_root /path/to/Wan2.2-I2V-A14B/ \
        --num_inference_steps 40 \
        --sigma_shift 5.0 \
        --output_path ./cache_params.json

The generated JSON file will contain:
    - K, retention_ratio, thresh: Default values (0), need to be adjusted by user
    - sigma_shift: The sigma shift value used during calibration
    - num_inference_steps: Number of inference steps used during calibration
    - cond_mag_ratios: Mag ratios for conditional path (with 1.0 prepended)
    - uncond_mag_ratios: Mag ratios for unconditional path (with 1.0 prepended)

These parameters are used by AdaTaylorCache (including when n_derivatives=0
for simple residual caching).
"""

import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan22_video import (
    Wan22VideoPipeline,
    Wan22VideoPipelineConfig,
)
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_image_size, save_video

# Default configuration
PPL_CONFIG = dict(
    name="wan22_14B_i2v_cache_calibrate",
    negative_prompt="Overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="720p",
    cfg_scale_high=3.5,
    cfg_scale_low=3.5,
    sigma_shift=5.0,
    tiled=False,
    target_fps=16,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    dit_high_path_list=[
        "dit_high_noise_model_bfloat16_5d6fd.safetensors",
    ],
    dit_low_path_list=[
        "dit_low_noise_model_bfloat16_c55d6.safetensors",
    ],
    model_type="Wan2_2-I2V-A14B",
)


def get_pipeline(parallelism: int = 1, model_root: str = "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B"):
    """
    Create and initialize the video generation pipeline.

    Args:
        parallelism: Number of parallel GPUs for inference
        model_root: Root directory of the model files
    """
    module_manager = ModuleManager(device="cpu")
    module_manager.load_model(
        f"{model_root}/Wan2.1_VAE.pth",
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_model(
        [os.path.join(model_root, filename) for filename in PPL_CONFIG["dit_high_path_list"]],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_model(
        [os.path.join(model_root, filename) for filename in PPL_CONFIG["dit_low_path_list"]],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )

    pipe = Wan22VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan22VideoPipelineConfig()
    pipe_config.dit_high_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_low_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]

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


def run_calibration(
    pipeline: Wan22VideoPipeline,
    prompt: str,
    image: Image.Image,
    negative_prompt: str = "",
    seed: int = 42,
    resolution: str = PPL_CONFIG["resolution"],
    model_name: str = "Wan2.2-I2V-A14B",
    output_path: str | None = None,
):
    """
    Run cache calibration.

    This function runs the pipeline once in calibration mode to collect
    residual data and generate cache parameters for AdaTaylorCache.

    Args:
        pipeline: Preloaded video generation pipeline
        prompt: Positive guidance text prompt
        image: Input PIL Image for video generation
        negative_prompt: Negative guidance prompt
        seed: Random seed
        resolution: Resolution such as 720p, 480p
        model_name: Model name for the output file
        output_path: Output path for the JSON file

    Returns:
        Generated video frames
    """
    width, height = get_target_image_size(image.size[0], image.size[1], resolution=resolution)

    pipeline.denoise_high_stage.dit.set_ada_taylor_cache_calibrator(
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        sigma_shift=PPL_CONFIG["sigma_shift"],
        model_name=model_name,
        output_path=output_path,
    )

    # Share the same calibrator between dit_high and dit_low for dual-branch architecture
    shared_calibrator = pipeline.denoise_high_stage.dit.feature_cache
    pipeline.denoise_low_stage.dit._feature_cache = shared_calibrator

    logger.info(
        f"Starting cache calibration with {PPL_CONFIG['num_inference_steps']} steps, sigma_shift={PPL_CONFIG['sigma_shift']}"
    )
    logger.info(f"Output will be saved to: {output_path or 'default params directory'}")

    video = pipeline(
        prompt=prompt,
        input_image=image,
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
    "--image_path",
    default=None,
    help="Input image path (default: examples/data/101235-video-720_0.png)",
)
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--seed", default=42, help="Random seed")
@click.option("--resolution", default=PPL_CONFIG["resolution"], help="Resolution (480p, 720p)")
@click.option(
    "--model_root",
    default="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B",
    help="Root directory of the model files",
)
@click.option("--model_name", default="Wan2.2-I2V-A14B", help="Model name for the output file")
@click.option("--output_path", default=None, help="Output path for the JSON file (default: params directory)")
def main(
    gpu_num: int,
    image_path: str | None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    resolution: str,
    model_root: str,
    model_name: str,
    output_path: str | None,
):
    """Run cache calibration for Wan2.2 14B Image-to-Video model."""
    logger.info("=" * 60)
    logger.info("Cache Calibrator for Wan2.2 14B Image-to-Video")
    logger.info("=" * 60)

    if image_path is None:
        image_path = os.path.join(os.path.dirname(__file__), "../data/101235-video-720_0.png")

    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded input image from: {image_path}")

    pipe = get_pipeline(gpu_num, model_root)

    start = time.time()
    video = run_calibration(
        pipe,
        prompt,
        image,
        negative_prompt,
        seed,
        resolution,
        model_name,
        output_path,
    )
    elapsed_time = time.time() - start

    logger.info(f"Calibration completed in {elapsed_time:.2f} seconds")
    logger.info(f"Generated {len(video)} frames")

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    sample_path = os.path.join(output_dir, get_example_name(__file__, "mp4"))

    save_video(video, sample_path, fps=PPL_CONFIG["target_fps"], quality=6)
    logger.info(f"Sample video saved to: {sample_path}")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
