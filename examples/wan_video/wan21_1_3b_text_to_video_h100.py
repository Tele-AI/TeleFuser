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
    name="wan21_1.3B_t2v_h100",
    negative_prompt="Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="480p",
    cfg_scale=5.0,
    tiled=True,
    target_fps=16,  # Interpolated to 30fps
    sample_solver="euler",
    attn_impl=AttnImplType.SAGE_ATTN_2_8_8_SM90,
    model_type="Wan2.1-I2V-1.3B-720P",
    sigma_shift=8.0,
    enable_vfi=False,  # Enable video frame interpolation
    vfi_model_path="/dev/shm/Wan2.1-T2V-1.3B/flownet.pkl",  # RIFE model path
)


def get_pipeline(parallelism=1, model_root="/dev/shm/Wan2.1-T2V-1.3B/"):
    """
    Args:
        parallelism (int): Number of parallel GPUs for inference: 2, 4 or 8
        model_root (str): Root directory of the model files
    """
    # Load models
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,  # VAE is loaded with bfloat16
    )
    module_manager.load_models(
        [[f"{model_root}/diffusion_pytorch_model.safetensors"]],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    module_manager.load_models(
        [
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        ],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    # Load VFI model if enabled
    if PPL_CONFIG["enable_vfi"]:
        module_manager.load_models(
            [PPL_CONFIG["vfi_model_path"]],
            torch_dtype=torch.bfloat16,
        )
    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_clip_stage = True
    pipe_config.enable_vfi = PPL_CONFIG["enable_vfi"]
    pipe_config.enable_metrics = True
    pipe_config.dit_config.compile_config.enabled = True
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
    Convert static images to video sequences using video generation model.
    Args:
        pipeline (VideoGenerationPipeline): Preloaded video generation pipeline object
        image (PIL.Image/ndarray): Input image, resolution should match height/width parameters
        prompt (str): Positive guidance text prompt
        negative_prompt (str, optional): Negative guidance prompt, will be merged with base negative prompt. Default is empty
        seed (int, optional): Random seed. Default is 42
        resolution(str): Resolution such as 720p, 480p
        aspect_ratio (str): Aspect ratio such as 16:9

    Returns:
        List[PIL.Image]: Generated video sequence
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
        target_fps=PPL_CONFIG["target_fps"] if PPL_CONFIG["enable_vfi"] else None,
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
    video = run(
        pipeline,
        prompt,
        aspect_ratio=aspect_ratio,
        negative_prompt=negative_prompt,
        seed=seed,
        resolution=resolution,
    )
    logger.info(f"save target video to {output_path}")
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
def main(
    gpu_num,
    prompt,
    negative_prompt,
    seed,
    resolution,
    aspect_ratio,
    model_root,
):
    """Text to video conversion using Wan2.1 1.3B model"""
    pipe = get_pipeline(gpu_num, model_root)

    video = run(
        pipe,
        prompt,
        negative_prompt,
        seed,
        resolution,
        aspect_ratio,
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
    print(pipe.get_prometheus_metrics())

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
