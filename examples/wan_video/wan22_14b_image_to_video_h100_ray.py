import os
import time

import click
import ray
import torch
from PIL import Image

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    ParallelConfig,
    RayConfig,
    RayGPUConfig,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan22_video import (
    Wan22VideoPipeline,
    Wan22VideoPipelineConfig,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_image_size, save_video

PPL_CONFIG = dict(
    name="wan22_A14B_i2v_h100_distill",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B",
    negative_prompt="Overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=8,
    num_frames=81,
    resolution="720p",
    cfg_scale_high=1.0,
    cfg_scale_low=1.0,
    seed=42,
    tiled=False,
    attn_impl=AttnImplType.TORCH_SDPA,
    model_type="Wan2.2-I2V-A14B",
    sigma_shift=5.0,
    boundary=0.9,
    sample_solver="euler",
    dit_high_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/dit_high_noise_distill_model_bf16_1022_ecab7.safetensors",
    ],
    dit_low_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B/dit_low_noise_distill_model_bf16_1022_200c2.safetensors",
    ],
)


def get_pipeline(num_gpus=2):
    """
    Pipeline configuration for distributed inference using Ray

    Args:
        num_gpus (int): Number of GPUs to use, default is 2 GPUs
    """
    model_root = PPL_CONFIG["model_root"]

    # Load models
    module_manager = ModuleManager(device="cpu")

    # vae
    module_manager.load_model(
        f"{model_root}/Wan2.1_VAE.pth",
        torch_dtype=torch.bfloat16,
    )
    # dit high
    module_manager.load_model(
        PPL_CONFIG["dit_high_path_list"],
        torch_dtype=torch.bfloat16,
    )
    # dit low
    module_manager.load_model(
        PPL_CONFIG["dit_low_path_list"],
        torch_dtype=torch.bfloat16,
    )
    # t5
    module_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )

    pipe = Wan22VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan22VideoPipelineConfig()

    # Set basic configuration
    pipe_config.dit_high_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_high_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_low_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_low_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_vae_ray = True
    vae_ray_config = RayConfig(gpu_config=RayGPUConfig(num_gpus=num_gpus), num_cpus=8)
    pipe_config.vae_config.ray_config = vae_ray_config

    if num_gpus > 1:
        parallel_config = ParallelConfig()
        parallel_config.device_ids = list(range(num_gpus))
        # Configure parallel based on cfg_scale
        # For cfg_scale > 1: cfg_degree=2, sp_ulysses_degree=parallelism//2
        # For cfg_scale == 1: cfg_degree=1, sp_ulysses_degree=parallelism
        cfg_scale_high = PPL_CONFIG["cfg_scale_high"]
        cfg_scale_low = PPL_CONFIG["cfg_scale_low"]

        if cfg_scale_high > 1:
            parallel_config.cfg_degree = 2
            parallel_config.sp_ulysses_degree = num_gpus // 2
        else:
            parallel_config.sp_ulysses_degree = num_gpus

        pipe_config.vae_config.parallel_config = parallel_config
        pipe_config.enable_vae_parallel = True
        pipe_config.dit_high_config.parallel_config = parallel_config
        pipe_config.enable_denoising_parallel = True
        cfg_scale_low = PPL_CONFIG["cfg_scale_low"]

        if cfg_scale_low > 1:
            parallel_config.cfg_degree = 2
            parallel_config.sp_ulysses_degree = num_gpus // 2
        else:
            parallel_config.sp_ulysses_degree = num_gpus
        pipe_config.dit_low_config.parallel_config = parallel_config
    pipe.init(module_manager, pipe_config)
    return pipe


def run(
    pipeline,
    image,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
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

    Returns:
        List[PIL.Image]: Generated video sequence
    """
    width, height = get_target_image_size(image.size[0], image.size[1], resolution=resolution)
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
    "--image_path", default=f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png", help="Input image path"
)
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--resolution", default=PPL_CONFIG["resolution"], help="480p or 720p")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Root directory of the model files")
def main(
    gpu_num,
    image_path,
    prompt,
    negative_prompt,
    resolution,
    seed,
    model_root,
):
    """Image to video conversion using Wan2.2 14B Ray distributed model"""
    # Update PPL_CONFIG with command line parameters
    PPL_CONFIG["model_root"] = model_root
    PPL_CONFIG["seed"] = seed
    PPL_CONFIG["resolution"] = resolution

    # Initialize Ray
    ray.init(num_gpus=gpu_num)
    pipe = get_pipeline(gpu_num)
    image = Image.open(image_path).convert("RGB")

    # Run inference
    start_time = time.time()
    video = run(pipe, image, prompt, negative_prompt, seed, resolution)
    elapsed_time = time.time() - start_time

    print(f"Inference completed, time taken: {elapsed_time:.2f} seconds")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_ray_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=16, quality=6)
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
