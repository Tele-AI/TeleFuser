import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_image_size, save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="wan21_14B_i2v_h100",
    model_root=TF_MODEL_ZOO_PATH + "/Wan2.1-I2V-14B-720P",
    negative_prompt="Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="720p",
    cfg_scale=5.0,
    cfg_skip_ratio=0.0,
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
        f"{model_root}/diffusion_pytorch_model-00001-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00002-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00003-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00004-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00005-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00006-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00007-of-00007.safetensors",
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
        [f"{model_root}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        torch_dtype=torch.float16,  # Image Encoder is loaded with float16
    )
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [dit_path_list],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    module_manager.load_models(
        [
            f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        ],
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
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
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
    )
    return video


def run_with_file(
    pipeline: Wan21VideoPipeline,
    first_image_path: str,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = PPL_CONFIG["seed"],
    output_path: str = "",
    resolution: str = PPL_CONFIG["resolution"],
    **kwargs,
):
    """Run pipeline from an input image path and save to file."""
    if not first_image_path:
        raise ValueError("run_with_file requires first_image_path")

    image = Image.open(first_image_path).convert("RGB")
    video = run(
        pipeline,
        image,
        prompt,
        negative_prompt,
        seed,
        resolution=resolution,
    )
    print(f"Saving video to {output_path}")
    save_video(
        video,
        output_path,
        fps=PPL_CONFIG["target_fps"],
        quality=6,
    )


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
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Root directory of the model")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
def main(
    gpu_num,
    image_path,
    prompt,
    negative_prompt,
    resolution,
    model_root,
    seed,
):
    """Image to video conversion using Wan2.1 14B model"""
    pipe = get_pipeline(gpu_num, model_root)
    image = Image.open(image_path).convert("RGB")

    # Run inference
    while True:
        start = time.time()
        video = run(
            pipe,
            image,
            prompt,
            negative_prompt,
            seed=seed,
            resolution=resolution,
        )
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
