"""HunyuanVideo Image-to-Video generation example.

This example demonstrates how to use HunyuanVideo for image-to-video generation
with Telefuser internal model implementations.
"""

import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.hunyuan_video_dit import HunyuanVideoDiT
from telefuser.models.hunyuan_video_image_encoder import HunyuanVideoImageEncoder
from telefuser.models.hunyuan_video_text_encoder import HunyuanVideoTextEncoder
from telefuser.models.hunyuan_video_vae import HunyuanVideoVAE
from telefuser.pipelines.hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
    HunyuanVideo15PipelineConfig,
)
from telefuser.schedulers.flow_match_discrete import FlowMatchDiscreteScheduler
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_image_size, save_video

PPL_CONFIG = dict(
    name="hunyuan_video_i2v",
    negative_prompt="",
    transformer_version="480p_i2v",
    sample_solver="euler",
    resolution="480p",
    model_type="HunyuanVideo15-I2V-480P",
    attn_impl=AttnImplType.SAGE_ATTN_2_8_8_SM90,
    sigma_shift=7.0,
    target_fps=24,
    num_inference_steps=50,
    num_frames=121,
    cfg_scale=6.0,
    enable_feature_cache=True,
)


def get_pipeline(parallelism: int = 1, model_root: str = "/root/models/HunyuanVideo-1.5"):
    """Create and initialize the HunyuanVideo I2V pipeline.

    Args:
        parallelism: Number of parallel GPUs for inference
        model_root: Root directory of the model checkpoints

    Returns:
        Initialized HunyuanVideoPipeline
    """
    module_manager = ModuleManager(device="cpu")

    # Load VAE
    vae_path = os.path.join(model_root, "vae")
    logger.info(f"Loading VAE from {vae_path}")
    vae = HunyuanVideoVAE.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
    module_manager.add_module(vae, name="vae")

    # Load Text Encoder
    text_encoder_path = os.path.join(model_root, "text_encoder", "llm")
    logger.info(f"Loading Text Encoder from {text_encoder_path}")
    text_encoder = HunyuanVideoTextEncoder.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
    module_manager.add_module(text_encoder, name="text_encoder")

    # Load Scheduler
    scheduler_path = os.path.join(model_root, "scheduler")
    logger.info(f"Loading Scheduler from {scheduler_path}")
    scheduler = FlowMatchDiscreteScheduler.from_pretrained(scheduler_path, shift=PPL_CONFIG["sigma_shift"])
    module_manager.add_module(scheduler, name="scheduler")

    # Load Vision Encoder (for I2V) - uses TeleFuser's internal implementation
    vision_encoder_path = os.path.join(model_root, "vision_encoder", "siglip")
    logger.info(f"Loading Vision Encoder from {vision_encoder_path}")
    vision_encoder = HunyuanVideoImageEncoder.from_pretrained(
        vision_encoder_path,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    module_manager.add_module(vision_encoder, name="vision_encoder")

    # Load Transformer (I2V version)
    transformer_path = os.path.join(model_root, "transformer", PPL_CONFIG["transformer_version"])
    logger.info(f"Loading Transformer from {transformer_path}")
    transformer = HunyuanVideoDiT.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    module_manager.add_module(transformer, name="hunyuan_video_dit")

    # Create pipeline
    pipe = HunyuanVideo15Pipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = HunyuanVideo15PipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    if PPL_CONFIG["enable_feature_cache"]:
        pipe_config.dit_config.feature_cache_config.enabled = True
        pipe_config.dit_config.feature_cache_config.model_type = PPL_CONFIG["model_type"]
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_image_encoding = True  # Enable for I2V

    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism
        pipe_config.enable_denoising_parallel = True

    pipe.init(module_manager, pipe_config)

    return pipe


def run(
    pipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    resolution: str = PPL_CONFIG["resolution"],
):
    """Generate video from image and text prompt."""
    width, height = get_target_image_size(image.size[0], image.size[1], resolution, 16, 16)
    video = pipeline(
        prompt=prompt,
        input_image=image,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}".strip(),
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        height=height,
        width=width,
        rand_device="cpu",
    )

    return video


def run_with_file(
    pipeline,
    image,
    prompt,
    negative_prompt="",
    seed=42,
    resolution: str = PPL_CONFIG["resolution"],
    output_path=None,
):
    """Generate video from image and save to file."""
    video = run(pipeline, image, prompt, negative_prompt, seed, resolution=resolution)

    if output_path:
        logger.info(f"Saving video to {output_path}")
        save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)

    return video


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use")
@click.option(
    "--image_path", default=f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png", help="Input image path"
)
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative prompt")
@click.option("--seed", default=42, help="Random seed")
@click.option("--model_root", default="/nvfile-heatstorage/model_zoo/HunyuanVideo-1.5", help="Model checkpoint root")
@click.option("--resolution", default="480p", help="480p or 720p")
def main(gpu_num, image_path, prompt, negative_prompt, seed, model_root, resolution):
    """HunyuanVideo Image-to-Video generation."""
    pipe = get_pipeline(gpu_num, model_root)

    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded reference image: {image.size}")

    start = time.time()
    video = run(pipe, image, prompt, negative_prompt, seed, resolution=resolution)
    elapsed_time = time.time() - start

    print(f"Video generation time: {elapsed_time:.2f} seconds")

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
