"""Wan2.2 T2V (Text-to-Video) 5B example.

Usage:
    python wan22_t2v_5b.py --prompt "A cat playing piano"
"""

import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan22_ti2v import (
    Wan22TI2VPipeline,
    Wan22TI2VPipelineConfig,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

PPL_CONFIG = dict(
    name="wan22_t2v_5b",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-TI2V-5B",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=50,
    num_frames=121,
    fps=24,
    resolution="480p",
    aspect_ratio="16:9",
    cfg_scale=5.0,
    seed=42,
    tiled=False,
    sigma_shift=5.0,
    sample_solver="unipc",
    attn_impl=AttnImplType.TORCH_SDPA,
    dit_path=[
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
    ],
    vae_path="Wan2.2_VAE.pth",
    t5_path="models_t5_umt5-xxl-enc-bf16.pth",
)


def get_pipeline(parallelism: int = 1, model_root: str = PPL_CONFIG["model_root"]):
    """Create and initialize the Wan2.2 T2V pipeline.

    Args:
        parallelism: Number of parallel GPUs for inference (1, 2, 4, or 8)
        model_root: Root directory containing model weights

    Returns:
        Initialized Wan22TI2VPipeline instance
    """
    ppl_config = PPL_CONFIG

    # Load models using ModuleManager
    module_manager = ModuleManager(device="cpu")

    # VAE (Wan2.2 VAE with 48 latent channels)
    module_manager.load_model(
        os.path.join(model_root, ppl_config["vae_path"]),
        torch_dtype=torch.bfloat16,
    )

    # DiT
    module_manager.load_model(
        [os.path.join(model_root, filename) for filename in ppl_config["dit_path"]],
        torch_dtype=torch.bfloat16,
    )

    # T5 Text Encoder
    module_manager.load_model(
        os.path.join(model_root, ppl_config["t5_path"]),
        torch_dtype=torch.bfloat16,
    )

    # Create pipeline
    pipe = Wan22TI2VPipeline(device="cuda", torch_dtype=torch.bfloat16)

    # Configure pipeline
    pipe_config = Wan22TI2VPipelineConfig()
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(ppl_config["attn_impl"])
    pipe_config.sample_solver = ppl_config["sample_solver"]

    # Configure parallelism
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


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--resolution", default=PPL_CONFIG["resolution"], help="Output resolution: 480p or 720p")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio: 16:9, 9:16, 1:1, etc.")
@click.option("--num_frames", default=121, help="Number of frames (should be 4n+1)")
@click.option(
    "--model_root",
    default="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-TI2V-5B",
    help="Root directory of the model files",
)
def main(
    gpu_num: int,
    prompt: str,
    negative_prompt: str,
    resolution: str,
    aspect_ratio: str,
    num_frames: int,
    model_root: str,
):
    """Generate video from text using Wan2.2 T2V 5B model."""
    # Initialize pipeline
    pipe = get_pipeline(gpu_num, model_root)

    # Run inference
    start = time.time()
    width, height = get_target_video_size_from_ratio(aspect_ratio, resolution, 32, 32)
    video = pipe(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=num_frames,
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=PPL_CONFIG["seed"],
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
    )
    elapsed_time = time.time() - start
    print(f"Video generation time: {elapsed_time:.2f} seconds")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=PPL_CONFIG["fps"], quality=6)
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
