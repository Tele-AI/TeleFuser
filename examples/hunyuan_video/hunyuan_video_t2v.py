"""HunyuanVideo Text-to-Video generation example.

This example demonstrates how to use HunyuanVideo for text-to-video generation
with Telefuser internal model implementations.

Supports:
- Optional ByT5 for glyph text rendering (text in quotes will be rendered in the generated video)
- Optional Super-Resolution (SR) for 480p -> 720p upscaling
"""

import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.hunyuan_video_byt5 import HunyuanVideoByT5Encoder
from telefuser.models.hunyuan_video_dit import HunyuanVideoDiT
from telefuser.models.hunyuan_video_text_encoder import HunyuanVideoTextEncoder
from telefuser.models.hunyuan_video_upsampler import SRTo720pUpsampler
from telefuser.models.hunyuan_video_vae import HunyuanVideoVAE
from telefuser.pipelines.hunyuan_video_1_5 import (
    HunyuanVideo15Pipeline,
    HunyuanVideo15PipelineConfig,
)
from telefuser.schedulers.flow_match_discrete import FlowMatchDiscreteScheduler
from telefuser.utils.logging import logger
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

# SR configuration: 480p -> 720p
SR_CONFIG = {
    "base_resolution": "480p",
    "sr_resolution": "720p",
    "sr_version": "720p_sr_distilled",
    "flow_shift": 2.0,
    "num_inference_steps": 6,
    "lq_noise_strength": 0.7,
}

PPL_CONFIG = dict(
    name="hunyuan_video_t2v",
    negative_prompt="",
    transformer_version="480p_t2v",
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    sigma_shift=5.0,
    target_fps=24,
    num_inference_steps=50,
    num_frames=121,
    cfg_scale=6.0,
    model_type="HunyuanVideo15-T2V-480P",
    enable_feature_cache=True,
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = "/root/models/HunyuanVideo-1.5",
    enable_byt5: bool = False,
    enable_sr: bool = False,
):
    """Create and initialize the HunyuanVideo pipeline.

    Args:
        parallelism: Number of parallel GPUs for inference (1, 2, 4, or 8)
        model_root: Root directory of the model checkpoints
        enable_byt5: Enable ByT5 for glyph text rendering
        enable_sr: Enable Super-Resolution (480p -> 720p)

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

    # Load ByT5 (optional, for glyph text rendering)
    if enable_byt5:
        try:
            byt5_encoder = HunyuanVideoByT5Encoder.from_pretrained(
                model_root,
                torch_dtype=torch.bfloat16,
            )
            module_manager.add_module(byt5_encoder.model, name="byt5_model")
            module_manager.add_module(byt5_encoder.tokenizer, name="byt5_tokenizer")
            logger.info("ByT5 loaded for glyph text rendering")
        except Exception as e:
            logger.warning(f"Failed to load ByT5: {e}, glyph text rendering disabled")

    # Load base Transformer
    if enable_sr:
        # For SR mode, use 480p transformer for base generation
        transformer_version = f"{SR_CONFIG['base_resolution']}_t2v"
    else:
        transformer_version = PPL_CONFIG["transformer_version"]

    transformer_path = os.path.join(model_root, "transformer", transformer_version)
    logger.info(f"Loading Transformer from {transformer_path}")
    transformer = HunyuanVideoDiT.from_pretrained(
        transformer_path,
        torch_dtype=torch.bfloat16,
    )
    module_manager.add_module(transformer, name="hunyuan_video_dit")

    # Load SR components (optional, for super-resolution)
    if enable_sr:
        # Load upsampler
        upsampler_path = os.path.join(model_root, "upsampler", "720p_sr_distilled")
        logger.info(f"Loading Upsampler from {upsampler_path}")
        upsampler = SRTo720pUpsampler.from_pretrained(upsampler_path, torch_dtype=torch.float32)
        module_manager.add_module(upsampler, name="upsampler")

        # Load SR transformer (distilled version for SR)
        sr_transformer_version = SR_CONFIG["sr_version"]
        sr_transformer_path = os.path.join(model_root, "transformer", sr_transformer_version)
        logger.info(f"Loading SR Transformer from {sr_transformer_path}")
        sr_transformer = HunyuanVideoDiT.from_pretrained(
            sr_transformer_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        module_manager.add_module(sr_transformer, name="sr_dit")

    # Create pipeline
    pipe = HunyuanVideo15Pipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = HunyuanVideo15PipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    if PPL_CONFIG["enable_feature_cache"]:
        pipe_config.dit_config.feature_cache_config.enabled = True
        pipe_config.dit_config.feature_cache_config.model_type = PPL_CONFIG["model_type"]
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    if enable_sr:
        pipe_config.sr_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])

    # SR configuration
    if enable_sr:
        pipe_config.enable_sr = True
        pipe_config.lq_noise_strength = SR_CONFIG["lq_noise_strength"]

    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism
        pipe_config.enable_denoising_parallel = True

    pipe.init(module_manager, pipe_config)

    return pipe


def run(
    pipeline,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
):
    """Generate video from text prompt.

    For glyph text rendering, enclose text in quotes within the prompt:
        prompt = 'A beautiful sunset with "Hello World" in the sky'

    The quoted text will be rendered with special styling in the generated video.

    """
    # For SR mode, use base resolution (480p) for initial generation
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=16,
        width_division_factor=16,
    )

    video = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}".strip(),
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        sr_num_inference_steps=SR_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        height=height,
        width=width,
        rand_device="cpu",
    )

    return video


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use")
@click.option("--prompt", default="一只可爱的猫咪在阳光下打盹，毛发在微风中轻轻飘动，画面温馨治愈", help="Text prompt")
@click.option("--negative_prompt", default="", help="Negative prompt")
@click.option("--seed", default=42, help="Random seed")
@click.option("--resolution", default="720p", help="Target resolution (ignored if --enable_sr)")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio")
@click.option("--model_root", default="/nvfile-heatstorage/model_zoo/HunyuanVideo-1.5", help="Model checkpoint root")
@click.option("--enable_byt5", is_flag=True, help="Enable ByT5 for glyph text rendering")
@click.option("--enable_sr", is_flag=True, help="Enable Super-Resolution (480p -> 720p)")
def main(
    gpu_num,
    prompt,
    negative_prompt,
    seed,
    resolution,
    aspect_ratio,
    model_root,
    enable_byt5,
    enable_sr,
):
    """HunyuanVideo Text-to-Video generation.

    Examples:
        # Basic usage
        python hunyuan_video_t2v.py --prompt "A beautiful sunset"

        # With glyph text rendering
        python hunyuan_video_t2v.py --enable_byt5 --prompt 'A sunset scene with "Hello" in the sky'

        # With Super-Resolution (480p -> 720p)
        python hunyuan_video_t2v.py --enable_sr --prompt "A beautiful sunset"
    """
    pipe = get_pipeline(gpu_num, model_root, enable_byt5=enable_byt5, enable_sr=enable_sr)

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

    print(f"Video generation time: {elapsed_time:.2f} seconds")

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    suffix_parts = []
    if enable_byt5:
        suffix_parts.append("byt5")
    if enable_sr:
        suffix_parts.append("sr")
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    filename = get_example_name(__file__).replace(".py", f"{suffix}_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
