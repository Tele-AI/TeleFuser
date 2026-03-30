import os
import time

import click
import torch
from transformers import UMT5EncoderModel

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.longcat_video import (
    LongCatVideoPipeline,
    LongCatVideoPipelineConfig,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

PPL_CONFIG = dict(
    name="longcat_t2v",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/LongCat-Video",
    negative_prompt="色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走",
    num_inference_steps=50,
    num_frames=93,
    resolution="720p",
    cfg_scale=4.0,
    seed=42,
    tiled=False,
    target_fps=15,
    attn_impl=AttnImplType.TORCH_SDPA,
    tokenizer_path="/nvfile-heatstorage/model_zoo/huggingface/LongCat-Video/tokenizer",
    text_encoder_path="/nvfile-heatstorage/model_zoo/huggingface/LongCat-Video/text_encoder",
    dit_path_list=[
        "/nvfile-heatstorage/model_zoo/modelscope/LongCat-Video/dit/diffusion_pytorch_model-00001-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/LongCat-Video/dit/diffusion_pytorch_model-00002-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/LongCat-Video/dit/diffusion_pytorch_model-00003-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/LongCat-Video/dit/diffusion_pytorch_model-00004-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/LongCat-Video/dit/diffusion_pytorch_model-00005-of-00006.safetensors",
        "/nvfile-heatstorage/model_zoo/modelscope/LongCat-Video/dit/diffusion_pytorch_model-00006-of-00006.safetensors",
    ],
    vae_path="/nvfile-heatstorage/model_zoo/modelscope/Wan2___1-T2V-14B/Wan2.1_VAE.pth",
)


def get_pipeline(parallelism=1):
    """
    Args:
        parallelism (int): Number of parallel GPUs for inference: 1, 2, 4 or 8
    """
    # Load models
    module_manager = ModuleManager(device="cpu")

    module_manager.load_models(
        [PPL_CONFIG["vae_path"]],
        torch_dtype=torch.bfloat16,
    )

    module_manager.load_models(
        [PPL_CONFIG["dit_path_list"]],
        torch_dtype=torch.bfloat16,
    )

    module_manager.load_from_huggingface(
        PPL_CONFIG["tokenizer_path"],
        module_source="transformers",
        module_name="longcat_tokenizer",
    )

    module_manager.load_from_huggingface(
        PPL_CONFIG["text_encoder_path"],
        module_source="transformers",
        module_name="longcat_text_encoder",
        module_class=UMT5EncoderModel,
        torch_dtype=torch.bfloat16,
    )

    if parallelism == 1:
        parallel_config = ParallelConfig()
    elif parallelism == 2:
        parallel_config = ParallelConfig(device_ids=[0, 1], cfg_degree=2, sp_ulysses_degree=1, timeout=1800)
    elif parallelism == 4:
        parallel_config = ParallelConfig(device_ids=[0, 1, 2, 3], cfg_degree=2, sp_ulysses_degree=2, timeout=1800)
    elif parallelism == 8:
        parallel_config = ParallelConfig(device_ids=list(range(8)), cfg_degree=2, sp_ulysses_degree=4, timeout=1800)
    else:
        raise ValueError(f"Unsupported parallelism: {parallelism}. Must be 1, 2, 4, or 8.")

    # Pipeline configuration
    dit_config = ModelRuntimeConfig(torch_dtype=torch.bfloat16)
    dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    dit_config.parallel_config = parallel_config

    pipe_config = LongCatVideoPipelineConfig(
        vae_config=ModelRuntimeConfig(torch_dtype=torch.bfloat16),
        dit_config=dit_config,
        text_encoding_config=ModelRuntimeConfig(torch_dtype=torch.bfloat16),
        sample_solver="euler",
        enable_denoising_parallel=(parallelism > 1),
    )

    pipe = LongCatVideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.init(module_manager, pipe_config)
    pipe.enable_cpu_offload()

    return pipe


def run(
    pipeline,
    prompt,
    height,
    width,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
):
    """
    Generate video from text prompt.
    Args:
        pipeline (LongCatVideoPipeline): Preloaded video generation pipeline object
        prompt (str): Positive guidance text prompt
        height (int): Video height
        width (int): Video width
        negative_prompt (str, optional): Negative guidance prompt, will be merged with base negative prompt. Default is empty
        seed (int, optional): Random seed. Default is 42

    Returns:
        List[PIL.Image]: Generated video sequence
    """
    video, _ = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        attn_impl=PPL_CONFIG["attn_impl"],
    )
    return video


def run_with_file(
    pipeline,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    output_path="output.mp4",
    resolution="480p",
    aspect_ratio="16:9",
    **kwargs,
):
    """Service-compatible entry point for task-based execution."""
    width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
    video = run(pipeline, prompt, height, width, negative_prompt, seed)
    save_video(video, output_path, fps=15, quality=6)


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--prompt",
    default="A stylish woman walking down a Tokyo street filled with warm golden sunlight and cherry blossoms floating in the wind. The camera follows her from behind as she strolls leisurely, creating a cinematic atmosphere.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--resolution", default="720p", help="480p or 720p")
@click.option("--aspect_ratio", default="16:9", help="Aspect ratio: 16:9, 9:16, 1:1, etc.")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
def main(gpu_num, prompt, negative_prompt, resolution, aspect_ratio, seed):
    """Text to video conversion using LongCat Video model"""
    pipe = get_pipeline(gpu_num)
    width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)

    # Run inference
    start = time.time()
    video = run(pipe, prompt, height, width, negative_prompt, seed)
    elapsed_time = time.time() - start

    print(f"Video generation time: {elapsed_time:.2f} seconds")

    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    save_video(video, output_path, fps=15, quality=6)
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
