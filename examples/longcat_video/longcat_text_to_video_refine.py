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

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="longcat_t2v_refine",
    model_root=TF_MODEL_ZOO_PATH + "/LongCat-Video",
    negative_prompt="色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走",
    num_inference_steps=50,
    num_frames=93,
    cfg_scale=4.0,
    seed=42,
    tiled=False,
    target_fps=15,
    attn_impl=AttnImplType.TORCH_SDPA,
    dit_filename_list=[
        "dit/diffusion_pytorch_model-00001-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00002-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00003-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00004-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00005-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00006-of-00006.safetensors",
    ],
    refinement_lora_filename="lora/refinement_lora.safetensors",
    vae_path=os.path.join(TF_MODEL_ZOO_PATH, "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
    # Refine defaults
    refine_height=720,
    refine_width=1280,
    refine_num_steps=50,
    refine_t_thresh=0.5,
    enable_refine_bsa=False,
    refine_num_frames=None,
    spatial_refine_only=False,
)


def get_pipeline(parallelism: int = 1, model_root: str = PPL_CONFIG["model_root"]):
    """Build LongCat pipeline with refinement enabled.

    Args:
        parallelism: Number of parallel GPUs (1, 2, 4, or 8).
        model_root: Root directory of the model files.
    """
    # Build absolute paths from model_root
    dit_paths = [os.path.join(model_root, f) for f in PPL_CONFIG["dit_filename_list"]]
    refinement_lora_path = os.path.join(model_root, PPL_CONFIG["refinement_lora_filename"])

    module_manager = ModuleManager(device="cpu")

    module_manager.load_model(
        PPL_CONFIG["vae_path"],
        torch_dtype=torch.bfloat16,
    )

    module_manager.load_model(
        dit_paths,
        torch_dtype=torch.bfloat16,
    )

    module_manager.load_from_huggingface(
        os.path.join(model_root, "tokenizer"),
        module_source="transformers",
        module_name="longcat_tokenizer",
    )

    module_manager.load_from_huggingface(
        os.path.join(model_root, "text_encoder"),
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

    dit_config = ModelRuntimeConfig(torch_dtype=torch.bfloat16)
    dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    dit_config.parallel_config = parallel_config

    refine_dit_config = ModelRuntimeConfig(torch_dtype=torch.bfloat16)
    refine_dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    refine_dit_config.parallel_config = parallel_config

    pipe_config = LongCatVideoPipelineConfig(
        vae_config=ModelRuntimeConfig(torch_dtype=torch.bfloat16),
        dit_config=dit_config,
        text_encoding_config=ModelRuntimeConfig(torch_dtype=torch.bfloat16),
        sample_solver="euler",
        enable_denoising_parallel=(parallelism > 1),
        enable_refine=True,
        refine_dit_config=refine_dit_config,
        refine_num_steps=PPL_CONFIG["refine_num_steps"],
        refine_t_thresh=PPL_CONFIG["refine_t_thresh"],
        refine_lora_path=refinement_lora_path,
        enable_refine_bsa=PPL_CONFIG["enable_refine_bsa"],
        refine_num_frames=PPL_CONFIG["refine_num_frames"],
        spatial_refine_only=PPL_CONFIG["spatial_refine_only"],
    )

    pipe = LongCatVideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.init(module_manager, pipe_config)
    pipe.enable_cpu_offload()

    return pipe


def run(
    pipeline: LongCatVideoPipeline,
    prompt: str,
    height: int | None = None,
    width: int | None = None,
    negative_prompt: str = "",
    seed: int = PPL_CONFIG["seed"],
    resolution: str = "720p",
    aspect_ratio: str = "16:9",
):
    """Generate video at base resolution, then refine to higher resolution.

    Args:
        pipeline: Preloaded LongCat video pipeline with refinement enabled.
        prompt: Text prompt.
        height: Base generation height. If None, computed from resolution/aspect_ratio.
        width: Base generation width. If None, computed from resolution/aspect_ratio.
        negative_prompt: Negative guidance prompt.
        seed: Random seed.
        resolution: Target resolution (e.g. "480p", "720p"). Default is "720p".
        aspect_ratio: Aspect ratio (e.g. "16:9"). Default is "16:9".

    Returns:
        List[PIL.Image]: Generated and refined video frames.
    """
    if height is None or width is None:
        width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
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
        enable_refine=True,
        refine_height=PPL_CONFIG["refine_height"],
        refine_width=PPL_CONFIG["refine_width"],
        refine_num_steps=PPL_CONFIG["refine_num_steps"],
        refine_t_thresh=PPL_CONFIG["refine_t_thresh"],
        refine_num_frames=PPL_CONFIG["refine_num_frames"],
        enable_refine_bsa=PPL_CONFIG["enable_refine_bsa"],
        spatial_refine_only=PPL_CONFIG["spatial_refine_only"],
    )
    return video


def run_with_file(
    pipeline: LongCatVideoPipeline,
    prompt: str,
    negative_prompt: str = "",
    seed: int = PPL_CONFIG["seed"],
    output_path: str = "output.mp4",
    resolution: str = "480p",
    aspect_ratio: str = "16:9",
    **kwargs,
):
    """Service-compatible entry point for task-based execution."""
    width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
    video = run(pipeline, prompt, height, width, negative_prompt, seed)
    save_video(video, output_path, fps=15, quality=6)


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use")
@click.option("--height", default=480, help="Base generation height")
@click.option("--width", default=832, help="Base generation width")
@click.option("--refine_height", default=PPL_CONFIG["refine_height"], help="Refine target height")
@click.option("--refine_width", default=PPL_CONFIG["refine_width"], help="Refine target width")
@click.option("--refine_num_steps", default=PPL_CONFIG["refine_num_steps"], help="Refine denoising steps")
@click.option("--refine_t_thresh", default=PPL_CONFIG["refine_t_thresh"], help="Refine noise threshold [0,1]")
@click.option("--enable_bsa", is_flag=True, default=False, help="Enable block sparse attention for refine")
@click.option("--refine_num_frames", default=None, type=int, help="Target frame count for temporal refine (None=auto)")
@click.option(
    "--spatial_refine_only", is_flag=True, default=False, help="Only do spatial upsampling (no temporal frame doubling)"
)
@click.option(
    "--prompt",
    default="A small boat is bravely battling the waves, forging ahead. The vast blue sea is tumultuous, with white spray crashing against the hull, but the little boat shows no fear, steadfastly sailing towards the distant horizon.",
    help="Text prompt",
)
@click.option("--negative_prompt", default="", help="Negative prompt")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model root directory")
def main(
    gpu_num,
    height,
    width,
    refine_height,
    refine_width,
    refine_num_steps,
    refine_t_thresh,
    enable_bsa,
    refine_num_frames,
    spatial_refine_only,
    prompt,
    negative_prompt,
    seed,
    model_root,
):
    """Text-to-Video with official LongCat refinement.

    Generates video at base resolution (e.g. 480p), then refines to higher
    resolution (e.g. 720p) using refinement LoRA. Optionally enables BSA
    acceleration and temporal frame upsampling.
    """
    # Override PPL_CONFIG with CLI args
    PPL_CONFIG["refine_height"] = refine_height
    PPL_CONFIG["refine_width"] = refine_width
    PPL_CONFIG["refine_num_steps"] = refine_num_steps
    PPL_CONFIG["refine_t_thresh"] = refine_t_thresh
    PPL_CONFIG["enable_refine_bsa"] = enable_bsa
    PPL_CONFIG["refine_num_frames"] = refine_num_frames
    PPL_CONFIG["spatial_refine_only"] = spatial_refine_only

    pipe = get_pipeline(gpu_num, model_root)

    start = time.time()
    video = run(pipe, prompt, height, width, negative_prompt, seed)
    elapsed = time.time() - start

    print(f"Video generation time (base + refine): {elapsed:.2f} seconds")
    print(f"Base: {width}x{height} -> Refine: {refine_width}x{refine_height}")

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__)
    output_path = os.path.join(output_dir, filename)

    # Temporal refine doubles frames → output at 30fps; spatial-only stays at 15fps
    output_fps = 15 if spatial_refine_only else 30
    save_video(video, output_path, fps=output_fps, quality=6)
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
