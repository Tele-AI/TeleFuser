import gc
import json
import os
import time

import click
import torch
from tqdm import tqdm
from transformers import UMT5EncoderModel

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.longcat_video import (
    LongCatVideoPipeline,
    LongCatVideoPipelineConfig,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import VideoData, get_target_video_size_from_ratio, save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="longcat_vc",
    model_root=TF_MODEL_ZOO_PATH + "/LongCat-Video",
    negative_prompt="色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走",
    num_inference_steps=50,
    num_frames=93,
    resolution="720p",
    cfg_scale=4.0,
    seed=42,
    tiled=False,
    attn_impl=AttnImplType.TORCH_SDPA,
    base_fps=15,
    target_fps=24,
    use_kv_cache=True,
    dit_filename_list=[
        "dit/diffusion_pytorch_model-00001-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00002-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00003-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00004-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00005-of-00006.safetensors",
        "dit/diffusion_pytorch_model-00006-of-00006.safetensors",
    ],
    vae_path=os.path.join(TF_MODEL_ZOO_PATH, "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
    vfi_model_path=os.path.join(TF_MODEL_ZOO_PATH, "RIFEv4.26_0921/flownet.pkl"),
)


def get_pipeline(parallelism: int = 1, model_root: str = PPL_CONFIG["model_root"]):
    """
    Args:
        parallelism (int): Number of parallel GPUs for inference: 1, 2, 4 or 8
        model_root (str): Root directory of the model files
    """
    # Build absolute paths from model_root
    dit_paths = [os.path.join(model_root, f) for f in PPL_CONFIG["dit_filename_list"]]

    # Load models
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

    # Load VFI model into the same module_manager
    module_manager.load_model(
        PPL_CONFIG["vfi_model_path"],
        torch_dtype=torch.bfloat16,
    )

    # Parallel configuration
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
        vfi_config=ModelRuntimeConfig(torch_dtype=torch.bfloat16),
        sample_solver="euler",
        enable_denoising_parallel=(parallelism > 1),
        enable_vfi=True,
    )

    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.ASYNC_CPU_OFFLOAD

    if parallelism > 1:
        pipe_config.dit_config.parallel_config.enable_fsdp = True
    pipe = LongCatVideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.init(module_manager, pipe_config)
    pipe.enable_cpu_offload()

    return pipe


def log_gpu_memory(tag: str, device_count: int = 1):
    """Log GPU memory usage for debugging."""
    for i in range(device_count):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        print(f"  [{tag}] GPU {i}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


def clear_gpu_cache(device_count: int = 1):
    """Force garbage collection and clear CUDA cache on all devices."""
    gc.collect()
    for i in range(device_count):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


def run(
    pipeline,
    input_video,
    prompt,
    height: int | None = None,
    width: int | None = None,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    aspect_ratio="16:9",
    return_latents=False,
    need_encode=True,
):
    """Video continuation from input video.
    Args:
        pipeline (LongCatVideoPipeline): Preloaded video generation pipeline object
        input_video (VideoData): Input video for continuation
        prompt (str): Positive guidance text prompt
        height (int | None): Video height. If None, computed from resolution/aspect_ratio.
        width (int | None): Video width. If None, computed from resolution/aspect_ratio.
        negative_prompt (str, optional): Negative guidance prompt. Default is empty
        seed (int, optional): Random seed. Default is 42
        resolution (str, optional): Target resolution. Default is PPL_CONFIG["resolution"]
        aspect_ratio (str, optional): Aspect ratio. Default is "16:9"
        return_latents (bool, optional): Whether to return latents. Default is False
        need_encode (bool, optional): Whether to encode input. Default is True

    Returns:
        List[PIL.Image] | tuple: Generated video, optionally with latents
    """
    if height is None or width is None:
        width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
    video, latents = pipeline(
        input_video=input_video,
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
        use_kv_cache=PPL_CONFIG["use_kv_cache"],
        return_latents=return_latents,
        need_encode=need_encode,
        target_fps=PPL_CONFIG["target_fps"] if PPL_CONFIG["target_fps"] > PPL_CONFIG["base_fps"] else None,
    )

    if return_latents:
        return video, latents
    return video


def run_with_file(
    pipeline,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    output_path="output.mp4",
    ref_video_path="",
    **kwargs,
):
    """Service-compatible entry point for task-based execution."""
    input_video = VideoData(ref_video_path)
    height, width = input_video.height, input_video.width
    video = run(pipeline, input_video, prompt, height, width, negative_prompt, seed)
    save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--video_path",
    default=f"{os.path.dirname(__file__)}/../data/sample_video.mp4",
    help="Input video path",
)
@click.option(
    "--prompt",
    default="The camera continues to follow the stylish woman as she walks through the cherry blossom-filled Tokyo street, capturing more of the warm golden sunlight and the gentle breeze carrying petals through the air.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model root directory")
@click.option(
    "--benchmark",
    is_flag=True,
    help="Run benchmark test with multiple video segments",
)
def main(gpu_num, video_path, prompt, negative_prompt, seed, model_root, benchmark):
    """Video continuation using LongCat Video model"""
    pipe = get_pipeline(gpu_num, model_root)
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")

    if benchmark:
        # Run benchmark test
        benchmark_file = "/nvfile-heatstorage/AIGC_H100/ICET/cjf/TeleFuser/examples/data/ContinueBmk_2_translated.json"
        if not os.path.exists(benchmark_file):
            print(f"Benchmark file not found: {benchmark_file}")
            return

        with open(benchmark_file) as f:
            bmk = json.load(f)

        for item in tqdm(bmk["items"]):
            item_id = item["id"]
            log_gpu_memory(f"Before item {item_id}", device_count=gpu_num)

            cond = item["condition"]
            prompt_list = cond["prompt_en"]
            ref_video_path = cond["ref_video"]["1"]
            all_frames = []

            latents = None
            for i in range(3):
                if i == 0:
                    input_video = VideoData(ref_video_path)
                    height, width = input_video.height, input_video.width
                    all_frames.extend(input_video.raw_data())
                    need_encode = True
                else:
                    input_video = latents
                    need_encode = False

                prompt = prompt_list[str(i + 1)]
                start = time.time()
                video, latents = run(
                    pipe, input_video, prompt, height, width, return_latents=True, need_encode=need_encode
                )
                elapsed_time = time.time() - start
                print(f"Segment {i + 1} generation time: {elapsed_time:.2f} seconds")

                all_frames.extend(video)

                # Clean up intermediate references after each segment
                del video
                if i > 0:
                    del input_video
                gc.collect()

            result_path = os.path.join(output_dir, f"{item_id}_all.mp4")
            save_video(all_frames, result_path, fps=PPL_CONFIG["target_fps"], quality=6)
            print(f"Final video saved to: {result_path}")

            # === Critical: clean up between items ===
            del all_frames
            del latents
            clear_gpu_cache(device_count=gpu_num)
            log_gpu_memory(f"After cleanup item {item_id}", device_count=gpu_num)

    else:
        # Single video continuation
        input_video = VideoData(video_path)
        height, width = input_video.height, input_video.width

        # Run inference
        start = time.time()
        video = run(pipe, input_video, prompt, height, width, negative_prompt, seed)
        elapsed_time = time.time() - start

        print(f"Video generation time: {elapsed_time:.2f} seconds")

        # Save results
        output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
        filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
        output_path = os.path.join(output_dir, filename)

        save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)
        print(f"Video saved to: {output_path}")

    del pipe
    clear_gpu_cache(device_count=gpu_num)


if __name__ == "__main__":
    main()
