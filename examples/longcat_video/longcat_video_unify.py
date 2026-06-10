import json
import os
import time

import click
import torch
from PIL import Image
from tqdm import tqdm
from transformers import UMT5EncoderModel

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.longcat_video import (
    LongCatVideoPipeline,
    LongCatVideoPipelineConfig,
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import (
    VideoData,
    get_target_image_size,
    get_target_video_size_from_ratio,
    save_video,
)

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="longcat_unify",
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

    pipe = LongCatVideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.init(module_manager, pipe_config)
    pipe.enable_cpu_offload()

    return pipe


def run(
    pipeline,
    prompt,
    height: int | None = None,
    width: int | None = None,
    first_image=None,
    input_video=None,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    aspect_ratio="16:9",
    return_latents=False,
    need_encode=True,
):
    """Unified video generation interface supporting t2v, i2v, and video continuation.
    Args:
        pipeline (LongCatVideoPipeline): Preloaded video generation pipeline object
        prompt (str): Positive guidance text prompt
        height (int | None): Video height. If None, computed from resolution/aspect_ratio.
        width (int | None): Video width. If None, computed from resolution/aspect_ratio.
        first_image (PIL.Image, optional): Input image for i2v mode
        input_video (VideoData, optional): Input video for continuation mode
        negative_prompt (str, optional): Negative guidance prompt. Default is empty
        seed (int, optional): Random seed. Default is 42
        resolution (str, optional): Target resolution. Default is PPL_CONFIG["resolution"]
        aspect_ratio (str, optional): Aspect ratio. Default is "16:9"
        return_latents (bool, optional): Whether to return latents. Default is False
        need_encode (bool, optional): Whether to encode input. Default is True

    Returns:
        tuple: (video frames, latents) if return_latents=True, else video frames only
    """
    if height is None or width is None:
        width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
    num_frames = 77 if input_video is None else 93

    video, latents = pipeline(
        input_image=first_image,
        input_video=input_video,
        prompt=prompt,
        num_frames=num_frames,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        attn_impl=PPL_CONFIG["attn_impl"],
        use_kv_cache=(first_image is not None or input_video is not None),
        return_latents=return_latents,
        need_encode=need_encode,
        target_fps=PPL_CONFIG["target_fps"] if PPL_CONFIG["target_fps"] > PPL_CONFIG["base_fps"] else None,
    )

    return (video, latents) if return_latents else video


def run_with_file(
    pipeline,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    output_path="output.mp4",
    resolution="480p",
    aspect_ratio="16:9",
    first_image_path=None,
    ref_video_path=None,
    **kwargs,
):
    """Service-compatible entry point for task-based execution."""
    if ref_video_path:
        # VC mode
        input_video = VideoData(ref_video_path)
        height, width = input_video.height, input_video.width
        video = run(
            pipeline, prompt, height, width, input_video=input_video, negative_prompt=negative_prompt, seed=seed
        )
    elif first_image_path:
        # I2V mode
        image = Image.open(first_image_path).convert("RGB")
        width, height = get_target_image_size(image.width, image.height, resolution)
        video = run(pipeline, prompt, height, width, first_image=image, negative_prompt=negative_prompt, seed=seed)
    else:
        # T2V mode
        width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
        video = run(pipeline, prompt, height, width, negative_prompt=negative_prompt, seed=seed)

    save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)


@click.command()
@click.option(
    "--task",
    type=click.Choice(["t2v", "i2v", "vc"]),
    default="t2v",
    help="Task type: t2v (text to video), i2v (image to video), vc (video continue)",
)
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option("--video_length", default=5, help="Video length in seconds (5, 10, 15)")
@click.option(
    "--resolution",
    default="720p",
    help="Video resolution (480p, 720p)",
)
@click.option(
    "--aspect_ratio",
    default="16:9",
    help="Aspect ratio (16:9, 9:16, 1:1)",
)
@click.option(
    "--prompt",
    default="A stylish woman walking down a Tokyo street filled with warm golden sunlight and cherry blossoms floating in the wind. The camera follows her from behind as she strolls leisurely, creating a cinematic atmosphere.",
    help="Positive guidance text prompt",
)
@click.option(
    "--image_path",
    default=f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png",
    help="Input image path for i2v task",
)
@click.option(
    "--video_path",
    default=f"{os.path.dirname(__file__)}/../data/sample_video.mp4",
    help="Input video path for vc task",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model root directory")
@click.option(
    "--benchmark",
    is_flag=True,
    help="Run benchmark test",
)
def main(
    task,
    gpu_num,
    video_length,
    resolution,
    aspect_ratio,
    prompt,
    image_path,
    video_path,
    negative_prompt,
    seed,
    model_root,
    benchmark,
):
    """Unified LongCat video generation interface"""
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
            cond = item["condition"]
            prompt_list = cond["prompt_en"]
            final_frames = []
            final_result_path = os.path.join(output_dir, f"{task}_{item['id']}_{video_length}.mp4")

            if os.path.exists(final_result_path):
                continue

            latents = None

            if task == "t2v":
                prompt = prompt_list["0"]
                width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
                start = time.time()
                video, latents = run(
                    pipe,
                    prompt=prompt,
                    height=height,
                    width=width,
                    return_latents=True,
                )
                print(f"T2V generation time: {time.time() - start:.2f} seconds")
                final_frames.extend(video)
                vc_len = video_length - 5

            elif task == "i2v":
                prompt = prompt_list["0"]
                image_path = cond["first_image"]["0"]
                image = Image.open(image_path).convert("RGB")
                width, height = get_target_image_size(image.width, image.height, resolution)
                start = time.time()
                video, latents = run(pipe, prompt, height, width, first_image=image, return_latents=True)
                print(f"I2V generation time: {time.time() - start:.2f} seconds")
                final_frames.extend(video)
                vc_len = video_length - 5

            elif task == "vc":
                ref_video_path = cond["ref_video"]["1"]
                input_video = VideoData(ref_video_path)
                final_frames.extend(input_video.raw_data())
                height = input_video.height
                width = input_video.width
                vc_len = video_length
            else:
                print(f"Unsupported task: {task}")
                return

            if vc_len == 0:
                save_video(final_frames, final_result_path, fps=PPL_CONFIG["target_fps"], quality=6)
                continue

            vc_num = vc_len // 5
            if task != "vc":
                input_video = latents

            for i in range(vc_num):
                prompt = prompt_list[str(i + 1)]
                start = time.time()
                video, latents = run(
                    pipe,
                    prompt,
                    height,
                    width,
                    input_video=input_video,
                    return_latents=True,
                    need_encode=(latents is None),
                )
                print(f"VC segment {i + 1} generation time: {time.time() - start:.2f} seconds")
                input_video = latents
                final_frames.extend(video)

            save_video(final_frames, final_result_path, fps=PPL_CONFIG["target_fps"], quality=6)
            print(f"Final video saved to: {final_result_path}")
    else:
        # Single generation based on task type
        filename = get_example_name(__file__).replace(".py", f"_{task}_{gpu_num}gpu.mp4")
        output_path = os.path.join(output_dir, filename)

        if task == "t2v":
            width, height = get_target_video_size_from_ratio(aspect_ratio, resolution)
            start = time.time()
            video = run(pipe, prompt, height, width, negative_prompt=negative_prompt, seed=seed)
            elapsed_time = time.time() - start
            print(f"T2V generation time: {elapsed_time:.2f} seconds")
            save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)
            print(f"Video saved to: {output_path}")

        elif task == "i2v":
            image = Image.open(image_path).convert("RGB")
            width, height = get_target_image_size(image.width, image.height, resolution)
            start = time.time()
            video = run(pipe, prompt, height, width, first_image=image, negative_prompt=negative_prompt, seed=seed)
            elapsed_time = time.time() - start
            print(f"I2V generation time: {elapsed_time:.2f} seconds")
            save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)
            print(f"Video saved to: {output_path}")

        elif task == "vc":
            input_video = VideoData(video_path)
            height, width = input_video.height, input_video.width
            start = time.time()
            video = run(
                pipe, prompt, height, width, input_video=input_video, negative_prompt=negative_prompt, seed=seed
            )
            elapsed_time = time.time() - start
            print(f"VC generation time: {elapsed_time:.2f} seconds")
            save_video(video, output_path, fps=PPL_CONFIG["target_fps"], quality=6)
            print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
